"""
Deepfake Tool Testing Module

Tests PrivacyShield protection against real deepfake pipelines:
  1. InsightFace inswapper_128 (used by Roop/ReaFace)
  2. IP-Adapter FaceID Plus v2 (generative deepfakes via Stable Diffusion)

Both tools are lazy-loaded on first call to avoid heavy startup costs.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import cv2


@dataclass
class DeepfakeToolResult:
    """Result from a single deepfake tool test."""
    tool_name: str
    clean_output: Optional[np.ndarray] = None       # BGR output image from clean input
    protected_output: Optional[np.ndarray] = None    # BGR output image from protected input
    clean_similarity: float = 0.0                    # Cosine sim: clean_output vs clean_identity
    protected_similarity: float = 0.0                # Cosine sim: protected_output vs clean_identity
    protection_effective: bool = False                # True if protected_similarity < threshold
    error: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class DeepfakeTestResult:
    """Combined result from all deepfake tool tests."""
    inswapper: Optional[DeepfakeToolResult] = None
    ipadapter: Optional[DeepfakeToolResult] = None
    overall_verdict: str = "untested"


class DeepfakeTestRegistry:
    """
    Lazy-loaded registry for deepfake testing tools.

    Models are loaded on first use, not at import or startup.
    """

    def __init__(self):
        self._inswapper_model = None
        self._face_analysis = None
        self._ipadapter_pipe = None
        self._ipadapter_available = None  # None = unchecked, True/False = checked

    @property
    def inswapper_loaded(self) -> bool:
        return self._inswapper_model is not None

    @property
    def ipadapter_loaded(self) -> bool:
        return self._ipadapter_pipe is not None

    def _ensure_face_analysis(self):
        """Load InsightFace FaceAnalysis (shared by inswapper and embedding extraction)."""
        if self._face_analysis is not None:
            return self._face_analysis

        from insightface.app import FaceAnalysis
        self._face_analysis = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
        )
        self._face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_analysis

    def _ensure_inswapper(self):
        """Load inswapper_128 model from weights directory."""
        if self._inswapper_model is not None:
            return self._inswapper_model

        import insightface
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
        model_path = os.path.join(weights_dir, "inswapper_128.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"inswapper_128.onnx not found at {model_path}. "
                "Download from HuggingFace: "
                "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
            )

        self._inswapper_model = insightface.model_zoo.get_model(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._ensure_face_analysis()
        print(f"Inswapper loaded from {model_path}")
        return self._inswapper_model

    def _ensure_ipadapter(self):
        """Load IP-Adapter FaceID Plus v2 pipeline."""
        if self._ipadapter_pipe is not None:
            return self._ipadapter_pipe

        try:
            from diffusers import StableDiffusionPipeline, DDIMScheduler
        except ImportError:
            raise ImportError(
                "IP-Adapter requires: pip install diffusers transformers accelerate"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print("Loading Stable Diffusion 1.5...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        # Load FaceID Plus v2 LoRA weights (required for FaceID Plus variants)
        print("Loading FaceID Plus v2 LoRA weights...")
        pipe.load_lora_weights(
            "h94/IP-Adapter-FaceID",
            weight_name="ip-adapter-faceid-plusv2_sd15_lora.safetensors",
        )
        pipe.fuse_lora()

        pipe.to(device)

        # Load IP-Adapter FaceID Plus v2 weights from HF repo
        # image_encoder_folder=None skips auto-loading CLIP (we load it separately)
        print("Loading IP-Adapter FaceID Plus v2 weights...")
        pipe.load_ip_adapter(
            "h94/IP-Adapter-FaceID",
            subfolder=None,
            weight_name="ip-adapter-faceid-plusv2_sd15.bin",
            image_encoder_folder=None,
        )
        pipe.set_ip_adapter_scale(0.7)

        # Load CLIP image encoder for the "Plus" face image path
        print("Loading CLIP ViT-H image encoder...")
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        self._clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        ).to(device, dtype=dtype)
        self._clip_image_processor = CLIPImageProcessor()

        self._ipadapter_pipe = pipe
        print("IP-Adapter FaceID Plus v2 loaded successfully")
        return self._ipadapter_pipe

    def check_ipadapter_available(self) -> bool:
        """Check if diffusers is installed without loading models."""
        if self._ipadapter_available is not None:
            return self._ipadapter_available
        try:
            import diffusers  # noqa: F401
            import transformers  # noqa: F401
            self._ipadapter_available = True
        except ImportError:
            self._ipadapter_available = False
        return self._ipadapter_available

    def _detect_face(self, img_bgr: np.ndarray):
        """Detect the largest face in a BGR image. Returns insightface Face object or None."""
        app = self._ensure_face_analysis()
        faces = app.get(img_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def _get_embedding_from_face(self, face) -> Optional[np.ndarray]:
        """Extract normalized embedding from an insightface Face object."""
        if face is None or not hasattr(face, "normed_embedding"):
            return None
        emb = face.normed_embedding
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _cosine_sim(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two normalized embeddings."""
        return float(np.dot(emb1, emb2))

    def run_inswapper_test(
        self,
        clean_bgr: np.ndarray,
        protected_bgr: np.ndarray,
        target_bgr: np.ndarray,
        threshold: float = 0.3,
    ) -> DeepfakeToolResult:
        """
        Test inswapper face swap with clean vs protected source images.

        Args:
            clean_bgr: Original unprotected face (BGR)
            protected_bgr: Protected face (BGR)
            target_bgr: Target image to swap face onto (BGR)
            threshold: Cosine similarity threshold for protection

        Returns:
            DeepfakeToolResult with swap outputs and similarity metrics
        """
        start = time.time()
        try:
            swapper = self._ensure_inswapper()

            # Detect faces in all images
            clean_face = self._detect_face(clean_bgr)
            protected_face = self._detect_face(protected_bgr)
            target_face = self._detect_face(target_bgr)

            if clean_face is None:
                return DeepfakeToolResult(tool_name="inswapper", error="No face detected in clean image")
            if protected_face is None:
                return DeepfakeToolResult(tool_name="inswapper", error="No face detected in protected image")
            if target_face is None:
                return DeepfakeToolResult(tool_name="inswapper", error="No face detected in target image")

            # Get clean identity embedding for comparison
            clean_emb = self._get_embedding_from_face(clean_face)
            if clean_emb is None:
                return DeepfakeToolResult(tool_name="inswapper", error="Could not extract clean embedding")

            # Swap clean face onto target
            swap_from_clean = target_bgr.copy()
            swap_from_clean = swapper.get(swap_from_clean, target_face, clean_face, paste_back=True)

            # Swap protected face onto target
            swap_from_protected = target_bgr.copy()
            swap_from_protected = swapper.get(swap_from_protected, target_face, protected_face, paste_back=True)

            # Detect faces in swap outputs and compare to clean identity
            clean_swap_face = self._detect_face(swap_from_clean)
            protected_swap_face = self._detect_face(swap_from_protected)

            clean_sim = 0.0
            protected_sim = 0.0

            if clean_swap_face is not None:
                clean_swap_emb = self._get_embedding_from_face(clean_swap_face)
                if clean_swap_emb is not None:
                    clean_sim = self._cosine_sim(clean_emb, clean_swap_emb)

            if protected_swap_face is not None:
                protected_swap_emb = self._get_embedding_from_face(protected_swap_face)
                if protected_swap_emb is not None:
                    protected_sim = self._cosine_sim(clean_emb, protected_swap_emb)

            elapsed = (time.time() - start) * 1000
            return DeepfakeToolResult(
                tool_name="inswapper",
                clean_output=swap_from_clean,
                protected_output=swap_from_protected,
                clean_similarity=round(clean_sim, 4),
                protected_similarity=round(protected_sim, 4),
                protection_effective=protected_sim < threshold,
                processing_time_ms=round(elapsed, 1),
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return DeepfakeToolResult(
                tool_name="inswapper",
                error=str(e),
                processing_time_ms=round(elapsed, 1),
            )

    def run_ipadapter_test(
        self,
        clean_bgr: np.ndarray,
        protected_bgr: np.ndarray,
        prompt: str = "a photo of a person",
        threshold: float = 0.3,
        seed: int = 42,
        num_inference_steps: int = 30,
    ) -> DeepfakeToolResult:
        """
        Test IP-Adapter FaceID Plus v2 with clean vs protected face embeddings.

        Args:
            clean_bgr: Original unprotected face (BGR)
            protected_bgr: Protected face (BGR)
            prompt: Text prompt for generation
            threshold: Cosine similarity threshold
            seed: Fixed seed for reproducible comparison
            num_inference_steps: Diffusion steps

        Returns:
            DeepfakeToolResult with generated outputs and similarity metrics
        """
        start = time.time()
        try:
            pipe = self._ensure_ipadapter()
            self._ensure_face_analysis()

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Detect faces and get embeddings
            clean_face = self._detect_face(clean_bgr)
            protected_face = self._detect_face(protected_bgr)

            if clean_face is None:
                return DeepfakeToolResult(tool_name="ipadapter", error="No face detected in clean image")
            if protected_face is None:
                return DeepfakeToolResult(tool_name="ipadapter", error="No face detected in protected image")

            clean_emb = self._get_embedding_from_face(clean_face)
            protected_emb = self._get_embedding_from_face(protected_face)

            if clean_emb is None:
                return DeepfakeToolResult(tool_name="ipadapter", error="Could not extract clean embedding")
            if protected_emb is None:
                return DeepfakeToolResult(tool_name="ipadapter", error="Could not extract protected embedding")

            # Face ID embeddings: (batch, 1, 512) for the FaceID projection
            dtype = next(pipe.unet.parameters()).dtype
            clean_id = torch.from_numpy(clean_emb).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
            prot_id = torch.from_numpy(protected_emb).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)

            # CLIP hidden states for the Plus path: (batch, 257, 1280) from penultimate layer
            from PIL import Image as PILImage
            clean_face_pil = PILImage.fromarray(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))
            protected_face_pil = PILImage.fromarray(cv2.cvtColor(protected_bgr, cv2.COLOR_BGR2RGB))

            clip_clean = self._clip_image_processor(images=clean_face_pil, return_tensors="pt").pixel_values.to(device, dtype=dtype)
            clip_prot = self._clip_image_processor(images=protected_face_pil, return_tensors="pt").pixel_values.to(device, dtype=dtype)

            with torch.no_grad():
                clean_clip_hidden = self._clip_image_encoder(clip_clean, output_hidden_states=True).hidden_states[-2]
                prot_clip_hidden = self._clip_image_encoder(clip_prot, output_hidden_states=True).hidden_states[-2]

            # For classifier-free guidance: concat [negative_zeros, positive] on dim=0
            neg_id = torch.zeros_like(clean_id)
            clean_id_cfg = torch.cat([neg_id, clean_id], dim=0)    # (2, 1, 512)
            prot_id_cfg = torch.cat([neg_id, prot_id], dim=0)      # (2, 1, 512)

            proj_layer = pipe.unet.encoder_hid_proj.image_projection_layers[0]

            # clip_embeds must be 4D: (batch, num_images, seq_len, hidden_dim)
            # Generate from clean face identity
            neg_clip = torch.zeros_like(clean_clip_hidden)
            proj_layer.clip_embeds = torch.cat(
                [neg_clip.unsqueeze(1), clean_clip_hidden.unsqueeze(1)], dim=0
            )  # (2, 1, 257, 1280)
            gen_clean = pipe(
                prompt=prompt,
                ip_adapter_image_embeds=[clean_id_cfg],
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=device).manual_seed(seed),
            ).images[0]

            # Generate from protected face identity (same seed for fair comparison)
            neg_clip = torch.zeros_like(prot_clip_hidden)
            proj_layer.clip_embeds = torch.cat(
                [neg_clip.unsqueeze(1), prot_clip_hidden.unsqueeze(1)], dim=0
            )  # (2, 1, 257, 1280)
            gen_protected = pipe(
                prompt=prompt,
                ip_adapter_image_embeds=[prot_id_cfg],
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=device).manual_seed(seed),
            ).images[0]

            # Convert PIL to BGR for face detection
            gen_clean_bgr = cv2.cvtColor(np.array(gen_clean), cv2.COLOR_RGB2BGR)
            gen_protected_bgr = cv2.cvtColor(np.array(gen_protected), cv2.COLOR_RGB2BGR)

            # Detect faces in generated images and compare to clean identity
            gen_clean_face = self._detect_face(gen_clean_bgr)
            gen_protected_face = self._detect_face(gen_protected_bgr)

            clean_sim = 0.0
            protected_sim = 0.0

            if gen_clean_face is not None:
                gen_clean_emb = self._get_embedding_from_face(gen_clean_face)
                if gen_clean_emb is not None:
                    clean_sim = self._cosine_sim(clean_emb, gen_clean_emb)

            if gen_protected_face is not None:
                gen_protected_emb = self._get_embedding_from_face(gen_protected_face)
                if gen_protected_emb is not None:
                    protected_sim = self._cosine_sim(clean_emb, gen_protected_emb)

            elapsed = (time.time() - start) * 1000
            return DeepfakeToolResult(
                tool_name="ipadapter",
                clean_output=gen_clean_bgr,
                protected_output=gen_protected_bgr,
                clean_similarity=round(clean_sim, 4),
                protected_similarity=round(protected_sim, 4),
                protection_effective=protected_sim < threshold,
                processing_time_ms=round(elapsed, 1),
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return DeepfakeToolResult(
                tool_name="ipadapter",
                error=str(e),
                processing_time_ms=round(elapsed, 1),
            )

    def run_full_test(
        self,
        clean_bgr: np.ndarray,
        protected_bgr: np.ndarray,
        target_bgr: Optional[np.ndarray] = None,
        run_inswapper: bool = True,
        run_ipadapter: bool = True,
        prompt: str = "a photo of a person",
        threshold: float = 0.3,
    ) -> DeepfakeTestResult:
        """
        Run all enabled deepfake tool tests.

        Args:
            clean_bgr: Original unprotected face (BGR)
            protected_bgr: Protected face (BGR)
            target_bgr: Target image for face swap (defaults to clean_bgr)
            run_inswapper: Whether to test inswapper
            run_ipadapter: Whether to test IP-Adapter
            prompt: Text prompt for IP-Adapter generation
            threshold: Cosine similarity threshold

        Returns:
            DeepfakeTestResult with per-tool results and overall verdict
        """
        if target_bgr is None:
            target_bgr = clean_bgr

        result = DeepfakeTestResult()

        if run_inswapper:
            result.inswapper = self.run_inswapper_test(
                clean_bgr, protected_bgr, target_bgr, threshold
            )

        if run_ipadapter:
            if self.check_ipadapter_available():
                result.ipadapter = self.run_ipadapter_test(
                    clean_bgr, protected_bgr, prompt, threshold
                )
            else:
                result.ipadapter = DeepfakeToolResult(
                    tool_name="ipadapter",
                    error="diffusers not installed (pip install diffusers transformers accelerate)",
                )

        # Determine overall verdict
        tested = []
        if result.inswapper and result.inswapper.error is None:
            tested.append(result.inswapper.protection_effective)
        if result.ipadapter and result.ipadapter.error is None:
            tested.append(result.ipadapter.protection_effective)

        if not tested:
            result.overall_verdict = "error"
        elif all(tested):
            result.overall_verdict = "protected"
        elif any(tested):
            result.overall_verdict = "partial"
        else:
            result.overall_verdict = "vulnerable"

        return result
