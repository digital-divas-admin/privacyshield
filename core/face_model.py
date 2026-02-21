"""
Face Embedding Model Wrapper

Wraps ArcFace into a fully differentiable PyTorch module so gradients
flow from the embedding loss back through the input image.

Two backends:
  1. "arcface_torch" — native PyTorch ArcFace (preferred for training/attack)
  2. "insightface"   — InsightFace ONNX (for evaluation/comparison only)

The key insight from PhotoGuard/Anti-DreamBooth: we need the face recognition
model to be fully differentiable. InsightFace's ONNX runtime doesn't support
backprop, so we use a PyTorch re-implementation of ArcFace (iresnet100) with
the same pretrained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# ArcFace backbone: IResNet (Improved ResNet for face recognition)
# Architecture matches InsightFace's iresnet100 used in buffalo_l
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))  # Global average pool
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.view(b, c, 1, 1)


class IResBlock(nn.Module):
    """IResNet residual block with optional SE."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet100(nn.Module):
    """
    IResNet-100 backbone for ArcFace.
    Input:  (B, 3, 112, 112) — aligned face images, pixel range [0, 1]
    Output: (B, 512) — L2-normalized face embeddings
    """
    def __init__(self, embedding_dim: int = 512, use_se: bool = True):
        super().__init__()
        # Layer config for iresnet100: [3, 13, 30, 3]
        layers = [3, 13, 30, 3]
        channels = [64, 128, 256, 512]

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU(64)

        # Build residual stages
        self.layer1 = self._make_layer(64, channels[0], layers[0], stride=2, use_se=use_se)
        self.layer2 = self._make_layer(channels[0], channels[1], layers[1], stride=2, use_se=use_se)
        self.layer3 = self._make_layer(channels[1], channels[2], layers[2], stride=2, use_se=use_se)
        self.layer4 = self._make_layer(channels[2], channels[3], layers[3], stride=2, use_se=use_se)

        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride, use_se):
        layers = [IResBlock(in_ch, out_ch, stride=stride, use_se=use_se)]
        for _ in range(1, num_blocks):
            layers.append(IResBlock(out_ch, out_ch, stride=1, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) in [0, 1] range
        Returns:
            (B, 512) L2-normalized embeddings
        """
        # Normalize to [-1, 1] as ArcFace expects
        x = (x - 0.5) / 0.5

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.bn3(x)
        return F.normalize(x, dim=1)


# ---------------------------------------------------------------------------
# Face detection + alignment (uses InsightFace for detection only)
# ---------------------------------------------------------------------------

class FaceAligner:
    """
    Detect and align faces to 112x112 for ArcFace input.
    Uses InsightFace's RetinaFace detector but alignment is done in
    a differentiable way with kornia for the attack pipeline.
    """

    def __init__(self, det_size: Tuple[int, int] = (640, 640)):
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection"],
            )
            self.app.prepare(ctx_id=0, det_size=det_size)
            self._available = True
        except ImportError:
            self._available = False
            print("Warning: insightface not installed. Using center-crop alignment fallback.")

    def detect_and_align(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect face in BGR image, return aligned 112x112 crop and affine matrix.

        Returns:
            aligned: (112, 112, 3) aligned face in RGB, or None
            M: (2, 3) affine matrix used for alignment, or None
        """
        if not self._available:
            return self._center_crop_fallback(image)

        import cv2
        from insightface.utils.face_align import norm_crop

        faces = self.app.get(image)
        if len(faces) == 0:
            return None, None

        # Pick largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        kps = face.kps  # (5, 2) landmarks

        # Standard alignment using insightface's norm_crop
        aligned = norm_crop(image, kps, image_size=112)
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        # Compute affine matrix for differentiable warping
        from skimage.transform import SimilarityTransform
        dst_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

        tform = SimilarityTransform()
        tform.estimate(kps, dst_pts)
        M = tform.params[:2]

        return aligned_rgb, M.astype(np.float32)

    @staticmethod
    def _center_crop_fallback(
        image: np.ndarray,
    ) -> Tuple[np.ndarray, None]:
        """Simple center-crop fallback when detection is unavailable."""
        import cv2
        h, w = image.shape[:2]
        s = min(h, w)
        y, x = (h - s) // 2, (w - s) // 2
        crop = image[y : y + s, x : x + s]
        aligned = cv2.resize(crop, (112, 112))
        if len(aligned.shape) == 3 and aligned.shape[2] == 3:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        return aligned, None


# ---------------------------------------------------------------------------
# Combined differentiable face embedder
# ---------------------------------------------------------------------------

class FaceEmbedder(nn.Module):
    """
    Full pipeline: takes a (B, 3, H, W) image tensor, extracts face region,
    and returns (B, 512) embeddings — all differentiable.

    For the attack loop, images should already be aligned to 112x112.
    The align_from_numpy() method handles detection + alignment for new images.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        embedding_dim: int = 512,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Official ArcFace iresnet100 weights don't include SE blocks
        self.backbone = IResNet100(embedding_dim=embedding_dim, use_se=False)
        self.device = device
        self.aligner = FaceAligner()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=device, weights_only=True)
            # Remap keys from official InsightFace naming to ours
            state = self._remap_state_dict(state)
            self.backbone.load_state_dict(state, strict=False)
            print(f"Loaded ArcFace weights from {weights_path}")
        else:
            print(
                "No weights loaded — using random init. "
                "Download ArcFace weights for real protection."
            )

        self.backbone = self.backbone.to(device)
        self.backbone.eval()
        # Freeze — we never train the face model
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) aligned face tensor in [0, 1]
        Returns:
            (B, 512) L2-normalized embeddings
        """
        return self.backbone(x)

    @torch.no_grad()
    def align_from_numpy(self, image_bgr: np.ndarray) -> Optional[torch.Tensor]:
        """
        Detect, align, and convert a single BGR image to tensor.
        Returns: (1, 3, 112, 112) tensor in [0, 1] or None if no face found.
        """
        aligned, _ = self.aligner.detect_and_align(image_bgr)
        if aligned is None:
            return None
        # (112, 112, 3) uint8 RGB -> (1, 3, 112, 112) float [0,1]
        tensor = torch.from_numpy(aligned).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    @staticmethod
    def _remap_state_dict(state: dict) -> dict:
        """Remap official InsightFace arcface_torch key names to ours."""
        remapped = {}
        for k, v in state.items():
            new_key = k
            if k == "prelu.weight":
                new_key = "prelu1.weight"
            elif k.startswith("features."):
                new_key = k.replace("features.", "bn3.")
            remapped[new_key] = v
        return remapped

    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between two embedding batches."""
        return F.cosine_similarity(emb1, emb2, dim=1)


# ---------------------------------------------------------------------------
# FaceNet Wrapper (InceptionResNet-V1, pretrained on VGGFace2)
# ---------------------------------------------------------------------------

class FaceNetWrapper(nn.Module):
    """
    Wraps facenet_pytorch's InceptionResnetV1 for use as an ensemble member.

    FaceNet expects 160x160 input, but all callers pass 112x112.
    The resize happens internally via differentiable F.interpolate(bilinear),
    so gradients flow through it — no caller changes needed.

    Weights auto-download from facenet_pytorch on first use.
    """

    def __init__(self, device: Optional[str] = None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = None
        self._available = False

        try:
            from facenet_pytorch import InceptionResnetV1
            self._model = InceptionResnetV1(pretrained="vggface2").to(device).eval()
            for p in self._model.parameters():
                p.requires_grad = False
            self._available = True
            print("FaceNet (InceptionResnetV1-VGGFace2) loaded")
        except ImportError:
            print("Warning: facenet_pytorch not installed. FaceNet unavailable.")
        except Exception as e:
            print(f"Warning: FaceNet load failed: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) aligned face in [0, 1]
        Returns:
            (B, 512) L2-normalized embeddings
        """
        # Resize 112 -> 160 (FaceNet's expected input size)
        if x.shape[2] != 160 or x.shape[3] != 160:
            x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        # FaceNet expects [-1, 1] normalized input
        x = (x - 0.5) / 0.5
        emb = self._model(x)
        return F.normalize(emb, dim=1)


# ---------------------------------------------------------------------------
# AdaFace Wrapper (IR-101 backbone)
# ---------------------------------------------------------------------------

class AdaFaceWrapper(nn.Module):
    """
    Wraps the vendored AdaFace IR-101 backbone.

    Loads weights from a checkpoint file. Input is 112x112 (same as ArcFace).
    Returns (B, 512) L2-normalized embeddings.
    """

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = None
        self._available = False

        try:
            from .adaface_backbone import build_adaface_ir101
            # CVLFace checkpoint doesn't include SE blocks, so build without them
            self._model = build_adaface_ir101(embedding_dim=512, use_se=False)

            if weights_path is not None:
                state = torch.load(weights_path, map_location=device, weights_only=False)
                # AdaFace checkpoints may wrap state_dict under 'state_dict' key
                if "state_dict" in state:
                    state = state["state_dict"]
                # Strip 'model.' or 'net.' prefix if present
                state = {k.replace("model.", "").replace("net.", ""): v for k, v in state.items()}
                # Remap CVLFace format to our backbone format
                state = self._remap_cvlface_state_dict(state)
                missing, unexpected = self._model.load_state_dict(state, strict=False)
                if missing:
                    print(f"AdaFace: {len(missing)} missing keys (expected for head layers)")
                if unexpected:
                    print(f"AdaFace: {len(unexpected)} unexpected keys")
                print(f"AdaFace IR-101 weights loaded from {weights_path}")
                self._available = True
            else:
                print("AdaFace IR-101 initialized (no weights — random init, excluded from ensemble)")

            self._model = self._model.to(device).eval()
            for p in self._model.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"Warning: AdaFace load failed: {e}")

    @staticmethod
    def _remap_cvlface_state_dict(state: dict) -> dict:
        """Remap CVLFace/HuggingFace key format to our backbone format.

        CVLFace block uses res_layer Sequential:
            res_layer.0 = BN1, .1 = Conv1, .2 = BN2, .3 = PReLU, .4 = Conv2, .5 = BN3
        Our block uses named attributes:
            bn1, conv1, bn2, prelu, conv2, bn3

        CVLFace top-level:
            input_layer.{0=Conv, 1=BN, 2=PReLU}
            body.{0..48} (flat block indices)
            output_layer.{0=BN2d, 1=Dropout, 2=Flatten, 3=Linear, 4=BN1d}
        Our top-level:
            conv1, bn1, prelu, layer{1..4}.{N}, bn2, fc, bn3
        """
        # Layer block boundaries: [3, 13, 30, 3] = 49 total
        layer_sizes = [3, 13, 30, 3]
        layer_starts = [0, 3, 16, 46]  # cumulative starts

        # res_layer index → named attribute
        res_map = {
            "res_layer.0.": "bn1.",
            "res_layer.1.": "conv1.",
            "res_layer.2.": "bn2.",
            "res_layer.3.": "prelu.",
            "res_layer.4.": "conv2.",
            "res_layer.5.": "bn3.",
        }

        remapped = {}
        for k, v in state.items():
            new_key = k

            # Input layer
            if k.startswith("input_layer.0."):
                new_key = k.replace("input_layer.0.", "conv1.")
            elif k.startswith("input_layer.1."):
                new_key = k.replace("input_layer.1.", "bn1.")
            elif k.startswith("input_layer.2."):
                new_key = k.replace("input_layer.2.", "prelu.")

            # Output layer: 0=BN2d, 1=Dropout, 2=Flatten, 3=Linear, 4=BN1d
            elif k.startswith("output_layer.0."):
                new_key = k.replace("output_layer.0.", "bn2.")
            elif k.startswith("output_layer.3."):
                new_key = k.replace("output_layer.3.", "fc.")
            elif k.startswith("output_layer.4."):
                new_key = k.replace("output_layer.4.", "bn3.")

            # Body blocks → layer{1..4}
            elif k.startswith("body."):
                parts = k.split(".", 2)  # ['body', 'N', 'rest...']
                body_idx = int(parts[1])
                rest = parts[2]

                # Find which layer this body index belongs to
                layer_num = 0
                for i, start in enumerate(layer_starts):
                    if body_idx >= start:
                        layer_num = i + 1
                        local_idx = body_idx - start

                # Remap res_layer.N → named attributes
                for old_prefix, new_prefix in res_map.items():
                    if rest.startswith(old_prefix):
                        rest = rest.replace(old_prefix, new_prefix, 1)
                        break

                # Remap shortcut_layer → downsample
                rest = rest.replace("shortcut_layer.", "downsample.")

                new_key = f"layer{layer_num}.{local_idx}.{rest}"

            remapped[new_key] = v
        return remapped

    @property
    def is_available(self) -> bool:
        return self._available

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) aligned face in [0, 1]
        Returns:
            (B, 512) L2-normalized embeddings
        """
        # AdaFace expects [-1, 1] range
        x = (x - 0.5) / 0.5
        emb = self._model(x)
        return F.normalize(emb, dim=1)


# ---------------------------------------------------------------------------
# Ensemble Face Model
# ---------------------------------------------------------------------------

class EnsembleFaceModel(nn.Module):
    """
    Wraps multiple face recognition models behind the same interface as FaceEmbedder.

    Dual interface:
      - forward(x) -> (B, 512): returns ArcFace embedding only (backward compatible)
      - ensemble_cosine_loss(clean_embs_dict, x_adv) -> (loss, per_model_metrics):
        weighted ensemble loss for attack optimization
      - get_all_clean_embeddings(x) -> dict: pre-compute embeddings from all models

    Graceful degradation: missing models are skipped. If only ArcFace is loaded,
    behavior is identical to using FaceEmbedder directly.

    Default weights: arcface=1.0, facenet=0.5, adaface=0.5 (normalized).
    """

    def __init__(
        self,
        arcface_model: FaceEmbedder,
        facenet_model: Optional[FaceNetWrapper] = None,
        adaface_model: Optional[AdaFaceWrapper] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.arcface = arcface_model
        self.facenet = facenet_model if (facenet_model and facenet_model.is_available) else None
        self.adaface = adaface_model if (adaface_model and adaface_model.is_available) else None

        # Build active models dict
        self._models: Dict[str, nn.Module] = {"arcface": self.arcface}
        if self.facenet is not None:
            self._models["facenet"] = self.facenet
        if self.adaface is not None:
            self._models["adaface"] = self.adaface

        # Set weights (default: arcface=1.0, facenet=0.5, adaface=0.5)
        default_weights = {"arcface": 1.0, "facenet": 0.5, "adaface": 0.5}
        if weights is not None:
            default_weights.update(weights)

        # Filter to active models and normalize
        active_weights = {k: default_weights.get(k, 0.5) for k in self._models}
        total = sum(active_weights.values())
        self._weights = {k: v / total for k, v in active_weights.items()}

        models_str = ", ".join(f"{k}({v:.2f})" for k, v in self._weights.items())
        print(f"Ensemble: {models_str}")

    @property
    def active_model_names(self):
        return list(self._models.keys())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backward-compatible: returns ArcFace embedding only."""
        return self.arcface(x)

    def align_from_numpy(self, image_bgr: np.ndarray) -> Optional[torch.Tensor]:
        """Delegate to ArcFace's aligner."""
        return self.arcface.align_from_numpy(image_bgr)

    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Delegate to ArcFace's cosine_similarity."""
        return self.arcface.cosine_similarity(emb1, emb2)

    @torch.no_grad()
    def get_all_clean_embeddings(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pre-compute embeddings from all active models."""
        embs = {}
        for name, model in self._models.items():
            embs[name] = model(x)
        return embs

    def ensemble_cosine_loss(
        self,
        clean_embs: Dict[str, torch.Tensor],
        x_adv: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted ensemble cosine similarity loss.

        Args:
            clean_embs: dict of pre-computed clean embeddings per model
            x_adv: (B, 3, H, W) adversarial image (gradients flow through)

        Returns:
            loss: weighted sum of per-model cosine similarities
            metrics: dict with per-model cosine similarity values
        """
        total_loss = torch.tensor(0.0, device=x_adv.device)
        metrics = {}

        for name, model in self._models.items():
            if name not in clean_embs:
                continue
            adv_emb = model(x_adv)
            cos_sim = F.cosine_similarity(clean_embs[name], adv_emb, dim=1).mean()
            total_loss = total_loss + self._weights[name] * cos_sim
            metrics[f"{name}_cos_sim"] = cos_sim.item()

        metrics["ensemble_loss"] = total_loss.item()
        return total_loss, metrics
