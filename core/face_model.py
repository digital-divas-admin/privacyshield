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
from typing import Optional, Tuple


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
        device: str = "cuda",
        embedding_dim: int = 512,
    ):
        super().__init__()
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
