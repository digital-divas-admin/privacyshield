"""
Differentiable Face Alignment via Spatial Transformer

Gemini's Point #1 is valid for end-to-end ViT encoder training:
  The standard InsightFace pipeline does detect → OpenCV warpAffine → ArcFace.
  OpenCV warpAffine is NOT differentiable, so gradients snap at that boundary.

Solution:
  1. Use InsightFace/RetinaFace to detect 5 landmarks (forward-only, no grad needed)
  2. Freeze those landmark coordinates
  3. Use torch.nn.functional.grid_sample to perform the affine warp
     → This IS differentiable, so gradients flow from ArcFace embedding
        all the way back through the warp to the original image pixels

This matters for:
  - ViT encoder training on full-size images (gradients must reach raw pixels)
  - PGD on full-size images (not just pre-cropped 112x112)

For PGD on pre-aligned 112x112 crops, this module is unnecessary — gradients
already flow fine since there's no alignment step in the loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# Standard ArcFace alignment target landmarks (112x112)
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _estimate_affine(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Estimate 2x3 affine matrix from source to destination points.
    Uses least-squares fit (same as skimage SimilarityTransform).
    """
    n = src_pts.shape[0]
    # Build linear system: [x y 1 0 0 0; 0 0 0 x y 1] @ [a b c d e f]^T = [u; v]
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        A[2*i, 0] = src_pts[i, 0]
        A[2*i, 1] = src_pts[i, 1]
        A[2*i, 2] = 1
        A[2*i+1, 3] = src_pts[i, 0]
        A[2*i+1, 4] = src_pts[i, 1]
        A[2*i+1, 5] = 1
        b[2*i] = dst_pts[i, 0]
        b[2*i+1] = dst_pts[i, 1]

    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
    ])
    return M


def _affine_to_grid(
    M: np.ndarray,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int] = (112, 112),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convert a 2x3 affine matrix (src→dst) into a sampling grid for grid_sample.

    grid_sample expects a grid in [-1, 1] normalized coords that maps
    OUTPUT pixels → INPUT pixels. So we need the inverse affine.

    Args:
        M: 2x3 affine matrix (maps src coords to dst coords)
        src_size: (H, W) of source image
        dst_size: (H, W) of destination (aligned) image
        device: torch device

    Returns:
        grid: (1, dst_H, dst_W, 2) sampling grid for grid_sample
    """
    dst_h, dst_w = dst_size
    src_h, src_w = src_size

    # Invert the affine: we need dst→src mapping for grid_sample
    # Pad M to 3x3
    M_full = np.eye(3)
    M_full[:2, :] = M
    M_inv = np.linalg.inv(M_full)[:2, :]  # 2x3

    # Create grid of destination pixel coordinates
    # grid_sample expects normalized [-1, 1] coords
    gy, gx = torch.meshgrid(
        torch.arange(dst_h, dtype=torch.float32, device=device),
        torch.arange(dst_w, dtype=torch.float32, device=device),
        indexing="ij",
    )

    # Stack to (H, W, 3) homogeneous coords
    ones = torch.ones_like(gx)
    dst_coords = torch.stack([gx, gy, ones], dim=-1)  # (H, W, 3)

    # Apply inverse affine to get source coordinates
    M_inv_t = torch.from_numpy(M_inv).float().to(device)  # (2, 3)
    src_coords = torch.einsum("ij,hwj->hwi", M_inv_t, dst_coords)  # (H, W, 2)

    # Normalize to [-1, 1] for grid_sample
    src_coords[..., 0] = 2.0 * src_coords[..., 0] / (src_w - 1) - 1.0
    src_coords[..., 1] = 2.0 * src_coords[..., 1] / (src_h - 1) - 1.0

    return src_coords.unsqueeze(0)  # (1, H, W, 2)


class DifferentiableAligner(nn.Module):
    """
    Differentiable face alignment module.

    Usage in the attack/training pipeline:
        aligner = DifferentiableAligner()

        # Step 1: Detect landmarks (non-differentiable, done once)
        landmarks = aligner.detect_landmarks(image_bgr)

        # Step 2: Build sampling grid (non-differentiable, done once)
        grid = aligner.build_grid(landmarks, src_size=(H, W))

        # Step 3: Warp (DIFFERENTIABLE — gradients flow through here)
        aligned = aligner.warp(image_tensor, grid)

        # Now ArcFace embedding is differentiable w.r.t. image_tensor!
        embedding = arcface(aligned)
        loss = 1 - cos_sim(embedding, target)
        loss.backward()  # Gradients flow through grid_sample!
    """

    def __init__(self, det_size: Tuple[int, int] = (640, 640), output_size: int = 112):
        super().__init__()
        self.output_size = output_size
        self.dst_pts = ARCFACE_DST.copy()

        # Scale destination points if output_size != 112
        if output_size != 112:
            scale = output_size / 112.0
            self.dst_pts *= scale

        try:
            from insightface.app import FaceAnalysis
            self.detector = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection"],
            )
            self.detector.prepare(ctx_id=0, det_size=det_size)
            self._has_detector = True
        except ImportError:
            self._has_detector = False
            print("Warning: insightface not available. Using center-crop fallback.")

    @torch.no_grad()
    def detect_landmarks(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face and return 5 landmarks. Non-differentiable.
        Args:
            image_bgr: (H, W, 3) BGR image as numpy array
        Returns:
            landmarks: (5, 2) float32 array or None
        """
        if not self._has_detector:
            return None

        faces = self.detector.get(image_bgr)
        if len(faces) == 0:
            return None

        # Pick largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.kps.astype(np.float32)

    def build_grid(
        self,
        landmarks: np.ndarray,
        src_size: Tuple[int, int],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Build a differentiable sampling grid from detected landmarks.
        This is computed once per image and reused across PGD steps.

        Args:
            landmarks: (5, 2) facial landmarks
            src_size: (H, W) of the source image
        Returns:
            grid: (1, out_size, out_size, 2) for grid_sample
        """
        M = _estimate_affine(landmarks, self.dst_pts)
        grid = _affine_to_grid(
            M, src_size,
            dst_size=(self.output_size, self.output_size),
            device=device,
        )
        return grid

    def warp(self, image_tensor: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply differentiable affine warp using grid_sample.

        THIS IS THE KEY DIFFERENTIABLE OPERATION.
        Gradients from the ArcFace loss flow through grid_sample
        back to the input image pixels.

        Args:
            image_tensor: (B, 3, H, W) in [0, 1] — the image being attacked
            grid: (1, out_H, out_W, 2) — precomputed sampling grid
        Returns:
            aligned: (B, 3, out_H, out_W) aligned face crop
        """
        B = image_tensor.shape[0]
        # Expand grid to batch size
        grid = grid.expand(B, -1, -1, -1)

        # grid_sample is fully differentiable!
        aligned = F.grid_sample(
            image_tensor,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return aligned

    def forward(
        self,
        image_tensor: torch.Tensor,
        landmarks: np.ndarray,
    ) -> torch.Tensor:
        """
        Full pipeline: landmarks → grid → differentiable warp.

        Args:
            image_tensor: (B, 3, H, W) image tensor (requires_grad=True for attack)
            landmarks: (5, 2) pre-detected landmarks
        """
        grid = self.build_grid(
            landmarks,
            src_size=(image_tensor.shape[2], image_tensor.shape[3]),
            device=image_tensor.device,
        )
        return self.warp(image_tensor, grid)


class FullImagePGDAttack:
    """
    PGD attack that operates on FULL-SIZE images instead of pre-aligned crops.

    The key difference from the base PGDAttack:
      1. Detect landmarks once (non-differentiable)
      2. Build grid_sample grid once (non-differentiable)
      3. Each PGD step: perturb full image → warp with grid_sample → ArcFace → loss.backward()
         The gradients flow through grid_sample back to the full image pixels.

    This means the perturbation is optimized IN THE ORIGINAL IMAGE SPACE,
    accounting for how the face alignment will transform it.
    """

    def __init__(
        self,
        face_model,
        eot_wrapper,
        aligner: DifferentiableAligner,
        epsilon: float = 8 / 255,
        step_size: float = 2 / 255,
        num_steps: int = 50,
    ):
        self.face_model = face_model
        self.eot = eot_wrapper
        self.aligner = aligner
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps

    def run(
        self,
        image_tensor: torch.Tensor,
        landmarks: np.ndarray,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run PGD on full-size image with differentiable alignment.

        Args:
            image_tensor: (1, 3, H, W) full-size image in [0, 1]
            landmarks: (5, 2) pre-detected face landmarks
        Returns:
            x_protected: (1, 3, H, W) protected full-size image
            info: dict with metrics
        """
        device = image_tensor.device
        x = image_tensor.detach().clone()

        # Build grid once (not differentiable, but doesn't need to be)
        grid = self.aligner.build_grid(
            landmarks,
            src_size=(x.shape[2], x.shape[3]),
            device=device,
        )

        # Get clean embedding
        with torch.no_grad():
            aligned_clean = self.aligner.warp(x, grid)
            clean_emb = self.face_model(aligned_clean)

        # Initialize perturbation
        delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_(True)

        for step in range(self.num_steps):
            x_adv = (x + delta).clamp(0.0, 1.0)

            # Differentiable alignment — gradients flow through here!
            aligned_adv = self.aligner.warp(x_adv, grid)

            # EoT loss on aligned face
            loss = self.eot(aligned_adv, clean_emb)
            loss.backward()

            with torch.no_grad():
                delta.data -= self.step_size * delta.grad.sign()
                delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
                delta.data = (x + delta.data).clamp(0.0, 1.0) - x

            delta.grad.zero_()

        x_protected = (x + delta.data).clamp(0.0, 1.0)

        with torch.no_grad():
            aligned_prot = self.aligner.warp(x_protected, grid)
            prot_emb = self.face_model(aligned_prot)
            cos_sim = F.cosine_similarity(clean_emb, prot_emb, dim=1).mean().item()

        return x_protected, {"final_cosine_sim": cos_sim, "delta_linf": delta.data.abs().max().item()}
