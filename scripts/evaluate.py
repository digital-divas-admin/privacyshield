"""
Evaluation Script

Measures protection effectiveness across multiple conditions:
1. Clean cosine similarity (no transforms)
2. Post-JPEG similarity at various quality levels
3. Post-resize similarity at various scales
4. Post-blur similarity
5. Combined transform robustness

A good perturbation should make ALL of these similarities low (< 0.3),
meaning the face recognition system fails to match the identity.

Usage:
  python scripts/evaluate.py \
    --clean-dir ./data/clean \
    --protected-dir ./data/protected \
    --device cuda
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.face_model import FaceEmbedder
from core.diff_jpeg import DiffJPEG
from core.eot import RandomResize, RandomGaussianBlur


def load_images(directory: str, size: int = 112) -> torch.Tensor:
    """Load all images from directory as a tensor."""
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    paths = sorted(Path(directory).rglob("*.png")) + sorted(Path(directory).rglob("*.jpg"))
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(tfm(img))
    return torch.stack(tensors) if tensors else torch.empty(0)


@torch.no_grad()
def evaluate(args):
    device = args.device
    face_model = FaceEmbedder(device=device)
    diff_jpeg = DiffJPEG().to(device)

    clean_imgs = load_images(args.clean_dir).to(device)
    protected_imgs = load_images(args.protected_dir).to(device)

    assert len(clean_imgs) == len(protected_imgs), \
        f"Mismatch: {len(clean_imgs)} clean vs {len(protected_imgs)} protected"

    n = len(clean_imgs)
    print(f"Evaluating {n} image pairs")
    print(f"{'='*60}")

    # Get clean embeddings
    clean_embs = face_model(clean_imgs)

    # 1. No-transform similarity
    prot_embs = face_model(protected_imgs)
    cos_clean = F.cosine_similarity(clean_embs, prot_embs, dim=1)
    print(f"\n1. Clean (no transforms):")
    print(f"   Cosine sim: {cos_clean.mean():.4f} ± {cos_clean.std():.4f}")
    print(f"   Match rate (>0.4): {(cos_clean > 0.4).float().mean():.1%}")

    # 2. Post-JPEG at various qualities
    print(f"\n2. Post-JPEG compression:")
    for q in [95, 85, 75, 50]:
        jpeg_imgs = diff_jpeg(protected_imgs, quality=q)
        jpeg_embs = face_model(jpeg_imgs)
        cos_jpeg = F.cosine_similarity(clean_embs, jpeg_embs, dim=1)
        print(f"   Q={q:3d}: cos_sim={cos_jpeg.mean():.4f} ± {cos_jpeg.std():.4f}, "
              f"match_rate={( cos_jpeg > 0.4).float().mean():.1%}")

    # 3. Post-resize
    print(f"\n3. Post-resize:")
    for scale in [1.0, 0.75, 0.5, 0.25]:
        h, w = int(112 * scale), int(112 * scale)
        down = F.interpolate(protected_imgs, size=(h, w), mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=(112, 112), mode="bilinear", align_corners=False)
        resize_embs = face_model(up)
        cos_resize = F.cosine_similarity(clean_embs, resize_embs, dim=1)
        print(f"   Scale={scale:.2f}: cos_sim={cos_resize.mean():.4f} ± {cos_resize.std():.4f}, "
              f"match_rate={(cos_resize > 0.4).float().mean():.1%}")

    # 4. Post-blur
    print(f"\n4. Post-Gaussian blur:")
    blur = RandomGaussianBlur(sigma_range=(0, 0))  # We'll set sigma manually
    for sigma in [0.5, 1.0, 2.0]:
        import torch.nn.functional as F_nn
        k = 5
        ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.outer(kernel_1d).view(1, 1, k, k).expand(3, -1, -1, -1)
        blurred = F_nn.conv2d(protected_imgs, kernel_2d, padding=k//2, groups=3)
        blur_embs = face_model(blurred.clamp(0, 1))
        cos_blur = F.cosine_similarity(clean_embs, blur_embs, dim=1)
        print(f"   σ={sigma:.1f}: cos_sim={cos_blur.mean():.4f} ± {cos_blur.std():.4f}, "
              f"match_rate={(cos_blur > 0.4).float().mean():.1%}")

    # 5. Combined: JPEG(75) + resize(0.75)
    print(f"\n5. Combined transforms (JPEG Q=75 + resize 0.75):")
    down = F.interpolate(protected_imgs, size=(84, 84), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(112, 112), mode="bilinear", align_corners=False)
    combo = diff_jpeg(up, quality=75)
    combo_embs = face_model(combo)
    cos_combo = F.cosine_similarity(clean_embs, combo_embs, dim=1)
    print(f"   cos_sim={cos_combo.mean():.4f} ± {cos_combo.std():.4f}, "
          f"match_rate={(cos_combo > 0.4).float().mean():.1%}")

    # 6. Perturbation stats
    delta = protected_imgs - clean_imgs
    print(f"\n6. Perturbation statistics:")
    print(f"   L∞ norm:  {delta.abs().max():.4f}")
    print(f"   L2 norm:  {delta.norm(p=2, dim=(1,2,3)).mean():.4f}")
    print(f"   PSNR:     {10 * torch.log10(1.0 / (delta ** 2).mean()):.1f} dB")

    print(f"\n{'='*60}")
    print("Protection goal: cos_sim < 0.3 and match_rate = 0% across all conditions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--protected-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate(args)
