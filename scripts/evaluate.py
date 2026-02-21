"""
Evaluation Script — Cross-Model Robustness

Measures protection effectiveness across multiple conditions AND
multiple face recognition models (ArcFace, FaceNet, AdaFace).

Conditions tested:
1. Clean cosine similarity (no transforms)
2. Post-JPEG similarity at various quality levels
3. Post-resize similarity at various scales
4. Post-blur similarity
5. Combined transform robustness

A good perturbation should make ALL of these similarities low (< 0.3),
meaning the face recognition system fails to match the identity.

Cross-model evaluation proves transferability: perturbations optimized
against the ensemble should also fool models not seen during optimization.

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

from core.face_model import FaceEmbedder, FaceNetWrapper, AdaFaceWrapper
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


def eval_model_condition(model, clean_embs, transformed_imgs):
    """Evaluate a single model on transformed images."""
    embs = model(transformed_imgs)
    cos = F.cosine_similarity(clean_embs, embs, dim=1)
    return cos


@torch.no_grad()
def evaluate(args):
    device = args.device

    # Load all available models
    models = {}

    # ArcFace (always available)
    face_model = FaceEmbedder(device=device)
    models["arcface"] = face_model

    # FaceNet
    facenet = FaceNetWrapper(device=device)
    if facenet.is_available:
        models["facenet"] = facenet

    # AdaFace
    adaface_weights = args.adaface_weights if hasattr(args, "adaface_weights") else None
    adaface = AdaFaceWrapper(weights_path=adaface_weights, device=device)
    if adaface.is_available:
        models["adaface"] = adaface

    diff_jpeg = DiffJPEG().to(device)

    clean_imgs = load_images(args.clean_dir).to(device)
    protected_imgs = load_images(args.protected_dir).to(device)

    assert len(clean_imgs) == len(protected_imgs), \
        f"Mismatch: {len(clean_imgs)} clean vs {len(protected_imgs)} protected"

    n = len(clean_imgs)
    model_names = list(models.keys())
    print(f"Evaluating {n} image pairs across {len(models)} models: {', '.join(model_names)}")
    print(f"{'='*80}")

    # Pre-compute clean embeddings for all models
    clean_embs = {}
    for name, model in models.items():
        clean_embs[name] = model(clean_imgs)

    def print_condition(label, transformed):
        """Print results for all models on a given condition."""
        for name in model_names:
            cos = eval_model_condition(models[name], clean_embs[name], transformed)
            pad = " " * (10 - len(name))
            print(f"   {name}:{pad}cos_sim={cos.mean():.4f} ± {cos.std():.4f}, "
                  f"match_rate={(cos > 0.4).float().mean():.1%}")

    # 1. No-transform similarity
    print(f"\n1. Clean (no transforms):")
    print_condition("clean", protected_imgs)

    # 2. Post-JPEG at various qualities
    print(f"\n2. Post-JPEG compression:")
    for q in [95, 85, 75, 50]:
        jpeg_imgs = diff_jpeg(protected_imgs, quality=q)
        print(f"   --- JPEG Q={q} ---")
        print_condition(f"jpeg_q{q}", jpeg_imgs)

    # 3. Post-resize
    print(f"\n3. Post-resize:")
    for scale in [1.0, 0.75, 0.5, 0.25]:
        h, w = int(112 * scale), int(112 * scale)
        if h < 1:
            continue
        down = F.interpolate(protected_imgs, size=(h, w), mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=(112, 112), mode="bilinear", align_corners=False)
        print(f"   --- Scale={scale:.2f} ---")
        print_condition(f"resize_{scale}", up)

    # 4. Post-blur
    print(f"\n4. Post-Gaussian blur:")
    for sigma in [0.5, 1.0, 2.0]:
        k = 5
        ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.outer(kernel_1d).view(1, 1, k, k).expand(3, -1, -1, -1)
        blurred = F.conv2d(protected_imgs, kernel_2d, padding=k // 2, groups=3).clamp(0, 1)
        print(f"   --- Blur σ={sigma:.1f} ---")
        print_condition(f"blur_{sigma}", blurred)

    # 5. Combined: JPEG(75) + resize(0.75)
    print(f"\n5. Combined transforms (JPEG Q=75 + resize 0.75):")
    down = F.interpolate(protected_imgs, size=(84, 84), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(112, 112), mode="bilinear", align_corners=False)
    combo = diff_jpeg(up, quality=75)
    print_condition("combined", combo)

    # 6. Perturbation stats
    delta = protected_imgs - clean_imgs
    print(f"\n6. Perturbation statistics:")
    print(f"   L∞ norm:  {delta.abs().max():.4f}")
    print(f"   L2 norm:  {delta.norm(p=2, dim=(1,2,3)).mean():.4f}")
    print(f"   PSNR:     {10 * torch.log10(1.0 / (delta ** 2).mean()):.1f} dB")

    print(f"\n{'='*80}")
    print("Protection goal: cos_sim < 0.3 and match_rate = 0% across all conditions and models")
    if len(models) > 1:
        print(f"Cross-model transferability: check that non-ArcFace models also show low cos_sim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--protected-dir", required=True)
    parser.add_argument("--adaface-weights", default=None, help="Path to AdaFace IR-101 weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate(args)
