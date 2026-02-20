"""
Quick CLI for protecting a single image.

Usage:
  python protect.py input.jpg output.png --epsilon 0.031 --steps 50
"""

import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from core.face_model import FaceEmbedder
from core.eot import EoTWrapper
from core.attacks import PGDAttack
from core.encoder import NoiseEncoder
from config import config


def main():
    parser = argparse.ArgumentParser(description="Protect a face image")
    parser.add_argument("input", type=str, help="Input image path")
    parser.add_argument("output", type=str, help="Output image path")
    parser.add_argument("--mode", choices=["pgd", "encoder"], default="pgd")
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eot-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoder-weights", type=str, default=None)
    parser.add_argument("--arcface-weights", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    # Load face model
    print("Loading face model...")
    face_model = FaceEmbedder(
        weights_path=args.arcface_weights,
        device=device,
    )

    # Load and align image
    print(f"Loading {args.input}...")
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        print(f"Error: Could not read {args.input}")
        sys.exit(1)

    x = face_model.align_from_numpy(img_bgr)
    if x is None:
        print("Error: No face detected")
        sys.exit(1)

    print(f"Face detected and aligned to 112x112")

    if args.mode == "encoder":
        # Single-pass encoder mode
        if args.encoder_weights is None:
            print("Error: --encoder-weights required for encoder mode")
            sys.exit(1)

        encoder = NoiseEncoder(epsilon=args.epsilon).to(device)
        encoder.load_state_dict(torch.load(args.encoder_weights, map_location=device))
        encoder.eval()

        with torch.no_grad():
            x_protected = encoder.protect(x)

        # Quick eval
        clean_emb = face_model(x)
        prot_emb = face_model(x_protected)
        cos_sim = torch.nn.functional.cosine_similarity(clean_emb, prot_emb, dim=1).item()
        print(f"Encoder protection: cos_sim = {cos_sim:.4f}")

    else:
        # PGD mode
        print("Setting up EoT wrapper...")
        eot = EoTWrapper(
            model=face_model,
            num_samples=args.eot_samples,
            jpeg_quality_range=(50, 95),
            resize_scale_range=(0.5, 1.0),
            gaussian_sigma_range=(0.0, 1.0),
            crop_fraction_range=(0.8, 1.0),
        )

        print(f"Running PGD attack (ε={args.epsilon:.4f}, steps={args.steps}, EoT={args.eot_samples})...")
        attack = PGDAttack(
            face_model=face_model,
            eot_wrapper=eot,
            epsilon=args.epsilon,
            step_size=args.epsilon / 4,
            num_steps=args.steps,
        )

        x_protected, info = attack.run(x)
        print(f"Protection complete:")
        print(f"  Cosine similarity: {info['final_cosine_sim']:.4f}")
        print(f"  Robust cosine sim: {info['robust_cosine_sim']:.4f}")

    # Save output
    img_out = x_protected.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_out = (img_out * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_out).save(args.output)
    print(f"Saved to {args.output}")

    # Also save a side-by-side comparison
    comparison_path = str(Path(args.output).with_suffix("")) + "_comparison.png"
    img_clean = x.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_clean = (img_clean * 255).clip(0, 255).astype(np.uint8)
    delta_vis = (x_protected - x).squeeze(0).cpu().permute(1, 2, 0).numpy()
    delta_vis = ((delta_vis * 10 + 0.5) * 255).clip(0, 255).astype(np.uint8)  # Amplified delta

    comparison = np.concatenate([img_clean, img_out, delta_vis], axis=1)
    Image.fromarray(comparison).save(comparison_path)
    print(f"Comparison saved to {comparison_path}")
    print("  (left: clean | center: protected | right: perturbation ×10)")


if __name__ == "__main__":
    main()
