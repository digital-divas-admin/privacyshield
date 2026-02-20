"""
Train the Noise Encoder

Two-phase training:
  Phase 1 (Distillation): Train on PGD-generated (x, δ*) pairs
  Phase 2 (End-to-End):   Fine-tune with the actual embedding loss + EoT

Phase 2 is what makes the encoder actually good — it learns to make
perturbations that work, not just perturbations that look like PGD output.

Usage:
  # Phase 1: Distill from PGD pairs
  python scripts/train_encoder.py --phase distill --data-dir ./faces --epochs 50

  # Phase 2: End-to-end with embedding loss
  python scripts/train_encoder.py --phase e2e --checkpoint ./checkpoints/distill_best.pt --epochs 50
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.encoder import NoiseEncoder, EncoderLoss
from core.face_model import FaceEmbedder
from core.eot import EoTWrapper
from core.attacks import PGDAttack
from core.losses import PrivacyShieldLoss, LPIPSLoss, CLIPVisionWrapper
from core.semantic_mask import SemanticMask, MaskedPerturbation
from config import config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FaceDataset(Dataset):
    """Load aligned face images from a directory."""

    def __init__(self, root_dir: str, image_size: int = 112):
        self.paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            self.paths.extend(Path(root_dir).rglob(ext))
        self.paths = sorted(self.paths)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class PGDPairsDataset(Dataset):
    """Load precomputed (clean, delta) pairs."""

    def __init__(self, pairs_dir: str):
        self.clean_dir = Path(pairs_dir) / "clean"
        self.delta_dir = Path(pairs_dir) / "delta"
        self.filenames = sorted(os.listdir(self.clean_dir))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        clean = self.transform(Image.open(self.clean_dir / name).convert("RGB"))
        delta = self.transform(Image.open(self.delta_dir / name).convert("RGB"))
        # Delta was saved as (delta + 0.5) to fit in [0, 1] image format
        delta = delta - 0.5
        return clean, delta


# ---------------------------------------------------------------------------
# Phase 1: Distillation from PGD pairs
# ---------------------------------------------------------------------------

def train_distill(args):
    """Train encoder to match PGD-generated perturbations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Load face model for embedding loss
    face_model = FaceEmbedder(device=device)

    # Initialize encoder
    encoder = NoiseEncoder(epsilon=args.epsilon).to(device)

    # Loss
    criterion = EncoderLoss(face_model, lambda_mse=1.0, lambda_emb=5.0, lambda_per=0.5)

    # Optimizer
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Dataset
    dataset = PGDPairsDataset(args.pairs_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        encoder.train()
        total_loss = 0
        total_cos = 0
        n = 0

        for clean, delta_target in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            clean = clean.to(device)
            delta_target = delta_target.to(device)

            # Forward
            delta_pred = encoder(clean)

            # Loss
            loss, loss_dict = criterion(delta_pred, delta_target, clean)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_dict["total"].item() * clean.shape[0]
            total_cos += loss_dict["cos_sim"].item() * clean.shape[0]
            n += clean.shape[0]

        scheduler.step()
        avg_loss = total_loss / n
        avg_cos = total_cos / n

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, cos_sim={avg_cos:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, "distill_best.pt"))
            print(f"  Saved best checkpoint (loss={best_loss:.4f})")

        torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, "distill_latest.pt"))


# ---------------------------------------------------------------------------
# Phase 2: End-to-end with EoT
# ---------------------------------------------------------------------------

def train_e2e(args):
    """Fine-tune encoder with actual embedding loss + EoT."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    face_model = FaceEmbedder(device=device)
    eot = EoTWrapper(
        model=face_model,
        num_samples=5,  # Fewer for training speed
        enable_jpeg=True,
        enable_resize=True,
        enable_gaussian=True,
        enable_crop=False,
    )

    # Load encoder from Phase 1
    encoder = NoiseEncoder(epsilon=args.epsilon).to(device)
    if args.checkpoint:
        encoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded encoder from {args.checkpoint}")

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr * 0.1)

    # Dataset (just clean images, no PGD pairs needed)
    dataset = FaceDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_cos = 1.0  # Lower is better

    for epoch in range(args.epochs):
        encoder.train()
        total_cos = 0
        n = 0

        for clean in tqdm(loader, desc=f"E2E Epoch {epoch+1}/{args.epochs}"):
            clean = clean.to(device)

            # Get clean embeddings (frozen)
            with torch.no_grad():
                clean_emb = face_model(clean)

            # Generate perturbation
            delta = encoder(clean)
            x_adv = (clean + delta).clamp(0, 1)

            # EoT loss: minimize cosine sim through transforms
            cos_loss = eot(x_adv, clean_emb)

            # Imperceptibility regularizer
            reg = delta.abs().mean() * 10.0  # Keep perturbation small

            loss = cos_loss + reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            total_cos += cos_loss.item() * clean.shape[0]
            n += clean.shape[0]

        avg_cos = total_cos / n
        print(f"  Epoch {epoch+1}: avg_cos_sim={avg_cos:.4f}")

        if avg_cos < best_cos:
            best_cos = avg_cos
            torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  Saved best checkpoint (cos_sim={best_cos:.4f})")

        torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, "latest.pt"))


# ---------------------------------------------------------------------------
# Phase 3: V2 End-to-End with LPIPS + CLIP + Semantic Mask
# ---------------------------------------------------------------------------

def train_v2_e2e(args):
    """
    v2 end-to-end training with all four Gemini fixes:
      - Semantic mask (concentrate noise in textured regions)
      - LPIPS loss (visual quality guard)
      - CLIP dual-targeting (blind both ArcFace + CLIP)
      - Full unified loss

    This produces the highest-quality encoder — perturbations are
    invisible AND robust AND blind both biometric + style channels.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load all models ---
    print("Loading ArcFace...")
    face_model = FaceEmbedder(device=device)

    print("Loading CLIP Vision...")
    clip_model = CLIPVisionWrapper(device=device)

    print("Loading LPIPS...")
    lpips_loss = LPIPSLoss(device=device)

    print("Loading EoT wrapper...")
    eot = EoTWrapper(model=face_model, num_samples=args.eot_samples)

    # Unified loss
    unified_loss = PrivacyShieldLoss(
        face_model=face_model,
        clip_model=clip_model if clip_model.is_available else None,
        lpips_loss=lpips_loss,
        alpha_arcface=1.0,
        beta_clip=0.5,
        lambda_lpips=args.lambda_lpips,
        lambda_reg=0.01,
    )

    # Semantic mask
    semantic_mask = SemanticMask() if args.use_mask else None

    # --- Load encoder ---
    encoder = NoiseEncoder(epsilon=args.epsilon).to(device)
    if args.checkpoint:
        encoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded encoder from {args.checkpoint}")

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr * 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dataset = FaceDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        encoder.train()
        running = {"loss": 0, "arcface": 0, "clip": 0, "lpips": 0}
        n = 0

        for clean in tqdm(loader, desc=f"V2-E2E Epoch {epoch+1}/{args.epochs}"):
            clean = clean.to(device)

            # Clean embeddings (frozen)
            with torch.no_grad():
                clean_arcface = face_model(clean)
                clean_clip = clip_model(clean) if clip_model.is_available else None

                # Compute semantic mask (no grad, fixed per batch)
                mask = semantic_mask(clean) if semantic_mask is not None else None

            # Generate perturbation
            delta = encoder(clean)

            # Apply semantic mask
            if mask is not None:
                delta = delta * mask

            x_adv = (clean + delta).clamp(0, 1)

            # EoT-averaged unified loss
            total_loss = torch.tensor(0.0, device=device)
            step_metrics = {}

            for _ in range(args.eot_samples):
                x_t = eot.apply_random_transform(x_adv)
                loss_t, metrics_t = unified_loss(clean, x_t, clean_arcface, clean_clip)
                total_loss = total_loss + loss_t
                step_metrics = metrics_t  # Keep last sample's metrics

            avg_loss = total_loss / args.eot_samples

            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            bs = clean.shape[0]
            running["loss"] += avg_loss.item() * bs
            running["arcface"] += step_metrics.get("arcface_cos_sim", 0) * bs
            running["clip"] += step_metrics.get("clip_cos_sim", 0) * bs
            running["lpips"] += step_metrics.get("lpips", 0) * bs
            n += bs

        scheduler.step()

        avg = {k: v / n for k, v in running.items()}
        print(f"  Epoch {epoch+1}: loss={avg['loss']:.4f} "
              f"arc={avg['arcface']:.4f} clip={avg['clip']:.4f} "
              f"lpips={avg['lpips']:.4f}")

        if avg["loss"] < best_loss:
            best_loss = avg["loss"]
            torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, "v2_best.pt"))
            print(f"  Saved v2 best (loss={best_loss:.4f})")

        torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, "v2_latest.pt"))


# ---------------------------------------------------------------------------
# Generate PGD pairs
# ---------------------------------------------------------------------------
    """Pre-compute PGD perturbation pairs for Phase 1 training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    face_model = FaceEmbedder(device=device)
    eot = EoTWrapper(model=face_model, num_samples=args.eot_samples)
    attack = PGDAttack(
        face_model=face_model,
        eot_wrapper=eot,
        epsilon=args.epsilon,
        num_steps=args.steps,
        verbose=False,
    )

    dataset = FaceDataset(args.data_dir)
    clean_dir = Path(args.output_dir) / "clean"
    delta_dir = Path(args.output_dir) / "delta"
    clean_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Generating PGD pairs"):
        x = dataset[idx].unsqueeze(0).to(device)
        x_adv, _ = attack.run(x)

        delta = x_adv - x
        # Save delta as image (shift to [0,1] by adding 0.5)
        delta_img = (delta.squeeze(0).cpu() + 0.5).clamp(0, 1)
        clean_img = x.squeeze(0).cpu()

        from torchvision.utils import save_image
        save_image(clean_img, clean_dir / f"{idx:06d}.png")
        save_image(delta_img, delta_dir / f"{idx:06d}.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PrivacyShield Noise Encoder")
    parser.add_argument("--phase", choices=["distill", "e2e", "v2_e2e", "generate"], required=True)
    parser.add_argument("--data-dir", type=str, default="./data/faces")
    parser.add_argument("--pairs-dir", type=str, default="./data/pairs")
    parser.add_argument("--output-dir", type=str, default="./data/pairs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eot-samples", type=int, default=10)
    # v2-specific
    parser.add_argument("--lambda-lpips", type=float, default=0.1, help="LPIPS penalty weight")
    parser.add_argument("--use-mask", action="store_true", help="Enable semantic face mask")

    args = parser.parse_args()

    if args.phase == "generate":
        generate_pairs(args)
    elif args.phase == "distill":
        train_distill(args)
    elif args.phase == "e2e":
        train_e2e(args)
    elif args.phase == "v2_e2e":
        train_v2_e2e(args)
