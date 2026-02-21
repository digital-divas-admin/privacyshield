"""
Train the Noise Encoder

Three-phase training:
  Phase 0 (Generate): Pre-compute PGD perturbation pairs
  Phase 1 (Distillation): Train on PGD-generated (x, δ*) pairs
  Phase 2 (End-to-End):   Fine-tune with the actual embedding loss + EoT
  Phase 3 (V2 E2E):       Full pipeline with LPIPS + CLIP + semantic mask

Supports both U-Net (~50ms) and ViT-S/8 (~170ms) encoder architectures.

Usage:
  # Generate PGD pairs
  python scripts/train_encoder.py --phase generate --data-dir ./data/faces/train --output-dir ./data/pairs

  # Distill into U-Net
  python scripts/train_encoder.py --phase distill --encoder-type unet --pairs-dir ./data/pairs --data-dir ./data/faces --epochs 50

  # Distill into ViT
  python scripts/train_encoder.py --phase distill --encoder-type vit --pairs-dir ./data/pairs --data-dir ./data/faces --epochs 50

  # V2 E2E fine-tune
  python scripts/train_encoder.py --phase v2_e2e --encoder-type unet --data-dir ./data/faces --checkpoint ./checkpoints/distill_best.pt --use-mask --epochs 50
"""

import os
import sys
import platform
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
# Encoder factory
# ---------------------------------------------------------------------------

def create_encoder(encoder_type: str, epsilon: float, device: str):
    """
    Create encoder by type.

    Returns:
        (encoder, image_size) tuple. ViT handles 112x112 via internal resize,
        so PGD pairs at 112x112 work for both architectures.
    """
    if encoder_type == "vit":
        from core.vit_encoder import vit_noise_encoder_small
        return vit_noise_encoder_small(epsilon=epsilon).to(device), 224
    else:
        return NoiseEncoder(epsilon=epsilon).to(device), 112


def checkpoint_prefix(encoder_type: str) -> str:
    """Return filename prefix for checkpoints."""
    return "vit_" if encoder_type == "vit" else ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FaceDataset(Dataset):
    """Load aligned face images from a directory with optional train/val split."""

    def __init__(self, root_dir: str, image_size: int = 112, split: str = None):
        """
        Args:
            root_dir: Path to face images directory.
            image_size: Resize target.
            split: "train", "val", or None (use all images).
                   If root_dir/train/ exists, uses subdirectory.
                   Otherwise, 90/10 split on sorted filenames.
        """
        # Check for pre-split directories
        train_subdir = Path(root_dir) / "train"
        val_subdir = Path(root_dir) / "val"

        if split and train_subdir.exists() and val_subdir.exists():
            scan_dir = train_subdir if split == "train" else val_subdir
            self.paths = self._scan_images(scan_dir)
        else:
            all_paths = self._scan_images(Path(root_dir))
            if split is None:
                self.paths = all_paths
            else:
                split_idx = int(len(all_paths) * 0.9)
                if split == "train":
                    self.paths = all_paths[:split_idx]
                else:
                    self.paths = all_paths[split_idx:]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1]
        ])

    @staticmethod
    def _scan_images(root: Path) -> list:
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(root.rglob(ext))
        return sorted(paths)

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
# DataLoader helpers
# ---------------------------------------------------------------------------

def default_num_workers() -> int:
    """0 on Windows (multiprocessing issues), 4 on Linux."""
    return 0 if platform.system() == "Windows" else 4


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create DataLoader with pin_memory on CUDA."""
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )


# ---------------------------------------------------------------------------
# TensorBoard helper
# ---------------------------------------------------------------------------

def create_writer(args, phase: str):
    """Create TensorBoard SummaryWriter if enabled."""
    if args.no_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(args.checkpoint_dir, "logs", f"{phase}_{args.encoder_type}")
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to {log_dir}")
        return writer
    except ImportError:
        print("tensorboard not installed — logging disabled")
        return None


def tb_log(writer, tag: str, value: float, step: int):
    """Log scalar if writer exists."""
    if writer is not None:
        writer.add_scalar(tag, value, step)


# ---------------------------------------------------------------------------
# Phase 0: Generate PGD pairs
# ---------------------------------------------------------------------------

def generate_pairs(args):
    """Pre-compute PGD perturbation pairs for Phase 1 training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    face_model = FaceEmbedder(weights_path=args.arcface_weights, device=device)
    eot = EoTWrapper(model=face_model, num_samples=args.eot_samples)
    attack = PGDAttack(
        face_model=face_model,
        eot_wrapper=eot,
        epsilon=args.epsilon,
        num_steps=args.steps,
        verbose=False,
    )

    # Only generate from training split to avoid val data leak
    dataset = FaceDataset(args.data_dir, split="train")
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
# Phase 1: Distillation from PGD pairs
# ---------------------------------------------------------------------------

def train_distill(args):
    """Train encoder to match PGD-generated perturbations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Load face model for embedding loss
    face_model = FaceEmbedder(weights_path=args.arcface_weights, device=device)

    # Initialize encoder
    encoder, image_size = create_encoder(args.encoder_type, args.epsilon, device)
    prefix = checkpoint_prefix(args.encoder_type)

    # Loss
    criterion = EncoderLoss(face_model, lambda_mse=1.0, lambda_emb=5.0, lambda_per=0.5)

    # Optimizer
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-5)

    # Scheduler — ViT gets linear warmup before cosine annealing
    if args.encoder_type == "vit":
        warmup_epochs = 5
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - warmup_epochs, 1)
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Dataset — pairs for training, face images for validation
    train_dataset = PGDPairsDataset(args.pairs_dir)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_dataset = FaceDataset(args.data_dir, image_size=image_size, split="val") if args.data_dir else None
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers) if val_dataset and len(val_dataset) > 0 else None

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = create_writer(args, "distill")
    best_loss = float("inf")
    patience_counter = 0

    try:
        for epoch in range(args.epochs):
            encoder.train()
            total_loss = 0
            total_cos = 0
            n = 0

            for clean, delta_target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
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
            tb_log(writer, "train/loss", avg_loss, epoch)
            tb_log(writer, "train/cos_sim", avg_cos, epoch)
            tb_log(writer, "train/lr", optimizer.param_groups[0]["lr"], epoch)

            # Validation
            val_loss = avg_loss  # fallback if no val set
            if val_loader is not None:
                val_loss = _validate_distill(encoder, face_model, val_loader, device, criterion)
                print(f"  Val loss={val_loss:.4f}")
                tb_log(writer, "val/loss", val_loss, epoch)

            # Save checkpoint (use val loss when available)
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, f"{prefix}distill_best.pt"))
                print(f"  Saved best checkpoint (loss={best_loss:.4f})")
            else:
                patience_counter += 1

            torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, f"{prefix}distill_latest.pt"))

            if patience_counter >= args.patience:
                print(f"  Early stopping after {args.patience} epochs without improvement")
                break
    finally:
        if writer:
            writer.close()


def _validate_distill(encoder, face_model, val_loader, device, criterion):
    """Run validation loop for distill phase (using face images, not pairs)."""
    encoder.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            clean_emb = face_model(batch)
            delta = encoder(batch)
            x_adv = (batch + delta).clamp(0, 1)
            adv_emb = face_model(x_adv)
            cos_sim = F.cosine_similarity(clean_emb, adv_emb, dim=1).mean()
            total_loss += cos_sim.item() * batch.shape[0]
            n += batch.shape[0]
    return total_loss / n if n > 0 else float("inf")


# ---------------------------------------------------------------------------
# Phase 2: End-to-end with EoT
# ---------------------------------------------------------------------------

def train_e2e(args):
    """Fine-tune encoder with actual embedding loss + EoT."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    face_model = FaceEmbedder(weights_path=args.arcface_weights, device=device)
    eot = EoTWrapper(
        model=face_model,
        num_samples=5,  # Fewer for training speed
        enable_jpeg=True,
        enable_resize=True,
        enable_gaussian=True,
        enable_crop=False,
    )

    # Load encoder from Phase 1
    encoder, image_size = create_encoder(args.encoder_type, args.epsilon, device)
    prefix = checkpoint_prefix(args.encoder_type)
    if args.checkpoint:
        encoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded encoder from {args.checkpoint}")

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr * 0.1)

    # Dataset with train/val split
    train_dataset = FaceDataset(args.data_dir, image_size=image_size, split="train")
    val_dataset = FaceDataset(args.data_dir, image_size=image_size, split="val")
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers) if len(val_dataset) > 0 else None

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = create_writer(args, "e2e")
    best_cos = 1.0  # Lower is better
    patience_counter = 0

    try:
        for epoch in range(args.epochs):
            encoder.train()
            total_cos = 0
            n = 0

            for clean in tqdm(train_loader, desc=f"E2E Epoch {epoch+1}/{args.epochs}"):
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
            tb_log(writer, "train/cos_sim", avg_cos, epoch)
            tb_log(writer, "train/lr", optimizer.param_groups[0]["lr"], epoch)

            # Validation
            val_cos = avg_cos
            if val_loader is not None:
                val_cos = _validate_e2e(encoder, face_model, val_loader, device)
                print(f"  Val cos_sim={val_cos:.4f}")
                tb_log(writer, "val/cos_sim", val_cos, epoch)

            if val_cos < best_cos:
                best_cos = val_cos
                patience_counter = 0
                torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, f"{prefix}best.pt"))
                print(f"  Saved best checkpoint (cos_sim={best_cos:.4f})")
            else:
                patience_counter += 1

            torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, f"{prefix}latest.pt"))

            if patience_counter >= args.patience:
                print(f"  Early stopping after {args.patience} epochs without improvement")
                break
    finally:
        if writer:
            writer.close()


def _validate_e2e(encoder, face_model, val_loader, device):
    """Validation loop for e2e phase."""
    encoder.eval()
    total_cos = 0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            clean_emb = face_model(batch)
            delta = encoder(batch)
            x_adv = (batch + delta).clamp(0, 1)
            adv_emb = face_model(x_adv)
            cos_sim = F.cosine_similarity(clean_emb, adv_emb, dim=1).mean()
            total_cos += cos_sim.item() * batch.shape[0]
            n += batch.shape[0]
    return total_cos / n if n > 0 else 1.0


# ---------------------------------------------------------------------------
# Phase 3: V2 End-to-End with LPIPS + CLIP + Semantic Mask
# ---------------------------------------------------------------------------

def train_v2_e2e(args):
    """
    v2 end-to-end training with all improvements:
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
    face_model = FaceEmbedder(weights_path=args.arcface_weights, device=device)

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
    encoder, image_size = create_encoder(args.encoder_type, args.epsilon, device)
    prefix = checkpoint_prefix(args.encoder_type)
    if args.checkpoint:
        encoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded encoder from {args.checkpoint}")

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr * 0.1)

    # Scheduler — ViT gets linear warmup before cosine annealing
    if args.encoder_type == "vit":
        warmup_epochs = 5
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - warmup_epochs, 1)
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Dataset with train/val split
    train_dataset = FaceDataset(args.data_dir, image_size=image_size, split="train")
    val_dataset = FaceDataset(args.data_dir, image_size=image_size, split="val")
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers) if len(val_dataset) > 0 else None

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = create_writer(args, "v2_e2e")
    best_loss = float("inf")
    patience_counter = 0

    try:
        for epoch in range(args.epochs):
            encoder.train()
            running = {"loss": 0, "arcface": 0, "clip": 0, "lpips": 0}
            n = 0

            for clean in tqdm(train_loader, desc=f"V2-E2E Epoch {epoch+1}/{args.epochs}"):
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

            tb_log(writer, "train/loss", avg["loss"], epoch)
            tb_log(writer, "train/arcface_cos", avg["arcface"], epoch)
            tb_log(writer, "train/clip_cos", avg["clip"], epoch)
            tb_log(writer, "train/lpips", avg["lpips"], epoch)
            tb_log(writer, "train/lr", optimizer.param_groups[0]["lr"], epoch)

            # Validation
            val_loss = avg["loss"]
            if val_loader is not None:
                val_loss = _validate_v2(encoder, face_model, val_loader, device, semantic_mask)
                print(f"  Val cos_sim={val_loss:.4f}")
                tb_log(writer, "val/cos_sim", val_loss, epoch)

            # Save as production checkpoint (best.pt / vit_best.pt)
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, f"{prefix}best.pt"))
                print(f"  Saved best (loss={best_loss:.4f})")
            else:
                patience_counter += 1

            torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, f"{prefix}latest.pt"))

            if patience_counter >= args.patience:
                print(f"  Early stopping after {args.patience} epochs without improvement")
                break
    finally:
        if writer:
            writer.close()


def _validate_v2(encoder, face_model, val_loader, device, semantic_mask):
    """Validation loop for v2_e2e phase (ArcFace cos_sim)."""
    encoder.eval()
    total_cos = 0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            clean_emb = face_model(batch)
            delta = encoder(batch)
            if semantic_mask is not None:
                mask = semantic_mask(batch)
                delta = delta * mask
            x_adv = (batch + delta).clamp(0, 1)
            adv_emb = face_model(x_adv)
            cos_sim = F.cosine_similarity(clean_emb, adv_emb, dim=1).mean()
            total_cos += cos_sim.item() * batch.shape[0]
            n += batch.shape[0]
    return total_cos / n if n > 0 else 1.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PrivacyShield Noise Encoder")
    parser.add_argument("--phase", choices=["distill", "e2e", "v2_e2e", "generate"], required=True)
    parser.add_argument("--encoder-type", choices=["unet", "vit"], default="unet",
                        help="Encoder architecture: unet (~50ms) or vit (~170ms)")
    parser.add_argument("--data-dir", type=str, default="./data/faces")
    parser.add_argument("--pairs-dir", type=str, default="./data/pairs")
    parser.add_argument("--output-dir", type=str, default="./data/pairs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--arcface-weights", type=str, default="./weights/arcface_r100.pth",
                        help="Path to ArcFace IResNet-100 weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: 16 for unet, 8 for vit)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 1e-4 for unet, 5e-5 for vit)")
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eot-samples", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: 0 on Windows, 4 on Linux)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    # v2-specific
    parser.add_argument("--lambda-lpips", type=float, default=0.1, help="LPIPS penalty weight")
    parser.add_argument("--use-mask", action="store_true", help="Enable semantic face mask")

    args = parser.parse_args()

    # Smart defaults based on encoder type
    if args.batch_size is None:
        args.batch_size = 8 if args.encoder_type == "vit" else 16
    if args.lr is None:
        args.lr = 5e-5 if args.encoder_type == "vit" else 1e-4
    if args.num_workers is None:
        args.num_workers = default_num_workers()

    print(f"Encoder: {args.encoder_type}, Batch: {args.batch_size}, LR: {args.lr}, "
          f"Workers: {args.num_workers}")

    if args.phase == "generate":
        generate_pairs(args)
    elif args.phase == "distill":
        train_distill(args)
    elif args.phase == "e2e":
        train_e2e(args)
    elif args.phase == "v2_e2e":
        train_v2_e2e(args)
