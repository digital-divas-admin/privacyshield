"""
Download and prepare face datasets for PrivacyShield encoder training.

Supports:
  - FFHQ (70k faces) via HuggingFace `marcosv/ffhq-dataset`

Usage:
  # Full FFHQ dataset
  python scripts/prepare_data.py --output-dir ./data/faces --dataset ffhq

  # 5k subset for quick validation
  python scripts/prepare_data.py --output-dir ./data/faces --dataset ffhq --max-images 5000

  # Custom image size
  python scripts/prepare_data.py --output-dir ./data/faces --dataset ffhq --image-size 112
"""

import argparse
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def download_ffhq(output_dir: str, max_images: int = None, image_size: int = 256, val_split: float = 0.1):
    """Download FFHQ from HuggingFace and split into train/val."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: `datasets` package not installed.")
        print("Run: pip install datasets>=2.14.0")
        return

    # Create output dirs
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Use streaming to avoid downloading entire dataset upfront
    print("Loading FFHQ from HuggingFace (marcosv/ffhq-dataset) with streaming...")
    dataset = load_dataset("marcosv/ffhq-dataset", split="train", streaming=True)

    total = max_images or 70000
    split_idx = int(total * (1 - val_split))
    train_count = 0
    val_count = 0

    for idx, sample in enumerate(tqdm(dataset, total=total, desc="Downloading")):
        if max_images and idx >= max_images:
            break

        img = sample["image"]

        # Resize
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.LANCZOS)

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Split into train/val
        if idx < split_idx:
            img.save(train_dir / f"{idx:06d}.png")
            train_count += 1
        else:
            img.save(val_dir / f"{idx:06d}.png")
            val_count += 1

    print(f"Done: {train_count} train, {val_count} val images in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare face datasets")
    parser.add_argument("--output-dir", type=str, default="./data/faces",
                        help="Output directory for processed images")
    parser.add_argument("--dataset", choices=["ffhq"], default="ffhq",
                        help="Dataset to download")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to download (for subset training)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Resize images to this size (default: 256)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")

    args = parser.parse_args()

    if args.dataset == "ffhq":
        download_ffhq(args.output_dir, args.max_images, args.image_size, args.val_split)


if __name__ == "__main__":
    main()
