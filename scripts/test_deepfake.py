"""
Deepfake Tool Testing Script

Tests PrivacyShield protection against real deepfake pipelines:
  1. InsightFace inswapper_128 (what Roop uses)
  2. IP-Adapter FaceID Plus v2 (generative deepfakes)

Usage:
  python scripts/test_deepfake.py \
    --clean-image ./data/clean_face.jpg \
    --protected-image ./data/protected_face.png \
    --target-image ./data/target.jpg \
    --output-dir ./test_outputs

  # Skip IP-Adapter (for low-VRAM systems):
  python scripts/test_deepfake.py \
    --clean-image face.jpg \
    --protected-image face_protected.png \
    --target-image face.jpg \
    --skip-ipadapter
"""

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.deepfake_test import DeepfakeTestRegistry


def main():
    parser = argparse.ArgumentParser(
        description="Test PrivacyShield protection against real deepfake tools"
    )
    parser.add_argument("--clean-image", required=True, help="Path to clean (unprotected) face image")
    parser.add_argument("--protected-image", required=True, help="Path to protected face image")
    parser.add_argument("--target-image", default=None, help="Target image for face swap (defaults to clean image)")
    parser.add_argument("--output-dir", default=None, help="Directory to save output images")
    parser.add_argument("--skip-ipadapter", action="store_true", help="Skip IP-Adapter test (saves VRAM)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Cosine similarity threshold for protection")
    parser.add_argument("--prompt", default="a photo of a person", help="Prompt for IP-Adapter generation")
    args = parser.parse_args()

    # Load images
    clean_bgr = cv2.imread(args.clean_image)
    if clean_bgr is None:
        print(f"Error: Could not load clean image: {args.clean_image}")
        sys.exit(1)

    protected_bgr = cv2.imread(args.protected_image)
    if protected_bgr is None:
        print(f"Error: Could not load protected image: {args.protected_image}")
        sys.exit(1)

    target_bgr = None
    if args.target_image:
        target_bgr = cv2.imread(args.target_image)
        if target_bgr is None:
            print(f"Error: Could not load target image: {args.target_image}")
            sys.exit(1)
    else:
        target_bgr = clean_bgr.copy()

    print(f"{'='*70}")
    print("PrivacyShield -- Deepfake Tool Testing")
    print(f"{'='*70}")
    print(f"  Clean image:     {args.clean_image} ({clean_bgr.shape[1]}x{clean_bgr.shape[0]})")
    print(f"  Protected image: {args.protected_image} ({protected_bgr.shape[1]}x{protected_bgr.shape[0]})")
    print(f"  Target image:    {args.target_image or '(same as clean)'}")
    print(f"  Threshold:       {args.threshold}")
    print(f"  IP-Adapter:      {'skipped' if args.skip_ipadapter else 'enabled'}")
    print()

    # Run tests
    registry = DeepfakeTestRegistry()
    result = registry.run_full_test(
        clean_bgr=clean_bgr,
        protected_bgr=protected_bgr,
        target_bgr=target_bgr,
        run_inswapper=True,
        run_ipadapter=not args.skip_ipadapter,
        prompt=args.prompt,
        threshold=args.threshold,
    )

    # Save outputs if requested
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if result.inswapper and result.inswapper.clean_output is not None:
            cv2.imwrite(str(out_dir / "inswapper_from_clean.png"), result.inswapper.clean_output)
            cv2.imwrite(str(out_dir / "inswapper_from_protected.png"), result.inswapper.protected_output)
            print(f"Saved inswapper outputs to {out_dir}")

        if result.ipadapter and result.ipadapter.clean_output is not None:
            cv2.imwrite(str(out_dir / "ipadapter_from_clean.png"), result.ipadapter.clean_output)
            cv2.imwrite(str(out_dir / "ipadapter_from_protected.png"), result.ipadapter.protected_output)
            print(f"Saved IP-Adapter outputs to {out_dir}")

        print()

    # Print results
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    if result.inswapper:
        r = result.inswapper
        print()
        print(f"  Inswapper (Roop)")
        print(f"  {'-'*40}")
        if r.error:
            print(f"    Error: {r.error}")
        else:
            verdict = "PROTECTED" if r.protection_effective else "VULNERABLE"
            color = "\033[92m" if r.protection_effective else "\033[91m"
            reset = "\033[0m"
            print(f"    Swap from clean:     cos_sim = {r.clean_similarity:.4f}")
            print(f"    Swap from protected: cos_sim = {r.protected_similarity:.4f}")
            print(f"    Verdict: {color}{verdict}{reset} (threshold={args.threshold})")
            print(f"    Time: {r.processing_time_ms:.0f}ms")

    if result.ipadapter:
        r = result.ipadapter
        print()
        print(f"  IP-Adapter FaceID Plus v2")
        print(f"  {'-'*40}")
        if r.error:
            print(f"    Error: {r.error}")
        else:
            verdict = "PROTECTED" if r.protection_effective else "VULNERABLE"
            color = "\033[92m" if r.protection_effective else "\033[91m"
            reset = "\033[0m"
            print(f"    Gen from clean:     cos_sim = {r.clean_similarity:.4f}")
            print(f"    Gen from protected: cos_sim = {r.protected_similarity:.4f}")
            print(f"    Verdict: {color}{verdict}{reset} (threshold={args.threshold})")
            print(f"    Time: {r.processing_time_ms:.0f}ms")

    print()
    print(f"  {'-'*40}")
    verdict_map = {
        "protected": "\033[92mPROTECTED\033[0m -- all tested tools defeated",
        "partial": "\033[93mPARTIAL\033[0m -- some tools defeated",
        "vulnerable": "\033[91mVULNERABLE\033[0m -- protection ineffective",
        "error": "\033[91mERROR\033[0m -- tests could not complete",
        "untested": "UNTESTED",
    }
    print(f"  Overall: {verdict_map.get(result.overall_verdict, result.overall_verdict)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
