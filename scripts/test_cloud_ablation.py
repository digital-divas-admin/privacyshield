"""
Cloud Model Loss Ablation Test

Systematic ablation to find WHICH loss components transfer to cloud models,
testing against BOTH Gemini 3 Pro AND SeedReam 4.5.

The brute force ceiling test showed eps=64/255 breaks Gemini but with heavy
visible noise (PSNR=17dB). At eps=8/255, Gemini avg_sim=0.428 (VULNERABLE).
We have 5 loss components implemented but disabled (weight=0.0) in all prior
cloud tests: DINOv2 CLS, DINOv2 patch, multi-layer CLIP, Laplacian highfreq,
and attack LPIPS.

Three phases: generate (GPU), test (API), summary.

Usage:
  python scripts/test_cloud_ablation.py --phase generate
  python scripts/test_cloud_ablation.py --phase test --num-trials 3
  python scripts/test_cloud_ablation.py --phase all --num-trials 3
  python scripts/test_cloud_ablation.py --summary-only
"""

import sys
import os
import json
import time
import math
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Config definitions (10 ablation configs)
# ---------------------------------------------------------------------------

def build_configs() -> Dict[str, dict]:
    """
    Return config overrides for each ablation configuration.
    Keys are PipelineConfig field names; values override defaults.
    All use 50 PGD steps, v2_full mode. Step size = epsilon / 4.
    """
    configs = {}

    # 1. Baseline: current production config (eps=8/255)
    configs["baseline_default"] = dict(
        epsilon=8 / 255, step_size=2 / 255, num_steps=50,
    )

    # 2-7. Single-component additions (eps=8/255, add ONE to default)
    configs["+dino_cls"] = dict(
        epsilon=8 / 255, step_size=2 / 255, num_steps=50,
        gamma_dino_cls=0.5,
    )
    configs["+dino_patch"] = dict(
        epsilon=8 / 255, step_size=2 / 255, num_steps=50,
        gamma_dino_patch=0.3,
    )
    configs["+dino_both"] = dict(
        epsilon=8 / 255, step_size=2 / 255, num_steps=50,
        gamma_dino_cls=0.5, gamma_dino_patch=0.3,
    )
    configs["+clip_multi"] = dict(
        epsilon=8 / 255, step_size=2 / 255, num_steps=50,
        gamma_clip_multi_layer=0.3,
    )
    configs["+highfreq"] = dict(
        epsilon=8 / 255, step_size=2 / 255, num_steps=50,
        gamma_highfreq=0.3,
    )
    configs["+atk_lpips"] = dict(
        epsilon=8 / 255, step_size=2 / 255, num_steps=50,
        attack_lpips=0.3, lambda_lpips=0.0,
    )

    # 8. Kitchen sink at eps=8/255 (all components)
    configs["kitchen_sink_8"] = dict(
        epsilon=8 / 255, step_size=2 / 255, num_steps=50,
        gamma_dino_cls=0.5, gamma_dino_patch=0.3,
        gamma_clip_multi_layer=0.3, gamma_highfreq=0.2,
        attack_lpips=0.3, lambda_lpips=0.0,
    )

    # 9-10. Kitchen sink at elevated epsilon
    configs["kitchen_sink_12"] = dict(
        epsilon=12 / 255, step_size=3 / 255, num_steps=50,
        gamma_dino_cls=0.5, gamma_dino_patch=0.3,
        gamma_clip_multi_layer=0.3, gamma_highfreq=0.2,
        attack_lpips=0.3, lambda_lpips=0.0,
    )
    configs["kitchen_sink_16"] = dict(
        epsilon=16 / 255, step_size=4 / 255, num_steps=50,
        gamma_dino_cls=0.5, gamma_dino_patch=0.3,
        gamma_clip_multi_layer=0.3, gamma_highfreq=0.2,
        attack_lpips=0.3, lambda_lpips=0.0,
    )

    return configs


ALL_CONFIG_NAMES = list(build_configs().keys())
CLOUD_MODELS = ["gemini", "seedream"]


# ---------------------------------------------------------------------------
# Google AI Studio client (Gemini 3 Pro)
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = "AIzaSyBQ5qrCj3juJPlLnqAJJB_QCHPB3-Yt60M"
GEMINI_MODEL = "gemini-3-pro-image-preview"


def _ensure_genai_client():
    from google import genai
    api_key = os.environ.get("GOOGLE_API_KEY", GOOGLE_API_KEY)
    return genai.Client(api_key=api_key)


def generate_from_reference_gemini(client, reference_bgr, prompt):
    """Send reference image + prompt to Gemini, return (output_bgr, error)."""
    from google.genai import types
    from PIL import Image

    try:
        rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        max_dim = 2048
        if max(pil_img.size) > max_dim:
            ratio = max_dim / max(pil_img.size)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, pil_img],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        if response.parts:
            for part in response.parts:
                if part.inline_data is not None:
                    raw_bytes = part.inline_data.data
                    nparr = np.frombuffer(raw_bytes, np.uint8)
                    out_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if out_bgr is not None:
                        return out_bgr, None

        text_parts = [p.text for p in (response.parts or []) if p.text]
        if text_parts:
            return None, f"No image found. Text: {' '.join(text_parts)[:200]}"
        return None, "No image found in model response"

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# SeedReam 4.5 client (OpenRouter)
# ---------------------------------------------------------------------------

_openrouter_client = None


def _ensure_openrouter_client():
    global _openrouter_client
    if _openrouter_client is not None:
        return _openrouter_client
    from core.openrouter import OpenRouterClient
    _openrouter_client = OpenRouterClient()
    if not _openrouter_client.is_available:
        print("WARNING: OPENROUTER_API_KEY not set. SeedReam tests will fail.")
    return _openrouter_client


def generate_from_reference_seedream(client, reference_bgr, prompt):
    """Send reference image + prompt to SeedReam 4.5, return (output_bgr, error)."""
    return client.generate_from_reference("seedream", reference_bgr, prompt)


# ---------------------------------------------------------------------------
# Face similarity (insightface buffalo_l)
# ---------------------------------------------------------------------------

_face_app = None


def _ensure_face_app():
    global _face_app
    if _face_app is not None:
        return _face_app
    from insightface.app import FaceAnalysis
    _face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def _get_face_sim(clean_bgr, output_bgr):
    """Cosine similarity between clean identity and output face."""
    app = _ensure_face_app()

    clean_faces = app.get(clean_bgr)
    if not clean_faces:
        return 0.0
    clean_face = max(clean_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    clean_emb = clean_face.normed_embedding
    if clean_emb is None:
        return 0.0
    clean_emb = clean_emb / (np.linalg.norm(clean_emb) + 1e-8)

    out_faces = app.get(output_bgr)
    if not out_faces:
        return 0.0
    out_face = max(out_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    out_emb = out_face.normed_embedding
    if out_emb is None:
        return 0.0
    out_emb = out_emb / (np.linalg.norm(out_emb) + 1e-8)

    return float(np.dot(clean_emb, out_emb))


def classify_trial(output_bgr, error, clean_bgr, threshold=0.3):
    """Returns (status, sim)."""
    if error and "No image found" in error:
        return "REFUSED", -1.0
    if error:
        return "ERROR", 0.0
    sim = _get_face_sim(clean_bgr, output_bgr)
    if sim < threshold:
        return "PROTECTED", sim
    return "VULNERABLE", sim


# ---------------------------------------------------------------------------
# Results I/O (atomic save)
# ---------------------------------------------------------------------------

def _save_results(results, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    shutil.move(tmp, str(path))


def _load_results(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_psnr(clean_bgr, prot_bgr):
    """PSNR in dB between two uint8 BGR images."""
    mse = np.mean((clean_bgr.astype(np.float64) - prot_bgr.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(255.0 ** 2 / mse)


def tensor_to_bgr(x_tensor):
    """Convert (1, 3, H, W) float [0,1] tensor to uint8 BGR numpy."""
    import torch
    with torch.no_grad():
        img = x_tensor.squeeze(0).cpu().clamp(0, 1)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Phase 1: Generate protected images
# ---------------------------------------------------------------------------

def run_generate(args):
    import torch
    from core.pipeline import ProtectionPipeline, PipelineConfig

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [int(x.strip()) for x in args.images.split(",")]
    all_configs = build_configs()

    config_names = args.configs.split(",") if args.configs else ALL_CONFIG_NAMES
    config_names = [c.strip() for c in config_names]
    for c in config_names:
        if c not in all_configs:
            print(f"ERROR: Unknown config '{c}'. Available: {ALL_CONFIG_NAMES}")
            sys.exit(1)

    # Load source images
    source_images = {}
    for idx in images:
        path = source_dir / f"{idx}_clean.png"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping image {idx}")
            continue
        bgr = cv2.imread(str(path))
        source_images[idx] = bgr
        print(f"Loaded image {idx}: {bgr.shape[1]}x{bgr.shape[0]}")

    if not source_images:
        print("ERROR: No source images found")
        sys.exit(1)

    # Load generation results for resume
    gen_results_path = output_dir / "generation_results.json"
    gen_results = _load_results(gen_results_path) if args.resume else None
    if gen_results is None:
        gen_results = {"images": {}, "configs": {}}

    # Determine which (config, image) pairs need generation
    work = []
    for cname in config_names:
        for idx in source_images:
            out_path = output_dir / f"{idx}_{cname}.png"
            key = f"{idx}_{cname}"
            if args.resume and out_path.exists() and key in gen_results.get("images", {}):
                print(f"  SKIP (exists): {key}")
                continue
            work.append((cname, idx))

    if not work:
        print("All images already generated. Use without --resume to regenerate.")
        return

    print(f"\n{len(work)} (config, image) pairs to generate")

    # Setup pipeline once with DINOv2 enabled so all configs can use it.
    base_cfg = PipelineConfig(
        gamma_dino_cls=0.1,  # >0 triggers DINOv2 loading
        gamma_dino_patch=0.1,
        verbose=True,
    )
    # Resolve weight paths relative to project root
    project_root = Path(__file__).parent.parent
    arcface_weights = str(project_root / "weights" / "arcface_r100.pth")
    adaface_weights = str(project_root / "weights" / "adaface_ir101.pth")
    bisenet_weights = str(project_root / "weights" / "bisenet_face.pth")

    if not Path(arcface_weights).exists():
        print(f"WARNING: ArcFace weights not found at {arcface_weights}")
    if Path(adaface_weights).exists():
        base_cfg.adaface_weights = adaface_weights
    if Path(bisenet_weights).exists():
        base_cfg.bisenet_weights = bisenet_weights

    print("\nInitializing pipeline (loading all models once)...")
    pipeline = ProtectionPipeline(base_cfg)
    pipeline.setup(arcface_weights=arcface_weights)
    print("Pipeline ready.\n")

    # Group work by config to minimize config switching
    from collections import OrderedDict
    work_by_config = OrderedDict()
    for cname, idx in work:
        work_by_config.setdefault(cname, []).append(idx)

    done = 0
    total = len(work)

    for cname, img_indices in work_by_config.items():
        overrides = all_configs[cname]
        print(f"\n{'='*60}")
        print(f"Config: {cname}")
        print(f"  Overrides: {overrides}")
        print(f"{'='*60}")

        # Reset to defaults, then apply overrides
        default_cfg = PipelineConfig()
        for field_name in vars(default_cfg):
            if field_name.startswith("_"):
                continue
            setattr(pipeline.config, field_name, getattr(default_cfg, field_name))
        for k, v in overrides.items():
            if hasattr(pipeline.config, k):
                setattr(pipeline.config, k, v)
            else:
                print(f"  WARNING: unknown config field '{k}'")
        pipeline.config.verbose = True

        # Store config info
        gen_results["configs"][cname] = {
            k: (float(v) if isinstance(v, float) else v)
            for k, v in overrides.items()
        }

        for idx in img_indices:
            done += 1
            key = f"{idx}_{cname}"
            print(f"\n[{done}/{total}] Image {idx}, config {cname}")

            image_bgr = source_images[idx]

            try:
                x_prot, metrics = pipeline.protect_full(image_bgr)
            except Exception as e:
                print(f"  ERROR: {e}")
                gen_results["images"][key] = {"error": str(e)}
                _save_results(gen_results, gen_results_path)
                continue

            if x_prot is None:
                print(f"  FAILED: {metrics.get('error', 'unknown')}")
                gen_results["images"][key] = {"error": metrics.get("error", "no face")}
                _save_results(gen_results, gen_results_path)
                continue

            # Convert to BGR and save
            prot_bgr = tensor_to_bgr(x_prot)
            out_path = output_dir / f"{idx}_{cname}.png"
            cv2.imwrite(str(out_path), prot_bgr)

            # Compute PSNR
            psnr = compute_psnr(image_bgr, prot_bgr)

            # Record metrics
            record = {
                "arcface_cos_sim": metrics.get("arcface_cos_sim"),
                "clip_patch_cos_sim": metrics.get("clip_patch_cos_sim"),
                "delta_linf": metrics.get("delta_linf"),
                "processing_time_s": round(metrics.get("processing_time_s", 0), 1),
                "psnr_db": round(psnr, 1),
            }
            if "per_model_similarity" in metrics:
                record["per_model_similarity"] = {
                    k: round(v, 4) for k, v in metrics["per_model_similarity"].items()
                }

            gen_results["images"][key] = record
            _save_results(gen_results, gen_results_path)

            sim_str = f"{metrics.get('arcface_cos_sim', 0):.4f}"
            per_model = metrics.get("per_model_similarity", {})
            extra = ", ".join(f"{k}={v:.3f}" for k, v in per_model.items() if k != "arcface")
            print(f"  Saved: {out_path.name}")
            print(f"  ArcFace={sim_str}, PSNR={psnr:.1f}dB, "
                  f"time={metrics.get('processing_time_s', 0):.1f}s")
            if extra:
                print(f"  Ensemble: {extra}")

    print(f"\nGeneration complete. Results: {gen_results_path}")
    _print_generation_summary(gen_results, images, config_names)


def _print_generation_summary(gen_results, images, config_names):
    """Print local metrics summary table."""
    print(f"\n{'='*80}")
    print("LOCAL METRICS (generation phase)")
    print(f"{'='*80}")
    print(f"{'Config':<20s}| {'Eps':>9s} | {'ArcFace':>8s} | {'PSNR':>7s} | {'Time':>6s} | Per-model")
    print(f"{'-'*20}+{'-'*11}+{'-'*10}+{'-'*9}+{'-'*8}+{'-'*30}")

    for cname in config_names:
        cfg_info = gen_results.get("configs", {}).get(cname, {})
        eps = cfg_info.get("epsilon", 8 / 255)
        eps_str = f"{eps*255:.0f}/255"

        arcface_vals = []
        psnr_vals = []
        time_vals = []
        per_model_all = {}

        for idx in images:
            key = f"{idx}_{cname}"
            rec = gen_results.get("images", {}).get(key, {})
            if "error" in rec:
                continue
            if rec.get("arcface_cos_sim") is not None:
                arcface_vals.append(rec["arcface_cos_sim"])
            if rec.get("psnr_db") is not None:
                psnr_vals.append(rec["psnr_db"])
            if rec.get("processing_time_s") is not None:
                time_vals.append(rec["processing_time_s"])
            for k, v in rec.get("per_model_similarity", {}).items():
                per_model_all.setdefault(k, []).append(v)

        arc_str = f"{np.mean(arcface_vals):.3f}" if arcface_vals else "--"
        psnr_str = f"{np.mean(psnr_vals):.1f}" if psnr_vals else "--"
        time_str = f"{np.mean(time_vals):.0f}s" if time_vals else "--"

        pm_parts = []
        for k, vals in per_model_all.items():
            if k != "arcface":
                pm_parts.append(f"{k}={np.mean(vals):.3f}")
        pm_str = ", ".join(pm_parts) if pm_parts else ""

        print(f"{cname:<20s}| {eps_str:>9s} | {arc_str:>8s} | {psnr_str:>7s} | {time_str:>6s} | {pm_str}")

    print()


# ---------------------------------------------------------------------------
# Phase 2: Test against cloud models
# ---------------------------------------------------------------------------

def _run_cloud_test_for_model(
    model_name: str,
    args,
    config_names: List[str],
    images: List[int],
    clean_images: Dict[int, np.ndarray],
    output_dir: Path,
):
    """Run cloud API tests for a single model (gemini or seedream)."""
    results_path = output_dir / f"{model_name}_results.json"

    # Init client
    if model_name == "gemini":
        print("Initializing Gemini client...")
        client = _ensure_genai_client()
        print(f"Model: {GEMINI_MODEL}")
        gen_fn = lambda ref, prompt: generate_from_reference_gemini(client, ref, prompt)
    elif model_name == "seedream":
        print("Initializing SeedReam 4.5 client (OpenRouter)...")
        client = _ensure_openrouter_client()
        if not client.is_available:
            print("ERROR: OPENROUTER_API_KEY not set. Skipping SeedReam tests.")
            return
        print("Model: SeedReam 4.5 (bytedance-seed/seedream-4.5)")
        gen_fn = lambda ref, prompt: generate_from_reference_seedream(client, ref, prompt)
    else:
        print(f"ERROR: Unknown model '{model_name}'")
        return

    # Load / init results
    results = _load_results(results_path) if args.resume else None
    if results is None:
        results = {
            "config": {
                "num_trials": args.num_trials,
                "images": images,
                "configs": config_names,
                "prompt": args.prompt,
                "threshold": args.threshold,
                "model": model_name,
            },
            "results": {},
        }
    else:
        results["config"]["num_trials"] = args.num_trials

    total_cells = len(config_names) * len(clean_images)
    cell = 0
    total_calls = 0
    skipped = 0

    print(f"\n{'='*74}")
    print(f"CLOUD ABLATION — {model_name.upper()} TEST")
    print(f"N={args.num_trials}, {len(config_names)} configs, {len(clean_images)} images")
    print(f"{'='*74}")

    for cname in config_names:
        for idx in clean_images:
            cell += 1
            key = f"{idx}_{cname}"

            prot_path = output_dir / f"{key}.png"
            if not prot_path.exists():
                print(f"[{cell}/{total_cells}] {key}: protected image missing, skip")
                continue

            prot_bgr = cv2.imread(str(prot_path))
            clean_bgr = clean_images[idx]

            if key not in results["results"]:
                results["results"][key] = []

            existing = len(results["results"][key])
            remaining = args.num_trials - existing

            if remaining <= 0 and args.resume:
                skipped += args.num_trials
                print(f"[{cell}/{total_cells}] {key}: {existing}/{args.num_trials} done (skip)")
                continue
            elif remaining <= 0:
                results["results"][key] = []
                existing = 0

            if args.resume and existing > 0:
                skipped += existing
                print(f"[{cell}/{total_cells}] {key}: resuming from {existing + 1}")

            start_trial = existing if args.resume else 0
            for t in range(start_trial, args.num_trials):
                label = f"[{cell}/{total_cells}] {key} trial {t+1}/{args.num_trials}"
                print(f"  {label}...", end=" ", flush=True)

                retry = 0
                while True:
                    output_bgr, err = gen_fn(prot_bgr, args.prompt)
                    if err and ("429" in str(err) or "RESOURCE_EXHAUSTED" in str(err)
                                or "rate" in str(err).lower()):
                        retry += 1
                        if retry > 3:
                            print("rate limited, giving up")
                            break
                        backoff = args.sleep * (2 ** retry)
                        print(f"rate limited, backoff {backoff:.0f}s...", end=" ", flush=True)
                        time.sleep(backoff)
                        continue
                    break

                status, sim = classify_trial(output_bgr, err, clean_bgr, args.threshold)

                # Save output image for inspection
                if output_bgr is not None:
                    out_img_path = output_dir / f"{key}_{model_name}_t{t}.png"
                    cv2.imwrite(str(out_img_path), output_bgr)

                results["results"][key].append({
                    "status": status,
                    "sim": round(sim, 4),
                    "error": err[:200] if err else None,
                })
                total_calls += 1

                sim_str = f"{sim:.4f}" if sim != -1.0 else "REFUSED"
                print(f"{status} (sim={sim_str})")

                _save_results(results, results_path)
                time.sleep(args.sleep)

    print(f"\n{model_name.upper()} done. API calls: {total_calls}, skipped: {skipped}")
    _save_results(results, results_path)


def run_test(args):
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [int(x.strip()) for x in args.images.split(",")]
    config_names = args.configs.split(",") if args.configs else ALL_CONFIG_NAMES
    config_names = [c.strip() for c in config_names]

    models = [m.strip() for m in args.models.split(",")]
    for m in models:
        if m not in CLOUD_MODELS:
            print(f"ERROR: Unknown model '{m}'. Available: {CLOUD_MODELS}")
            sys.exit(1)

    # Load clean images for similarity comparison
    clean_images = {}
    for idx in images:
        path = source_dir / f"{idx}_clean.png"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping image {idx}")
            continue
        clean_images[idx] = cv2.imread(str(path))

    if not clean_images:
        print("ERROR: No clean images found")
        sys.exit(1)

    for model_name in models:
        _run_cloud_test_for_model(
            model_name, args, config_names, images, clean_images, output_dir,
        )

    print_summary(args)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _load_model_results(output_dir: Path, model_name: str):
    """Load results for a cloud model, return dict or None."""
    return _load_results(output_dir / f"{model_name}_results.json")


def _compute_avg_sim(model_results, images, cname, threshold):
    """Compute (avg_sim, refusals, total_trials) for a config across images."""
    if model_results is None:
        return None, 0, 0

    refusals = 0
    non_refused_sims = []
    total_trials = 0

    for idx in images:
        key = f"{idx}_{cname}"
        trials = model_results.get("results", {}).get(key, [])
        for t in trials:
            if t["status"] == "ERROR":
                continue
            total_trials += 1
            if t["status"] == "REFUSED":
                refusals += 1
            else:
                non_refused_sims.append(t["sim"])

    avg_sim = np.mean(non_refused_sims) if non_refused_sims else None
    return avg_sim, refusals, total_trials


def print_summary(args):
    output_dir = Path(args.output_dir)
    gen_results = _load_results(output_dir / "generation_results.json")
    gemini_results = _load_model_results(output_dir, "gemini")
    seedream_results = _load_model_results(output_dir, "seedream")

    if not gen_results and not gemini_results and not seedream_results:
        print("No results found.")
        return

    images = [int(x.strip()) for x in args.images.split(",")]
    config_names = args.configs.split(",") if args.configs else ALL_CONFIG_NAMES
    config_names = [c.strip() for c in config_names]
    all_configs = build_configs()
    threshold = args.threshold

    print(f"\n{'='*100}")
    print(f"CLOUD MODEL LOSS ABLATION TEST")
    print(f"{'='*100}")

    # Main table header
    print(f"\n{'Config':<20s}| {'Eps':>9s} | {'PSNR':>7s} | "
          f"{'Gemini Sim':>10s} | {'SR4.5 Sim':>10s} | {'Local Arc':>10s} | Components")
    print(f"{'-'*20}+{'-'*11}+{'-'*9}+{'-'*12}+{'-'*12}+{'-'*12}+{'-'*25}")

    # Collect per-config averages for marginal analysis
    baseline_gemini_sim = None
    baseline_seedream_sim = None
    config_sims = {}  # cname -> (gemini_avg, seedream_avg)

    for cname in config_names:
        overrides = all_configs.get(cname, {})
        eps = overrides.get("epsilon", 8 / 255)
        eps_str = f"{eps*255:.0f}/255"

        # Gemini avg sim
        gem_avg, gem_ref, gem_total = _compute_avg_sim(
            gemini_results, images, cname, threshold,
        )
        gem_str = f"{gem_avg:.3f}" if gem_avg is not None else "--"
        if gem_ref > 0:
            gem_str += f" R{gem_ref}"

        # SeedReam avg sim
        sr_avg, sr_ref, sr_total = _compute_avg_sim(
            seedream_results, images, cname, threshold,
        )
        sr_str = f"{sr_avg:.3f}" if sr_avg is not None else "--"
        if sr_ref > 0:
            sr_str += f" R{sr_ref}"

        config_sims[cname] = (gem_avg, sr_avg)

        if cname == "baseline_default":
            baseline_gemini_sim = gem_avg
            baseline_seedream_sim = sr_avg

        # Local metrics
        local_arcface = []
        local_psnr = []
        if gen_results:
            for idx in images:
                key = f"{idx}_{cname}"
                rec = gen_results.get("images", {}).get(key, {})
                if rec.get("arcface_cos_sim") is not None:
                    local_arcface.append(rec["arcface_cos_sim"])
                if rec.get("psnr_db") is not None:
                    local_psnr.append(rec["psnr_db"])

        arc_str = f"{np.mean(local_arcface):.3f}" if local_arcface else "--"
        psnr_str = f"{np.mean(local_psnr):.1f}" if local_psnr else "--"

        # Components description
        components = []
        if overrides.get("gamma_dino_cls", 0) > 0:
            components.append("dino_cls")
        if overrides.get("gamma_dino_patch", 0) > 0:
            components.append("dino_patch")
        if overrides.get("gamma_clip_multi_layer", 0) > 0:
            components.append("clip_ml")
        if overrides.get("gamma_highfreq", 0) > 0:
            components.append("highfreq")
        if overrides.get("attack_lpips", 0) > 0:
            components.append("atk_lpips")
        comp_str = "+".join(components) if components else "default"

        print(f"{cname:<20s}| {eps_str:>9s} | {psnr_str:>7s} | "
              f"{gem_str:>10s} | {sr_str:>10s} | {arc_str:>10s} | {comp_str}")

    # Marginal impact analysis
    if baseline_gemini_sim is not None or baseline_seedream_sim is not None:
        print(f"\n{'='*100}")
        print("MARGINAL IMPACT (delta from baseline_default):")
        print(f"  {'Config':<20s}  {'Gemini d':>10s}  {'SR4.5 d':>10s}  Verdict")
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*30}")

        for cname in config_names:
            if cname == "baseline_default":
                continue
            gem_avg, sr_avg = config_sims.get(cname, (None, None))

            gem_delta_str = "--"
            sr_delta_str = "--"
            verdict_parts = []

            if gem_avg is not None and baseline_gemini_sim is not None:
                gem_delta = gem_avg - baseline_gemini_sim
                gem_delta_str = f"{gem_delta:+.3f}"
                if gem_delta < -0.1:
                    verdict_parts.append("Gemini HELPS")
                elif gem_delta > 0.05:
                    verdict_parts.append("Gemini HURTS")

            if sr_avg is not None and baseline_seedream_sim is not None:
                sr_delta = sr_avg - baseline_seedream_sim
                sr_delta_str = f"{sr_delta:+.3f}"
                if sr_delta < -0.1:
                    verdict_parts.append("SR HELPS")
                elif sr_delta > 0.05:
                    verdict_parts.append("SR HURTS")

            verdict = ", ".join(verdict_parts) if verdict_parts else ""
            print(f"  {cname:<20s}  {gem_delta_str:>10s}  {sr_delta_str:>10s}  {verdict}")

    # Per-image detail
    has_any_cloud = gemini_results or seedream_results
    if has_any_cloud:
        print(f"\n{'-'*100}")
        print("Per-image detail:")
        for idx in images:
            print(f"\n  Image {idx}:")
            for cname in config_names:
                key = f"{idx}_{cname}"
                parts = []

                # Gemini
                if gemini_results:
                    trials = gemini_results.get("results", {}).get(key, [])
                    if trials:
                        statuses = "".join(t["status"][0] for t in trials)
                        sims = [t["sim"] for t in trials if t["status"] not in ("REFUSED", "ERROR")]
                        avg = f"{np.mean(sims):.3f}" if sims else "--"
                        ref = sum(1 for t in trials if t["status"] == "REFUSED")
                        parts.append(f"Gem={avg}[{statuses}]")
                        if ref > 0:
                            parts[-1] += f" R{ref}"

                # SeedReam
                if seedream_results:
                    trials = seedream_results.get("results", {}).get(key, [])
                    if trials:
                        statuses = "".join(t["status"][0] for t in trials)
                        sims = [t["sim"] for t in trials if t["status"] not in ("REFUSED", "ERROR")]
                        avg = f"{np.mean(sims):.3f}" if sims else "--"
                        ref = sum(1 for t in trials if t["status"] == "REFUSED")
                        parts.append(f"SR={avg}[{statuses}]")
                        if ref > 0:
                            parts[-1] += f" R{ref}"

                if parts:
                    print(f"    {cname:<20s}: {', '.join(parts)}")

    # Interpretation
    print(f"\n{'='*100}")
    print("INTERPRETATION:")

    # Find best single-component addition
    best_gem_delta = 0
    best_gem_name = None
    best_sr_delta = 0
    best_sr_name = None

    single_additions = ["+dino_cls", "+dino_patch", "+dino_both",
                        "+clip_multi", "+highfreq", "+atk_lpips"]

    for cname in single_additions:
        gem_avg, sr_avg = config_sims.get(cname, (None, None))
        if gem_avg is not None and baseline_gemini_sim is not None:
            delta = gem_avg - baseline_gemini_sim
            if delta < best_gem_delta:
                best_gem_delta = delta
                best_gem_name = cname
        if sr_avg is not None and baseline_seedream_sim is not None:
            delta = sr_avg - baseline_seedream_sim
            if delta < best_sr_delta:
                best_sr_delta = delta
                best_sr_name = cname

    if best_gem_name and best_gem_delta < -0.05:
        print(f"  Best single addition for Gemini: {best_gem_name} (d={best_gem_delta:+.3f})")
    elif baseline_gemini_sim is not None:
        print(f"  No single addition meaningfully improves Gemini protection at eps=8/255")

    if best_sr_name and best_sr_delta < -0.05:
        print(f"  Best single addition for SeedReam: {best_sr_name} (d={best_sr_delta:+.3f})")
    elif baseline_seedream_sim is not None:
        print(f"  No single addition meaningfully improves SeedReam protection at eps=8/255")

    # Kitchen sink analysis
    ks8_gem, ks8_sr = config_sims.get("kitchen_sink_8", (None, None))
    ks16_gem, ks16_sr = config_sims.get("kitchen_sink_16", (None, None))

    if ks8_gem is not None and baseline_gemini_sim is not None:
        if ks8_gem < threshold:
            print(f"  kitchen_sink_8 PROTECTS against Gemini (sim={ks8_gem:.3f} < {threshold})")
        elif ks16_gem is not None and ks16_gem < threshold:
            print(f"  kitchen_sink_8 VULNERABLE but kitchen_sink_16 PROTECTED against Gemini")
            print(f"    -> Need full loss ensemble + elevated epsilon (16/255)")
        else:
            print(f"  Even kitchen_sink_16 VULNERABLE against Gemini")
            print(f"    -> Current loss targets cannot break cloud models at eps<=16/255")

    # Gemini vs SeedReam comparison
    if gemini_results and seedream_results:
        gem_avgs = [v[0] for v in config_sims.values() if v[0] is not None]
        sr_avgs = [v[1] for v in config_sims.values() if v[1] is not None]
        if gem_avgs and sr_avgs:
            gem_mean = np.mean(gem_avgs)
            sr_mean = np.mean(sr_avgs)
            if abs(gem_mean - sr_mean) > 0.1:
                harder = "SeedReam" if sr_mean > gem_mean else "Gemini"
                print(f"  {harder} is harder to attack (avg sim: Gemini={gem_mean:.3f}, SR={sr_mean:.3f})")
                print(f"    -> Models use different identity encoders; may need model-specific strategies")
            else:
                print(f"  Gemini and SeedReam respond similarly (avg sim: Gemini={gem_mean:.3f}, SR={sr_mean:.3f})")

    print(f"{'='*100}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cloud model loss ablation test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", choices=["generate", "test", "all"], default="all")
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--output-dir", default="./test_outputs/cloud_ablation")
    parser.add_argument("--source-dir", default="./test_outputs/injection_ab")
    parser.add_argument("--images", default="0,1")
    parser.add_argument("--configs", default=None,
                        help=f"Comma-separated config names (default: all). Available: {ALL_CONFIG_NAMES}")
    parser.add_argument("--models", default="gemini,seedream",
                        help=f"Comma-separated cloud models to test (default: gemini,seedream). Available: {CLOUD_MODELS}")
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--prompt", default="a photo of this person wearing a business suit in an office")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    if args.summary_only:
        print_summary(args)
        return

    if args.phase in ("generate", "all"):
        run_generate(args)

    if args.phase in ("test", "all"):
        run_test(args)

    if args.phase == "generate":
        # Print local summary even without test phase
        gen_results = _load_results(Path(args.output_dir) / "generation_results.json")
        if gen_results:
            images = [int(x.strip()) for x in args.images.split(",")]
            config_names = args.configs.split(",") if args.configs else ALL_CONFIG_NAMES
            _print_generation_summary(gen_results, images, config_names)


if __name__ == "__main__":
    main()
