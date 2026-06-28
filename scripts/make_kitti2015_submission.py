#!/usr/bin/env python3
"""Generate a KITTI Stereo 2015 submission-ready disparity package.

This script writes first-frame disparity predictions as 16-bit PNG files under
``disp_0/`` and archives them as a zip file. It does not submit to the KITTI
server; the output is a ready-to-upload artifact plus a manifest for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import cv2
import numpy as np
import torch

code_dir = Path(__file__).resolve().parent
sys.path.append(str(code_dir.parent))

from core.utils import frame_utils  # noqa: E402
from scripts.eval_generalization import load_model  # noqa: E402
from scripts.eval_multiscale_refine import fuse_predictions, infer_at_scale, refine_prediction  # noqa: E402
from Utils import set_logging_format, set_seed  # noqa: E402


METHOD_PRESETS: dict[str, dict[str, Any]] = {
    "ffs8": {
        "valid_iters": 8,
        "scales": [1.0],
        "fusion": "native",
        "uncertainty_thresh": 0.35,
        "clip_pred": False,
        "median_kernel": 0,
        "local_consistency_refine": False,
        "median_residual_thresh": 1.0,
        "spread_refine_thresh": 0.0,
        "image_edge_percentile": 70.0,
        "adaptive_guided_refine": False,
        "guided_radius": 7,
        "guided_eps": 1e-3,
        "guided_blend": 0.65,
        "guided_residual_thresh": 1.5,
        "photometric_refine": False,
        "photometric_thresh": 0.08,
        "lr_refine_residual_thresh": 1.0,
        "smooth_percentile": 65.0,
        "metric_max_disp": 0.0,
    },
    "ffsomega_q": {
        "valid_iters": 16,
        "scales": [1.0],
        "fusion": "native",
        "uncertainty_thresh": 0.35,
        "clip_pred": True,
        "median_kernel": 3,
        "local_consistency_refine": True,
        "median_residual_thresh": 2.0,
        "spread_refine_thresh": 0.0,
        "image_edge_percentile": 90.0,
        "adaptive_guided_refine": False,
        "guided_radius": 7,
        "guided_eps": 1e-3,
        "guided_blend": 0.65,
        "guided_residual_thresh": 1.5,
        "photometric_refine": False,
        "photometric_thresh": 0.08,
        "lr_refine_residual_thresh": 1.0,
        "smooth_percentile": 65.0,
        "metric_max_disp": 0.0,
    },
    "kitti_light_aqr": {
        "valid_iters": 8,
        "scales": [1.0],
        "fusion": "native",
        "uncertainty_thresh": 0.35,
        "clip_pred": True,
        "median_kernel": 3,
        "local_consistency_refine": True,
        "median_residual_thresh": 0.5,
        "spread_refine_thresh": 0.0,
        "image_edge_percentile": 100.0,
        "adaptive_guided_refine": False,
        "guided_radius": 7,
        "guided_eps": 1e-3,
        "guided_blend": 0.65,
        "guided_residual_thresh": 1.5,
        "photometric_refine": False,
        "photometric_thresh": 0.08,
        "lr_refine_residual_thresh": 1.0,
        "smooth_percentile": 65.0,
        "metric_max_disp": 0.0,
    },
}


def parse_scales(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one scale is required")
    if any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("scales must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--base_model_dir", default=None, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--method", choices=sorted(METHOD_PRESETS), default="kitti_light_aqr")
    parser.add_argument("--frame_suffix", default="_10.png", type=str)
    parser.add_argument("--expected_count", default=200, type=int)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--max_disp", default=192, type=int)
    parser.add_argument("--metric_max_disp", default=None, type=float)
    parser.add_argument("--hiera", default=0, type=int)
    parser.add_argument("--small_ratio", default=0.5, type=float)
    parser.add_argument("--low_memory", action="store_true")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--valid_iters", default=None, type=int)
    parser.add_argument("--scales", default=None, type=parse_scales)
    parser.add_argument("--fusion", choices=["native", "mean", "median", "trimmed", "uncertainty_median"], default=None)
    parser.add_argument("--uncertainty_thresh", default=None, type=float)
    parser.add_argument("--clip_pred", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--median_kernel", default=None, type=int)
    parser.add_argument("--local_consistency_refine", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--median_residual_thresh", default=None, type=float)
    parser.add_argument("--spread_refine_thresh", default=None, type=float)
    parser.add_argument("--image_edge_percentile", default=None, type=float)
    parser.add_argument("--adaptive_guided_refine", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--guided_radius", default=None, type=int)
    parser.add_argument("--guided_eps", default=None, type=float)
    parser.add_argument("--guided_blend", default=None, type=float)
    parser.add_argument("--guided_residual_thresh", default=None, type=float)
    parser.add_argument("--photometric_refine", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--photometric_thresh", default=None, type=float)
    parser.add_argument("--lr_refine_residual_thresh", default=None, type=float)
    parser.add_argument("--smooth_percentile", default=None, type=float)
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = METHOD_PRESETS[args.method]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_test_pairs(args: argparse.Namespace) -> list[dict[str, Path | str]]:
    root = Path(args.data_root)
    left_dir = root / "testing" / "image_2"
    right_dir = root / "testing" / "image_3"
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise FileNotFoundError(f"KITTI testing/image_2 and testing/image_3 are required under {root}")

    pairs: list[dict[str, Path | str]] = []
    for left_path in sorted(left_dir.glob(f"*{args.frame_suffix}")):
        right_path = right_dir / left_path.name
        if not right_path.exists():
            raise FileNotFoundError(f"Missing right image for {left_path.name}: {right_path}")
        pairs.append({"id": left_path.name, "left": left_path, "right": right_path})

    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]

    if args.max_samples is None and len(pairs) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} KITTI test pairs matching *{args.frame_suffix}, found {len(pairs)}."
        )
    if not pairs:
        raise RuntimeError(f"No KITTI test pairs found in {left_dir} with suffix {args.frame_suffix}")
    return pairs


def write_kitti_disparity_png(path: Path, disp: np.ndarray) -> None:
    encoded = np.clip(disp.astype(np.float32) * 256.0, 0.0, 65535.0).round().astype(np.uint16)
    if not cv2.imwrite(str(path), encoded):
        raise IOError(f"Failed to write {path}")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_zip(disp_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for png_path in sorted(disp_dir.glob("*.png")):
            zf.write(png_path, arcname=f"disp_0/{png_path.name}")


@torch.no_grad()
def main() -> None:
    args = apply_preset(parse_args())
    set_logging_format()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for KITTI submission generation.")

    out_dir = Path(args.out_dir)
    disp_dir = out_dir / "disp_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    disp_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_test_pairs(args)
    model = load_model(args, torch.device("cuda"))
    logging.info("method: %s", args.method)
    logging.info("test pairs: %d", len(pairs))
    logging.info("out_dir: %s", out_dir)

    start_wall = time.perf_counter()
    total_infer_seconds = 0.0
    per_sample: list[dict[str, Any]] = []

    for idx, pair in enumerate(pairs):
        left = frame_utils.read_gen(str(pair["left"])).astype(np.uint8)
        right = frame_utils.read_gen(str(pair["right"])).astype(np.uint8)
        if left.ndim == 2:
            left = np.tile(left[..., None], (1, 1, 3))
        if right.ndim == 2:
            right = np.tile(right[..., None], (1, 1, 3))
        left = left[..., :3]
        right = right[..., :3]

        preds = []
        elapsed = 0.0
        for scale in args.scales:
            pred_s, elapsed_s = infer_at_scale(model, left, right, scale, args)
            preds.append(pred_s)
            elapsed += elapsed_s
        pred, spread = fuse_predictions(preds, args)
        pred = refine_prediction(pred, args, left=left, right=right, spread=spread)

        out_png = disp_dir / str(pair["id"])
        write_kitti_disparity_png(out_png, pred)
        total_infer_seconds += elapsed
        per_sample.append(
            {
                "id": str(pair["id"]),
                "shape": list(pred.shape),
                "infer_seconds": elapsed,
                "disp_min": float(np.nanmin(pred)),
                "disp_max": float(np.nanmax(pred)),
                "disp_mean": float(np.nanmean(pred)),
                "scale_spread_p95": float(np.nanpercentile(spread, 95)),
                "png_bytes": out_png.stat().st_size,
            }
        )
        logging.info("[%03d/%03d] %s %.1f ms", idx + 1, len(pairs), pair["id"], elapsed * 1000.0)

    zip_path = out_dir / f"kitti2015_{args.method}_disp_0.zip"
    make_zip(disp_dir, zip_path)
    png_count = len(list(disp_dir.glob("*.png")))
    if png_count != len(pairs):
        raise RuntimeError(f"PNG count mismatch: wrote {png_count}, expected {len(pairs)}")

    manifest = {
        "artifact_type": "KITTI2015 stereo submission-ready disparity package",
        "status": "prepared_not_submitted",
        "method": args.method,
        "notes": [
            "This package was generated locally and has not been submitted to the KITTI online evaluation server.",
            "Only first-frame stereo test pairs matching *_10.png are included.",
            "Disparities are encoded as uint16 PNG using disparity * 256, matching the project readDispKITTI convention.",
        ],
        "kitti_official_context": {
            "benchmark_url": "https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo",
            "test_pairs": len(pairs),
            "metric_family": "D1 bad pixels with 3px / 5% outlier rule",
        },
        "model_dir": str(Path(args.model_dir).resolve()),
        "base_model_dir": str(Path(args.base_model_dir).resolve()) if args.base_model_dir else None,
        "data_root": str(Path(args.data_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "disp_dir": str(disp_dir.resolve()),
        "zip_path": str(zip_path.resolve()),
        "zip_sha256": sha256_file(zip_path),
        "png_count": png_count,
        "frame_suffix": args.frame_suffix,
        "expected_count": args.expected_count,
        "avg_infer_ms": 1000.0 * total_infer_seconds / max(1, len(pairs)),
        "wall_seconds": time.perf_counter() - start_wall,
        "parameters": {
            "valid_iters": args.valid_iters,
            "max_disp": args.max_disp,
            "scales": args.scales,
            "fusion": args.fusion,
            "uncertainty_thresh": args.uncertainty_thresh,
            "clip_pred": bool(args.clip_pred),
            "median_kernel": args.median_kernel,
            "local_consistency_refine": bool(args.local_consistency_refine),
            "median_residual_thresh": args.median_residual_thresh,
            "spread_refine_thresh": args.spread_refine_thresh,
            "image_edge_percentile": args.image_edge_percentile,
            "adaptive_guided_refine": bool(args.adaptive_guided_refine),
            "guided_radius": args.guided_radius,
            "guided_eps": args.guided_eps,
            "guided_blend": args.guided_blend,
            "guided_residual_thresh": args.guided_residual_thresh,
            "photometric_refine": bool(args.photometric_refine),
            "photometric_thresh": args.photometric_thresh,
            "lr_refine_residual_thresh": args.lr_refine_residual_thresh,
            "smooth_percentile": args.smooth_percentile,
            "metric_max_disp": args.metric_max_disp,
            "hiera": args.hiera,
            "small_ratio": args.small_ratio,
            "low_memory": bool(args.low_memory),
            "seed": args.seed,
        },
        "per_sample": per_sample,
    }
    manifest_path = out_dir / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("zip: %s", zip_path)
    logging.info("manifest: %s", manifest_path)
    logging.info("zip sha256: %s", manifest["zip_sha256"])


if __name__ == "__main__":
    main()

