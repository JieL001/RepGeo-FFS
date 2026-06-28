import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    class _ImageIOFallback:
        @staticmethod
        def imread(path):
            return np.array(Image.open(path))

        @staticmethod
        def imwrite(path, array):
            Image.fromarray(array).save(path)

    imageio = _ImageIOFallback()

code_dir = Path(__file__).resolve().parent
sys.path.append(str(code_dir.parent))

from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity
from core.foundation_stereo import FastFoundationStereo
from core.utils import frame_utils
from core.utils.utils import InputPadder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="Serialized model_best.pth or checkpoint state dict.",
    )
    parser.add_argument(
        "--base_model_dir",
        default=None,
        type=str,
        help="Required only when model_dir points to a state-dict checkpoint such as last_state.pth.",
    )
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument(
        "--image_root",
        default=None,
        type=str,
        help="Optional image root when RGB images and GT are stored in different roots.",
    )
    parser.add_argument(
        "--gt_root",
        default=None,
        type=str,
        help="Optional GT root when disparities/masks are stored in a separate root.",
    )
    parser.add_argument("--dataset_name", default="custom", type=str)
    parser.add_argument("--left_name", default="im0.png", type=str)
    parser.add_argument("--right_name", default="im1.png", type=str)
    parser.add_argument("--disp_name", default="disp0GT.pfm", type=str)
    parser.add_argument(
        "--mask_name",
        default="mask0nocc.png",
        type=str,
        help="Set to empty string to skip visible-mask loading and evaluate on all valid GT pixels.",
    )
    parser.add_argument("--valid_iters", default=8, type=int)
    parser.add_argument("--max_disp", default=192, type=int)
    parser.add_argument(
        "--metric_max_disp",
        default=0.0,
        type=float,
        help="Optional GT disparity cap for evaluation. Use 0 to disable.",
    )
    parser.add_argument("--scale", default=1.0, type=float)
    parser.add_argument("--hiera", default=0, type=int)
    parser.add_argument("--small_ratio", default=0.5, type=float)
    parser.add_argument("--low_memory", action="store_true")
    parser.add_argument("--recursive", default=1, type=int)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--report_json", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def load_model(args, device):
    loaded = torch.load(args.model_dir, map_location="cpu", weights_only=False)

    if isinstance(loaded, FastFoundationStereo):
        model = loaded
    elif isinstance(loaded, dict) and "model_state" in loaded:
        if args.base_model_dir is None:
            raise ValueError("--base_model_dir is required when model_dir is a state-dict checkpoint.")
        base_model = torch.load(args.base_model_dir, map_location="cpu", weights_only=False)
        if not isinstance(base_model, FastFoundationStereo):
            raise TypeError(f"Expected FastFoundationStereo base model, got: {type(base_model)}")
        base_model.load_state_dict(loaded["model_state"])
        model = base_model
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(loaded)}")

    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.update_block.hidden_dim = model.update_block.disp_head.conv[0].in_channels
    model.update_block._ensure_refine_gate()
    model = model.to(device).eval()
    return model


def find_samples(args):
    default_root = Path(args.data_root)
    image_root = Path(args.image_root) if args.image_root is not None else default_root
    gt_root = Path(args.gt_root) if args.gt_root is not None else default_root
    if not image_root.exists():
        raise FileNotFoundError(image_root)
    if not gt_root.exists():
        raise FileNotFoundError(gt_root)

    disp_paths = gt_root.rglob(args.disp_name) if args.recursive else gt_root.glob(f"*/{args.disp_name}")
    samples = []
    for disp_path in sorted(disp_paths):
        gt_scene_dir = disp_path.parent
        rel_dir = gt_scene_dir.relative_to(gt_root)
        image_scene_dir = image_root / rel_dir
        left_path = image_scene_dir / args.left_name
        right_path = image_scene_dir / args.right_name
        mask_path = (gt_scene_dir / args.mask_name) if args.mask_name else None
        if not left_path.exists() or not right_path.exists():
            continue
        if mask_path is not None and not mask_path.exists():
            mask_path = None
        sample_id = str(rel_dir).replace("\\", "__").replace("/", "__")
        if not sample_id or sample_id == ".":
            sample_id = gt_scene_dir.name
        samples.append(
            {
                "id": sample_id,
                "scene_dir": image_scene_dir,
                "left": left_path,
                "right": right_path,
                "disp": disp_path,
                "mask": mask_path,
            }
        )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(
            f"No samples found with image_root={image_root} gt_root={gt_root} "
            f"and names {args.left_name}, {args.right_name}, {args.disp_name}, {args.mask_name}"
        )
    return samples


def read_sample(sample, scale):
    left = frame_utils.read_gen(str(sample["left"])).astype(np.uint8)
    right = frame_utils.read_gen(str(sample["right"])).astype(np.uint8)
    if left.ndim == 2:
        left = np.tile(left[..., None], (1, 1, 3))
    if right.ndim == 2:
        right = np.tile(right[..., None], (1, 1, 3))
    left = left[..., :3]
    right = right[..., :3]

    disp_gt = frame_utils.readPFM(str(sample["disp"])).astype(np.float32)
    if disp_gt.ndim == 3:
        disp_gt = disp_gt[..., 0]

    if sample["mask"] is not None:
        mask_noc = imageio.imread(sample["mask"])
        if mask_noc.ndim == 3:
            mask_noc = mask_noc[..., 0]
        mask_noc = mask_noc > 0
    else:
        mask_noc = np.isfinite(disp_gt) & (disp_gt > 0)

    orig_h, orig_w = left.shape[:2]
    if scale != 1.0:
        left = cv2.resize(left, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, dsize=(left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)

    return left, right, disp_gt, mask_noc, (orig_h, orig_w)


def infer_disparity(model, left_np, right_np, args):
    left = torch.from_numpy(left_np).permute(2, 0, 1).float().unsqueeze(0).cuda()
    right = torch.from_numpy(right_np).permute(2, 0, 1).float().unsqueeze(0).cuda()
    padder = InputPadder(left.shape, divis_by=32, force_square=False)
    left, right = padder.pad(left, right)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        if args.hiera:
            pred = model.run_hierachical(
                left,
                right,
                iters=args.valid_iters,
                test_mode=True,
                low_memory=args.low_memory,
                small_ratio=args.small_ratio,
            )
        else:
            pred = model.forward(
                left,
                right,
                iters=args.valid_iters,
                test_mode=True,
                low_memory=args.low_memory,
                optimize_build_volume="pytorch1",
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    pred = padder.unpad(pred.float()).squeeze(0).squeeze(0).cpu().numpy()
    return pred, elapsed


def aggregate_errors(error, disp_gt, mask):
    if not np.any(mask):
        return {"count": 0}

    err = error[mask]
    gt = np.maximum(np.abs(disp_gt[mask]), 1e-6)
    return {
        "count": int(mask.sum()),
        "epe_sum": float(err.sum()),
        "bad1_sum": int((err > 1.0).sum()),
        "bad2_sum": int((err > 2.0).sum()),
        "bad3_sum": int((err > 3.0).sum()),
        "d1_sum": int(((err > 3.0) & ((err / gt) > 0.05)).sum()),
    }


def merge_stats(dst, src):
    dst["count"] += src.get("count", 0)
    for key in ["epe_sum", "bad1_sum", "bad2_sum", "bad3_sum", "d1_sum"]:
        dst[key] += src.get(key, 0.0)


def finalize_stats(stats):
    if stats["count"] == 0:
        return {
            "count": 0,
            "epe": float("nan"),
            "bad1": float("nan"),
            "bad2": float("nan"),
            "bad3": float("nan"),
            "d1": float("nan"),
        }
    count = stats["count"]
    return {
        "count": count,
        "epe": stats["epe_sum"] / count,
        "bad1": stats["bad1_sum"] / count,
        "bad2": stats["bad2_sum"] / count,
        "bad3": stats["bad3_sum"] / count,
        "d1": stats["d1_sum"] / count,
    }


def save_visualization(sample_id, left, right, disp_gt, pred, valid_occ, out_dir):
    disp_min = float(np.min(disp_gt[valid_occ])) if np.any(valid_occ) else 0.0
    disp_max = float(np.max(disp_gt[valid_occ])) if np.any(valid_occ) else 1.0
    gt_vis = vis_disparity(disp_gt, min_val=disp_min, max_val=disp_max)
    pred_vis = vis_disparity(pred, min_val=disp_min, max_val=disp_max)
    canvas = np.concatenate([left, right, gt_vis, pred_vis], axis=1)
    imageio.imwrite(out_dir / f"{sample_id}_vis.png", canvas)


@torch.no_grad()
def main():
    args = parse_args()
    set_logging_format()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for evaluation.")

    if args.save_vis and args.out_dir is None:
        raise ValueError("--out_dir is required when --save_vis is enabled.")

    out_dir = None
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args, device)
    samples = find_samples(args)
    logging.info(f"dataset: {args.dataset_name}")
    logging.info(f"samples: {len(samples)}")
    logging.info(f"model: {args.model_dir}")

    total_occ = {"count": 0, "epe_sum": 0.0, "bad1_sum": 0, "bad2_sum": 0, "bad3_sum": 0, "d1_sum": 0}
    total_vis = {"count": 0, "epe_sum": 0.0, "bad1_sum": 0, "bad2_sum": 0, "bad3_sum": 0, "d1_sum": 0}
    total_all = {"count": 0, "epe_sum": 0.0, "bad1_sum": 0, "bad2_sum": 0, "bad3_sum": 0, "d1_sum": 0}
    per_sample = []
    total_time = 0.0

    for idx, sample in enumerate(samples):
        left, right, disp_gt, valid_noc, orig_shape = read_sample(sample, args.scale)
        pred, elapsed = infer_disparity(model, left, right, args)
        total_time += elapsed

        if args.scale != 1.0:
            pred = cv2.resize(pred, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR) / args.scale
            left = cv2.resize(left, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)

        valid_occ = np.isfinite(disp_gt) & (disp_gt > 0)
        if args.metric_max_disp > 0:
            valid_occ &= disp_gt < args.metric_max_disp
        valid_noc = valid_noc & valid_occ
        occ_mask = valid_occ & (~valid_noc)

        error = np.abs(pred - disp_gt)
        stats_all = aggregate_errors(error, disp_gt, valid_occ)
        stats_vis = aggregate_errors(error, disp_gt, valid_noc)
        stats_occ = aggregate_errors(error, disp_gt, occ_mask)
        merge_stats(total_all, stats_all)
        merge_stats(total_vis, stats_vis)
        merge_stats(total_occ, stats_occ)

        sample_metrics = {
            "id": sample["id"],
            "all": finalize_stats(stats_all),
            "visible": finalize_stats(stats_vis),
            "occluded": finalize_stats(stats_occ),
            "seconds": elapsed,
        }
        per_sample.append(sample_metrics)

        logging.info(
            f"[{idx + 1:03d}/{len(samples):03d}] {sample['id']} "
            f"EPE={sample_metrics['all']['epe']:.4f} "
            f"D1={sample_metrics['all']['d1']:.4f} "
            f"EPE_vis={sample_metrics['visible']['epe']:.4f} "
            f"EPE_occ={sample_metrics['occluded']['epe']:.4f} "
            f"time={elapsed * 1000.0:.1f}ms"
        )

        if args.save_vis:
            save_visualization(sample["id"], left, right, disp_gt, pred, valid_occ, out_dir)

    summary = {
        "dataset_name": args.dataset_name,
        "data_root": str(Path(args.data_root).resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "samples": len(samples),
        "avg_time_ms": 1000.0 * total_time / max(1, len(samples)),
        "all": finalize_stats(total_all),
        "visible": finalize_stats(total_vis),
        "occluded": finalize_stats(total_occ),
        "per_sample": per_sample,
    }

    logging.info("")
    logging.info(
        f"summary: EPE={summary['all']['epe']:.4f} "
        f"D1={summary['all']['d1']:.4f} "
        f"bad1={summary['all']['bad1']:.4f} "
        f"bad2={summary['all']['bad2']:.4f} "
        f"bad3={summary['all']['bad3']:.4f}"
    )
    logging.info(
        f"visible: EPE={summary['visible']['epe']:.4f} "
        f"D1={summary['visible']['d1']:.4f}"
    )
    logging.info(
        f"occluded: EPE={summary['occluded']['epe']:.4f} "
        f"D1={summary['occluded']['d1']:.4f}"
    )
    logging.info(f"avg_time_ms: {summary['avg_time_ms']:.2f}")

    report_path = None
    if args.report_json is not None:
        report_path = Path(args.report_json)
    elif out_dir is not None:
        report_path = out_dir / "eval_report.json"

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"report: {report_path}")


if __name__ == "__main__":
    main()

