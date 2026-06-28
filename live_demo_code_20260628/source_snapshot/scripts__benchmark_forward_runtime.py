import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

code_dir = Path(__file__).resolve().parent
sys.path.append(str(code_dir.parent))

from Utils import AMP_DTYPE, set_logging_format, set_seed
from core.foundation_stereo import FastFoundationStereo
from core.lora import apply_lora_to_conv_modules, count_lora_parameters, set_norm_layers_eval
from core.utils.utils import InputPadder


TRAIN_GROUPS = {
    "feature": ["feature", "proj_cmb", "stem_2"],
    "context": ["cnet", "sam", "cam"],
    "cost": ["corr_stem", "corr_feature_att", "cost_agg", "classifier"],
    "update": ["update_block"],
    "upsample": ["spx_2_gru", "spx_gru"],
    "all": [],
}


OFFICIAL_RUNTIME_TABLE = {
    ("23-36-37", 8, 480, 640): 49.4,
    ("23-36-37", 4, 480, 640): 41.1,
    ("20-26-39", 8, 480, 640): 43.6,
    ("20-26-39", 4, 480, 640): 37.5,
    ("20-30-48", 8, 480, 640): 38.4,
    ("20-30-48", 4, 480, 640): 29.3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile Fast-FoundationStereo forward runtime with the same core protocol "
            "as the official scripts/profile_speed.py: random 640x480 stereo tensors, "
            "AMP, warmup iterations, synchronized PyTorch forward timing."
        )
    )
    parser.add_argument(
        "--model_dir",
        default=str(code_dir.parent / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"),
        type=str,
        help="Serialized FastFoundationStereo model or a checkpoint dict with model_state.",
    )
    parser.add_argument(
        "--base_model_dir",
        default=None,
        type=str,
        help="Base serialized model path required when --model_dir is a state-dict checkpoint.",
    )
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=640, type=int)
    parser.add_argument("--valid_iters", default=8, type=int)
    parser.add_argument("--max_disp", default=192, type=int)
    parser.add_argument("--warmup", default=30, type=int)
    parser.add_argument("--repeat", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--precision",
        choices=["amp", "fp32"],
        default="amp",
        help="Official profile_speed.py uses CUDA autocast; keep amp for paper-comparable PyTorch timing.",
    )
    parser.add_argument(
        "--timer",
        choices=["sync", "event"],
        default="sync",
        help="sync matches scripts/profile_speed.py; event uses CUDA events for a lower-overhead cross-check.",
    )
    parser.add_argument(
        "--optimize_build_volume",
        choices=["triton", "pytorch1"],
        default="triton",
        help="Official profile_speed.py uses triton.",
    )
    parser.add_argument("--hiera", action="store_true")
    parser.add_argument("--small_ratio", default=0.5, type=float)
    parser.add_argument("--low_memory", action="store_true")
    parser.add_argument(
        "--lora_adapt",
        action="store_true",
        help="Attach Conv-LoRA wrappers before loading a model_state checkpoint.",
    )
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--lora_alpha", default=8.0, type=float)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--lora_min_channels", default=8, type=int)
    parser.add_argument("--lora_targets", nargs="+", default=["cost", "update", "upsample"])
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--tag", default=None, type=str)
    return parser.parse_args()


def resolve_lora_prefixes(lora_targets):
    if "all" in lora_targets:
        return []
    prefixes = []
    for target in lora_targets:
        prefixes.extend(TRAIN_GROUPS.get(target, [target]))
    deduped = []
    for prefix in prefixes:
        if prefix and prefix not in deduped:
            deduped.append(prefix)
    return deduped


def maybe_load_cfg(model_dir, args):
    cfg_path = Path(model_dir).resolve().parent / "cfg.yaml"
    if not cfg_path.exists():
        return args
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in vars(args).items():
        cfg[key] = value
    return OmegaConf.create(cfg)


def load_model(args, device):
    loaded = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    load_kind = "serialized_model"
    lora_wrapped = []

    if isinstance(loaded, FastFoundationStereo):
        model = loaded
    elif isinstance(loaded, dict) and "model_state" in loaded:
        load_kind = "state_dict"
        if args.base_model_dir is None:
            raise ValueError("--base_model_dir is required when --model_dir is a model_state checkpoint.")
        model = torch.load(args.base_model_dir, map_location="cpu", weights_only=False)
        if not isinstance(model, FastFoundationStereo):
            raise TypeError(f"Expected FastFoundationStereo base model, got {type(model)}")
        if args.lora_adapt:
            lora_wrapped = apply_lora_to_conv_modules(
                model,
                resolve_lora_prefixes(args.lora_targets),
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                min_channels=args.lora_min_channels,
            )
            if not lora_wrapped:
                raise RuntimeError(f"No LoRA modules matched targets={args.lora_targets}")
        missing, unexpected = model.load_state_dict(loaded["model_state"], strict=False)
        logging.info(f"loaded state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        raise TypeError(f"Unsupported checkpoint type from {args.model_dir}: {type(loaded)}")

    if not isinstance(model, FastFoundationStereo):
        raise TypeError(f"Expected FastFoundationStereo checkpoint, got {type(model)}")

    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    if hasattr(model, "update_block") and hasattr(model.update_block, "disp_head"):
        model.update_block.hidden_dim = model.update_block.disp_head.conv[0].in_channels
        if hasattr(model.update_block, "_ensure_refine_gate"):
            model.update_block._ensure_refine_gate()
    set_norm_layers_eval(model)
    model = model.to(device).eval()
    return model, {"load_kind": load_kind, "lora_wrapped": lora_wrapped}


def forward_once(model, img0, img1, args):
    if args.hiera:
        return model.run_hierachical(
            img0,
            img1,
            iters=args.valid_iters,
            test_mode=True,
            low_memory=args.low_memory,
            small_ratio=args.small_ratio,
        )
    return model.forward(
        img0,
        img1,
        iters=args.valid_iters,
        test_mode=True,
        low_memory=args.low_memory,
        optimize_build_volume=args.optimize_build_volume,
    )


def time_forward(model, img0, img1, args):
    use_amp = args.precision == "amp"
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=AMP_DTYPE):
            if args.timer == "event":
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                disp = forward_once(model, img0, img1, args)
                ender.record()
                torch.cuda.synchronize()
                elapsed_ms = starter.elapsed_time(ender)
            else:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                disp = forward_once(model, img0, img1, args)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, disp


def summarize(times):
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std(ddof=0)),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def checkpoint_family(path):
    parts = Path(path).parts
    for part in parts:
        if part in {"23-36-37", "20-26-39", "20-30-48"}:
            return part
    return "unknown"


def write_reports(out_dir, payload):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "runtime_forward_report.json"
    md_path = out_dir / "runtime_forward_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    official = payload.get("official_reference_ms")
    official_line = (
        f"- Official README PyTorch runtime for this exact profile: **{official:.1f} ms on RTX 3090**.\n"
        if official is not None
        else "- Official README has no exact reference row for this profile.\n"
    )
    stats = payload["stats"]
    lines = [
        "# Forward Runtime Benchmark\n",
        "\n",
        "This is a pure model-forward benchmark aligned with `scripts/profile_speed.py`, not a dataset-evaluation wall-clock measurement.\n",
        "\n",
        "## Protocol\n",
        f"- Model: `{payload['model_dir']}`\n",
        f"- Base model: `{payload.get('base_model_dir') or ''}`\n",
        f"- Image size: {payload['height']}x{payload['width']}\n",
        f"- valid_iters: {payload['valid_iters']}\n",
        f"- optimize_build_volume: `{payload['optimize_build_volume']}`\n",
        f"- precision: `{payload['precision']}`\n",
        f"- timer: `{payload['timer']}`\n",
        f"- warmup/repeat: {payload['warmup']} / {payload['repeat']}\n",
        official_line,
        "\n",
        "## Result\n",
        f"- Mean: **{stats['mean_ms']:.2f} ms**\n",
        f"- Median: **{stats['median_ms']:.2f} ms**\n",
        f"- Std: {stats['std_ms']:.2f} ms\n",
        f"- P10/P90: {stats['p10_ms']:.2f} / {stats['p90_ms']:.2f} ms\n",
        f"- Min/Max: {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms\n",
        f"- Peak CUDA memory: {payload['peak_memory_mb']:.1f} MB\n",
        "\n",
        "## Environment\n",
        f"- GPU: {payload['gpu_name']}\n",
        f"- Torch: {payload['torch_version']}\n",
        f"- CUDA runtime: {payload['torch_cuda_version']}\n",
        f"- cuDNN: {payload['cudnn_version']}\n",
        f"- Python: {payload['python_version']}\n",
        "\n",
        "## Interpretation\n",
        "- Use this number when comparing with the FFS README runtime table.\n",
        "- Do not compare the older `avg_time_ms` from dataset evaluation JSONs against the README table; those measurements include a different evaluation pipeline and were only same-protocol within our own evaluation scripts.\n",
    ]
    md_path.write_text("".join(lines), encoding="utf-8")
    return json_path, md_path


def main():
    args = parse_args()
    set_logging_format()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_grad_enabled(False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runtime benchmarking.")

    device = torch.device("cuda")
    args = maybe_load_cfg(args.model_dir, args)
    tag = args.tag or Path(args.model_dir).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else code_dir.parent / "output_eval" / f"runtime_forward_{tag}_{timestamp}"

    logging.info("load model")
    model, load_info = load_model(args, device)

    logging.info(f"prepare synthetic input: {args.height}x{args.width}")
    img0 = torch.randint(0, 256, (1, 3, args.height, args.width), dtype=torch.float32, device=device)
    img1 = torch.randint(0, 256, (1, 3, args.height, args.width), dtype=torch.float32, device=device)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    total = int(args.warmup) + int(args.repeat)
    times = []
    checksum = None
    logging.info(
        f"start benchmark: warmup={args.warmup} repeat={args.repeat} valid_iters={args.valid_iters} "
        f"precision={args.precision} timer={args.timer}"
    )
    for idx in range(total):
        elapsed_ms, disp = time_forward(model, img0, img1, args)
        if idx >= args.warmup:
            times.append(elapsed_ms)
        if idx == total - 1:
            if isinstance(disp, (tuple, list)):
                disp = disp[-1]
            checksum = float(disp.float().mean().detach().cpu())
        label = "warmup" if idx < args.warmup else "measure"
        logging.info(f"{idx:04d} {label}: {elapsed_ms:.2f} ms")

    stats = summarize(times)
    family = checkpoint_family(args.model_dir)
    official_reference = OFFICIAL_RUNTIME_TABLE.get((family, int(args.valid_iters), int(args.height), int(args.width)))
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(Path(args.model_dir).resolve()),
        "base_model_dir": str(Path(args.base_model_dir).resolve()) if args.base_model_dir else None,
        "load_info": load_info,
        "height": int(args.height),
        "width": int(args.width),
        "padded_shape": list(img0.shape),
        "valid_iters": int(args.valid_iters),
        "max_disp": int(args.max_disp),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "precision": str(args.precision),
        "timer": str(args.timer),
        "optimize_build_volume": str(args.optimize_build_volume),
        "hiera": bool(args.hiera),
        "stats": stats,
        "all_times_ms": [float(x) for x in times],
        "checksum": checksum,
        "official_reference_ms": official_reference,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "peak_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "lora_parameter_count": int(count_lora_parameters(model)),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
    }
    json_path, md_path = write_reports(out_dir, payload)
    logging.info(f"mean={stats['mean_ms']:.2f} ms median={stats['median_ms']:.2f} ms peak_mem={payload['peak_memory_mb']:.1f} MB")
    logging.info(f"wrote {json_path}")
    logging.info(f"wrote {md_path}")


if __name__ == "__main__":
    main()

