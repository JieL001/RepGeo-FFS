import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

code_dir = Path(__file__).resolve().parent
sys.path.append(str(code_dir.parent))

from Utils import AMP_DTYPE, set_logging_format, set_seed
from core.foundation_stereo import FastFoundationStereo
from core.lora import LORA_CONV_TYPES, LoRAConv2d, LoRAConv3d, count_lora_parameters
from core.utils.utils import InputPadder
from scripts.train_kitti import configure_lora_adapters


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge Conv-LoRA adapters into their base Conv2d/Conv3d weights for inference. "
            "This preserves the eval-time function because LoRA dropout is inactive in eval mode."
        )
    )
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument(
        "--base_model_dir",
        default=None,
        type=str,
        help="Base FFS checkpoint used to materialize LoRA modules when --model_dir is a training state dict.",
    )
    parser.add_argument("--out_model", required=True, type=str)
    parser.add_argument("--report_json", default=None, type=str)
    parser.add_argument("--check", action="store_true", help="Run a random-input equivalence check before saving.")
    parser.add_argument("--height", default=256, type=int)
    parser.add_argument("--width", default=320, type=int)
    parser.add_argument("--valid_iters", default=4, type=int)
    parser.add_argument("--max_disp", default=192, type=int)
    parser.add_argument("--optimize_build_volume", default="triton", choices=["triton", "pytorch1"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--lora_alpha", default=8.0, type=float)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--lora_min_channels", default=1, type=int)
    parser.add_argument("--lora_targets", nargs="+", default=["cost", "update", "upsample"])
    parser.add_argument(
        "--repgeo_alpha_threshold",
        default=0.0,
        type=float,
        help=(
            "Prune RepGeo residual bases whose absolute alpha is below this threshold before folding. "
            "The default 0 keeps the original merge behavior."
        ),
    )
    parser.add_argument(
        "--dense_grouped",
        action="store_true",
        help=(
            "Allow merging grouped base convolutions by expanding them to dense groups=1 convolutions. "
            "This is exact but may not be faster. By default grouped LoRA modules are left unmerged."
        ),
    )
    return parser.parse_args()


def load_lora_or_model(args):
    loaded = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    if isinstance(loaded, FastFoundationStereo):
        return loaded
    if not (isinstance(loaded, dict) and "model_state" in loaded):
        raise TypeError(f"Unsupported checkpoint type: {type(loaded)}")
    if args.base_model_dir is None:
        raise ValueError("--base_model_dir is required when --model_dir is a training state dict.")
    model = torch.load(args.base_model_dir, map_location="cpu", weights_only=False)
    if not isinstance(model, FastFoundationStereo):
        raise TypeError(f"Expected FastFoundationStereo base checkpoint, got {type(model)}")
    configure_lora_adapters(
        model,
        SimpleNamespace(
            lora_targets=args.lora_targets,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_min_channels=args.lora_min_channels,
        ),
    )
    missing, unexpected = model.load_state_dict(loaded["model_state"], strict=False)
    logging.info("materialized state dict: missing=%d unexpected=%d", len(missing), len(unexpected))
    return model


def _clone_conv_common(base, groups):
    kwargs = {
        "in_channels": base.in_channels,
        "out_channels": base.out_channels,
        "kernel_size": base.kernel_size,
        "stride": base.stride,
        "padding": base.padding,
        "dilation": base.dilation,
        "groups": groups,
        "bias": base.bias is not None,
        "padding_mode": base.padding_mode,
    }
    return kwargs


def _dense_base_weight(base):
    weight = base.weight.detach()
    if base.groups == 1:
        return weight.clone()
    out_channels, in_per_group = weight.shape[:2]
    in_channels = base.in_channels
    groups = base.groups
    out_per_group = out_channels // groups
    dense_shape = (out_channels, in_channels, *weight.shape[2:])
    dense = weight.new_zeros(dense_shape)
    for group_idx in range(groups):
        out0 = group_idx * out_per_group
        out1 = out0 + out_per_group
        in0 = group_idx * in_per_group
        in1 = in0 + in_per_group
        dense[out0:out1, in0:in1] = weight[out0:out1]
    return dense


def _merge_conv2d(module, dense_grouped=False, repgeo_alpha_threshold=0.0):
    base = module.base
    repgeo_alpha = float(getattr(module, "repgeo_alpha", torch.ones(1)).detach().float().cpu().item())
    if abs(repgeo_alpha) < repgeo_alpha_threshold:
        return base, "pruned_base"
    if base.padding_mode != "zeros":
        return None, "skip_nonzero_padding"
    if base.groups != 1 and not dense_grouped:
        return None, "skip_grouped"
    groups = 1 if base.groups != 1 else base.groups
    merged = nn.Conv2d(**_clone_conv_common(base, groups=groups))
    base_weight = _dense_base_weight(base) if base.groups != groups else base.weight.detach().clone()
    down = module.lora_down.weight.detach()
    up = module.lora_up.weight.detach()[:, :, 0, 0]
    delta = torch.einsum("or,rihw->oihw", up, down) * module.scaling * repgeo_alpha
    merged.weight.data.copy_(base_weight + delta)
    if base.bias is not None:
        merged.bias.data.copy_(base.bias.detach())
    merged.to(device=base.weight.device, dtype=base.weight.dtype)
    return merged, "merged_dense" if base.groups != groups else "merged"


def _merge_conv3d(module, dense_grouped=False, repgeo_alpha_threshold=0.0):
    base = module.base
    repgeo_alpha = float(getattr(module, "repgeo_alpha", torch.ones(1)).detach().float().cpu().item())
    if abs(repgeo_alpha) < repgeo_alpha_threshold:
        return base, "pruned_base"
    if base.padding_mode != "zeros":
        return None, "skip_nonzero_padding"
    if base.groups != 1 and not dense_grouped:
        return None, "skip_grouped"
    groups = 1 if base.groups != 1 else base.groups
    merged = nn.Conv3d(**_clone_conv_common(base, groups=groups))
    base_weight = _dense_base_weight(base) if base.groups != groups else base.weight.detach().clone()
    down = module.lora_down.weight.detach()
    up = module.lora_up.weight.detach()[:, :, 0, 0, 0]
    delta = torch.einsum("or,ridhw->oidhw", up, down) * module.scaling * repgeo_alpha
    merged.weight.data.copy_(base_weight + delta)
    if base.bias is not None:
        merged.bias.data.copy_(base.bias.detach())
    merged.to(device=base.weight.device, dtype=base.weight.dtype)
    return merged, "merged_dense" if base.groups != groups else "merged"


def merge_lora_modules(model, dense_grouped=False, repgeo_alpha_threshold=0.0):
    report = []

    def visit(parent, prefix=""):
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRAConv2d):
                original_alpha = float(getattr(child, "repgeo_alpha", torch.ones(1)).detach().float().cpu().item())
                merged, status = _merge_conv2d(
                    child,
                    dense_grouped=dense_grouped,
                    repgeo_alpha_threshold=repgeo_alpha_threshold,
                )
                if merged is not None:
                    setattr(parent, child_name, merged)
                report.append(
                    {
                        "name": full_name,
                        "type": "LoRAConv2d",
                        "status": status,
                        "rank": int(child.rank),
                        "repgeo_alpha": original_alpha,
                        "repgeo_alpha_abs": abs(original_alpha),
                        "repgeo_alpha_threshold": float(repgeo_alpha_threshold),
                        "base_groups": int(child.base.groups),
                        "base_shape": list(child.base.weight.shape),
                    }
                )
                continue
            if isinstance(child, LoRAConv3d):
                original_alpha = float(getattr(child, "repgeo_alpha", torch.ones(1)).detach().float().cpu().item())
                merged, status = _merge_conv3d(
                    child,
                    dense_grouped=dense_grouped,
                    repgeo_alpha_threshold=repgeo_alpha_threshold,
                )
                if merged is not None:
                    setattr(parent, child_name, merged)
                report.append(
                    {
                        "name": full_name,
                        "type": "LoRAConv3d",
                        "status": status,
                        "rank": int(child.rank),
                        "repgeo_alpha": original_alpha,
                        "repgeo_alpha_abs": abs(original_alpha),
                        "repgeo_alpha_threshold": float(repgeo_alpha_threshold),
                        "base_groups": int(child.base.groups),
                        "base_shape": list(child.base.weight.shape),
                    }
                )
                continue
            visit(child, full_name)

    visit(model)
    return report


def prepare_model(model, args, device):
    if not isinstance(model, FastFoundationStereo):
        raise TypeError(f"Expected FastFoundationStereo, got {type(model)}")
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    if hasattr(model, "update_block") and hasattr(model.update_block, "disp_head"):
        model.update_block.hidden_dim = model.update_block.disp_head.conv[0].in_channels
        if hasattr(model.update_block, "_ensure_refine_gate"):
            model.update_block._ensure_refine_gate()
    return model.to(device).eval()


@torch.inference_mode()
def run_forward(model, left, right, args):
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        return model.forward(
            left,
            right,
            iters=args.valid_iters,
            test_mode=True,
            optimize_build_volume=args.optimize_build_volume,
        )


def equivalence_check(before, after, args, device):
    left = torch.randint(0, 256, (1, 3, args.height, args.width), dtype=torch.float32, device=device)
    right = torch.randint(0, 256, (1, 3, args.height, args.width), dtype=torch.float32, device=device)
    padder = InputPadder(left.shape, divis_by=32, force_square=False)
    left, right = padder.pad(left, right)
    pred_before = run_forward(before, left, right, args).float()
    pred_after = run_forward(after, left, right, args).float()
    diff = (pred_before - pred_after).abs()
    return {
        "shape": list(pred_before.shape),
        "max_abs": float(diff.max().detach().cpu()),
        "mean_abs": float(diff.mean().detach().cpu()),
        "before_mean": float(pred_before.mean().detach().cpu()),
        "after_mean": float(pred_after.mean().detach().cpu()),
    }


def main():
    args = parse_args()
    set_logging_format()
    set_seed(args.seed)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_lora_or_model(args)
    model = prepare_model(model, args, device)
    before_lora_params = int(count_lora_parameters(model))
    before_lora_modules = sum(1 for module in model.modules() if isinstance(module, LORA_CONV_TYPES))
    logging.info(f"before merge: lora_modules={before_lora_modules} lora_params={before_lora_params}")

    if args.check and device.type != "cuda":
        raise RuntimeError("--check requires CUDA for this model.")
    before_model = None
    if args.check:
        import copy

        before_model = copy.deepcopy(model).eval()

    merge_report = merge_lora_modules(
        model,
        dense_grouped=args.dense_grouped,
        repgeo_alpha_threshold=args.repgeo_alpha_threshold,
    )
    model = model.eval()
    after_lora_params = int(count_lora_parameters(model))
    after_lora_modules = sum(1 for module in model.modules() if isinstance(module, LORA_CONV_TYPES))
    logging.info(f"after merge: lora_modules={after_lora_modules} lora_params={after_lora_params}")

    check_report = None
    if args.check:
        check_report = equivalence_check(before_model, model, args, device)
        logging.info(
            "equivalence: max_abs=%.6g mean_abs=%.6g",
            check_report["max_abs"],
            check_report["mean_abs"],
        )

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu(), out_model)
    logging.info(f"wrote merged model: {out_model}")

    payload = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "out_model": str(out_model.resolve()),
        "dense_grouped": bool(args.dense_grouped),
        "repgeo_alpha_threshold": float(args.repgeo_alpha_threshold),
        "pruned_lora_modules": sum(1 for row in merge_report if str(row.get("status", "")).startswith("pruned")),
        "before_lora_modules": before_lora_modules,
        "before_lora_params": before_lora_params,
        "after_lora_modules": after_lora_modules,
        "after_lora_params": after_lora_params,
        "merge_report": merge_report,
        "equivalence_check": check_report,
    }
    report_json = Path(args.report_json) if args.report_json else out_model.with_suffix(".merge_report.json")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"wrote report: {report_json}")


if __name__ == "__main__":
    main()

