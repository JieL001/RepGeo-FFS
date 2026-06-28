#!/usr/bin/env python3
"""Create interpolated RepGeo checkpoints.

The generated model keeps the original FFS architecture and interpolates weights
between the released base checkpoint and a compiled RepGeo checkpoint:

    W_gamma = W_base + gamma * (W_compiled - W_base)

This is useful for auditing whether a target-domain residual can be attenuated
to reduce out-of-domain degradation without adding inference-time branches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

code_dir = Path(__file__).resolve().parent
sys.path.append(str(code_dir.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True, type=Path)
    parser.add_argument("--compiled_model", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument(
        "--gammas",
        required=True,
        type=str,
        help="Comma-separated gamma values, e.g. 0,0.1,0.25,0.5,1.0",
    )
    return parser.parse_args()


def load_state(model):
    if hasattr(model, "update_block") and hasattr(model.update_block, "_ensure_refine_gate"):
        model.update_block._ensure_refine_gate()
    if hasattr(model, "state_dict"):
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in model.state_dict().items()
        }
    if isinstance(model, dict) and "model_state" in model:
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in model["model_state"].items()
        }
    raise TypeError(f"Unsupported checkpoint type: {type(model)}")


def gamma_label(gamma: float) -> str:
    raw = f"{gamma:.4f}".rstrip("0").rstrip(".")
    return raw.replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gammas = [float(item.strip()) for item in args.gammas.split(",") if item.strip()]

    base_model = torch.load(args.base_model, map_location="cpu", weights_only=False)
    compiled_model = torch.load(args.compiled_model, map_location="cpu", weights_only=False)
    base_state = load_state(base_model)
    compiled_state = load_state(compiled_model)

    common = sorted(set(base_state.keys()) & set(compiled_state.keys()))
    missing_in_compiled = sorted(set(base_state.keys()) - set(compiled_state.keys()))
    missing_in_base = sorted(set(compiled_state.keys()) - set(base_state.keys()))
    if missing_in_compiled or missing_in_base:
        raise RuntimeError(
            "State-dict keys differ; refusing to interpolate. "
            f"missing_in_compiled={missing_in_compiled[:5]} "
            f"missing_in_base={missing_in_base[:5]}"
        )

    float_keys = [
        key
        for key in common
        if torch.is_tensor(base_state[key])
        and torch.is_tensor(compiled_state[key])
        and base_state[key].shape == compiled_state[key].shape
        and base_state[key].dtype.is_floating_point
        and compiled_state[key].dtype.is_floating_point
    ]

    report = {
        "base_model": str(args.base_model),
        "compiled_model": str(args.compiled_model),
        "gammas": gammas,
        "num_state_keys": len(common),
        "num_interpolated_float_keys": len(float_keys),
        "outputs": [],
    }

    for gamma in gammas:
        new_state = {}
        l2_delta_sq = 0.0
        for key in common:
            base_value = base_state[key]
            compiled_value = compiled_state[key]
            if key in float_keys:
                delta = compiled_value.to(torch.float32) - base_value.to(torch.float32)
                value = base_value.to(torch.float32) + gamma * delta
                value = value.to(dtype=base_value.dtype)
                new_state[key] = value
                l2_delta_sq += float((gamma * delta).pow(2).sum().item())
            else:
                new_state[key] = base_value

        base_model.load_state_dict(new_state, strict=True)
        out_path = args.out_dir / f"repgeo_gamma_{gamma_label(gamma)}.pth"
        torch.save(base_model, out_path)
        report["outputs"].append(
            {
                "gamma": gamma,
                "path": str(out_path),
                "l2_delta": l2_delta_sq**0.5,
            }
        )

    (args.out_dir / "gamma_checkpoint_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.out_dir / "gamma_checkpoint_report.json")


if __name__ == "__main__":
    main()

