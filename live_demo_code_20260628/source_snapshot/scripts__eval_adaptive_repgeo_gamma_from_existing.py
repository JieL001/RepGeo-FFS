#!/usr/bin/env python3
"""Evaluate an input-adaptive RepGeo gamma policy from existing result files.

This script does not rerun FFS.  It combines:

1. Stage21 RGF target-likeness scores, and
2. already evaluated fixed-gamma endpoint metrics.

For each clean sample, the hard adaptive policy is

    gamma(x) = gamma_target, if score(x) <= threshold
             = gamma_min,    otherwise

where score(x) is the RGF feature distance to KITTI calibration data.

The output is an auditable estimate of whether manual gamma selection can be
replaced by an automatic input-conditioned scalar.  It should be reported as an
adaptive policy audit, not as a newly rerun dynamic-checkpoint benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rgf_report",
        type=Path,
        default=ROOT
        / "output_eval"
        / "remote_stage21_rgf_feature_ablation_20260616"
        / "rgf_feature_ablation_report.json",
    )
    parser.add_argument("--rgf_setting", default="full_q")
    parser.add_argument(
        "--kitti_base",
        type=Path,
        default=ROOT / "output_eval" / "stage29_repgeo_gamma_sweep_20260617" / "kitti_fine" / "ffs_base_kitti.json",
    )
    parser.add_argument(
        "--kitti_min",
        type=Path,
        default=ROOT / "output_eval" / "stage29_repgeo_gamma_sweep_20260617" / "kitti_fine" / "gamma_0p01_kitti.json",
    )
    parser.add_argument(
        "--kitti_target",
        type=Path,
        default=ROOT
        / "output_eval"
        / "stage29_repgeo_gamma_sweep_20260617"
        / "kitti_fine"
        / "stage23_compiled_kitti.json",
    )
    parser.add_argument(
        "--eth3d_base",
        type=Path,
        default=ROOT
        / "output_eval"
        / "stage28_cross_dataset_repgeo_20260617"
        / "full_eval_local4060_20260617"
        / "ffs_base_eth3d.json",
    )
    parser.add_argument(
        "--eth3d_min",
        type=Path,
        default=ROOT
        / "output_eval"
        / "stage29_repgeo_gamma_sweep_20260617"
        / "full_cross_fine"
        / "gamma_0p01_eth3d.json",
    )
    parser.add_argument(
        "--eth3d_target",
        type=Path,
        default=ROOT
        / "output_eval"
        / "stage28_cross_dataset_repgeo_20260617"
        / "full_eval_local4060_20260617"
        / "stage23_compiled_eth3d.json",
    )
    parser.add_argument(
        "--middlebury_base",
        type=Path,
        default=ROOT
        / "output_eval"
        / "stage28_cross_dataset_repgeo_20260617"
        / "full_eval_local4060_20260617"
        / "ffs_base_middlebury.json",
    )
    parser.add_argument(
        "--middlebury_min",
        type=Path,
        default=ROOT
        / "output_eval"
        / "stage29_repgeo_gamma_sweep_20260617"
        / "full_cross_fine"
        / "gamma_0p01_middlebury.json",
    )
    parser.add_argument(
        "--middlebury_target",
        type=Path,
        default=ROOT
        / "output_eval"
        / "stage28_cross_dataset_repgeo_20260617"
        / "full_eval_local4060_20260617"
        / "stage23_compiled_middlebury.json",
    )
    parser.add_argument("--gamma_min", default=0.01, type=float)
    parser.add_argument("--gamma_target", default=1.0, type=float)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT / "output_eval" / "stage30_adaptive_repgeo_gamma_20260619",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metric_rows(path: Path) -> dict[str, dict[str, Any]]:
    obj = load_json(path)
    return {row["id"]: row for row in obj["per_sample"]}


def clean_scores(report: dict[str, Any], setting: str) -> tuple[float, dict[tuple[str, str], dict[str, Any]]]:
    item = report["settings"][setting]
    threshold = float(item["threshold"])
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in item["rows"]:
        if row.get("transform") != "clean":
            continue
        rows[(row["dataset"], row["id"])] = row
    return threshold, rows


def safe_metric(row: dict[str, Any], split: str, key: str) -> float:
    value = row[split][key]
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    return float(value)


def aggregate(rows: list[dict[str, Any]], split: str = "all") -> dict[str, float]:
    total = sum(int(row[split]["count"]) for row in rows)
    out: dict[str, float] = {"count": float(total)}
    for key in ["epe", "bad1", "bad2", "bad3", "d1"]:
        if total == 0:
            out[key] = float("nan")
        else:
            out[key] = sum(safe_metric(row, split, key) * int(row[split]["count"]) for row in rows) / total
    return out


def dataset_score_name(dataset: str) -> str:
    if dataset == "kitti":
        return "KITTI2015 local-val40"
    if dataset == "eth3d":
        return "ETH3D full27"
    if dataset == "middlebury":
        return "Middlebury full24"
    raise ValueError(dataset)


def build_adaptive_rows(
    dataset: str,
    base_rows: dict[str, dict[str, Any]],
    min_rows: dict[str, dict[str, Any]],
    target_rows: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
    threshold: float,
    gamma_min: float,
    gamma_target: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_dataset = dataset_score_name(dataset)
    chosen_metrics: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for sample_id in sorted(base_rows.keys()):
        score_row = scores.get((score_dataset, sample_id))
        if score_row is None:
            # Be conservative if a score is missing.
            score = float("inf")
            route = "min"
        else:
            score = float(score_row["score"])
            route = "target" if score <= threshold else "min"
        gamma = gamma_target if route == "target" else gamma_min
        metric = target_rows[sample_id] if route == "target" else min_rows[sample_id]
        chosen_metrics.append(metric)
        audit_rows.append(
            {
                "dataset": dataset,
                "id": sample_id,
                "score": score,
                "threshold": threshold,
                "gamma": gamma,
                "route": route,
                "chosen_epe": metric["all"]["epe"],
                "chosen_d1": metric["all"]["d1"],
                "base_epe": base_rows[sample_id]["all"]["epe"],
                "base_d1": base_rows[sample_id]["all"]["d1"],
            }
        )
    return chosen_metrics, audit_rows


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.3f}%"


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Stage30 Adaptive RepGeo Gamma Audit",
        "",
        "This report replaces manual gamma selection with an input-conditioned scalar policy.",
        "",
        "Policy:",
        "",
        "```text",
        "score(x) = RGF feature distance to KITTI calibration distribution",
        "gamma(x) = gamma_target if score(x) <= threshold else gamma_min",
        "```",
        "",
        "Important scope: this is an adaptive-policy audit assembled from already evaluated fixed-gamma endpoints. "
        "It is not a newly rerun dynamic-checkpoint benchmark. If deployed per input, it adds a lightweight "
        "input-conditioned scalar gate; if strict static deployment is required, use the automatically selected "
        "global gamma checkpoint instead.",
        "",
        f"- RGF setting: `{payload['rgf_setting']}`",
        f"- Threshold: `{payload['threshold']:.6f}`",
        f"- gamma_min: `{payload['gamma_min']}`",
        f"- gamma_target: `{payload['gamma_target']}`",
        "",
        "## Main Metrics",
        "",
        "| Dataset | FFS base EPE / D1 | Adaptive gamma EPE / D1 | Target-route samples | Interpretation |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for dataset, item in payload["datasets"].items():
        base = item["base"]
        adaptive = item["adaptive"]
        lines.append(
            f"| {item['display_name']} | {base['epe']:.4f} / {fmt_pct(base['d1'])} | "
            f"{adaptive['epe']:.4f} / {fmt_pct(adaptive['d1'])} | "
            f"{item['target_count']}/{item['sample_count']} | {item['interpretation']} |"
        )
    lines += [
        "",
        "## Defense Wording",
        "",
        "> Gamma is no longer manually picked. We estimate how close an input is to the KITTI target domain using the same "
        "label-free validity features used in RGF, then automatically choose a strong or conservative RepGeo residual strength. "
        "This keeps KITTI-like inputs strongly calibrated while preventing the full target residual from being blindly applied to "
        "ETH3D/Middlebury-like inputs.",
        "",
        "## What Not To Claim",
        "",
        "- Do not claim this is a single pure static checkpoint if per-input adaptive gamma is used.",
        "- Do not claim the adaptive policy was rerun end-to-end unless a later script actually performs dynamic-weight inference.",
        "- Do not claim universal SOTA; the point is automatic deployment-strength selection under the FFS/RepGeo setting.",
        "",
    ]
    (out_dir / "adaptive_gamma_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rgf = load_json(args.rgf_report)
    threshold, scores = clean_scores(rgf, args.rgf_setting)

    datasets = {
        "kitti": {
            "display_name": "KITTI local-val40",
            "base": metric_rows(args.kitti_base),
            "min": metric_rows(args.kitti_min),
            "target": metric_rows(args.kitti_target),
            "interpretation": "KITTI-like inputs receive the strong target residual.",
        },
        "eth3d": {
            "display_name": "ETH3D full27",
            "base": metric_rows(args.eth3d_base),
            "min": metric_rows(args.eth3d_min),
            "target": metric_rows(args.eth3d_target),
            "interpretation": "OOD-like inputs are kept near the conservative residual.",
        },
        "middlebury": {
            "display_name": "Middlebury full24",
            "base": metric_rows(args.middlebury_base),
            "min": metric_rows(args.middlebury_min),
            "target": metric_rows(args.middlebury_target),
            "interpretation": "OOD-like inputs are kept near the conservative residual.",
        },
    }

    all_audit_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "rgf_report": str(args.rgf_report),
        "rgf_setting": args.rgf_setting,
        "threshold": threshold,
        "gamma_min": args.gamma_min,
        "gamma_target": args.gamma_target,
        "datasets": {},
    }

    for name, item in datasets.items():
        chosen_rows, audit_rows = build_adaptive_rows(
            name,
            item["base"],
            item["min"],
            item["target"],
            scores,
            threshold,
            args.gamma_min,
            args.gamma_target,
        )
        base_agg = aggregate(list(item["base"].values()))
        adaptive_agg = aggregate(chosen_rows)
        target_count = sum(1 for row in audit_rows if row["route"] == "target")
        payload["datasets"][name] = {
            "display_name": item["display_name"],
            "sample_count": len(audit_rows),
            "target_count": target_count,
            "min_count": len(audit_rows) - target_count,
            "base": base_agg,
            "adaptive": adaptive_agg,
            "interpretation": item["interpretation"],
        }
        all_audit_rows.extend(audit_rows)

    (args.out_dir / "adaptive_gamma_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.out_dir / "adaptive_gamma_sample_routes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "id",
                "score",
                "threshold",
                "gamma",
                "route",
                "chosen_epe",
                "chosen_d1",
                "base_epe",
                "base_d1",
            ],
        )
        writer.writeheader()
        writer.writerows(all_audit_rows)
    write_report(args.out_dir, payload)
    print(args.out_dir / "adaptive_gamma_summary.md")


if __name__ == "__main__":
    main()

