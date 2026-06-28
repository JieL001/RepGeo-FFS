import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity
from core.utils.utils import InputPadder


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_H2 = font(21, True)
F_BODY = font(17)
F_SMALL = font(13)


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)


def load_model(path, valid_iters, max_disp, precision):
    cfg_path = Path(path).parent / "cfg.yaml"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
    cfg.update({"model_dir": str(path), "valid_iters": valid_iters, "max_disp": max_disp})
    _ = OmegaConf.create(cfg)

    model = torch.load(path, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    if precision == "fp32":
        model.args.mixed_precision = False
        model.dtype = torch.float32
    else:
        model.args.mixed_precision = True
    return model.to("cuda").eval()


def infer_one(model, left, right, valid_iters, optimize_build_volume, precision):
    h, w = left.shape[:2]
    lt = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    rt = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(lt.shape, divis_by=32, force_square=False)
    lt, rt = padder.pad(lt, rt)
    amp_context = (
        torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE)
        if precision == "amp"
        else nullcontext()
    )
    with torch.no_grad(), amp_context:
        disp = model.forward(
            lt,
            rt,
            iters=valid_iters,
            test_mode=True,
            optimize_build_volume=optimize_build_volume,
        )
    disp = padder.unpad(disp.float()).detach().cpu().numpy().reshape(h, w).clip(0, None)
    return disp


def fit_np_image(arr, w, h, bg=(255, 255, 255)):
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    canvas = Image.new("RGB", (w, h), bg)
    scale = min(w / img.width, h / img.height)
    nw = max(1, int(img.width * scale))
    nh = max(1, int(img.height * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(img, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def collect_samples(max_samples):
    samples = [("demo_data", ROOT / "demo_data" / "left.png", ROOT / "demo_data" / "right.png")]
    left_dir = ROOT / "data_scene_flow" / "training" / "image_2"
    right_dir = ROOT / "data_scene_flow" / "training" / "image_3"
    if left_dir.exists() and right_dir.exists():
        for left in sorted(left_dir.glob("*.png"))[: max(0, max_samples - 1)]:
            right = right_dir / left.name
            if right.exists():
                samples.append((f"KITTI {left.stem}", left, right))
    return samples[:max_samples]


def draw_board(sample_outputs, out_path):
    models = list(sample_outputs[0]["outputs"].keys())
    cols = ["Sample / left input"] + models
    cell_w = 430
    cell_h = 260
    label_h = 48
    margin = 52
    gap = 18
    title_h = 96
    W = margin * 2 + len(cols) * cell_w + (len(cols) - 1) * gap
    H = title_h + label_h + len(sample_outputs) * (cell_h + 78) + margin
    board = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(board)
    draw.text((margin, 32), "Multi-sample live inference board", fill=(14, 35, 61), font=F_TITLE)
    draw.text((margin, 70), "Rows are real forward passes. Columns compare checkpoints on the same stereo pairs.", fill=(86, 100, 118), font=F_BODY)

    y = title_h
    for c, name in enumerate(cols):
        x = margin + c * (cell_w + gap)
        draw.rounded_rectangle((x, y, x + cell_w, y + label_h), radius=12, fill=(18, 38, 63))
        draw.text((x + 16, y + 14), name, fill=(255, 255, 255), font=F_BODY)

    y += label_h + 16
    for row in sample_outputs:
        for c, name in enumerate(cols):
            x = margin + c * (cell_w + gap)
            draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=14, fill=(255, 255, 255), outline=(205, 214, 226), width=2)
            if c == 0:
                panel = fit_np_image(row["left"], cell_w - 24, cell_h - 58)
                board.paste(panel, (x + 12, y + 12))
                draw.text((x + 14, y + cell_h - 38), row["name"], fill=(34, 49, 68), font=F_H2)
            else:
                disp = row["outputs"][name]
                vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
                panel = fit_np_image(vis, cell_w - 24, cell_h - 58)
                board.paste(panel, (x + 12, y + 12))
                stat = f"mean {disp.mean():.2f}, max {disp.max():.2f}"
                draw.text((x + 14, y + cell_h - 34), stat, fill=(83, 96, 112), font=F_SMALL)
        y += cell_h + 78

    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "live_demo_code_20260628" / "outputs" / "boards" / "multi_sample_live_inference.png"))
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--valid_iters", type=int, default=2)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    parser.add_argument("--optimize_build_volume", choices=["pytorch1", "triton"], default="pytorch1")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for live multi-sample inference.")
    set_logging_format()
    set_seed(0)

    model_specs = [
        ("FFS base", ROOT / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"),
        ("RepGeo target", ROOT / "output_eval" / "stage29_repgeo_gamma_sweep_20260617" / "checkpoints" / "repgeo_gamma_1.pth"),
        ("RepGeo calibrated", ROOT / "output_eval" / "stage29_repgeo_gamma_sweep_20260617" / "checkpoints_fine" / "repgeo_gamma_0p01.pth"),
    ]
    model_specs = [(name, path) for name, path in model_specs if path.exists()]
    if not model_specs:
        raise FileNotFoundError("No model checkpoints found.")
    samples = collect_samples(args.max_samples)

    sample_outputs = []
    for sample_name, left_path, right_path in samples:
        left = read_rgb(left_path)
        right = read_rgb(right_path)
        item = {"name": sample_name, "left": left, "outputs": {}}
        for model_name, model_path in model_specs:
            print(f"[infer] {sample_name} | {model_name}")
            model = load_model(model_path, args.valid_iters, args.max_disp, args.precision)
            disp = infer_one(model, left, right, args.valid_iters, args.optimize_build_volume, args.precision)
            item["outputs"][model_name] = disp
            del model
            torch.cuda.empty_cache()
        sample_outputs.append(item)

    out_path = Path(args.out)
    draw_board(sample_outputs, out_path)
    print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
