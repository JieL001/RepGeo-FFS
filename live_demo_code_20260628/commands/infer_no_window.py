import argparse
from contextlib import nullcontext
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity
from core.utils.utils import InputPadder


def parse_args():
    parser = argparse.ArgumentParser(description="No-window stereo inference demo for classroom/code inspection.")
    parser.add_argument("--model_dir", default=str(ROOT / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"))
    parser.add_argument("--left_file", default=str(ROOT / "demo_data" / "left.png"))
    parser.add_argument("--right_file", default=str(ROOT / "demo_data" / "right.png"))
    parser.add_argument("--out_dir", default=str(ROOT / "live_demo_code_20260628" / "outputs" / "infer_no_window"))
    parser.add_argument("--valid_iters", type=int, default=4)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    parser.add_argument("--optimize_build_volume", default="pytorch1", choices=["pytorch1", "triton"])
    return parser.parse_args()


def read_rgb(path):
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return image[..., :3]


def main():
    args = parse_args()
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    if not torch.cuda.is_available():
        raise RuntimeError("This inference demo expects CUDA. Use CPU only for code reading, not live inference.")

    os.makedirs(args.out_dir, exist_ok=True)

    cfg_path = Path(args.model_dir).parent / "cfg.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    else:
        cfg = {}
    cfg.update(vars(args))
    cfg = OmegaConf.create(cfg)

    model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    if args.precision == "fp32":
        model.args.mixed_precision = False
        model.dtype = torch.float32
    else:
        model.args.mixed_precision = True
    model = model.to("cuda").eval()
    cpu_params = [(name, str(param.device)) for name, param in model.named_parameters() if param.device.type != "cuda"]
    if cpu_params:
        raise RuntimeError(f"Model still has CPU parameters after to('cuda'): {cpu_params[:8]}")

    left = read_rgb(args.left_file)
    right = read_rgb(args.right_file)
    if args.scale != 1.0:
        left = cv2.resize(left, dsize=None, fx=args.scale, fy=args.scale)
        right = cv2.resize(right, dsize=(left.shape[1], left.shape[0]))
    height, width = left.shape[:2]

    imageio.imwrite(Path(args.out_dir) / "left.png", left)
    imageio.imwrite(Path(args.out_dir) / "right.png", right)

    left_tensor = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

    amp_context = (
        torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE)
        if args.precision == "amp"
        else nullcontext()
    )
    with amp_context:
        disp = model.forward(
            left_tensor,
            right_tensor,
            iters=args.valid_iters,
            test_mode=True,
            optimize_build_volume=args.optimize_build_volume,
        )

    disp = padder.unpad(disp.float()).detach().cpu().numpy().reshape(height, width).clip(0, None)
    np.save(Path(args.out_dir) / "disp.npy", disp)

    disp_vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
    imageio.imwrite(Path(args.out_dir) / "disp_vis.png", disp_vis)
    board = np.concatenate([left, right, disp_vis], axis=1)
    imageio.imwrite(Path(args.out_dir) / "infer_board.png", board)

    print(f"saved: {Path(args.out_dir) / 'infer_board.png'}")
    print(f"disp shape={disp.shape}, min={float(np.nanmin(disp)):.4f}, max={float(np.nanmax(disp)):.4f}")


if __name__ == "__main__":
    main()

