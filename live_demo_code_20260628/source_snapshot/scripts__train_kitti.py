import argparse
import logging
import math
import os
from pathlib import Path
import sys

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

code_dir = Path(__file__).resolve().parent
sys.path.append(str(code_dir.parent))


TRAIN_GROUPS = {
    "feature": ["feature", "proj_cmb", "stem_2"],
    "context": ["cnet", "sam", "cam"],
    "cost": ["corr_stem", "corr_feature_att", "cost_agg", "classifier"],
    "update": ["update_block"],
    "upsample": ["spx_2_gru", "spx_gru"],
    "all": [],
}


def cuda_autocast(enabled, dtype=torch.float16):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def cuda_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def set_logging_format(level=logging.INFO):
    logging.basicConfig(level=level, format="%(message)s", datefmt="%m-%d|%H:%M:%S")


def set_seed(random_seed):
    import random

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def parse_args():
    code_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=str(code_dir.parent / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"),
        type=str,
    )
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--out_dir", default=str(code_dir.parent / "output_train"), type=str)
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--crop_h", default=256, type=int)
    parser.add_argument("--crop_w", default=768, type=int)
    parser.add_argument(
        "--train_split",
        default="train",
        choices=["train", "trainval"],
        help="Training split. Use trainval only for final official-test training, not local validation claims.",
    )
    parser.add_argument(
        "--val_split",
        default="val",
        choices=["val", "train"],
        help="Validation split used for monitoring/checkpointing. Local-val metrics are invalid if train_split=trainval.",
    )
    parser.add_argument("--val_stride", default=5, type=int)
    parser.add_argument("--gamma", default=0.9, type=float, help="sequence loss weight decay")
    parser.add_argument("--valid_iters", default=4, type=int)
    parser.add_argument("--max_disp", default=192, type=int)
    parser.add_argument("--max_train_samples", default=None, type=int)
    parser.add_argument("--max_val_samples", default=None, type=int)
    parser.add_argument(
        "--allow_cpu_eval",
        action="store_true",
        help="Allow eval_only on CPU for small sanity checks when CUDA is unavailable.",
    )
    parser.add_argument(
        "--report_json",
        default=None,
        type=str,
        help="Optional path to write eval_only validation metrics as JSON.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument(
        "--reset_optimizer_on_resume",
        action="store_true",
        help="Resume model/EMA weights but start a fresh optimizer with the current --lr.",
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--narrow_band", action="store_true")
    parser.add_argument("--band_radius", default=8, type=int)
    parser.add_argument("--adaptive_refine", action="store_true")
    parser.add_argument("--adaptive_refine_warmup", default=1, type=int)
    parser.add_argument("--adaptive_refine_threshold", default=0.5, type=float)
    parser.add_argument("--adaptive_refine_hard", default=1, type=int)
    parser.add_argument("--gate_sparsity_weight", default=0.0, type=float)
    parser.add_argument("--init_loss_weight", default=0.0, type=float)
    parser.add_argument("--visibility_supervision", action="store_true")
    parser.add_argument("--occ_loss_weight", default=0.3, type=float)
    parser.add_argument("--boundary_loss_weight", default=0.5, type=float)
    parser.add_argument("--disp_edge_thresh", default=1.0, type=float)
    parser.add_argument("--boundary_kernel", default=5, type=int)
    parser.add_argument("--pseudo_self_distill", action="store_true")
    parser.add_argument(
        "--pseudo_on_labeled",
        action="store_true",
        help="Use the supervised training stereo pairs as unlabeled pairs for pseudo/self-distillation regularization.",
    )
    parser.add_argument("--teacher_model_dir", default=None, type=str)
    parser.add_argument("--teacher_base_model_dir", default=None, type=str)
    parser.add_argument("--unlabeled_root", default=None, type=str)
    parser.add_argument("--unlabeled_left_name", default="im0.png", type=str)
    parser.add_argument("--unlabeled_right_name", default="im1.png", type=str)
    parser.add_argument("--max_unlabeled_samples", default=None, type=int)
    parser.add_argument("--unlabeled_batch_size", default=1, type=int)
    parser.add_argument("--pseudo_loss_weight", default=0.2, type=float)
    parser.add_argument("--pseudo_confidence_temp", default=1.5, type=float)
    parser.add_argument("--pseudo_confidence_thresh", default=3.0, type=float)
    parser.add_argument("--pseudo_weak_aug", default=0.08, type=float)
    parser.add_argument("--pseudo_strong_aug", default=0.20, type=float)
    parser.add_argument(
        "--preserve_model_dir",
        default=None,
        type=str,
        help="Frozen foundation-prior checkpoint used for self-compiled student preservation.",
    )
    parser.add_argument(
        "--preserve_base_model_dir",
        default=None,
        type=str,
        help="Base serialized model used when --preserve_model_dir points to a train-state checkpoint.",
    )
    parser.add_argument(
        "--preserve_loss_weight",
        default=0.0,
        type=float,
        help="Weight for distilling the current student toward the frozen foundation prior on labeled crops.",
    )
    parser.add_argument(
        "--preserve_high_conf_prior",
        action="store_true",
        help=(
            "Preserve the frozen FFS prior only where its own disparity gives reliable "
            "left-right photometric reconstruction. This is the RepGeo geometry-prior mask."
        ),
    )
    parser.add_argument("--preserve_photo_temp", default=0.08, type=float)
    parser.add_argument("--preserve_photo_thresh", default=0.25, type=float)
    parser.add_argument("--preserve_edge_downweight", default=0.0, type=float)
    parser.add_argument("--preserve_min_conf", default=0.0, type=float)
    parser.add_argument(
        "--rcsa_pseudo",
        action="store_true",
        help="Reliability-calibrated pseudo supervision using teacher agreement, photometric reprojection, and edge risk.",
    )
    parser.add_argument("--rcsa_photo_temp", default=0.08, type=float)
    parser.add_argument("--rcsa_photo_thresh", default=0.25, type=float)
    parser.add_argument("--rcsa_edge_downweight", default=0.35, type=float)
    parser.add_argument("--rcsa_conf_power", default=1.0, type=float)
    parser.add_argument("--rcsa_min_conf", default=0.0, type=float)
    parser.add_argument("--rcsa_valid_conf_thresh", default=0.0, type=float)
    parser.add_argument("--ema_decay", default=0.0, type=float, help="Enable trainable-parameter EMA when > 0.")
    parser.add_argument("--lora_adapt", action="store_true")
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--lora_alpha", default=8.0, type=float)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--lora_min_channels", default=8, type=int)
    parser.add_argument("--lora_freeze_norm_stats", default=1, type=int)
    parser.add_argument(
        "--repgeo_sparse_weight",
        default=0.0,
        type=float,
        help="L1 penalty on learned RepGeo contribution coefficients alpha.",
    )
    parser.add_argument(
        "--repgeo_delta_weight",
        default=0.0,
        type=float,
        help="L2 penalty on low-rank residual weights to keep calibration close to the FFS prior.",
    )
    parser.add_argument(
        "--lora_targets",
        nargs="+",
        default=["cost", "update", "upsample"],
        help="Train-group names or explicit module prefixes to receive Conv-LoRA adapters.",
    )
    parser.add_argument(
        "--train_groups",
        nargs="+",
        default=["cost", "update", "upsample"],
        choices=sorted(TRAIN_GROUPS.keys()),
    )
    return parser.parse_args()


def count_parameters(parameters):
    return sum(param.numel() for param in parameters)


def configure_trainable_modules(model, train_groups):
    if "all" in train_groups:
        for param in model.parameters():
            param.requires_grad = True
        return ["all"]

    selected_prefixes = []
    for group in train_groups:
        selected_prefixes.extend(TRAIN_GROUPS[group])

    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in selected_prefixes)
    return selected_prefixes


def resolve_lora_prefixes(lora_targets):
    if "all" in lora_targets:
        return []

    selected_prefixes = []
    for target in lora_targets:
        if target in TRAIN_GROUPS:
            selected_prefixes.extend(TRAIN_GROUPS[target])
        else:
            selected_prefixes.append(target)

    deduped = []
    for prefix in selected_prefixes:
        if prefix and prefix not in deduped:
            deduped.append(prefix)
    return deduped


def configure_lora_adapters(model, args):
    from core.lora import apply_lora_to_conv_modules, mark_only_lora_as_trainable

    lora_prefixes = resolve_lora_prefixes(args.lora_targets)
    replaced = apply_lora_to_conv_modules(
        model,
        lora_prefixes,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        min_channels=args.lora_min_channels,
    )
    if not replaced:
        raise RuntimeError(f"No Conv2d/Conv3d layers matched LoRA targets: {args.lora_targets}")
    mark_only_lora_as_trainable(model)
    return lora_prefixes, replaced


def sequence_loss(disp_preds, disp_gt, valid, gamma, max_disp, pixel_weight=None):
    valid_mask = valid & torch.isfinite(disp_gt) & (disp_gt > 0) & (disp_gt < max_disp)
    if valid_mask.sum() == 0:
        return None

    loss = disp_gt.new_tensor(0.0)
    num_preds = len(disp_preds)
    for i, pred in enumerate(disp_preds):
        weight = gamma ** (num_preds - i - 1)
        error = (pred.squeeze(1) - disp_gt).abs()
        if pixel_weight is None:
            loss = loss + weight * error[valid_mask].mean()
        else:
            weights = pixel_weight[valid_mask]
            denom = weights.sum()
            if denom.item() <= 0:
                continue
            loss = loss + weight * (error[valid_mask] * weights).sum() / denom.clamp_min(1e-6)
    return loss


def build_boundary_mask(valid_noc, occ_mask, disp_gt, edge_thresh=1.0, kernel_size=5):
    valid_occ = valid_noc | occ_mask
    occ = occ_mask.float().unsqueeze(1)
    vis = valid_noc.float().unsqueeze(1)
    pad = kernel_size // 2

    occ_neigh = torch.nn.functional.max_pool2d(occ, kernel_size=kernel_size, stride=1, padding=pad) > 0
    vis_neigh = torch.nn.functional.max_pool2d(vis, kernel_size=kernel_size, stride=1, padding=pad) > 0
    occ_boundary = occ_neigh & vis_neigh

    disp = disp_gt.unsqueeze(1)
    dx = torch.nn.functional.pad((disp[..., 1:] - disp[..., :-1]).abs(), (0, 1, 0, 0))
    dy = torch.nn.functional.pad((disp[:, :, 1:, :] - disp[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    disp_edge = (dx > edge_thresh) | (dy > edge_thresh)
    return ((occ_boundary | disp_edge) & valid_occ.unsqueeze(1)).squeeze(1)


def gate_sparsity_loss(aux_outputs, warmup_iters):
    gates = aux_outputs.get("refine_gates", [])
    if not gates:
        return None
    gated_iters = gates[warmup_iters:]
    if not gated_iters:
        return None
    return torch.stack([gate.mean() for gate in gated_iters]).mean()


def init_disp_loss(init_disp, disp_gt, valid, max_disp):
    init_disp_up = torch.nn.functional.interpolate(init_disp * 4.0, scale_factor=4, mode="bilinear", align_corners=True)
    init_disp_up = init_disp_up.squeeze(1)
    valid_mask = valid & torch.isfinite(disp_gt) & (disp_gt > 0) & (disp_gt < max_disp)
    if valid_mask.sum() == 0:
        return None
    return (init_disp_up - disp_gt).abs()[valid_mask].mean()


def visibility_aware_loss(
    disp_preds,
    disp_gt,
    valid_noc,
    occ_mask,
    gamma,
    max_disp,
    occ_loss_weight,
    boundary_loss_weight,
    edge_thresh,
    kernel_size,
):
    visible_loss = sequence_loss(disp_preds, disp_gt, valid_noc, gamma, max_disp)
    occ_loss = sequence_loss(disp_preds, disp_gt, occ_mask, gamma, max_disp)
    boundary_mask = build_boundary_mask(valid_noc, occ_mask, disp_gt, edge_thresh=edge_thresh, kernel_size=kernel_size)
    boundary_loss = sequence_loss(disp_preds, disp_gt, boundary_mask, gamma, max_disp)

    components = {}
    total = disp_gt.new_tensor(0.0)
    if visible_loss is not None:
        total = total + visible_loss
        components["visible"] = visible_loss.item()
    if occ_loss is not None:
        total = total + occ_loss_weight * occ_loss
        components["occ"] = occ_loss.item()
    if boundary_loss is not None:
        total = total + boundary_loss_weight * boundary_loss
        components["boundary"] = boundary_loss.item()
    if not components:
        return None, {}
    return total, components


def apply_stereo_photometric_augmentation(left, right, strength):
    if strength <= 0:
        return left, right

    batch = left.shape[0]
    device = left.device
    x_left = (left / 255.0).clamp(0.0, 1.0)
    x_right = (right / 255.0).clamp(0.0, 1.0)

    brightness = (torch.rand(batch, 1, 1, 1, device=device) * 2.0 - 1.0) * strength
    contrast = 1.0 + (torch.rand(batch, 1, 1, 1, device=device) * 2.0 - 1.0) * strength
    gamma = torch.exp((torch.rand(batch, 1, 1, 1, device=device) * 2.0 - 1.0) * strength)

    x_left = torch.clamp(x_left * contrast + brightness, 0.0, 1.0)
    x_right = torch.clamp(x_right * contrast + brightness, 0.0, 1.0)
    x_left = torch.pow(x_left.clamp_min(1e-4), gamma)
    x_right = torch.pow(x_right.clamp_min(1e-4), gamma)

    noise_std = torch.rand(batch, 1, 1, 1, device=device) * (0.02 + 0.08 * strength)
    shared_noise = torch.randn(batch, 1, left.shape[2], left.shape[3], device=device) * noise_std
    x_left = torch.clamp(x_left + shared_noise, 0.0, 1.0)
    x_right = torch.clamp(x_right + shared_noise, 0.0, 1.0)

    if strength >= 0.12:
        blur_mask = (torch.rand(batch, 1, 1, 1, device=device) < 0.35).float()
        left_blur = torch.nn.functional.avg_pool2d(x_left, kernel_size=3, stride=1, padding=1)
        right_blur = torch.nn.functional.avg_pool2d(x_right, kernel_size=3, stride=1, padding=1)
        x_left = left_blur * blur_mask + x_left * (1.0 - blur_mask)
        x_right = right_blur * blur_mask + x_right * (1.0 - blur_mask)

    return x_left * 255.0, x_right * 255.0


def _rgb_to_gray01(image):
    image01 = (image / 255.0).clamp(0.0, 1.0)
    if image01.shape[1] == 1:
        return image01
    weights = image01.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image01[:, :3] * weights).sum(dim=1, keepdim=True)


def warp_right_to_left(right, disp):
    """Warp right image/features into the left view with positive left disparity."""

    batch, _, height, width = right.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=right.device, dtype=right.dtype),
        torch.arange(width, device=right.device, dtype=right.dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(batch, -1, -1)
    yy = yy.unsqueeze(0).expand(batch, -1, -1)
    sample_x = xx - disp
    valid = (sample_x >= 0) & (sample_x <= width - 1)
    grid_x = sample_x / max(width - 1, 1) * 2.0 - 1.0
    grid_y = yy / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    warped = torch.nn.functional.grid_sample(
        right,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped, valid


def photometric_reliability(left, right, disp, temp, thresh):
    left_gray = _rgb_to_gray01(left)
    right_gray = _rgb_to_gray01(right)
    warped_right, warp_valid = warp_right_to_left(right_gray, disp)
    photo_err = (left_gray - warped_right).abs().squeeze(1)
    conf = torch.exp(-photo_err / max(float(temp), 1e-6))
    if thresh > 0:
        conf = conf * (photo_err < float(thresh)).float()
    conf = conf * warp_valid.float()
    return conf, photo_err


def edge_risk_reliability(left, downweight):
    """Downweight likely discontinuities for pseudo labels, not for supervised GT."""

    if downweight <= 0:
        return left.new_ones(left.shape[0], left.shape[2], left.shape[3])
    gray = _rgb_to_gray01(left)
    dx = torch.nn.functional.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = torch.nn.functional.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    grad = (dx + dy).squeeze(1)
    # A smooth soft attenuation is more stable than hard per-image quantiles.
    edge_risk = grad / (grad.flatten(1).mean(dim=1).view(-1, 1, 1).clamp_min(1e-6) * 4.0)
    edge_risk = edge_risk.clamp(0.0, 1.0)
    return (1.0 - float(downweight) * edge_risk).clamp(0.0, 1.0)


def reliability_calibrated_pseudo_weight(left, right, pseudo_disp, agreement, base_conf, args):
    photo_conf, photo_err = photometric_reliability(
        left,
        right,
        pseudo_disp,
        temp=args.rcsa_photo_temp,
        thresh=args.rcsa_photo_thresh,
    )
    edge_conf = edge_risk_reliability(left, args.rcsa_edge_downweight)
    conf = base_conf * photo_conf * edge_conf
    if args.rcsa_conf_power != 1.0:
        conf = conf.clamp_min(1e-8).pow(float(args.rcsa_conf_power))
    if args.rcsa_min_conf > 0:
        conf = conf.clamp_min(float(args.rcsa_min_conf))
    stats = {
        "pseudo_photo_err": photo_err.mean().item(),
        "pseudo_photo_conf": photo_conf.mean().item(),
        "pseudo_edge_conf": edge_conf.mean().item(),
        "pseudo_reliability": conf.mean().item(),
        "pseudo_agreement": agreement.mean().item(),
    }
    return conf, stats


def compute_pseudo_self_distill_loss(model, teacher_model, batch, device, args):
    left = batch["left"].to(device, non_blocking=True)
    right = batch["right"].to(device, non_blocking=True)

    weak_left_1, weak_right_1 = apply_stereo_photometric_augmentation(left, right, args.pseudo_weak_aug)
    weak_left_2, weak_right_2 = apply_stereo_photometric_augmentation(left, right, args.pseudo_weak_aug)
    strong_left, strong_right = apply_stereo_photometric_augmentation(left, right, args.pseudo_strong_aug)

    with torch.no_grad():
        with cuda_autocast(enabled=device.type == "cuda", dtype=torch.float16):
            pseudo_1 = teacher_model.forward(
                weak_left_1,
                weak_right_1,
                iters=args.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
            pseudo_2 = teacher_model.forward(
                weak_left_2,
                weak_right_2,
                iters=args.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )

    pseudo_1 = pseudo_1.squeeze(1).detach()
    pseudo_2 = pseudo_2.squeeze(1).detach()
    pseudo_disp = 0.5 * (pseudo_1 + pseudo_2)
    agreement = (pseudo_1 - pseudo_2).abs()
    conf = torch.exp(-agreement / max(args.pseudo_confidence_temp, 1e-6))
    if args.pseudo_confidence_thresh > 0:
        conf = conf * (agreement < args.pseudo_confidence_thresh).float()
    reliability_stats = {}
    if args.rcsa_pseudo:
        conf, reliability_stats = reliability_calibrated_pseudo_weight(
            left,
            right,
            pseudo_disp,
            agreement,
            conf,
            args,
        )

    valid = torch.isfinite(pseudo_disp) & (pseudo_disp > 0) & (pseudo_disp < args.max_disp)
    if args.rcsa_valid_conf_thresh > 0:
        valid = valid & (conf > float(args.rcsa_valid_conf_thresh))
    if valid.sum() == 0:
        return None, {}

    with cuda_autocast(enabled=device.type == "cuda", dtype=torch.float16):
        _, disp_preds, _ = model.forward(
            strong_left,
            strong_right,
            iters=args.valid_iters,
            test_mode=False,
            optimize_build_volume="pytorch1",
            return_aux=True,
        )
        pseudo_loss = sequence_loss(
            disp_preds,
            pseudo_disp,
            valid,
            args.gamma,
            args.max_disp,
            pixel_weight=conf,
        )

    if pseudo_loss is None:
        return None, {}

    valid_conf = conf[valid]
    stats = {
        "pseudo": pseudo_loss.item(),
        "pseudo_conf": valid_conf.mean().item() if valid_conf.numel() > 0 else math.nan,
        "pseudo_agreement": agreement[valid].mean().item() if valid.any() else math.nan,
    }
    stats.update(reliability_stats)
    return pseudo_loss, stats


def compute_preserve_loss(preserve_model, disp_preds, left, right, args):
    """Keep the compiled student near the released foundation prior.

    This is the SC-FFS "prior preservation" term.  It is intentionally a
    distillation loss on the already-computed student predictions, so the final
    deployed model can still be exported as a static FFS checkpoint.
    """

    if preserve_model is None or args.preserve_loss_weight <= 0:
        return None, {}

    with torch.no_grad():
        with cuda_autocast(enabled=left.device.type == "cuda", dtype=torch.float16):
            prior_disp = preserve_model.forward(
                left,
                right,
                iters=args.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
    prior_disp = prior_disp.squeeze(1).detach()
    valid = torch.isfinite(prior_disp) & (prior_disp > 0) & (prior_disp < args.max_disp)
    pixel_weight = None
    preserve_conf = None
    preserve_photo_err = None
    preserve_edge_conf = None
    if args.preserve_high_conf_prior:
        photo_conf, photo_err = photometric_reliability(
            left,
            right,
            prior_disp,
            temp=args.preserve_photo_temp,
            thresh=args.preserve_photo_thresh,
        )
        edge_conf = edge_risk_reliability(left, args.preserve_edge_downweight)
        preserve_conf = photo_conf * edge_conf
        if args.preserve_min_conf > 0:
            preserve_conf = preserve_conf * (preserve_conf >= float(args.preserve_min_conf)).float()
        valid = valid & (preserve_conf > 0)
        pixel_weight = preserve_conf
        preserve_photo_err = photo_err
        preserve_edge_conf = edge_conf
    if valid.sum() == 0:
        return None, {}
    preserve_loss = sequence_loss(
        disp_preds,
        prior_disp,
        valid,
        args.gamma,
        args.max_disp,
        pixel_weight=pixel_weight,
    )
    if preserve_loss is None:
        return None, {}
    stats = {
        "preserve": preserve_loss.item(),
        "preserve_valid": valid.float().mean().item(),
        "preserve_prior_mean": prior_disp[valid].mean().item() if valid.any() else math.nan,
    }
    if preserve_conf is not None:
        stats.update(
            {
                "preserve_conf": preserve_conf[valid].mean().item() if valid.any() else math.nan,
                "preserve_photo_err": preserve_photo_err[valid].mean().item() if valid.any() else math.nan,
                "preserve_edge_conf": preserve_edge_conf[valid].mean().item() if valid.any() else math.nan,
            }
        )
    return preserve_loss, stats


@torch.no_grad()
def compute_metrics(pred_disp, disp_gt, valid, max_disp):
    valid_mask = valid & torch.isfinite(disp_gt) & (disp_gt > 0) & (disp_gt < max_disp)
    if valid_mask.sum() == 0:
        return {"epe": math.nan, "d1": math.nan, "bad3": math.nan, "count": 0}

    error = (pred_disp.squeeze(1) - disp_gt).abs()
    error = error[valid_mask]
    gt = disp_gt[valid_mask]
    d1 = ((error > 3.0) & ((error / gt) > 0.05)).float().mean().item()
    bad3 = (error > 3.0).float().mean().item()
    return {
        "epe": error.mean().item(),
        "d1": d1,
        "bad3": bad3,
        "count": int(valid_mask.sum().item()),
    }


def create_dataloaders(args):
    from core.datasets import KittiStereoDataset, UnlabeledStereoDataset

    train_set = KittiStereoDataset(
        args.data_root,
        split=args.train_split,
        crop_size=(args.crop_h, args.crop_w),
        val_stride=args.val_stride,
        max_samples=args.max_train_samples,
    )
    val_set = KittiStereoDataset(
        args.data_root,
        split=args.val_split,
        crop_size=(args.crop_h, args.crop_w),
        val_stride=args.val_stride,
        max_samples=args.max_val_samples,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    unlabeled_loader = None
    if args.pseudo_self_distill and not args.eval_only:
        if args.pseudo_on_labeled:
            return train_loader, val_loader, None
        if args.unlabeled_root is None:
            raise ValueError("--unlabeled_root is required when --pseudo_self_distill is enabled.")
        unlabeled_set = UnlabeledStereoDataset(
            args.unlabeled_root,
            left_name=args.unlabeled_left_name,
            right_name=args.unlabeled_right_name,
            crop_size=(args.crop_h, args.crop_w),
            max_samples=args.max_unlabeled_samples,
        )
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=args.unlabeled_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return train_loader, val_loader, unlabeled_loader


def load_model(args, device):
    from core.foundation_stereo import FastFoundationStereo

    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
        if not args.lora_adapt:
            model.load_state_dict(resume_state["model_state"])
        start_epoch = resume_state["epoch"] + 1
        best_d1 = resume_state["best_d1"]
    else:
        model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
        start_epoch = 0
        best_d1 = float("inf")

    if not isinstance(model, FastFoundationStereo):
        raise TypeError(f"Expected FastFoundationStereo checkpoint, got: {type(model)}")

    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.update_block.hidden_dim = model.update_block.disp_head.conv[0].in_channels
    model.update_block._ensure_refine_gate()
    model.args.narrow_band = args.narrow_band
    model.args.band_radius = args.band_radius
    model.args.adaptive_refine = args.adaptive_refine
    model.args.adaptive_refine_warmup = args.adaptive_refine_warmup
    model.args.adaptive_refine_threshold = args.adaptive_refine_threshold
    model.args.adaptive_refine_hard = bool(args.adaptive_refine_hard)
    model = model.to(device)
    return model, start_epoch, best_d1, resume_state


def load_teacher_model(args, device):
    if not args.pseudo_self_distill or args.eval_only:
        return None

    from core.foundation_stereo import FastFoundationStereo

    teacher_path = args.teacher_model_dir or args.model_dir
    loaded = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, FastFoundationStereo):
        teacher_model = loaded
    elif isinstance(loaded, dict) and "model_state" in loaded:
        base_model_path = args.teacher_base_model_dir or args.model_dir
        base_model = torch.load(base_model_path, map_location="cpu", weights_only=False)
        if not isinstance(base_model, FastFoundationStereo):
            raise TypeError(f"Expected FastFoundationStereo teacher base model, got: {type(base_model)}")
        base_model.load_state_dict(loaded["model_state"])
        teacher_model = base_model
    else:
        raise TypeError(f"Unsupported teacher checkpoint type: {type(loaded)}")

    teacher_model.args.valid_iters = args.valid_iters
    teacher_model.args.max_disp = args.max_disp
    teacher_model.update_block.hidden_dim = teacher_model.update_block.disp_head.conv[0].in_channels
    teacher_model.update_block._ensure_refine_gate()
    teacher_model = teacher_model.to(device).eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model


def load_preserve_model(args, device):
    if args.eval_only or args.preserve_loss_weight <= 0 or args.preserve_model_dir is None:
        return None

    from core.foundation_stereo import FastFoundationStereo

    loaded = torch.load(args.preserve_model_dir, map_location="cpu", weights_only=False)
    if isinstance(loaded, FastFoundationStereo):
        preserve_model = loaded
    elif isinstance(loaded, dict) and "model_state" in loaded:
        base_model_path = args.preserve_base_model_dir or args.model_dir
        preserve_model = torch.load(base_model_path, map_location="cpu", weights_only=False)
        if not isinstance(preserve_model, FastFoundationStereo):
            raise TypeError(f"Expected FastFoundationStereo preserve base model, got: {type(preserve_model)}")
        preserve_model.load_state_dict(loaded["model_state"])
    else:
        raise TypeError(f"Unsupported preserve checkpoint type: {type(loaded)}")

    preserve_model.args.valid_iters = args.valid_iters
    preserve_model.args.max_disp = args.max_disp
    preserve_model.update_block.hidden_dim = preserve_model.update_block.disp_head.conv[0].in_channels
    preserve_model.update_block._ensure_refine_gate()
    preserve_model = preserve_model.to(device).eval()
    for param in preserve_model.parameters():
        param.requires_grad = False
    return preserve_model


def repgeo_regularization(model, args):
    if not args.lora_adapt or (args.repgeo_sparse_weight <= 0 and args.repgeo_delta_weight <= 0):
        return None, {}
    from core.lora import LORA_CONV_TYPES, repgeo_alpha_l1

    total = None
    stats = {}
    if args.repgeo_sparse_weight > 0:
        alpha_l1 = repgeo_alpha_l1(model)
        if alpha_l1 is not None:
            total = args.repgeo_sparse_weight * alpha_l1 if total is None else total + args.repgeo_sparse_weight * alpha_l1
            stats["repgeo_sparse"] = alpha_l1.detach().item()
    if args.repgeo_delta_weight > 0:
        delta_terms = []
        for module in model.modules():
            if isinstance(module, LORA_CONV_TYPES):
                delta_terms.append(module.lora_down.weight.float().pow(2).mean())
                delta_terms.append(module.lora_up.weight.float().pow(2).mean())
        if delta_terms:
            delta_l2 = torch.stack(delta_terms).mean()
            total = args.repgeo_delta_weight * delta_l2 if total is None else total + args.repgeo_delta_weight * delta_l2
            stats["repgeo_delta"] = delta_l2.detach().item()
    return total, stats


class TrainableEMA:
    """EMA over trainable parameters only; frozen backbone weights stay untouched."""

    def __init__(self, model, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("--ema_decay must be in (0, 1)")
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        if not self.shadow:
            raise RuntimeError("EMA requested but no trainable parameters were found.")

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        backup = {}
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            backup[name] = param.detach().clone()
            param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for name, param in model.named_parameters():
            if name in backup:
                param.copy_(backup[name].to(device=param.device, dtype=param.dtype))

    def state_dict(self):
        return {"decay": self.decay, "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()}}

    def load_state_dict(self, state):
        shadow = state.get("shadow", {})
        missing = sorted(set(self.shadow) - set(shadow))
        unexpected = sorted(set(shadow) - set(self.shadow))
        blocking_missing = [name for name in missing if not name.endswith(".repgeo_alpha")]
        if blocking_missing or unexpected:
            raise RuntimeError(
                f"EMA state mismatch: missing={blocking_missing[:5]} unexpected={unexpected[:5]}"
            )
        self.decay = float(state.get("decay", self.decay))
        for name, value in shadow.items():
            if name not in self.shadow:
                continue
            self.shadow[name].copy_(value.to(device=self.shadow[name].device, dtype=self.shadow[name].dtype))

    def validate(self, model, val_loader, device, args):
        backup = self.apply_to(model)
        try:
            return validate(model, val_loader, device, args)
        finally:
            self.restore(model, backup)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    args,
    epoch,
    teacher_model=None,
    preserve_model=None,
    unlabeled_loader=None,
    ema=None,
):
    model.train()
    if args.lora_adapt and args.lora_freeze_norm_stats:
        from core.lora import set_norm_layers_eval

        set_norm_layers_eval(model)
    running_loss = 0.0
    running_gate = 0.0
    running_visible = 0.0
    running_occ = 0.0
    running_boundary = 0.0
    running_pseudo = 0.0
    running_pseudo_conf = 0.0
    running_pseudo_agreement = 0.0
    running_pseudo_photo_conf = 0.0
    running_pseudo_photo_err = 0.0
    running_pseudo_edge_conf = 0.0
    running_pseudo_reliability = 0.0
    running_preserve = 0.0
    running_preserve_valid = 0.0
    running_preserve_conf = 0.0
    running_preserve_photo_err = 0.0
    running_preserve_edge_conf = 0.0
    running_repgeo_sparse = 0.0
    running_repgeo_delta = 0.0
    total_steps = 0
    pseudo_steps = 0
    preserve_steps = 0
    repgeo_steps = 0
    unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None

    for step, batch in enumerate(loader):
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        disp_gt = batch["disp"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True).bool()
        valid_noc = batch["valid_noc"].to(device, non_blocking=True).bool()
        occ_mask = batch["occ_mask"].to(device, non_blocking=True).bool()

        optimizer.zero_grad(set_to_none=True)
        loss = None
        pseudo_term = None
        pseudo_stats = {}
        with cuda_autocast(enabled=device.type == "cuda", dtype=torch.float16):
            init_disp, disp_preds, aux_outputs = model.forward(
                left,
                right,
                iters=args.valid_iters,
                test_mode=False,
                optimize_build_volume="pytorch1",
                return_aux=True,
            )
            loss_components = {}
            if args.visibility_supervision:
                loss, loss_components = visibility_aware_loss(
                    disp_preds,
                    disp_gt,
                    valid_noc,
                    occ_mask,
                    args.gamma,
                    args.max_disp,
                    args.occ_loss_weight,
                    args.boundary_loss_weight,
                    args.disp_edge_thresh,
                    args.boundary_kernel,
                )
            else:
                loss = sequence_loss(disp_preds, disp_gt, valid, args.gamma, args.max_disp)
            if args.init_loss_weight > 0:
                coarse_loss = init_disp_loss(init_disp, disp_gt, valid, args.max_disp)
                if coarse_loss is not None:
                    loss = loss + args.init_loss_weight * coarse_loss
            gate_loss = None
            if args.adaptive_refine and args.gate_sparsity_weight > 0:
                gate_loss = gate_sparsity_loss(aux_outputs, args.adaptive_refine_warmup)
                if gate_loss is not None:
                    loss = loss + args.gate_sparsity_weight * gate_loss
            preserve_loss, preserve_stats = compute_preserve_loss(
                preserve_model,
                disp_preds,
                left,
                right,
                args,
            )
            if preserve_loss is not None:
                loss = loss + args.preserve_loss_weight * preserve_loss
            repgeo_loss, repgeo_stats = repgeo_regularization(model, args)
            if repgeo_loss is not None:
                loss = loss + repgeo_loss

        if loss is not None:
            scaler.scale(loss).backward()

        if teacher_model is not None and (unlabeled_iter is not None or args.pseudo_on_labeled):
            if args.pseudo_on_labeled:
                unlabeled_batch = batch
            else:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)

            pseudo_loss, pseudo_stats = compute_pseudo_self_distill_loss(model, teacher_model, unlabeled_batch, device, args)
            if pseudo_loss is not None:
                pseudo_term = args.pseudo_loss_weight * pseudo_loss
                if pseudo_term.requires_grad:
                    scaler.scale(pseudo_term).backward()
                else:
                    pseudo_term = None

        if loss is None and pseudo_term is None:
            continue

        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        batch_loss_value = 0.0
        if loss is not None:
            batch_loss_value += loss.item()
        if pseudo_term is not None:
            batch_loss_value += pseudo_term.item()
        running_loss += batch_loss_value
        if args.visibility_supervision:
            if "visible" in loss_components:
                running_visible += loss_components["visible"]
            if "occ" in loss_components:
                running_occ += loss_components["occ"]
            if "boundary" in loss_components:
                running_boundary += loss_components["boundary"]
        if args.adaptive_refine:
            gate_mean = gate_sparsity_loss(aux_outputs, 0)
            if gate_mean is not None:
                running_gate += gate_mean.item()
        if preserve_model is not None and "preserve" in preserve_stats:
            running_preserve += preserve_stats["preserve"]
            running_preserve_valid += preserve_stats.get("preserve_valid", math.nan)
            running_preserve_conf += preserve_stats.get("preserve_conf", math.nan)
            running_preserve_photo_err += preserve_stats.get("preserve_photo_err", math.nan)
            running_preserve_edge_conf += preserve_stats.get("preserve_edge_conf", math.nan)
            preserve_steps += 1
        if "repgeo_sparse" in repgeo_stats or "repgeo_delta" in repgeo_stats:
            running_repgeo_sparse += repgeo_stats.get("repgeo_sparse", math.nan)
            running_repgeo_delta += repgeo_stats.get("repgeo_delta", math.nan)
            repgeo_steps += 1
        if pseudo_term is not None:
            running_pseudo += pseudo_stats.get("pseudo", math.nan)
            running_pseudo_conf += pseudo_stats.get("pseudo_conf", math.nan)
            running_pseudo_agreement += pseudo_stats.get("pseudo_agreement", math.nan)
            running_pseudo_photo_conf += pseudo_stats.get("pseudo_photo_conf", math.nan)
            running_pseudo_photo_err += pseudo_stats.get("pseudo_photo_err", math.nan)
            running_pseudo_edge_conf += pseudo_stats.get("pseudo_edge_conf", math.nan)
            running_pseudo_reliability += pseudo_stats.get("pseudo_reliability", math.nan)
            pseudo_steps += 1
        total_steps += 1

        if step % 20 == 0:
            message = (
                f"epoch {epoch:02d} step {step:04d}/{len(loader):04d} "
                f"loss={batch_loss_value:.4f}"
            )
            if args.visibility_supervision:
                if "visible" in loss_components:
                    message += f" vis={loss_components['visible']:.4f}"
                if "occ" in loss_components:
                    message += f" occ={loss_components['occ']:.4f}"
                if "boundary" in loss_components:
                    message += f" bnd={loss_components['boundary']:.4f}"
            if args.adaptive_refine:
                gate_mean = gate_sparsity_loss(aux_outputs, 0)
                if gate_mean is not None:
                    message += f" gate={gate_mean.item():.4f}"
            if preserve_model is not None and "preserve" in preserve_stats:
                message += (
                    f" preserve={preserve_stats['preserve']:.4f}"
                    f" pvalid={preserve_stats.get('preserve_valid', math.nan):.3f}"
                )
                if args.preserve_high_conf_prior:
                    message += (
                        f" pconf={preserve_stats.get('preserve_conf', math.nan):.4f}"
                        f" pphoto={preserve_stats.get('preserve_photo_err', math.nan):.4f}"
                    )
            if "repgeo_sparse" in repgeo_stats or "repgeo_delta" in repgeo_stats:
                message += (
                    f" alpha_l1={repgeo_stats.get('repgeo_sparse', math.nan):.4f}"
                    f" delta_l2={repgeo_stats.get('repgeo_delta', math.nan):.6f}"
                )
            if pseudo_term is not None:
                message += (
                    f" pseudo={pseudo_stats.get('pseudo', math.nan):.4f}"
                    f" pconf={pseudo_stats.get('pseudo_conf', math.nan):.4f}"
                )
                if args.rcsa_pseudo:
                    message += (
                        f" rconf={pseudo_stats.get('pseudo_reliability', math.nan):.4f}"
                        f" photo={pseudo_stats.get('pseudo_photo_conf', math.nan):.4f}"
                    )
            logging.info(message)

    mean_loss = running_loss / max(1, total_steps)
    metrics = {"loss": mean_loss}
    if args.visibility_supervision:
        metrics["visible"] = running_visible / max(1, total_steps)
        metrics["occ"] = running_occ / max(1, total_steps)
        metrics["boundary"] = running_boundary / max(1, total_steps)
    if args.adaptive_refine:
        metrics["gate"] = running_gate / max(1, total_steps)
    if preserve_steps > 0:
        metrics["preserve"] = running_preserve / preserve_steps
        metrics["preserve_valid"] = running_preserve_valid / preserve_steps
        metrics["preserve_conf"] = running_preserve_conf / preserve_steps
        metrics["preserve_photo_err"] = running_preserve_photo_err / preserve_steps
        metrics["preserve_edge_conf"] = running_preserve_edge_conf / preserve_steps
    if repgeo_steps > 0:
        metrics["repgeo_sparse"] = running_repgeo_sparse / repgeo_steps
        metrics["repgeo_delta"] = running_repgeo_delta / repgeo_steps
    if pseudo_steps > 0:
        metrics["pseudo"] = running_pseudo / pseudo_steps
        metrics["pseudo_conf"] = running_pseudo_conf / pseudo_steps
        metrics["pseudo_agreement"] = running_pseudo_agreement / pseudo_steps
        metrics["pseudo_photo_conf"] = running_pseudo_photo_conf / pseudo_steps
        metrics["pseudo_photo_err"] = running_pseudo_photo_err / pseudo_steps
        metrics["pseudo_edge_conf"] = running_pseudo_edge_conf / pseudo_steps
        metrics["pseudo_reliability"] = running_pseudo_reliability / pseudo_steps
    return metrics


@torch.no_grad()
def validate(model, loader, device, args):
    from core.utils.utils import InputPadder

    model.eval()
    total_epe = 0.0
    total_d1 = 0.0
    total_bad3 = 0.0
    total_gate = 0.0
    total_images = 0
    total_epe_vis = 0.0
    total_epe_occ = 0.0
    total_vis_images = 0
    total_occ_images = 0

    for batch in loader:
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        disp_gt = batch["disp"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True).bool()
        valid_noc = batch["valid_noc"].to(device, non_blocking=True).bool()
        occ_mask = batch["occ_mask"].to(device, non_blocking=True).bool()

        padder = InputPadder(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)
        pred, aux_outputs = model.forward(
            left,
            right,
            iters=args.valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
            return_aux=True,
        )
        pred = padder.unpad(pred)

        metrics = compute_metrics(pred, disp_gt, valid, args.max_disp)
        if metrics["count"] == 0:
            continue
        total_epe += metrics["epe"]
        total_d1 += metrics["d1"]
        total_bad3 += metrics["bad3"]
        error = (pred.squeeze(1) - disp_gt).abs()
        vis_valid = valid_noc & valid
        occ_valid = occ_mask & valid
        if vis_valid.any():
            total_epe_vis += error[vis_valid].mean().item()
            total_vis_images += 1
        if occ_valid.any():
            total_epe_occ += error[occ_valid].mean().item()
            total_occ_images += 1
        if args.adaptive_refine:
            gate_mean = gate_sparsity_loss(aux_outputs, 0)
            if gate_mean is not None:
                total_gate += gate_mean.item()
        total_images += 1

    if total_images == 0:
        return {
            "epe": math.nan,
            "d1": math.nan,
            "bad3": math.nan,
            "gate": math.nan,
            "epe_vis": math.nan,
            "epe_occ": math.nan,
        }

    metrics = {
        "epe": total_epe / total_images,
        "d1": total_d1 / total_images,
        "bad3": total_bad3 / total_images,
        "epe_vis": total_epe_vis / max(1, total_vis_images),
        "epe_occ": total_epe_occ / max(1, total_occ_images),
    }
    if args.adaptive_refine:
        metrics["gate"] = total_gate / total_images
    return metrics


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_d1,
    best_epe,
    out_dir,
    is_best_d1,
    is_best_epe,
    args=None,
    ema=None,
    best_ema_d1=None,
    best_ema_epe=None,
    is_best_ema_d1=False,
    is_best_ema_epe=False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_d1": best_d1,
        "best_epe": best_epe,
    }
    if best_ema_d1 is not None:
        state["best_ema_d1"] = best_ema_d1
    if best_ema_epe is not None:
        state["best_ema_epe"] = best_ema_epe
    if ema is not None:
        state["ema_state"] = ema.state_dict()
    if args is not None:
        state["args"] = vars(args)
    torch.save(state, out_dir / "last_state.pth")
    if is_best_d1:
        torch.save(model, out_dir / "model_best.pth")
    if is_best_epe:
        torch.save(model, out_dir / "model_best_epe.pth")
    if ema is not None and (is_best_ema_d1 or is_best_ema_epe):
        backup = ema.apply_to(model)
        try:
            if is_best_ema_d1:
                torch.save(model, out_dir / "model_best_ema.pth")
            if is_best_ema_epe:
                torch.save(model, out_dir / "model_best_ema_epe.pth")
        finally:
            ema.restore(model, backup)


def main():
    args = parse_args()
    set_logging_format()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not (args.eval_only and args.allow_cpu_eval):
        raise RuntimeError("CUDA is required for training. Use --eval_only --allow_cpu_eval for small CPU checks.")

    train_loader, val_loader, unlabeled_loader = create_dataloaders(args)
    model, start_epoch, best_d1, resume_state = load_model(args, device)
    best_epe = float(resume_state.get("best_epe", float("inf"))) if resume_state is not None else float("inf")
    best_ema_d1 = float(resume_state.get("best_ema_d1", float("inf"))) if resume_state is not None else float("inf")
    best_ema_epe = float(resume_state.get("best_ema_epe", float("inf"))) if resume_state is not None else float("inf")
    teacher_model = load_teacher_model(args, device)
    preserve_model = load_preserve_model(args, device)
    lora_replaced = []
    if args.lora_adapt:
        selected_prefixes, lora_replaced = configure_lora_adapters(model, args)
        if resume_state is not None:
            missing, unexpected = model.load_state_dict(resume_state["model_state"], strict=False)
            logging.info(f"resume LoRA state: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        selected_prefixes = configure_trainable_modules(model, args.train_groups)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were selected.")
    logging.info(f"device: {device}")
    logging.info(f"train split: {len(train_loader.dataset)} samples")
    logging.info(f"val split: {len(val_loader.dataset)} samples")
    if unlabeled_loader is not None:
        logging.info(f"unlabeled split: {len(unlabeled_loader.dataset)} samples")
    logging.info(f"train groups: {args.train_groups}")
    logging.info(
        f"pseudo distill: enabled={args.pseudo_self_distill} on_labeled={args.pseudo_on_labeled} "
        f"rcsa={args.rcsa_pseudo}"
    )
    logging.info(
        f"preserve prior: enabled={preserve_model is not None} "
        f"weight={args.preserve_loss_weight} model={args.preserve_model_dir}"
    )
    if args.lora_adapt:
        logging.info(
            f"LoRA: rank={args.lora_rank} alpha={args.lora_alpha} dropout={args.lora_dropout} "
            f"targets={args.lora_targets} adapters={len(lora_replaced)}"
        )
    logging.info(f"train prefixes: {selected_prefixes}")
    logging.info(
        f"trainable params: {count_parameters(trainable_params) / 1e6:.2f}M / "
        f"{count_parameters(model.parameters()) / 1e6:.2f}M"
    )
    ema = TrainableEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        logging.info(f"EMA: enabled decay={args.ema_decay} params={len(ema.shadow)}")
        if resume_state is not None and "ema_state" in resume_state:
            ema.load_state_dict(resume_state["ema_state"])
            logging.info("EMA: restored from resume state")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = cuda_grad_scaler(enabled=device.type == "cuda")

    if args.resume and not args.reset_optimizer_on_resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(resume_state["optimizer_state"])
    elif args.resume:
        logging.info("optimizer: reset on resume; using current lr/weight_decay settings")

    out_dir = Path(args.out_dir)
    if args.eval_only:
        val_metrics = validate(model, val_loader, device, args)
        message = (
            f"val: EPE={val_metrics['epe']:.4f} "
            f"D1={val_metrics['d1']:.4f} "
            f"bad3={val_metrics['bad3']:.4f} "
            f"EPE_vis={val_metrics['epe_vis']:.4f} "
            f"EPE_occ={val_metrics['epe_occ']:.4f}"
        )
        if args.adaptive_refine:
            message += f" gate={val_metrics['gate']:.4f}"
        logging.info(message)
        if args.report_json:
            import json

            report_path = Path(args.report_json)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_dir": args.model_dir,
                "resume": args.resume,
                "data_root": args.data_root,
                "device": str(device),
                "train_split": args.train_split,
                "val_split": args.val_split,
                "val_stride": args.val_stride,
                "max_val_samples": args.max_val_samples,
                "valid_iters": args.valid_iters,
                "max_disp": args.max_disp,
                "lora_adapt": bool(args.lora_adapt),
                "metrics": val_metrics,
            }
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            epoch,
            teacher_model=teacher_model,
            preserve_model=preserve_model,
            unlabeled_loader=unlabeled_loader,
            ema=ema,
        )
        val_metrics = validate(model, val_loader, device, args)
        ema_metrics = ema.validate(model, val_loader, device, args) if ema is not None else None

        is_best_d1 = val_metrics["d1"] < best_d1
        is_best_epe = val_metrics["epe"] < best_epe
        if is_best_d1:
            best_d1 = val_metrics["d1"]
        if is_best_epe:
            best_epe = val_metrics["epe"]

        is_best_ema_d1 = False
        is_best_ema_epe = False
        if ema_metrics is not None:
            is_best_ema_d1 = ema_metrics["d1"] < best_ema_d1
            is_best_ema_epe = ema_metrics["epe"] < best_ema_epe
            if is_best_ema_d1:
                best_ema_d1 = ema_metrics["d1"]
            if is_best_ema_epe:
                best_ema_epe = ema_metrics["epe"]

        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_d1,
            best_epe,
            out_dir,
            is_best_d1,
            is_best_epe,
            args=args,
            ema=ema,
            best_ema_d1=best_ema_d1 if ema is not None else None,
            best_ema_epe=best_ema_epe if ema is not None else None,
            is_best_ema_d1=is_best_ema_d1,
            is_best_ema_epe=is_best_ema_epe,
        )

        message = (
            f"epoch {epoch:02d} done: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_EPE={val_metrics['epe']:.4f} "
            f"val_D1={val_metrics['d1']:.4f} "
            f"val_bad3={val_metrics['bad3']:.4f} "
            f"val_EPE_vis={val_metrics['epe_vis']:.4f} "
            f"val_EPE_occ={val_metrics['epe_occ']:.4f}"
        )
        if ema_metrics is not None:
            message += (
                f" ema_EPE={ema_metrics['epe']:.4f}"
                f" ema_D1={ema_metrics['d1']:.4f}"
                f" ema_bad3={ema_metrics['bad3']:.4f}"
                f" ema_EPE_vis={ema_metrics['epe_vis']:.4f}"
                f" ema_EPE_occ={ema_metrics['epe_occ']:.4f}"
            )
        if args.visibility_supervision:
            message += (
                f" train_vis={train_metrics.get('visible', math.nan):.4f}"
                f" train_occ={train_metrics.get('occ', math.nan):.4f}"
                f" train_bnd={train_metrics.get('boundary', math.nan):.4f}"
            )
        if args.adaptive_refine:
            message += (
                f" train_gate={train_metrics.get('gate', math.nan):.4f}"
                f" val_gate={val_metrics.get('gate', math.nan):.4f}"
            )
        if args.pseudo_self_distill:
            message += (
                f" train_pseudo={train_metrics.get('pseudo', math.nan):.4f}"
                f" train_pconf={train_metrics.get('pseudo_conf', math.nan):.4f}"
                f" train_pagree={train_metrics.get('pseudo_agreement', math.nan):.4f}"
            )
            if args.rcsa_pseudo:
                message += (
                    f" train_rconf={train_metrics.get('pseudo_reliability', math.nan):.4f}"
                    f" train_photo_conf={train_metrics.get('pseudo_photo_conf', math.nan):.4f}"
                    f" train_photo_err={train_metrics.get('pseudo_photo_err', math.nan):.4f}"
                )
        if preserve_model is not None:
            message += (
                f" train_preserve={train_metrics.get('preserve', math.nan):.4f}"
                f" train_pvalid={train_metrics.get('preserve_valid', math.nan):.4f}"
            )
            if args.preserve_high_conf_prior:
                message += (
                    f" train_pconf={train_metrics.get('preserve_conf', math.nan):.4f}"
                    f" train_pphoto={train_metrics.get('preserve_photo_err', math.nan):.4f}"
                )
        if args.repgeo_sparse_weight > 0 or args.repgeo_delta_weight > 0:
            message += (
                f" train_alpha_l1={train_metrics.get('repgeo_sparse', math.nan):.4f}"
                f" train_delta_l2={train_metrics.get('repgeo_delta', math.nan):.6f}"
            )
        logging.info(message)


if __name__ == "__main__":
    main()

