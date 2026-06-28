import math

import torch
import torch.nn as nn


class _LoRAConvMixin:
    @property
    def in_channels(self):
        return self.base.in_channels

    @property
    def out_channels(self):
        return self.base.out_channels

    @property
    def kernel_size(self):
        return self.base.kernel_size

    @property
    def stride(self):
        return self.base.stride

    @property
    def padding(self):
        return self.base.padding

    @property
    def dilation(self):
        return self.base.dilation

    @property
    def groups(self):
        return self.base.groups

    @property
    def bias(self):
        return self.base.bias

    @property
    def weight(self):
        return self.base.weight

    def _ensure_repgeo_alpha(self):
        if not hasattr(self, "repgeo_alpha"):
            self.register_parameter(
                "repgeo_alpha",
                nn.Parameter(
                    torch.ones(
                        1,
                        device=self.base.weight.device,
                        dtype=self.base.weight.dtype,
                    )
                ),
            )
        return self.repgeo_alpha

    def lora_parameters(self):
        yield self._ensure_repgeo_alpha()
        yield from self.lora_down.parameters()
        yield from self.lora_up.parameters()

    def _freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False


class LoRAConv2d(nn.Module, _LoRAConvMixin):
    """Residual low-rank adapter for an existing Conv2d layer.

    The adapter is zero-initialized on the up projection, so attaching it keeps
    the original checkpoint output unchanged before fine-tuning.
    """

    def __init__(self, base, rank=4, alpha=8.0, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Conv2d):
            raise TypeError(f"LoRAConv2d expects nn.Conv2d, got {type(base)}")
        if rank < 1:
            raise ValueError("--lora_rank must be >= 1")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.repgeo_alpha = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Conv2d(
            base.in_channels,
            self.rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_up = nn.Conv2d(self.rank, base.out_channels, kernel_size=1, bias=False)
        self._reset_adapter_parameters()
        self._freeze_base()
        self._move_adapter_to_base()

    def _reset_adapter_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def _move_adapter_to_base(self):
        device = self.base.weight.device
        dtype = self.base.weight.dtype
        self.repgeo_alpha.data = self.repgeo_alpha.data.to(device=device, dtype=dtype)
        self.lora_down.to(device=device, dtype=dtype)
        self.lora_up.to(device=device, dtype=dtype)

    def forward(self, x):
        repgeo_alpha = self._ensure_repgeo_alpha()
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling * repgeo_alpha.view(1)


class LoRAConv3d(nn.Module, _LoRAConvMixin):
    """Residual low-rank adapter for an existing Conv3d layer."""

    def __init__(self, base, rank=4, alpha=8.0, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Conv3d):
            raise TypeError(f"LoRAConv3d expects nn.Conv3d, got {type(base)}")
        if rank < 1:
            raise ValueError("--lora_rank must be >= 1")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.repgeo_alpha = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Conv3d(
            base.in_channels,
            self.rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_up = nn.Conv3d(self.rank, base.out_channels, kernel_size=1, bias=False)
        self._reset_adapter_parameters()
        self._freeze_base()
        self._move_adapter_to_base()

    def _reset_adapter_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def _move_adapter_to_base(self):
        device = self.base.weight.device
        dtype = self.base.weight.dtype
        self.repgeo_alpha.data = self.repgeo_alpha.data.to(device=device, dtype=dtype)
        self.lora_down.to(device=device, dtype=dtype)
        self.lora_up.to(device=device, dtype=dtype)

    def forward(self, x):
        repgeo_alpha = self._ensure_repgeo_alpha()
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling * repgeo_alpha.view(1)


LORA_CONV_TYPES = (LoRAConv2d, LoRAConv3d)
NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
)


def _matches_prefix(module_name, prefixes):
    if not prefixes:
        return True
    return any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes)


def _wrap_conv(conv, rank, alpha, dropout):
    if isinstance(conv, nn.Conv2d):
        return LoRAConv2d(conv, rank=rank, alpha=alpha, dropout=dropout)
    if isinstance(conv, nn.Conv3d):
        return LoRAConv3d(conv, rank=rank, alpha=alpha, dropout=dropout)
    raise TypeError(f"Unsupported LoRA conv type: {type(conv)}")


def apply_lora_to_conv_modules(model, target_prefixes, rank=4, alpha=8.0, dropout=0.0, min_channels=8):
    """Replace targeted Conv2d/Conv3d modules with zero-initialized LoRA wrappers."""

    replaced = []

    def visit(module, prefix=""):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LORA_CONV_TYPES):
                if _matches_prefix(full_name, target_prefixes):
                    replaced.append(full_name)
                continue
            if isinstance(child, (nn.Conv2d, nn.Conv3d)) and _matches_prefix(full_name, target_prefixes):
                if min(child.in_channels, child.out_channels) >= min_channels:
                    setattr(module, child_name, _wrap_conv(child, rank=rank, alpha=alpha, dropout=dropout))
                    replaced.append(full_name)
                continue
            visit(child, full_name)

    visit(model)
    return replaced


def mark_only_lora_as_trainable(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LORA_CONV_TYPES):
            for param in module.lora_parameters():
                param.requires_grad = True


def set_norm_layers_eval(model):
    for module in model.modules():
        if isinstance(module, NORM_TYPES):
            module.eval()


def count_lora_parameters(model):
    return sum(param.numel() for module in model.modules() if isinstance(module, LORA_CONV_TYPES) for param in module.lora_parameters())


def repgeo_alpha_l1(model):
    terms = [module._ensure_repgeo_alpha().abs().sum() for module in model.modules() if isinstance(module, LORA_CONV_TYPES)]
    if not terms:
        return None
    return torch.stack(terms).sum()


def repgeo_alpha_stats(model):
    stats = []
    for name, module in model.named_modules():
        if isinstance(module, LORA_CONV_TYPES):
            stats.append(
                {
                    "name": name,
                    "alpha": float(module._ensure_repgeo_alpha().detach().float().cpu().item()),
                    "rank": int(module.rank),
                    "base_type": type(module.base).__name__,
                    "base_shape": list(module.base.weight.shape),
                    "base_groups": int(module.base.groups),
                    "lora_params": sum(param.numel() for param in module.lora_parameters()),
                }
            )
    return stats

