import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from core.utils import frame_utils


def _pad_to_min_size(image, min_h, min_w, value=0):
    h, w = image.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return image

    if image.ndim == 3:
        return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=value)


def _ensure_rgb_uint8(image):
    if image.ndim == 2:
        image = np.tile(image[..., None], (1, 1, 3))
    if image.shape[2] > 3:
        image = image[..., :3]
    return image.astype(np.uint8)


def _random_crop_stereo_pair(left, right, crop_size):
    crop_h, crop_w = crop_size
    left = _pad_to_min_size(left, crop_h, crop_w)
    right = _pad_to_min_size(right, crop_h, crop_w)

    h, w = left.shape[:2]
    y0 = random.randint(0, h - crop_h)
    x0 = random.randint(0, w - crop_w)
    y1 = y0 + crop_h
    x1 = x0 + crop_w
    return left[y0:y1, x0:x1], right[y0:y1, x0:x1]


class KittiStereoDataset(Dataset):
    """Minimal KITTI Stereo 2015 dataset for fine-tuning."""

    def __init__(
        self,
        root,
        split="train",
        crop_size=(256, 768),
        val_stride=5,
        max_samples=None,
    ):
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.val_stride = val_stride

        self.samples = self._build_samples()
        self.samples = self._select_split(self.samples)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise RuntimeError(f"No KITTI samples found under {self.root}")

    def _build_samples(self):
        left_dir = self.root / "training" / "image_2"
        right_dir = self.root / "training" / "image_3"
        disp_dir = self.root / "training" / "disp_occ_0"
        disp_noc_dir = self.root / "training" / "disp_noc_0"
        if not left_dir.is_dir() or not right_dir.is_dir() or not disp_dir.is_dir() or not disp_noc_dir.is_dir():
            raise RuntimeError(
                "KITTI root must contain training/image_2, training/image_3, training/disp_occ_0, and training/disp_noc_0"
            )

        samples = []
        for left_path in sorted(left_dir.glob("*.png")):
            right_path = right_dir / left_path.name
            disp_path = disp_dir / left_path.name
            disp_noc_path = disp_noc_dir / left_path.name
            if right_path.exists() and disp_path.exists() and disp_noc_path.exists():
                samples.append(
                    {
                        "id": left_path.stem,
                        "left": str(left_path),
                        "right": str(right_path),
                        "disp": str(disp_path),
                        "disp_noc": str(disp_noc_path),
                    }
                )
        return samples

    def _select_split(self, samples):
        if self.split == "train":
            return [sample for idx, sample in enumerate(samples) if idx % self.val_stride != 0]
        if self.split == "val":
            return [sample for idx, sample in enumerate(samples) if idx % self.val_stride == 0]
        if self.split == "trainval":
            return samples
        raise ValueError(f"Unsupported split: {self.split}")

    def _random_crop(self, left, right, disp, valid, valid_noc):
        crop_h, crop_w = self.crop_size
        left = _pad_to_min_size(left, crop_h, crop_w)
        right = _pad_to_min_size(right, crop_h, crop_w)
        disp = _pad_to_min_size(disp, crop_h, crop_w, value=0)
        valid = _pad_to_min_size(valid.astype(np.uint8), crop_h, crop_w, value=0).astype(bool)
        valid_noc = _pad_to_min_size(valid_noc.astype(np.uint8), crop_h, crop_w, value=0).astype(bool)

        h, w = left.shape[:2]
        y0 = random.randint(0, h - crop_h)
        x0 = random.randint(0, w - crop_w)
        y1 = y0 + crop_h
        x1 = x0 + crop_w

        left = left[y0:y1, x0:x1]
        right = right[y0:y1, x0:x1]
        disp = disp[y0:y1, x0:x1]
        valid = valid[y0:y1, x0:x1]
        valid_noc = valid_noc[y0:y1, x0:x1]
        return left, right, disp, valid, valid_noc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        left = frame_utils.read_gen(sample["left"]).astype(np.uint8)
        right = frame_utils.read_gen(sample["right"]).astype(np.uint8)
        disp, valid = frame_utils.readDispKITTI(sample["disp"])
        _, valid_noc = frame_utils.readDispKITTI(sample["disp_noc"])
        disp = disp.astype(np.float32)
        valid = valid.astype(bool)
        valid_noc = valid_noc.astype(bool)
        occ_mask = valid & (~valid_noc)

        if self.split in {"train", "trainval"}:
            left, right, disp, valid, valid_noc = self._random_crop(left, right, disp, valid, valid_noc)
            occ_mask = valid & (~valid_noc)

        left = torch.from_numpy(left).permute(2, 0, 1).float()
        right = torch.from_numpy(right).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp).float()
        valid = torch.from_numpy(valid)
        valid_noc = torch.from_numpy(valid_noc)
        occ_mask = torch.from_numpy(occ_mask)

        return {
            "left": left,
            "right": right,
            "disp": disp,
            "valid": valid,
            "valid_noc": valid_noc,
            "occ_mask": occ_mask,
            "sample_id": sample["id"],
        }


class UnlabeledStereoDataset(Dataset):
    """Recursive stereo-pair dataset for pseudo-label/self-distillation training."""

    def __init__(
        self,
        root,
        left_name="im0.png",
        right_name="im1.png",
        crop_size=(256, 768),
        recursive=True,
        max_samples=None,
    ):
        self.root = Path(root)
        self.left_name = left_name
        self.right_name = right_name
        self.crop_size = crop_size
        self.recursive = recursive

        self.samples = self._build_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise RuntimeError(
                f"No unlabeled stereo pairs found under {self.root} with names "
                f"{self.left_name} and {self.right_name}"
            )

    def _build_samples(self):
        if not self.root.is_dir():
            raise RuntimeError(f"Unlabeled stereo root does not exist: {self.root}")

        left_paths = self.root.rglob(self.left_name) if self.recursive else self.root.glob(f"*/{self.left_name}")
        samples = []
        for left_path in sorted(left_paths):
            right_path = left_path.with_name(self.right_name)
            if not right_path.exists():
                continue

            try:
                rel_dir = left_path.parent.relative_to(self.root)
                sample_id = str(rel_dir).replace("\\", "__").replace("/", "__")
                if not sample_id or sample_id == ".":
                    sample_id = left_path.stem
            except ValueError:
                sample_id = left_path.stem

            samples.append(
                {
                    "id": sample_id,
                    "left": str(left_path),
                    "right": str(right_path),
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        left = _ensure_rgb_uint8(frame_utils.read_gen(sample["left"]))
        right = _ensure_rgb_uint8(frame_utils.read_gen(sample["right"]))
        left, right = _random_crop_stereo_pair(left, right, self.crop_size)

        left = torch.from_numpy(left).permute(2, 0, 1).float()
        right = torch.from_numpy(right).permute(2, 0, 1).float()
        return {
            "left": left,
            "right": right,
            "sample_id": sample["id"],
        }

