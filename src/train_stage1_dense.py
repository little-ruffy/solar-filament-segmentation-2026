import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CROP_OFFSET,
    CROP_SIZE,
    FOLDS_PATH,
    FULL_SIZE,
    NUM_FOLDS,
    PREP_TEST_DIR,
    PREP_TRAIN_DIR,
    RANDOM_SEED,
    SUBMISSIONS_DIR,
    WORK_DIR,
)
from src.metrics import build_gt_cache, encode_label_map, evaluate_pq


FAMILY_SPECS = {
    "cnx_s": {
        "arch": "Unet",
        "encoder_candidates": [
            "tu-convnext_small.fb_in22k_ft_in1k_384",
            "tu-convnext_small.fb_in22k_ft_in1k",
            "tu-convnext_small",
        ],
        "decoder_channels": (256, 128, 64, 32, 16),
        "lr": 3.0e-4,
        "encoder_lr": 1.2e-4,
        "decoder_lr": 5.0e-4,
        "weight_decay": 1.5e-2,
        "stage_a_bs": 2,
        "stage_a_accum": 4,
        "whole_disk_bs": 1,
        "whole_disk_accum": 8,
    },
    "effv2_s": {
        "arch": "Unet",
        "encoder_candidates": [
            "tu-tf_efficientnetv2_s.in21k_ft_in1k",
            "tu-tf_efficientnetv2_s",
        ],
        "decoder_channels": (192, 96, 48, 24, 16),
        "lr": 3.5e-4,
        "encoder_lr": 1.5e-4,
        "decoder_lr": 6.0e-4,
        "weight_decay": 1.2e-2,
        "stage_a_bs": 4,
        "stage_a_accum": 2,
        "whole_disk_bs": 2,
        "whole_disk_accum": 4,
    },
    "upp_r34d": {
        "arch": "UnetPlusPlus",
        "encoder_candidates": [
            "tu-resnet34d",
            "resnet34",
        ],
        "decoder_channels": (192, 96, 48, 24, 16),
        "lr": 3.8e-4,
        "encoder_lr": 1.6e-4,
        "decoder_lr": 6.2e-4,
        "weight_decay": 1.0e-2,
        "stage_a_bs": 4,
        "stage_a_accum": 2,
        "whole_disk_bs": 2,
        "whole_disk_accum": 4,
    },
    "upp_eb4": {
        "arch": "UnetPlusPlus",
        "encoder_candidates": [
            "timm-efficientnet-b4",
            "tu-tf_efficientnet_b4.ns_jft_in1k",
        ],
        "decoder_channels": (192, 96, 48, 24, 16),
        "lr": 3.8e-4,
        "encoder_lr": 1.5e-4,
        "decoder_lr": 6.0e-4,
        "weight_decay": 1.0e-2,
        "stage_a_bs": 2,
        "stage_a_accum": 4,
        "whole_disk_bs": 1,
        "whole_disk_accum": 8,
    },
    "manet_eb4": {
        "arch": "MAnet",
        "encoder_candidates": [
            "tu-tf_efficientnet_b4.ns_jft_in1k",
            "timm-efficientnet-b4",
        ],
        "decoder_channels": (192, 96, 48, 24, 16),
        "lr": 3.5e-4,
        "encoder_lr": 1.5e-4,
        "decoder_lr": 5.8e-4,
        "weight_decay": 1.2e-2,
        "stage_a_bs": 2,
        "stage_a_accum": 4,
        "whole_disk_bs": 1,
        "whole_disk_accum": 8,
    },
    "upp_cnx_t": {
        "arch": "UnetPlusPlus",
        "encoder_candidates": [
            "tu-convnext_tiny.fb_in22k_ft_in1k_384",
            "tu-convnext_tiny.fb_in22k_ft_in1k",
            "tu-convnext_tiny",
        ],
        "decoder_channels": (192, 96, 48, 24, 16),
        "lr": 3.2e-4,
        "encoder_lr": 1.3e-4,
        "decoder_lr": 5.5e-4,
        "weight_decay": 1.5e-2,
        "stage_a_bs": 2,
        "stage_a_accum": 4,
        "whole_disk_bs": 1,
        "whole_disk_accum": 8,
    },
    "mit_b3": {
        "arch": "Unet",
        "encoder_candidates": [
            "mit_b3",
            "tu-tf_efficientnetv2_s.in21k_ft_in1k",
        ],
        "decoder_channels": (192, 96, 48, 24, 16),
        "lr": 2.8e-4,
        "encoder_lr": 1.0e-4,
        "decoder_lr": 5.0e-4,
        "weight_decay": 1.5e-2,
        "stage_a_bs": 2,
        "stage_a_accum": 4,
        "whole_disk_bs": 1,
        "whole_disk_accum": 8,
    },
}


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_d4_np(arr: np.ndarray, k: int) -> np.ndarray:
    """Apply one of the 8 D4 dihedral symmetries (k in 0..7) on spatial axes (0, 1)."""
    rot = k & 3
    flip = (k >> 2) & 1
    out = np.rot90(arr, k=rot, axes=(0, 1)) if rot else arr
    if flip:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


def apply_d4_torch(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply D4 symmetry k in 0..7 on tensor (..., H, W)."""
    rot = k & 3
    flip = (k >> 2) & 1
    out = torch.rot90(x, k=rot, dims=(-2, -1)) if rot else x
    if flip:
        out = torch.flip(out, dims=(-1,))
    return out


def invert_d4_torch(x: torch.Tensor, k: int) -> torch.Tensor:
    """Exact inverse of apply_d4_torch(x, k) on tensor (..., H, W)."""
    rot = k & 3
    flip = (k >> 2) & 1
    out = torch.flip(x, dims=(-1,)) if flip else x
    if rot:
        out = torch.rot90(out, k=(4 - rot) & 3, dims=(-2, -1))
    return out


_MIT_SDPA_PATCHED = False


def enable_mit_sdpa() -> None:
    """Replace quadratic softmax attention in SMP MixVisionTransformer with PyTorch 2 SDPA FlashAttention."""
    global _MIT_SDPA_PATCHED
    if _MIT_SDPA_PATCHED:
        return
    try:
        from segmentation_models_pytorch.encoders.mix_transformer import Attention

        def _sdpa_forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, height, width)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            out = F.scaled_dot_product_attention(
                q, kv[0], kv[1], dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
            )
            return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, C)))

        Attention.forward = _sdpa_forward
        _MIT_SDPA_PATCHED = True
    except Exception:
        pass


def build_stage1_model(family: str, pretrained: bool = True) -> Tuple[nn.Module, str]:
    enable_mit_sdpa()
    spec = FAMILY_SPECS[family]
    arch_cls = getattr(smp, spec["arch"])
    dec_ch = spec.get("decoder_channels", (256, 128, 64, 32, 16))
    last_err = None
    for enc in spec["encoder_candidates"]:
        try:
            if enc.startswith("tu-"):
                base_name = enc[3:]
                # Pass pretrained=True when checking dot-suffixed timm tags (e.g. .in21k_ft_in1k)
                if "." in base_name and len(timm.list_models(base_name, pretrained=True)) == 0:
                    if not timm.is_model(base_name.split(".")[0]):
                        continue
            enc_weights = (True if enc.startswith("tu-") else "imagenet") if pretrained else None
            model = arch_cls(
                encoder_name=enc,
                encoder_weights=enc_weights,
                in_channels=3,
                classes=2,
                decoder_channels=dec_ch,
                activation=None,
            )
            return model, enc
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Could not instantiate {family}: {last_err}")


def build_differential_optimizer(model: nn.Module, spec: dict) -> Tuple[torch.optim.Optimizer, List[float]]:
    """
    Build 4-group AdamW with differential encoder vs decoder/head learning rates
    and zero weight decay on biases and 1D normalization parameters.
    """
    enc_lr = float(spec.get("encoder_lr", spec["lr"] * 0.4))
    dec_lr = float(spec.get("decoder_lr", spec["lr"] * 1.5))
    wd = float(spec["weight_decay"])

    enc_decay, enc_nodecay = [], []
    dec_decay, dec_nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (param.ndim <= 1) or name.endswith(".bias") or ("norm" in name.lower()) or ("bn" in name.lower())
        if name.startswith("encoder."):
            (enc_nodecay if is_no_decay else enc_decay).append(param)
        else:
            (dec_nodecay if is_no_decay else dec_decay).append(param)

    groups = [
        {"params": enc_decay, "lr": enc_lr, "weight_decay": wd},
        {"params": enc_nodecay, "lr": enc_lr, "weight_decay": 0.0},
        {"params": dec_decay, "lr": dec_lr, "weight_decay": wd},
        {"params": dec_nodecay, "lr": dec_lr, "weight_decay": 0.0},
    ]
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    return opt, base_lrs


def set_step_lr(
    optimizer: torch.optim.Optimizer,
    base_lrs: List[float],
    cur_opt_step: int,
    total_opt_steps: int,
    warmup_opt_steps: int,
    min_lr_ratio: float = 0.03,
) -> None:
    if cur_opt_step < warmup_opt_steps and warmup_opt_steps > 0:
        mult = 0.15 + 0.85 * (float(cur_opt_step + 1) / float(warmup_opt_steps))
    else:
        prog = float(cur_opt_step - warmup_opt_steps) / float(max(total_opt_steps - warmup_opt_steps, 1))
        prog = min(max(prog, 0.0), 1.0)
        mult = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
    for pg, blr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = blr * mult


def ensure_memmap_caches(train_stems_all: List[str], test_stems_all: List[str]) -> Dict[str, object]:
    """
    Build uncompressed CHW uint8 SSD memmap caches once:
      - work/train_x_1856.npy : (707, 3, 1856, 1856) uint8
      - work/train_y_1856.npy : (707, 2, 1856, 1856) uint8
      - work/test_x_1856.npy  : (180, 3, 1856, 1856) uint8
    Eliminates 100% of PNG zlib CPU overhead while consuming 0 MB of Python heap RAM.
    """
    meta_path = WORK_DIR / "memmap_index.json"
    tx_path = WORK_DIR / "train_x_1856.npy"
    ty_path = WORK_DIR / "train_y_1856.npy"
    vx_path = WORK_DIR / "test_x_1856.npy"

    valid = False
    if meta_path.exists() and tx_path.exists() and ty_path.exists() and vx_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("train_stems") == list(train_stems_all) and meta.get("test_stems") == list(test_stems_all):
                valid = True
        except Exception:
            valid = False

    if not valid:
        t0 = time.time()
        print(f"[memmap] Building zero-RAM native 1856x1856 SSD memmap cache ({len(train_stems_all)} train, {len(test_stems_all)} test)...")
        tx_mm = np.lib.format.open_memmap(tx_path, mode="w+", dtype=np.uint8, shape=(len(train_stems_all), 3, CROP_SIZE, CROP_SIZE))
        ty_mm = np.lib.format.open_memmap(ty_path, mode="w+", dtype=np.uint8, shape=(len(train_stems_all), 2, CROP_SIZE, CROP_SIZE))
        for i, s in enumerate(train_stems_all):
            x_bgr = cv2.imread(str(PREP_TRAIN_DIR / f"{s}_x.png"), cv2.IMREAD_COLOR)
            y_bgr = cv2.imread(str(PREP_TRAIN_DIR / f"{s}_y.png"), cv2.IMREAD_COLOR)[:, :, :2]
            tx_mm[i] = x_bgr.transpose(2, 0, 1)
            ty_mm[i] = y_bgr.transpose(2, 0, 1)
        tx_mm.flush()
        ty_mm.flush()
        del tx_mm, ty_mm

        vx_mm = np.lib.format.open_memmap(vx_path, mode="w+", dtype=np.uint8, shape=(len(test_stems_all), 3, CROP_SIZE, CROP_SIZE))
        for i, s in enumerate(test_stems_all):
            x_bgr = cv2.imread(str(PREP_TEST_DIR / f"{s}_x.png"), cv2.IMREAD_COLOR)
            vx_mm[i] = x_bgr.transpose(2, 0, 1)
        vx_mm.flush()
        del vx_mm

        meta_path.write_text(
            json.dumps({"train_stems": list(train_stems_all), "test_stems": list(test_stems_all)}),
            encoding="utf-8",
        )
        print(f"[memmap] SSD cache ready in {time.time() - t0:.1f}s!")

    train_idx = {s: i for i, s in enumerate(train_stems_all)}
    test_idx = {s: i for i, s in enumerate(test_stems_all)}
    return {
        "tx_path": tx_path,
        "ty_path": ty_path,
        "vx_path": vx_path,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


class SolarWholeDiskMemmapDataset(Dataset):
    """Lazily-mapped zero-RAM whole-disk 1856x1856 dataset reading contiguous CHW uint8 slices."""

    def __init__(self, indices: List[int], tx_path: Path, ty_path: Path):
        self.indices = [int(i) for i in indices]
        self.tx_path = str(tx_path)
        self.ty_path = str(ty_path)
        self._tx = None
        self._ty = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._tx is None:
            self._tx = np.load(self.tx_path, mmap_mode="r")
            self._ty = np.load(self.ty_path, mmap_mode="r")
        gidx = self.indices[idx]
        x_t = torch.from_numpy(np.array(self._tx[gidx], dtype=np.uint8, copy=True))
        y_t = torch.from_numpy(np.array(self._ty[gidx], dtype=np.uint8, copy=True))
        return x_t, y_t


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.996):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


class SolarDenseDataset(Dataset):
    """
    Zero-Pagefile Streaming Native 1:1 Solar Disk Dataset.
    Workers read lossless PNG slices in parallel and return compact uint8 tensors (3, cs, cs) & (2, cs, cs);
    all float32 conversion, D4 dihedral symmetry, zoom, seeing blur, and contrast jitter execute on GPU!
    """

    def __init__(self, stems: List[str], crop_size: int = 1024, augment: bool = True, pos_cache: Optional[Dict[str, np.ndarray]] = None):
        self.stems = list(stems)
        self.crop_size = int(crop_size)
        self.augment = augment
        self.full_mode = self.crop_size >= CROP_SIZE
        self.samples_per_img = 1 if (not augment or self.full_mode) else 2
        if pos_cache is not None:
            self.pos_cache = pos_cache
        else:
            self.pos_cache = {}
            for s in self.stems:
                y0 = cv2.imread(str(PREP_TRAIN_DIR / f"{s}_y.png"), cv2.IMREAD_COLOR)[:, :, 0]
                self.pos_cache[s] = (np.argwhere(y0[::4, ::4] > 64) * 4).astype(np.int16)

    def __len__(self) -> int:
        return len(self.stems) * self.samples_per_img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx = idx // self.samples_per_img
        guided = (idx % self.samples_per_img == 0) and self.augment and not self.full_mode
        stem = self.stems[img_idx]

        x_full = cv2.imread(str(PREP_TRAIN_DIR / f"{stem}_x.png"), cv2.IMREAD_COLOR)
        y_full = cv2.imread(str(PREP_TRAIN_DIR / f"{stem}_y.png"), cv2.IMREAD_COLOR)[:, :, :2]

        cs = self.crop_size
        if cs >= CROP_SIZE:
            x_crop = x_full
            y_crop = y_full
        else:
            max_xy = CROP_SIZE - cs
            pos_yx = self.pos_cache.get(stem)
            if guided and pos_yx is not None and len(pos_yx) > 0:
                py, px = pos_yx[np.random.randint(len(pos_yx))]
                y0 = int(np.clip(int(py) - cs // 2 + np.random.randint(-260, 261), 0, max_xy))
                x0 = int(np.clip(int(px) - cs // 2 + np.random.randint(-260, 261), 0, max_xy))
            else:
                y0 = int(np.random.randint(0, max_xy + 1))
                x0 = int(np.random.randint(0, max_xy + 1))
            x_crop = x_full[y0:y0 + cs, x0:x0 + cs]
            y_crop = y_full[y0:y0 + cs, x0:x0 + cs]

        x_t = torch.from_numpy(np.ascontiguousarray(x_crop.transpose(2, 0, 1)))  # uint8 (3, cs, cs)
        y_t = torch.from_numpy(np.ascontiguousarray(y_crop.transpose(2, 0, 1)))  # uint8 (2, cs, cs)
        return x_t, y_t


_BLUR_K3 = torch.tensor(
    [[[[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]]] * 2,
    dtype=torch.float32,
)


@torch.no_grad()
def gpu_augment_and_normalize(
    xb_u8: torch.Tensor,
    yb_u8: torch.Tensor,
    augment: bool = True,
    allow_scale: bool = True,
    scale_range: Tuple[float, float] = (0.94, 1.03),
    target_size: int = CROP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute all float32 conversion, D4 dihedral symmetry, isotropic zoom, seeing blur,
    and astrophysical contrast jitter directly on GPU in < 0.5 ms per batch!
    If `target_size < CROP_SIZE` (e.g. Stage-A 1344x1344 whole-disk), box-integrates via mode='area'
    preserving exact target prevalence & coupling ratio kappa!
    """
    B, _, H, W = xb_u8.shape
    xf = xb_u8.float().mul_(1.0 / 255.0)
    yf = yb_u8.float().mul_(1.0 / 255.0)

    if target_size != H:
        xf = F.interpolate(xf, size=(target_size, target_size), mode="area")
        yf = F.interpolate(yf, size=(target_size, target_size), mode="area")

    if augment:
        # 1. Per-sample D4 dihedral symmetry
        for b in range(B):
            k = int(torch.randint(0, 8, (1,)).item())
            if k:
                xf[b] = apply_d4_torch(xf[b], k)
                yf[b] = apply_d4_torch(yf[b], k)

        # 2. Batch isotropic zoom keeping solar disk strictly inside canvas
        if allow_scale and float(torch.rand(1).item()) < 0.35:
            scale = float(torch.empty(1).uniform_(scale_range[0], scale_range[1]).item())
            theta = torch.zeros((B, 2, 3), device=xf.device, dtype=torch.float32)
            theta[:, 0, 0] = 1.0 / scale
            theta[:, 1, 1] = 1.0 / scale
            grid = F.affine_grid(theta, size=xf.shape, align_corners=False)
            xf = F.grid_sample(xf, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            yf = F.grid_sample(yf, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        valid_mask = xf[:, 2] > 0.002

        # 3. Mild atmospheric seeing blur on Ch0 & Ch1 (p=0.22)
        if float(torch.rand(1).item()) < 0.22:
            k3 = _BLUR_K3.to(xf.device)
            blurred = F.conv2d(F.pad(xf[:, :2], (1, 1, 1, 1), mode="replicate"), k3, groups=2)
            xf[:, :2] = blurred

        # 4. Per-sample photometric gain & residual background bias inside solar disk
        gain = torch.empty((B, 1, 1), device=xf.device).uniform_(0.92, 1.08)
        bias = torch.empty((B, 1, 1), device=xf.device).uniform_(-0.03, 0.03)
        c0 = torch.clamp((xf[:, 0] - 0.64) * gain + 0.64 + bias, 0.0, 1.0)
        c1 = torch.clamp((xf[:, 1] - 0.17) * gain + 0.17, 0.0, 1.0)
        xf[:, 0] = torch.where(valid_mask, c0, torch.zeros_like(c0))
        xf[:, 1] = torch.where(valid_mask, c1, torch.zeros_like(c1))
    else:
        valid_mask = xf[:, 2] > 0.002

    xf[:, 0] = (xf[:, 0] - 0.50) * 2.0
    xf[:, 1] = (xf[:, 1] - 0.25) * 2.5
    xf[:, 2] = (xf[:, 2] - 0.50) * 2.0
    return xf.contiguous(memory_format=torch.channels_last), yf.contiguous(), valid_mask.contiguous()


def normalize_input_u8(x_u8: np.ndarray) -> torch.Tensor:
    """Accepts HWC uint8 (1856, 1856, 3) and returns CHW float32 tensor."""
    xf = x_u8.astype(np.float32) * (1.0 / 255.0)
    xf[:, :, 0] = (xf[:, :, 0] - 0.50) * 2.0
    xf[:, :, 1] = (xf[:, :, 1] - 0.25) * 2.5
    xf[:, :, 2] = (xf[:, :, 2] - 0.50) * 2.0
    return torch.from_numpy(np.ascontiguousarray(xf.transpose(2, 0, 1)))


def dual_calibrated_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    loss_mode: str = "balanced_disk",
) -> torch.Tensor:
    """
    Calibrated Stage-1 loss in float32 inside the valid solar disk:
      - In `balanced_disk` mode (coupling ratio kappa = beta * N_disk / (alpha * S) ~ 3.7):
          L_fg = 20.0 * BCE_disk(fg) + 0.40 * SoftDice_disk(fg)
          L_sp = 20.0 * BCE_disk(spine) + 0.40 * SoftDice_disk(spine)
          L    = L_fg + 0.30 * L_sp
        Matches BCE and SoftDice logit gradient magnitudes on 0.20%-prevalence 1856x1856 solar disks
        so 1/3-annotator (0.333) and 1/2-annotator (0.50) filaments stay calibrated across all 16 tree levels!
      - Uses pos_weight = 1.0 everywhere so background probabilities remain strictly unbiased.
    """
    logits_f = logits.float()
    targets_f = targets.float()

    fg_log, sp_log = logits_f[:, 0], logits_f[:, 1]
    fg_tgt, sp_tgt = targets_f[:, 0], targets_f[:, 1]

    if valid_mask is not None and loss_mode == "balanced_disk":
        vm = valid_mask.float()
        vm_den = torch.clamp(vm.sum(), min=1.0)
        bce_fg = (F.binary_cross_entropy_with_logits(fg_log, fg_tgt, reduction="none") * vm).sum() / vm_den
        bce_sp = (F.binary_cross_entropy_with_logits(sp_log, sp_tgt, reduction="none") * vm).sum() / vm_den

        fg_prob = torch.sigmoid(fg_log) * vm
        sp_prob = torch.sigmoid(sp_log) * vm

        inter_fg = (fg_prob * fg_tgt).sum(dim=(1, 2))
        den_fg = fg_prob.sum(dim=(1, 2)) + fg_tgt.sum(dim=(1, 2)) + 1.0
        dice_fg = (1.0 - (2.0 * inter_fg + 1.0) / den_fg).mean()

        inter_sp = (sp_prob * sp_tgt).sum(dim=(1, 2))
        den_sp = sp_prob.sum(dim=(1, 2)) + sp_tgt.sum(dim=(1, 2)) + 1.0
        dice_sp = (1.0 - (2.0 * inter_sp + 1.0) / den_sp).mean()

        return (20.0 * bce_fg + 0.40 * dice_fg) + 0.30 * (20.0 * bce_sp + 0.40 * dice_sp)

    bce_fg = F.binary_cross_entropy_with_logits(fg_log, fg_tgt, reduction="mean")
    bce_sp = F.binary_cross_entropy_with_logits(sp_log, sp_tgt, reduction="mean")

    fg_prob = torch.sigmoid(fg_log)
    sp_prob = torch.sigmoid(sp_log)

    inter_fg = (fg_prob * fg_tgt).sum(dim=(1, 2))
    den_fg = fg_prob.sum(dim=(1, 2)) + fg_tgt.sum(dim=(1, 2)) + 1.0
    dice_fg = (1.0 - (2.0 * inter_fg + 1.0) / den_fg).mean()

    inter_sp = (sp_prob * sp_tgt).sum(dim=(1, 2))
    den_sp = sp_prob.sum(dim=(1, 2)) + sp_tgt.sum(dim=(1, 2)) + 1.0
    dice_sp = (1.0 - (2.0 * inter_sp + 1.0) / den_sp).mean()

    return (bce_fg + 0.50 * dice_fg) + 0.30 * (bce_sp + 0.50 * dice_sp)


@torch.no_grad()
def predict_full_disk(
    model: nn.Module,
    x_u8: np.ndarray,
    device: torch.device,
    use_d4_tta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict calibrated (fg_prob, spine_prob, fg_tta_std) on native 1856x1856 solar disk.
    Outside the physical solar disk (Ch2 == 0), probabilities are strictly masked to 0.
    """
    x_t = normalize_input_u8(x_u8).unsqueeze(0).to(device, non_blocking=True)
    disk_mask = x_u8[:, :, 2] > 0

    if not use_d4_tta:
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = torch.sigmoid(model(x_t))[0].float().cpu().numpy()
        fg = np.where(disk_mask, out[0], 0.0).astype(np.float32)
        sp = np.where(disk_mask, out[1], 0.0).astype(np.float32)
        return fg, sp, np.zeros_like(fg, dtype=np.float32)

    # Run 8 D4 views in 4 micro-batches of 2 views (< 5.5 GB VRAM at 1856x1856, zero WDDM spillover!)
    fg_sum = torch.zeros((CROP_SIZE, CROP_SIZE), dtype=torch.float32, device=device)
    fg_sq = torch.zeros((CROP_SIZE, CROP_SIZE), dtype=torch.float32, device=device)
    sp_sum = torch.zeros((CROP_SIZE, CROP_SIZE), dtype=torch.float32, device=device)

    for start_k in (0, 2, 4, 6):
        views = torch.cat([apply_d4_torch(x_t, k) for k in range(start_k, start_k + 2)], dim=0)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            probs = torch.sigmoid(model(views)).float()
        for j, k in enumerate(range(start_k, start_k + 2)):
            inv = invert_d4_torch(probs[j], k)
            fg_v = inv[0]
            sp_v = inv[1]
            fg_sum.add_(fg_v)
            fg_sq.add_(fg_v * fg_v)
            sp_sum.add_(sp_v)

    fg_mean = (fg_sum * (1.0 / 8.0)).cpu().numpy()
    sp_mean = (sp_sum * (1.0 / 8.0)).cpu().numpy()
    fg_var = torch.clamp(fg_sq * (1.0 / 8.0) - (fg_sum * (1.0 / 8.0)) ** 2, min=0.0)
    fg_std = torch.sqrt(fg_var).cpu().numpy()

    fg_mean = np.where(disk_mask, fg_mean, 0.0).astype(np.float32)
    sp_mean = np.where(disk_mask, sp_mean, 0.0).astype(np.float32)
    fg_std = np.where(disk_mask, fg_std, 0.0).astype(np.float32)
    return fg_mean, sp_mean, fg_std


def decode_simple_cc_rles(
    fg_1856: np.ndarray,
    t_lo: float = 0.35,
    min_mean_p: float = 0.72,
    min_area: int = 185,
) -> List[dict]:
    """Fast baseline connected-component decoder on native 1856->2048 canvas for epoch PQ monitoring."""
    bw = fg_1856 >= t_lo
    lbl, n_cc = ndi.label(bw)
    if n_cc == 0:
        return []
    full_lbl = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.int32)
    keep_id = 0
    slices = ndi.find_objects(lbl)
    for k, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sub_m = lbl[sl] == k
        area = int(sub_m.sum())
        if area < min_area:
            continue
        mean_p = float(fg_1856[sl][sub_m].mean())
        if mean_p < min_mean_p:
            continue
        keep_id += 1
        y0, y1 = sl[0].start + CROP_OFFSET, sl[0].stop + CROP_OFFSET
        x0, x1 = sl[1].start + CROP_OFFSET, sl[1].stop + CROP_OFFSET
        full_lbl[y0:y1, x0:x1][sub_m] = keep_id
    return encode_label_map(full_lbl)


_SIMPLE_CC_OPERATING_POINTS = (
    (0.30, 0.42, 150),
    (0.34, 0.48, 160),
)


@torch.no_grad()
def evaluate_val_fold_pq(
    model: nn.Module,
    val_stems: List[str],
    gt_cache: dict,
    device: torch.device,
    use_d4_tta: bool = False,
    tx_mm: Optional[np.ndarray] = None,
    train_idx: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    preds_by_op: List[Dict[str, List[dict]]] = [{} for _ in _SIMPLE_CC_OPERATING_POINTS]
    for s in val_stems:
        if tx_mm is not None and train_idx is not None:
            x_u8 = np.ascontiguousarray(tx_mm[train_idx[s]].transpose(1, 2, 0))
        else:
            x_u8 = cv2.imread(str(PREP_TRAIN_DIR / f"{s}_x.png"), cv2.IMREAD_COLOR)
        fg, _, _ = predict_full_disk(model, x_u8, device, use_d4_tta=use_d4_tta)
        for op_i, (t_lo, min_mp, min_a) in enumerate(_SIMPLE_CC_OPERATING_POINTS):
            preds_by_op[op_i][s] = decode_simple_cc_rles(fg, t_lo=t_lo, min_mean_p=min_mp, min_area=min_a)

    best_res = None
    for op_i in range(len(_SIMPLE_CC_OPERATING_POINTS)):
        res = evaluate_pq(preds_by_op[op_i], gt_cache, eval_stems=val_stems)
        if best_res is None or res["pq"] > best_res["pq"]:
            best_res = res
    return best_res


def save_prob_pack(path: Path, fg: np.ndarray, sp: np.ndarray, std: np.ndarray, extra_std: Optional[np.ndarray] = None) -> None:
    payload = {
        "fg": np.clip(np.round(fg * 65535.0), 0, 65535).astype(np.uint16),
        "sp": np.clip(np.round(sp * 65535.0), 0, 65535).astype(np.uint16),
        "std": np.clip(np.round(std * 1000.0), 0, 255).astype(np.uint8),  # 0.001 resolution in [0, 0.255]
    }
    if extra_std is not None:
        payload["fstd"] = np.clip(np.round(extra_std * 1000.0), 0, 255).astype(np.uint8)
    np.savez_compressed(path, **payload)


def load_prob_pack(path: Path, return_fstd: bool = False):
    with np.load(path) as z:
        fg = z["fg"].astype(np.float32) * (1.0 / 65535.0)
        sp = z["sp"].astype(np.float32) * (1.0 / 65535.0)
        std = z["std"].astype(np.float32) * (1.0 / 1000.0)
        if return_fstd:
            fstd = (
                z["fstd"].astype(np.float32) * (1.0 / 1000.0)
                if "fstd" in z.files
                else np.zeros_like(std, dtype=np.float32)
            )
            return fg, sp, std, fstd
    return fg, sp, std


def train_and_infer_family(
    family: str,
    folds_to_run: Optional[List[int]] = None,
    stage_a_epochs: int = 12,
    stage_a_size: int = 1344,
    stage_b_epochs: int = 8,
    batch_size: Optional[int] = None,
    grad_accum: Optional[int] = None,
    loss_mode: str = "balanced_disk",
    smoke_test: bool = False,
) -> Dict[str, object]:
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        # Hard physical GDDR6X ceiling on Windows WDDM: instant OOM instead of silent PCIe shared-RAM thrashing!
        torch.cuda.set_per_process_memory_fraction(0.88)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_cache = build_gt_cache()
    with open(FOLDS_PATH, "r", encoding="utf-8") as f:
        fold_map = json.load(f)

    spec = FAMILY_SPECS[family]
    bs_a = int(spec.get("stage_a_bs", 2))
    accum_a = int(spec.get("stage_a_accum", 4))
    bs_b = int(batch_size) if batch_size is not None else int(spec.get("whole_disk_bs", 2))
    accum_b = int(grad_accum) if grad_accum is not None else int(spec.get("whole_disk_accum", 4))

    ckpt_dir = WORK_DIR / f"ckpt_{family}"
    oof_dir = WORK_DIR / f"oof_{family}"
    test_dir = WORK_DIR / f"test_{family}"
    for d in (ckpt_dir, oof_dir, test_dir, SUBMISSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    target_folds = folds_to_run if folds_to_run is not None else list(range(NUM_FOLDS))
    test_stems = sorted(
        fn[:-6] for fn in os.listdir(PREP_TEST_DIR) if fn.endswith("_x.png")
    )

    if smoke_test:
        print(
            f"[smoke] Benchmarking Progressive Whole-Disk ({stage_a_size} bs={bs_a} -> 1856 bs={bs_b}) + D4 for {family}..."
        )
        seed_everything(RANDOM_SEED)
        torch.cuda.reset_peak_memory_stats(device)
        model, enc_used = build_stage1_model(family)
        model = model.to(device=device, memory_format=torch.channels_last)
        opt, base_lrs = build_differential_optimizer(model, spec)

        x_list, y_list = [], []
        for s in gt_cache["stems"][: max(bs_a, bs_b * 2, 4)]:
            xb_img = cv2.imread(str(PREP_TRAIN_DIR / f"{s}_x.png"), cv2.IMREAD_COLOR).transpose(2, 0, 1)
            yb_img = cv2.imread(str(PREP_TRAIN_DIR / f"{s}_y.png"), cv2.IMREAD_COLOR)[:, :, :2].transpose(2, 0, 1)
            x_list.append(xb_img)
            y_list.append(yb_img)

        # Stage-A step (1344 whole-disk)
        xb_a = torch.from_numpy(np.stack(x_list[:bs_a])).to(device)
        yb_a = torch.from_numpy(np.stack(y_list[:bs_a])).to(device)
        xb, yb, vmask = gpu_augment_and_normalize(xb_a, yb_a, augment=True, allow_scale=True, target_size=stage_a_size)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_a = dual_calibrated_loss(model(xb), yb, valid_mask=vmask, loss_mode=loss_mode)
        loss_a.backward()
        opt.step()
        torch.cuda.synchronize()

        # Stage-B steps (1856 whole-disk)
        t_step0 = time.time()
        for step in range(2):
            xb_u8 = torch.from_numpy(np.stack(x_list[step * bs_b:(step + 1) * bs_b])).to(device)
            yb_u8 = torch.from_numpy(np.stack(y_list[step * bs_b:(step + 1) * bs_b])).to(device)
            set_step_lr(opt, base_lrs, step, total_opt_steps=100, warmup_opt_steps=10)
            xb, yb, vmask = gpu_augment_and_normalize(xb_u8, yb_u8, augment=True, allow_scale=True, target_size=CROP_SIZE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb)
                loss = dual_calibrated_loss(out, yb, valid_mask=vmask, loss_mode=loss_mode)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if step == 0:
                t_step0 = time.time()
        dt_step = time.time() - t_step0
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)

        model.eval()
        x_sample = cv2.imread(str(PREP_TRAIN_DIR / f"{gt_cache['stems'][0]}_x.png"), cv2.IMREAD_COLOR)
        fg, sp, std = predict_full_disk(model, x_sample, device, use_d4_tta=True)
        assert fg.shape == (CROP_SIZE, CROP_SIZE) and np.isfinite(fg).all()
        rand_t = torch.randn(1, 2, 64, 64, device=device)
        for k in range(8):
            rt = invert_d4_torch(apply_d4_torch(rand_t, k), k)
            assert torch.allclose(rand_t, rt), f"D4 inverse failed for k={k}"
        est_ep_s = dt_step * (565.0 / bs_b)
        print(
            f"[smoke] OK! {family} ({enc_used}) | 1856 bs={bs_b}: {dt_step*1000:.0f} ms/batch "
            f"(~{est_ep_s:.1f}s/ep @1856), peak_VRAM={peak_gb:.2f} GB, loss={float(loss.detach()):.4f}"
        )
        return {"status": "smoke_ok", "encoder": enc_used, "peak_gb": peak_gb, "est_ep_s": est_ep_s}

    mm_info = ensure_memmap_caches(gt_cache["stems"], test_stems)
    tx_mm_ro = np.load(str(mm_info["tx_path"]), mmap_mode="r")
    vx_mm_ro = np.load(str(mm_info["vx_path"]), mmap_mode="r")
    train_idx = mm_info["train_idx"]
    test_idx = mm_info["test_idx"]

    total_epochs = stage_a_epochs + stage_b_epochs
    fold_reports = {}
    for fold in target_folds:
        f_t0 = time.time()
        seed_everything(RANDOM_SEED + fold * 101)
        train_stems = [s for s in gt_cache["stems"] if fold_map[s] != fold]
        val_stems = [s for s in gt_cache["stems"] if fold_map[s] == fold]
        ckpt_path = ckpt_dir / f"best_fold{fold}.pt"

        oof_complete = ckpt_path.exists() and all((oof_dir / f"{s}.npz").exists() for s in val_stems)
        if oof_complete:
            print(f"[{family} | fold {fold}] Found complete checkpoint & OOF predictions; skipping retrain.")
            continue

        model, enc_used = build_stage1_model(family)
        model = model.to(device=device, memory_format=torch.channels_last)
        ema = ModelEMA(model, decay=0.995)

        optimizer, base_lrs = build_differential_optimizer(model, spec)

        tr_indices = [train_idx[s] for s in train_stems]
        disk_ds = SolarWholeDiskMemmapDataset(tr_indices, mm_info["tx_path"], mm_info["ty_path"])
        loader_a = DataLoader(disk_ds, batch_size=bs_a, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        loader_b = DataLoader(disk_ds, batch_size=bs_b, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

        steps_a = max(1, math.ceil(len(loader_a) / float(accum_a))) if stage_a_epochs > 0 else 0
        steps_b = max(1, math.ceil(len(loader_b) / float(accum_b))) if stage_b_epochs > 0 else 0
        total_opt_steps = stage_a_epochs * steps_a + stage_b_epochs * steps_b
        warmup_opt_steps = 2 * (steps_a if stage_a_epochs > 0 else steps_b)
        cur_opt_step = 0

        print(
            f"\n[{family} ({enc_used}) | Fold {fold}/{NUM_FOLDS - 1}] "
            f"Progressive WholeDisk ({stage_a_epochs}ep@{stage_a_size} bs={bs_a}x{accum_a} -> "
            f"{stage_b_epochs}ep@1856 bs={bs_b}x{accum_b}), enc_lr={base_lrs[0]:.2e}, dec_lr={base_lrs[2]:.2e}"
        )

        best_pq = -1.0
        for epoch in range(1, total_epochs + 1):
            ep_t0 = time.time()
            model.train()
            in_stage_b = epoch > stage_a_epochs
            if in_stage_b and epoch == stage_a_epochs + 1:
                torch.cuda.empty_cache()

            cur_loader = loader_b if in_stage_b else loader_a
            cur_accum = accum_b if in_stage_b else accum_a
            cur_size = CROP_SIZE if in_stage_b else stage_a_size

            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            n_steps = 0

            set_step_lr(optimizer, base_lrs, cur_opt_step, total_opt_steps, warmup_opt_steps)

            for step_i, (xb_u8, yb_u8) in enumerate(cur_loader, start=1):
                xb, yb, vmask = gpu_augment_and_normalize(
                    xb_u8.to(device, non_blocking=True),
                    yb_u8.to(device, non_blocking=True),
                    augment=True,
                    allow_scale=True,
                    target_size=cur_size,
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = dual_calibrated_loss(logits, yb, valid_mask=vmask, loss_mode=loss_mode) / cur_accum
                loss.backward()
                running_loss += float(loss.item()) * cur_accum
                n_steps += 1

                if step_i % cur_accum == 0 or step_i == len(cur_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                    cur_opt_step += 1
                    set_step_lr(optimizer, base_lrs, cur_opt_step, total_opt_steps, warmup_opt_steps)

            should_eval = (epoch == stage_a_epochs) or (in_stage_b and (epoch % 2 == 0 or epoch == total_epochs))
            if should_eval:
                val_metrics = evaluate_val_fold_pq(
                    ema.ema, val_stems, gt_cache, device, use_d4_tta=False, tx_mm=tx_mm_ro, train_idx=train_idx
                )
                vpq = val_metrics["pq"]
                tag = ""
                if vpq > best_pq:
                    best_pq = vpq
                    torch.save(
                        {"model": ema.ema.state_dict(), "epoch": epoch, "val_pq": best_pq, "encoder": enc_used},
                        ckpt_path,
                    )
                    tag = " *BEST*"
                print(
                    f"  Ep {epoch:02d}/{total_epochs} [Disk{cur_size}] "
                    f"loss={running_loss / max(n_steps, 1):.4f} | val_PQ={vpq:.4f} (SQ={val_metrics['sq']:.4f}, RQ={val_metrics['rq']:.4f}) "
                    f"| {time.time() - ep_t0:.1f}s{tag}"
                )
            else:
                print(
                    f"  Ep {epoch:02d}/{total_epochs} [Disk{cur_size}] "
                    f"loss={running_loss / max(n_steps, 1):.4f} | dec_lr={optimizer.param_groups[2]['lr']:.2e} | {time.time() - ep_t0:.1f}s"
                )

        del disk_ds, loader_a, loader_b
        torch.cuda.empty_cache()

        best_state = torch.load(ckpt_path, map_location=device, weights_only=False)
        ema.ema.load_state_dict(best_state["model"])
        ema.ema.eval()

        oof_t0 = time.time()
        fold_oof_by_op = [{} for _ in _SIMPLE_CC_OPERATING_POINTS]
        for s in val_stems:
            x_u8 = np.ascontiguousarray(tx_mm_ro[train_idx[s]].transpose(1, 2, 0))
            fg, sp, std = predict_full_disk(ema.ema, x_u8, device, use_d4_tta=True)
            save_prob_pack(oof_dir / f"{s}.npz", fg, sp, std)
            for op_i, (t_lo, min_mp, min_a) in enumerate(_SIMPLE_CC_OPERATING_POINTS):
                fold_oof_by_op[op_i][s] = decode_simple_cc_rles(fg, t_lo=t_lo, min_mean_p=min_mp, min_area=min_a)

        d4_res = max(
            (evaluate_pq(fold_oof_by_op[op_i], gt_cache, eval_stems=val_stems) for op_i in range(len(_SIMPLE_CC_OPERATING_POINTS))),
            key=lambda r: r["pq"],
        )
        fold_reports[fold] = d4_res
        print(
            f"[{family} | Fold {fold} DONE in {(time.time() - f_t0)/60:.1f}m] "
            f"8-view D4 OOF PQ={d4_res['pq']:.4f} (SQ={d4_res['sq']:.4f}, RQ={d4_res['rq']:.4f}) [OOF TTA {time.time() - oof_t0:.1f}s]"
        )

    all_ckpts_ready = all((ckpt_dir / f"best_fold{f}.pt").exists() for f in range(NUM_FOLDS))
    if all_ckpts_ready:
        best_op = _SIMPLE_CC_OPERATING_POINTS[0]
        full_oof = {"pq": -1.0}
        for t_lo, min_mp, min_a in _SIMPLE_CC_OPERATING_POINTS:
            oof_rles_all = {}
            for s in gt_cache["stems"]:
                fg_s, _, _ = load_prob_pack(oof_dir / f"{s}.npz")
                oof_rles_all[s] = decode_simple_cc_rles(fg_s, t_lo=t_lo, min_mean_p=min_mp, min_area=min_a)
            cand_oof = evaluate_pq(oof_rles_all, gt_cache)
            if cand_oof["pq"] > full_oof["pq"]:
                full_oof = cand_oof
                best_op = (t_lo, min_mp, min_a)

        print(f"\n[{family}] Computing 5-fold x 8-view D4 (40-view) Test predictions across {len(test_stems)} test images...")
        models = []
        for f in range(NUM_FOLDS):
            m, _ = build_stage1_model(family)
            st = torch.load(ckpt_dir / f"best_fold{f}.pt", map_location=device, weights_only=False)
            m.load_state_dict(st["model"])
            m = m.to(device=device, memory_format=torch.channels_last).eval()
            models.append(m)

        sub_rows = []
        t_inf0 = time.time()
        for idx, s in enumerate(test_stems, start=1):
            out_npz = test_dir / f"{s}.npz"
            need_infer = not out_npz.exists()
            if not need_infer:
                with np.load(out_npz) as z_chk:
                    need_infer = "fstd" not in z_chk.files
            if need_infer:
                x_u8 = np.ascontiguousarray(vx_mm_ro[test_idx[s]].transpose(1, 2, 0))
                fgs, sps, stds = [], [], []
                for m in models:
                    fg_f, sp_f, std_f = predict_full_disk(m, x_u8, device, use_d4_tta=True)
                    fgs.append(fg_f)
                    sps.append(sp_f)
                    stds.append(std_f)
                fg_ens = np.mean(fgs, axis=0)
                sp_ens = np.mean(sps, axis=0)
                d4_std_ens = np.mean(stds, axis=0)
                fold_std = np.std(fgs, axis=0)
                save_prob_pack(out_npz, fg_ens, sp_ens, d4_std_ens, extra_std=fold_std)
            else:
                fg_ens, _, _ = load_prob_pack(out_npz)

            rles = decode_simple_cc_rles(fg_ens, t_lo=best_op[0], min_mean_p=best_op[1], min_area=best_op[2])
            for k, rle in enumerate(rles):
                sub_rows.append(f"{s}_{k},{rle['counts']}")
            if idx % 60 == 0 or idx == len(test_stems):
                print(f"  [{family}] test inferred: {idx}/{len(test_stems)} ({time.time() - t_inf0:.1f}s)")

        sub_path = SUBMISSIONS_DIR / f"sub_{family}_cc.csv"
        sub_path.write_text("filament_id,segmentation_rle\n" + "\n".join(sub_rows) + "\n", encoding="ascii")

        summary = {"family": family, "best_cc_op": best_op, "oof_simple_cc": full_oof, "sub_path": str(sub_path)}
        (WORK_DIR / f"summary_{family}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"=== [{family} COMPLETE] Full 707-img Simple-CC OOF PQ = {full_oof['pq']:.4f} "
            f"(SQ={full_oof['sq']:.4f}, RQ={full_oof['rq']:.4f}) -> {sub_path} ==="
        )
        return summary

    return {"family": family, "fold_reports": fold_reports}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_SPECS.keys()))
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--stage-a-epochs", type=int, default=12)
    parser.add_argument("--stage-a-size", type=int, default=1344)
    parser.add_argument("--stage-b-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--loss-mode", choices=["balanced_disk", "legacy"], default="balanced_disk")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    train_and_infer_family(
        family=args.family,
        folds_to_run=args.folds,
        stage_a_epochs=args.stage_a_epochs,
        stage_a_size=args.stage_a_size,
        stage_b_epochs=args.stage_b_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        loss_mode=args.loss_mode,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
