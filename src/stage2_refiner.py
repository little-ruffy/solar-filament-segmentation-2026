import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*eval_set.*")

import cv2
import lightgbm as lgb
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_util
from scipy import ndimage as ndi
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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
from src.hierarchy import encode_submask_rle_2048
from src.hurdle_scorer import load_blended_maps
from src.metrics import build_gt_cache, encode_label_map, evaluate_pq

S2_W = 384
S2_H = 128

STAGE2_SPECS = {
    "s2_upp_r34": {
        "arch": "UnetPlusPlus",
        "encoder": "resnet34",
        "lr": 4.0e-4,
    },
    "s2_unet_cnx": {
        "arch": "Unet",
        "encoder": "tu-convnext_tiny",
        "lr": 3.2e-4,
    },
    "s2_upp_effv2s": {
        "arch": "UnetPlusPlus",
        "encoder": "tu-tf_efficientnetv2_s.in21k_ft_in1k",
        "lr": 3.5e-4,
    },
    "s2_manet_r34d": {
        "arch": "MAnet",
        "encoder": "tu-resnet34d",
        "lr": 3.5e-4,
    },
}


def compute_oriented_crop_affine(mask_1856: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Given a binary instance mask on [1856, 1856], compute:
      1. Forward 2x3 affine matrix M_fwd: disk (x, y) -> canonical crop (u, v) in [0..384) x [0..128)
      2. Inverse 2x3 affine matrix M_inv_local: canonical crop (u, v) -> local ROI [0..rw) x [0..rh)
      3. Tight padded ROI bounds (rx0, ry0, rx1, ry1) on [1856, 1856]
    """
    ys, xs = np.nonzero(mask_1856)
    if len(xs) < 4:
        cx = float(xs.mean()) if len(xs) else CROP_SIZE * 0.5
        cy = float(ys.mean()) if len(ys) else CROP_SIZE * 0.5
        theta = 0.0
        span_u, span_v = 120.0, 60.0
    else:
        cx0, cy0 = float(xs.mean()), float(ys.mean())
        dx = xs.astype(np.float32) - cx0
        dy = ys.astype(np.float32) - cy0
        cxx = float(np.mean(dx * dx)) + 0.1
        cyy = float(np.mean(dy * dy)) + 0.1
        cxy = float(np.mean(dx * dy))
        theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        u_proj = dx * cos_t + dy * sin_t
        v_proj = -dx * sin_t + dy * cos_t

        u_min, u_max = float(u_proj.min()), float(u_proj.max())
        v_min, v_max = float(v_proj.min()), float(v_proj.max())
        u_mid = 0.5 * (u_min + u_max)
        v_mid = 0.5 * (v_min + v_max)
        cx = cx0 + u_mid * cos_t - v_mid * sin_t
        cy = cy0 + u_mid * sin_t + v_mid * cos_t

        span_u = max(1.38 * (u_max - u_min) + 28.0, 96.0)
        span_v = max(1.75 * (v_max - v_min) + 26.0, 48.0)

    su = S2_W / span_u
    sv = S2_H / span_v
    # Constrain anisotropy ratio within [0.5, 2.0] by expanding the tighter dimension
    if su > 2.0 * sv:
        su = 2.0 * sv
    elif sv > 2.0 * su:
        sv = 2.0 * su

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Forward: u = su * ((x - cx)*cos_t + (y - cy)*sin_t) + (S2_W - 1)*0.5
    #          v = sv * (-(x - cx)*sin_t + (y - cy)*cos_t) + (S2_H - 1)*0.5
    u_off = 0.5 * (S2_W - 1.0) - su * (cx * cos_t + cy * sin_t)
    v_off = 0.5 * (S2_H - 1.0) - sv * (-cx * sin_t + cy * cos_t)
    M_fwd = np.array(
        [
            [su * cos_t, su * sin_t, u_off],
            [-sv * sin_t, sv * cos_t, v_off],
        ],
        dtype=np.float32,
    )

    # Map the 4 corners of [0, S2_W-1] x [0, S2_H-1] back to disk space to get exact ROI bounds
    M3 = np.eye(3, dtype=np.float64)
    M3[:2, :] = M_fwd.astype(np.float64)
    M3_inv = np.linalg.inv(M3)

    corners_uv = np.array(
        [[0, 0, 1], [S2_W - 1, 0, 1], [0, S2_H - 1, 1], [S2_W - 1, S2_H - 1, 1]],
        dtype=np.float64,
    ).T
    corners_xy = M3_inv @ corners_uv
    rx0 = max(0, int(math.floor(corners_xy[0].min())) - 2)
    rx1 = min(CROP_SIZE, int(math.ceil(corners_xy[0].max())) + 3)
    ry0 = max(0, int(math.floor(corners_xy[1].min())) - 2)
    ry1 = min(CROP_SIZE, int(math.ceil(corners_xy[1].max())) + 3)
    if rx1 <= rx0:
        rx0, rx1 = 0, CROP_SIZE
    if ry1 <= ry0:
        ry0, ry1 = 0, CROP_SIZE

    # cv2.warpAffine with WARP_INVERSE_MAP=False maps src(u,v) -> dst(x_local, y_local) via M_inv_local!
    M_inv_local = M3_inv[:2, :].copy()
    M_inv_local[0, 2] -= float(rx0)
    M_inv_local[1, 2] -= float(ry0)
    return M_fwd, M_inv_local.astype(np.float32), (rx0, ry0, rx1, ry1)


def extract_5ch_crop(
    ff_norm: np.ndarray,
    fg_blend: np.ndarray,
    sp_blend: np.ndarray,
    seed_mask: np.ndarray,
    other_seeds_mask: np.ndarray,
    M_fwd: np.ndarray,
) -> np.ndarray:
    """
    Warp 5 channels into [5, 128, 384] float32 canonical oriented crop.
    Note: OpenCV `cv::Scalar` caps `borderValue` tuples at 4 channels (`len <= 4`), so we center
    `ff_norm` into `[-1, 1]` upfront (making border value 0.0 everywhere) and warp in 3ch + 2ch blocks.
    """
    src_a = np.stack(
        [
            (ff_norm - 0.5) * 2.0,
            fg_blend,
            sp_blend,
        ],
        axis=-1,
    ).astype(np.float32)
    src_b = np.stack(
        [
            seed_mask.astype(np.float32),
            other_seeds_mask.astype(np.float32),
        ],
        axis=-1,
    )
    wa = cv2.warpAffine(
        src_a,
        M_fwd,
        (S2_W, S2_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    wb = cv2.warpAffine(
        src_b,
        M_fwd,
        (S2_W, S2_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    out = np.concatenate([wa, wb], axis=-1).transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(out)


def join3_bridge_fragments(
    bin_roi: np.ndarray,
    fg_roi: np.ndarray,
    seed_roi: np.ndarray,
    max_gap_px: float = 15.0,
    min_bridge_fg: float = 0.11,
) -> np.ndarray:
    """
    Minimum-Spanning-Tree 3-px corridor bridging (`join3`):
    1. Retain connected components overlapping the selected Stage-1 seed OR covered >= 45% by high prob.
    2. If multiple kept sub-fragments lie within `max_gap_px` and the straight line between their
       closest boundary pixels stays inside elevated chromospheric support (`mean fg >= min_bridge_fg`),
       draw a 3-px corridor joining them so thin broken filaments never split into 2 FP pieces!
    """
    lbl, n_cc = ndi.label(bin_roi)
    if n_cc <= 1:
        return bin_roi.astype(np.uint8)

    keep_ids = []
    slices = ndi.find_objects(lbl)
    for cid, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        cm = lbl[sl] == cid
        area = int(cm.sum())
        if area < 12:
            continue
        seed_ov = int(seed_roi[sl][cm].sum())
        if seed_ov >= 4 or (area >= 35 and float(fg_roi[sl][cm].mean()) >= 0.45):
            keep_ids.append(cid)

    if len(keep_ids) == 0:
        # Keep largest component
        areas = [int((lbl == cid).sum()) for cid in range(1, n_cc + 1)]
        best_cid = int(np.argmax(areas)) + 1
        return (lbl == best_cid).astype(np.uint8)

    out = np.isin(lbl, keep_ids).astype(np.uint8)
    if len(keep_ids) == 1:
        return out

    # Extract subsampled boundary points per kept component for fast closest-pair MST
    comp_pts = []
    for cid in keep_ids:
        sl = slices[cid - 1]
        sub = (lbl[sl] == cid).astype(np.uint8)
        ero = cv2.erode(sub, np.ones((3, 3), dtype=np.uint8))
        b_yx = np.argwhere((sub > 0) & (ero == 0))
        if len(b_yx) == 0:
            b_yx = np.argwhere(sub > 0)
        b_yx[:, 0] += sl[0].start
        b_yx[:, 1] += sl[1].start
        if len(b_yx) > 80:
            b_yx = b_yx[:: max(1, len(b_yx) // 80)]
        comp_pts.append(b_yx)

    K = len(keep_ids)
    edges = []
    for i in range(K):
        pi = comp_pts[i]
        for j in range(i + 1, K):
            pj = comp_pts[j]
            diff = pi[:, None, :] - pj[None, :, :]
            d2 = (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
            flat_idx = int(np.argmin(d2))
            ii, jj = divmod(flat_idx, len(pj))
            dist = math.sqrt(float(d2[ii, jj]))
            if dist <= max_gap_px:
                edges.append((dist, i, j, pi[ii], pj[jj]))

    edges.sort(key=lambda e: e[0])
    parent_uf = list(range(K))

    def find_uf(x: int) -> int:
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x

    for dist, i, j, p1, p2 in edges:
        ri, rj = find_uf(i), find_uf(j)
        if ri == rj:
            continue
        n_samp = max(int(math.ceil(dist)) + 1, 3)
        ys_line = np.clip(np.round(np.linspace(p1[0], p2[0], n_samp)).astype(np.int32), 0, fg_roi.shape[0] - 1)
        xs_line = np.clip(np.round(np.linspace(p1[1], p2[1], n_samp)).astype(np.int32), 0, fg_roi.shape[1] - 1)
        line_fg = float(fg_roi[ys_line, xs_line].mean())
        if line_fg >= min_bridge_fg:
            cv2.line(out, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), color=1, thickness=3)
            parent_uf[ri] = rj

    return out


class Stage2CropDataset(Dataset):
    def __init__(self, items: List[dict], augment: bool = True):
        self.items = items
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        it = self.items[idx]
        x = it["x"].astype(np.float32).copy()  # (5, 128, 384)
        y = it["y"].astype(np.float32) * (1.0 / 255.0)  # (128, 384)
        w = np.float32(it["weight"])

        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
                y = y[:, ::-1]
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
                y = y[::-1, :]
            if np.random.rand() < 0.45:
                gain = float(np.random.uniform(0.92, 1.08))
                bias = float(np.random.uniform(-0.05, 0.05))
                x[0] = np.clip(x[0] * gain + bias, -1.0, 1.0)

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
            torch.tensor(w, dtype=torch.float32),
        )


def build_stage2_model(spec_name: str, pretrained: bool = True) -> nn.Module:
    sp = STAGE2_SPECS[spec_name]
    cls = getattr(smp, sp["arch"])
    enc = sp["encoder"]
    enc_weights = (True if enc.startswith("tu-") else "imagenet") if pretrained else None
    return cls(
        encoder_name=enc,
        encoder_weights=enc_weights,
        in_channels=5,
        classes=1,
        activation=None,
    )


def stage2_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    logits_f = logits[:, 0].float()
    targets_f = targets.float()
    w = weights.float() / weights.float().mean().clamp_min(1e-6)

    bce_per_sample = F.binary_cross_entropy_with_logits(logits_f, targets_f, reduction="none").mean(dim=(1, 2))
    probs = torch.sigmoid(logits_f)
    inter = (probs * targets_f).sum(dim=(1, 2))
    den = probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2)) + 1.0
    dice_per_sample = 1.0 - (2.0 * inter + 1.0) / den
    return ((bce_per_sample + 0.65 * dice_per_sample) * w).mean()


@torch.no_grad()
def predict_crop_4tta(model: nn.Module, x_batch: torch.Tensor, device: torch.device) -> np.ndarray:
    """Batch 4-flip TTA (id, H, V, HV) on canonical [B, 5, 128, 384] crops."""
    xb = x_batch.to(device, non_blocking=True)
    acc = torch.zeros((xb.shape[0], S2_H, S2_W), dtype=torch.float32, device=device)
    for fh in (False, True):
        for fv in (False, True):
            v = xb
            if fh:
                v = torch.flip(v, dims=(-1,))
            if fv:
                v = torch.flip(v, dims=(-2,))
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                p = torch.sigmoid(model(v)[:, 0]).float()
            if fv:
                p = torch.flip(p, dims=(-2,))
            if fh:
                p = torch.flip(p, dims=(-1,))
            acc.add_(p)
    return (acc * 0.25).cpu().numpy()


def build_stage2_training_crops(
    stems: List[str],
    oof_selected: Dict[str, List[dict]],
    families: List[str],
    blend_w: np.ndarray,
    gt_cache: dict,
    fold_map: Dict[str, int],
    fold_blend_w: Optional[Dict[int, np.ndarray]] = None,
    stage1_best_tau: float = 0.2700,
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """
    Build Stage-2 oriented 5-channel crops for ALL selected OOF instances (`infer_records`),
    cache compact local ROI slices (`fg_roi_u16`, `sp_roi_u16`, `ff_roi_u8`, `seed_roi`, `valid_roi`),
    and construct conditional matched-annotator targets q_c(x) for matched instances (`train_items`).
    """
    def _work_train_one(s: str) -> Tuple[str, List[dict], List[dict]]:
        sel_list = oof_selected.get(s, [])
        if not sel_list:
            return s, [], []

        x_u8 = cv2.imread(str(PREP_TRAIN_DIR / f"{s}_x.png"), cv2.IMREAD_COLOR)
        ff_u8 = x_u8[:, :, 0]
        ff_norm = ff_u8.astype(np.float32) * (1.0 / 255.0)
        valid_full = x_u8[:, :, 2] > 0
        w_s = fold_blend_w[fold_map[s]] if fold_blend_w is not None else blend_w
        fg_b, sp_b, _, _, _ = load_blended_maps(s, families, w_s, is_test=False)

        rles_b = [{"size": [FULL_SIZE, FULL_SIZE], "counts": it["rle"]["counts"].encode("ascii")} for it in sel_list]
        dec_full = mask_util.decode(rles_b)
        if dec_full.ndim == 2:
            dec_full = dec_full[:, :, None]
        dec_1856 = dec_full[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :]

        # Only include confident neighbor seeds in Ch4 (`other_m`) so speculative low-u_hat rescue
        # candidates never blind Stage-2 at the tips of adjacent real filaments!
        strong_flags = [float(it.get("u_hat", 1.0)) >= (stage1_best_tau - 0.015) for it in sel_list]
        strong_cov = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.int16)
        for k_i, is_str in enumerate(strong_flags):
            if is_str:
                strong_cov += dec_1856[:, :, k_i].astype(np.int16)

        gt_recs = gt_cache["records_by_stem"][s]
        gt_decoded_recs = []
        gt_rles_recs = []
        for _, g_raw in gt_recs:
            g_bytes = [{"size": [FULL_SIZE, FULL_SIZE], "counts": g["counts"].encode("ascii")} for g in g_raw]
            gt_rles_recs.append(g_bytes)
            gt_decoded_recs.append(
                mask_util.decode(g_bytes)[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_SIZE + CROP_OFFSET, :]
                if g_bytes else None
            )

        loc_infer: List[dict] = []
        loc_train: List[dict] = []
        for k in range(len(sel_list)):
            seed_m = dec_1856[:, :, k] > 0
            other_m = (strong_cov - (dec_1856[:, :, k].astype(np.int16) if strong_flags[k] else 0)) > 0
            M_fwd, M_inv_local, roi_box = compute_oriented_crop_affine(seed_m)
            rx0, ry0, rx1, ry1 = roi_box
            x_5ch = extract_5ch_crop(ff_norm, fg_b, sp_b, seed_m, other_m, M_fwd)
            u_k = float(sel_list[k]["u_hat"])

            rec_meta = {
                "x": x_5ch.astype(np.float16),
                "M_inv_local": M_inv_local,
                "roi_box": roi_box,
                "rle_stage1": sel_list[k]["rle"],
                "u_hat": u_k,
                "p_hit": float(sel_list[k].get("p_hit", min(1.0, u_k / 0.68))),
                "m_iou": float(sel_list[k].get("m_iou", 0.68)),
                "fg_roi_u16": np.clip(np.round(fg_b[ry0:ry1, rx0:rx1] * 65535.0), 0, 65535).astype(np.uint16),
                "sp_roi_u16": np.clip(np.round(sp_b[ry0:ry1, rx0:rx1] * 65535.0), 0, 65535).astype(np.uint16),
                "ff_roi_u8": ff_u8[ry0:ry1, rx0:rx1].copy(),
                "seed_roi": seed_m[ry0:ry1, rx0:rx1].copy(),
                "valid_roi": valid_full[ry0:ry1, rx0:rx1].copy(),
            }
            loc_infer.append(rec_meta)

            matched_masks = []
            cand_rle_1 = [rles_b[k]]
            for g_bytes, g_dec in zip(gt_rles_recs, gt_decoded_recs):
                if not g_bytes or g_dec is None:
                    continue
                ious = np.asarray(mask_util.iou(cand_rle_1, g_bytes, [0] * len(g_bytes)), dtype=np.float32)[0]
                best_g = int(np.argmax(ious))
                if float(ious[best_g]) >= 0.35:
                    matched_masks.append(g_dec[:, :, best_g].astype(np.float32))

            if len(matched_masks) >= 1:
                q_disk = np.mean(matched_masks, axis=0).astype(np.float32)
                q_crop = cv2.warpAffine(
                    q_disk, M_fwd, (S2_W, S2_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                )
                loc_train.append({
                    "stem": s,
                    "k_idx": k,
                    "fold": fold_map[s],
                    "x": x_5ch.astype(np.float16),
                    "y": np.clip(np.round(q_crop * 255.0), 0, 255).astype(np.uint8),
                    "weight": float(len(matched_masks)),
                })

        return s, loc_train, loc_infer

    train_items: List[dict] = []
    infer_records: Dict[str, List[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for s, tr_l, inf_l in ex.map(_work_train_one, stems):
            infer_records[s] = inf_l
            train_items.extend(tr_l)
    return train_items, infer_records


def build_stage2_test_crops(
    test_stems: List[str],
    test_selected: Dict[str, List[dict]],
    families: List[str],
    blend_w: np.ndarray,
    stage1_best_tau: float = 0.2700,
) -> Dict[str, List[dict]]:
    def _work_test_one(s: str) -> Tuple[str, List[dict]]:
        sel_list = test_selected.get(s, [])
        if not sel_list:
            return s, []
        x_u8 = cv2.imread(str(PREP_TEST_DIR / f"{s}_x.png"), cv2.IMREAD_COLOR)
        ff_u8 = x_u8[:, :, 0]
        ff_norm = ff_u8.astype(np.float32) * (1.0 / 255.0)
        valid_full = x_u8[:, :, 2] > 0
        fg_b, sp_b, _, _, _ = load_blended_maps(s, families, blend_w, is_test=True)

        rles_b = [{"size": [FULL_SIZE, FULL_SIZE], "counts": it["rle"]["counts"].encode("ascii")} for it in sel_list]
        dec_full = mask_util.decode(rles_b)
        if dec_full.ndim == 2:
            dec_full = dec_full[:, :, None]
        dec_1856 = dec_full[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :]

        strong_flags = [float(it.get("u_hat", 1.0)) >= (stage1_best_tau - 0.015) for it in sel_list]
        strong_cov = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.int16)
        for k_i, is_str in enumerate(strong_flags):
            if is_str:
                strong_cov += dec_1856[:, :, k_i].astype(np.int16)

        loc_out: List[dict] = []
        for k in range(len(sel_list)):
            seed_m = dec_1856[:, :, k] > 0
            other_m = (strong_cov - (dec_1856[:, :, k].astype(np.int16) if strong_flags[k] else 0)) > 0
            M_fwd, M_inv_local, roi_box = compute_oriented_crop_affine(seed_m)
            rx0, ry0, rx1, ry1 = roi_box
            x_5ch = extract_5ch_crop(ff_norm, fg_b, sp_b, seed_m, other_m, M_fwd)
            u_k = float(sel_list[k]["u_hat"])
            loc_out.append({
                "x": x_5ch.astype(np.float16),
                "M_inv_local": M_inv_local,
                "roi_box": roi_box,
                "rle_stage1": sel_list[k]["rle"],
                "u_hat": u_k,
                "p_hit": float(sel_list[k].get("p_hit", min(1.0, u_k / 0.68))),
                "m_iou": float(sel_list[k].get("m_iou", 0.68)),
                "fg_roi_u16": np.clip(np.round(fg_b[ry0:ry1, rx0:rx1] * 65535.0), 0, 65535).astype(np.uint16),
                "sp_roi_u16": np.clip(np.round(sp_b[ry0:ry1, rx0:rx1] * 65535.0), 0, 65535).astype(np.uint16),
                "ff_roi_u8": ff_u8[ry0:ry1, rx0:rx1].copy(),
                "seed_roi": seed_m[ry0:ry1, rx0:rx1].copy(),
                "valid_roi": valid_full[ry0:ry1, rx0:rx1].copy(),
            })
        return s, loc_out

    out: Dict[str, List[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for s, inf_l in ex.map(_work_test_one, test_stems):
            out[s] = inf_l
    return out


_RING_KERNEL_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def reconstruct_disjoint_stage2_image(
    stem: str,
    crop_recs: List[dict],
    crop_probs: List[np.ndarray],
    fg_blend: Optional[np.ndarray] = None,
    disk_valid_mask: Optional[np.ndarray] = None,
    alpha_s2: float = 0.80,
    t_pix: float = 0.42,
    use_join3: bool = True,
    crop_stds: Optional[List[np.ndarray]] = None,
    return_details: bool = False,
    active_indices: Optional[List[int]] = None,
    protect_strong_tau: Optional[float] = None,
    u_tie_scores: Optional[List[float]] = None,
) -> List[dict]:
    """
    Inverse-warp Stage-2 refined crop probabilities back to [1856, 1856], apply `join3` MST
    fragment bridging, enforce safety fallback if seed vanishes, and paint onto a strictly
    pixel-disjoint [2048, 2048] label map resolved by `argmax` probability ownership.

    - `active_indices`: when provided, ONLY those candidate indices paint onto `owner_map`
      (eliminating pixel stealing by discarded false-positive candidates!).
    - `protect_strong_tau`: when provided, confident seeds (`u_hat >= protect_strong_tau`) paint
      at full priority (`prior_mult = 1.0`) while speculative rescue seeds (`< protect_strong_tau`)
      paint at `prior_mult = 1e-3`, so rescue candidates can never steal pixels from confident
      filaments while still competing cleanly among themselves.
    """
    K = len(crop_recs)
    if K == 0:
        return []

    raw_indices = list(range(K)) if active_indices is None else [int(k) for k in active_indices if 0 <= int(k) < K]
    if not raw_indices:
        return []

    def _u_tie(k_idx: int) -> float:
        if u_tie_scores is not None and k_idx < len(u_tie_scores):
            return float(u_tie_scores[k_idx])
        return float(crop_recs[k_idx].get("u_hat", 0.5))

    def _tier(k_idx: int) -> int:
        if protect_strong_tau is None:
            return 0
        return 0 if float(crop_recs[k_idx].get("u_hat", 1.0)) >= protect_strong_tau else 1

    ordered_indices = sorted(raw_indices, key=lambda k_idx: (_tier(k_idx), -_u_tie(k_idx), k_idx))

    best_prob_map = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    owner_map = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.int32)

    need_s1_decode = any("seed_roi" not in crop_recs[k] for k in ordered_indices)
    if need_s1_decode:
        s1_rles = [{"size": [FULL_SIZE, FULL_SIZE], "counts": r["rle_stage1"]["counts"].encode("ascii")} for r in crop_recs]
        s1_dec = mask_util.decode(s1_rles)
        if s1_dec.ndim == 2:
            s1_dec = s1_dec[:, :, None]
        s1_dec = s1_dec[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :]
    else:
        s1_dec = None

    per_k_meta: Dict[int, dict] = {}
    for k in ordered_indices:
        rec = crop_recs[k]
        rx0, ry0, rx1, ry1 = rec["roi_box"]
        rw, rh = rx1 - rx0, ry1 - ry0
        q_roi = cv2.warpAffine(
            crop_probs[k].astype(np.float32),
            rec["M_inv_local"],
            (rw, rh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        fg_roi = (
            rec["fg_roi_u16"].astype(np.float32) * (1.0 / 65535.0)
            if "fg_roi_u16" in rec
            else fg_blend[ry0:ry1, rx0:rx1]
        )
        seed_roi = rec["seed_roi"] if "seed_roi" in rec else (s1_dec[ry0:ry1, rx0:rx1, k] > 0)
        valid_roi = rec["valid_roi"] if "valid_roi" in rec else disk_valid_mask[ry0:ry1, rx0:rx1]

        fused_roi = np.where(valid_roi, alpha_s2 * q_roi + (1.0 - alpha_s2) * fg_roi, 0.0)
        bw_roi = (fused_roi >= t_pix).astype(np.uint8)
        if use_join3 and bw_roi.any():
            bw_roi = join3_bridge_fragments(bw_roi, fg_roi, seed_roi)

        seed_area = int(seed_roi.sum())
        ref_area = int(bw_roi.sum())
        inter_seed = int(((bw_roi > 0) & seed_roi).sum())
        uni_seed = int(((bw_roi > 0) | seed_roi).sum())
        seed_iou = float(inter_seed / max(uni_seed, 1))
        used_fallback = 0.0

        if ref_area < 45 or (seed_area > 0 and inter_seed < 0.25 * min(seed_area, ref_area)):
            used_fallback = 1.0
            bw_roi = seed_roi.astype(np.uint8)
            base_score = np.maximum(fused_roi, 0.50)
        else:
            base_score = fused_roi

        prior_mult = 1e-3 if _tier(k) == 1 else 1.0
        tie_bonus = 1e-5 * min(max(_u_tie(k), 0.0), 1.0)
        score_roi = np.where(bw_roi > 0, (base_score + tie_bonus) * prior_mult, 0.0)

        cur_best = best_prob_map[ry0:ry1, rx0:rx1]
        cur_own = owner_map[ry0:ry1, rx0:rx1]
        win = (bw_roi > 0) & (score_roi > cur_best)
        cur_best[win] = score_roi[win]
        cur_own[win] = k + 1

        if return_details:
            if crop_stds is not None:
                d_roi = cv2.warpAffine(
                    crop_stds[k].astype(np.float32),
                    rec["M_inv_local"],
                    (rw, rh),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
            else:
                d_roi = np.zeros_like(q_roi)
            sp_roi = (
                rec["sp_roi_u16"].astype(np.float32) * (1.0 / 65535.0)
                if "sp_roi_u16" in rec
                else fg_roi
            )
            ff_roi = rec["ff_roi_u8"].astype(np.float32) if "ff_roi_u8" in rec else None
            per_k_meta[k] = {
                "q_roi": q_roi,
                "d_roi": d_roi,
                "fg_roi": fg_roi,
                "sp_roi": sp_roi,
                "ff_roi": ff_roi,
                "valid_roi": valid_roi,
                "seed_area": seed_area,
                "seed_iou": seed_iou,
                "used_fallback": used_fallback,
            }

    out_items: List[dict] = []
    slices = ndi.find_objects(owner_map)
    for k_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sub = owner_map[sl] == k_id
        area_fin = int(sub.sum())
        if area_fin < 35:
            continue
        y0, y1 = sl[0].start + CROP_OFFSET, sl[0].stop + CROP_OFFSET
        x0, x1 = sl[1].start + CROP_OFFSET, sl[1].stop + CROP_OFFSET
        rle_dict = {"size": [FULL_SIZE, FULL_SIZE], "counts": encode_submask_rle_2048(sub, y0, y1, x0, x1)}
        if not return_details:
            out_items.append(rle_dict)
        else:
            orig_k = k_id - 1
            rec = crop_recs[orig_k]
            mk = per_k_meta[orig_k]
            rx0, ry0, _, _ = rec["roi_box"]
            ly0, ly1 = sl[0].start - ry0, sl[0].stop - ry0
            lx0, lx1 = sl[1].start - rx0, sl[1].stop - rx0
            q_roi_k = mk["q_roi"]
            q_vals = q_roi_k[ly0:ly1, lx0:lx1][sub]
            d_vals = mk["d_roi"][ly0:ly1, lx0:lx1][sub]
            fg_vals = mk["fg_roi"][ly0:ly1, lx0:lx1][sub]
            sp_vals = mk["sp_roi"][ly0:ly1, lx0:lx1][sub]

            rh_k, rw_k = q_roi_k.shape
            py0, py1 = max(0, ly0 - 4), min(rh_k, ly1 + 4)
            px0, px1 = max(0, lx0 - 4), min(rw_k, lx1 + 4)
            sub_pad = np.zeros((py1 - py0, px1 - px0), dtype=np.uint8)
            sub_pad[ly0 - py0 : ly1 - py0, lx0 - px0 : lx1 - px0] = sub.astype(np.uint8)
            ring_m = (cv2.dilate(sub_pad, _RING_KERNEL_7) > 0) & (sub_pad == 0) & mk["valid_roi"][py0:py1, px0:px1]

            q_mean_v = float(q_vals.mean())
            if ring_m.any():
                q_ring_drop = q_mean_v - float(q_roi_k[py0:py1, px0:px1][ring_m].mean())
                if mk["ff_roi"] is not None:
                    ff_ring_m = float(mk["ff_roi"][py0:py1, px0:px1][ring_m].mean())
                    ff_in_m = float(mk["ff_roi"][ly0:ly1, lx0:lx1][sub].mean())
                    ff_contrast_drop = (ff_ring_m - ff_in_m) * (1.0 / 255.0)
                else:
                    ff_contrast_drop = 0.0
            else:
                q_ring_drop = q_mean_v
                ff_contrast_drop = 0.0

            u_s1_v = float(rec["u_hat"])
            out_items.append({
                "cand_idx": orig_k,
                "rle": rle_dict,
                "u_s1": u_s1_v,
                "p_hit_s1": float(rec.get("p_hit", min(1.0, u_s1_v / 0.68))),
                "m_iou_s1": float(rec.get("m_iou", 0.68)),
                "q_mean": q_mean_v,
                "q_q75": float(np.quantile(q_vals, 0.75)),
                "q_q90": float(np.quantile(q_vals, 0.90)),
                "q_max": float(q_vals.max()),
                "q_ring_drop": float(q_ring_drop),
                "s2_disagree": float(d_vals.mean()),
                "fg_in_mean": float(fg_vals.mean()),
                "sp_in_mean": float(sp_vals.mean()),
                "ff_contrast_drop": float(ff_contrast_drop),
                "seed_iou": float(mk["seed_iou"]),
                "log_area_fin": math.log1p(float(area_fin)),
                "log_area_ratio": math.log((float(area_fin) + 25.0) / (float(mk["seed_area"]) + 25.0)),
                "used_fallback": float(mk["used_fallback"]),
            })
    return out_items


S2_RESIDUAL_COLS = [
    "u_s1",
    "p_hit_s1",
    "m_iou_s1",
    "q_mean",
    "q_q75",
    "q_q90",
    "q_max",
    "q_ring_drop",
    "s2_disagree",
    "fg_in_mean",
    "sp_in_mean",
    "ff_contrast_drop",
    "seed_iou",
    "log_area_fin",
    "log_area_ratio",
    "used_fallback",
    "y_best_iou",
    "y_best_sc",
    "y_ixs",
]


def attach_yolo_agreement_to_details(details: List[dict], yolo_dets: List[dict]) -> None:
    if not details:
        return
    if not yolo_dets:
        for d in details:
            d["y_best_iou"] = 0.0
            d["y_best_sc"] = 0.0
            d["y_ixs"] = 0.0
        return
    yrles = [{"size": [FULL_SIZE, FULL_SIZE], "counts": y["rle"]["counts"].encode("ascii")} for y in yolo_dets]
    yscores = np.array([float(y["score"]) for y in yolo_dets], dtype=np.float32)
    prles = [{"size": [FULL_SIZE, FULL_SIZE], "counts": d["rle"]["counts"].encode("ascii")} for d in details]
    iou_m = np.asarray(mask_util.iou(prles, yrles, [0] * len(yrles)), dtype=np.float32)
    for i, d in enumerate(details):
        row = iou_m[i]
        bi = int(np.argmax(row))
        biou = float(row[bi])
        bsc = float(yscores[bi]) if biou > 0.0 else 0.0
        d["y_best_iou"] = biou
        d["y_best_sc"] = bsc
        d["y_ixs"] = float(np.max(row * yscores))


def run_stage2_pipeline(
    epochs: int = 22,
    batch_size: int = 24,
    ckpt_subdir: str = "ckpt_stage2_v3",
    save_infer_artifacts: bool = False,
) -> dict:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.88)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_cache = build_gt_cache()
    with open(FOLDS_PATH, "r", encoding="utf-8") as f:
        fold_map = json.load(f)
    with open(WORK_DIR / "stage1_dp_handoff.pkl", "rb") as f:
        handoff = pickle.load(f)

    oof_yolo, test_yolo = {}, {}
    if (WORK_DIR / "oof_yolo11l_seg.pkl").exists() and (WORK_DIR / "test_yolo11l_seg.pkl").exists():
        with open(WORK_DIR / "oof_yolo11l_seg.pkl", "rb") as f:
            oof_yolo = pickle.load(f)
        with open(WORK_DIR / "test_yolo11l_seg.pkl", "rb") as f:
            test_yolo = pickle.load(f)

    families = handoff["families"]
    blend_w = np.asarray(handoff["dense_blend_weights"], dtype=np.float32)
    raw_fbw = handoff.get("fold_dense_blend_weights")
    fold_blend_w = (
        {int(k): np.asarray(v, dtype=np.float32) for k, v in raw_fbw.items()}
        if raw_fbw is not None
        else None
    )
    oof_selected = handoff["oof_selected"]
    test_selected = handoff["test_selected"]
    handoff_tau = float(handoff.get("handoff_tau", handoff.get("best_tau", 0.2275)))
    stage1_best_tau = float(handoff.get("best_tau", 0.2700))

    stems = gt_cache["stems"]
    test_stems = sorted(fn[:-6] for fn in os.listdir(PREP_TEST_DIR) if fn.endswith("_x.png"))

    print(f"[stage2] Extracting 5-channel PCA-oriented 384x128 crops + local ROIs (families={families}, ckpt={ckpt_subdir})...")
    t0 = time.time()
    train_items, oof_infer_recs = build_stage2_training_crops(
        stems, oof_selected, families, blend_w, gt_cache, fold_map,
        fold_blend_w=fold_blend_w, stage1_best_tau=stage1_best_tau,
    )
    test_infer_recs = build_stage2_test_crops(
        test_stems, test_selected, families, blend_w, stage1_best_tau=stage1_best_tau
    )
    print(
        f"[stage2] Extracted {len(train_items)} matched training crops ({sum(len(v) for v in oof_infer_recs.values())} OOF, "
        f"{sum(len(v) for v in test_infer_recs.values())} Test) in {time.time() - t0:.1f}s."
    )

    ckpt_dir = WORK_DIR / ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    active_ckpt_dirs = [ckpt_subdir]
    if ckpt_subdir != "ckpt_stage2" and (WORK_DIR / "ckpt_stage2").exists():
        if all((WORK_DIR / "ckpt_stage2" / f"{sp}_fold{f}.pt").exists() for sp in STAGE2_SPECS for f in range(NUM_FOLDS)):
            active_ckpt_dirs.append("ckpt_stage2")

    slot_defs = [
        (
            f"{cdir}:{base_sp}" if len(active_ckpt_dirs) > 1 else base_sp,
            WORK_DIR / cdir,
            base_sp,
            STAGE2_SPECS[base_sp],
        )
        for cdir in active_ckpt_dirs
        for base_sp in STAGE2_SPECS
    ]
    spec_names = [sd[0] for sd in slot_defs]
    n_spec = len(spec_names)
    oof_spec_probs = {
        sp: {s: [None] * len(oof_infer_recs[s]) for s in stems}
        for sp in spec_names
    }
    test_spec_sum = {
        sp: {s: [np.zeros((S2_H, S2_W), dtype=np.float32) for _ in test_infer_recs[s]] for s in test_stems}
        for sp in spec_names
    }
    test_spec_sq = {
        sp: {s: [np.zeros((S2_H, S2_W), dtype=np.float32) for _ in test_infer_recs[s]] for s in test_stems}
        for sp in spec_names
    }

    for slot_name, slot_dir, base_spec_name, spec in slot_defs:
        val_ious_f = []
        for fold in range(NUM_FOLDS):
            ckpt_p = slot_dir / f"{base_spec_name}_fold{fold}.pt"
            tr_items = [it for it in train_items if it["fold"] != fold]
            va_items = [it for it in train_items if it["fold"] == fold]

            if not ckpt_p.exists():
                f_t0 = time.time()
                torch.manual_seed(RANDOM_SEED + fold * 31)
                model = build_stage2_model(base_spec_name).to(device)
                ema_m = copy.deepcopy(model).eval()
                for p in ema_m.parameters():
                    p.requires_grad_(False)

                opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-2)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=spec["lr"] * 0.03)
                tr_loader = DataLoader(Stage2CropDataset(tr_items, augment=True), batch_size=batch_size, shuffle=True, drop_last=True)
                va_loader = DataLoader(Stage2CropDataset(va_items, augment=False), batch_size=batch_size, shuffle=False)

                best_val_iou = -1.0
                updates = 0
                for ep in range(1, epochs + 1):
                    model.train()
                    for xb, yb, wb in tr_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        wb = wb.to(device, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            out = model(xb)
                            loss = stage2_loss(out, yb, wb)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        opt.step()
                        updates += 1
                        d = min(0.992, (1.0 + updates) / (10.0 + updates))
                        with torch.no_grad():
                            msd = model.state_dict()
                            for k_n, v_t in ema_m.state_dict().items():
                                if v_t.dtype.is_floating_point:
                                    v_t.mul_(d).add_(msd[k_n].detach(), alpha=1.0 - d)
                                else:
                                    v_t.copy_(msd[k_n])
                    sched.step()

                    if ep >= 10 and (ep % 3 == 0 or ep == epochs):
                        ema_m.eval()
                        ious = []
                        with torch.no_grad():
                            for xb, yb, _ in va_loader:
                                xb = xb.to(device)
                                yb_b = (yb.to(device) >= 0.5)
                                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                    pr_b = torch.sigmoid(ema_m(xb)[:, 0]) >= 0.42
                                inter = (pr_b & yb_b).sum(dim=(1, 2)).float()
                                uni = (pr_b | yb_b).sum(dim=(1, 2)).float().clamp_min(1.0)
                                ious.extend((inter / uni).cpu().tolist())
                        mean_iou = float(np.mean(ious))
                        if mean_iou > best_val_iou:
                            best_val_iou = mean_iou
                            torch.save({"model": ema_m.state_dict(), "val_iou": best_val_iou}, ckpt_p)
                print(
                    f"  [{slot_name} | Fold {fold}] Trained {epochs} eps in {time.time() - f_t0:.1f}s "
                    f"| best crop IoU = {best_val_iou:.4f}"
                )

            best_m = build_stage2_model(base_spec_name).to(device).eval()
            st_d = torch.load(ckpt_p, map_location=device, weights_only=False)
            best_m.load_state_dict(st_d["model"])
            val_ious_f.append(float(st_d.get("val_iou", 0.0)))

            val_stems = [s for s in stems if fold_map[s] == fold]
            for s in val_stems:
                recs = oof_infer_recs[s]
                if not recs:
                    continue
                xb = torch.from_numpy(np.stack([r["x"].astype(np.float32) for r in recs], axis=0))
                preds = predict_crop_4tta(best_m, xb, device)
                for k in range(len(recs)):
                    oof_spec_probs[slot_name][s][k] = preds[k].astype(np.float16)

            for s in test_stems:
                recs = test_infer_recs[s]
                if not recs:
                    continue
                xb = torch.from_numpy(np.stack([r["x"].astype(np.float32) for r in recs], axis=0))
                preds = predict_crop_4tta(best_m, xb, device)
                for k in range(len(recs)):
                    test_spec_sum[slot_name][s][k] += preds[k]
                    test_spec_sq[slot_name][s][k] += preds[k] * preds[k]

        print(f"[stage2] {slot_name} 5-fold mean crop IoU = {np.mean(val_ious_f):.4f}")

    # Optimize LOFO convex weights across the 4 Stage-2 architectures + exact Law-of-Total-Variance Stage-2 disagreement!
    ds_P_items = []
    ds_Y_items = []
    ds_F_items = np.array([it["fold"] for it in train_items], dtype=np.int32)
    for it in train_items:
        s_i, k_i = it["stem"], it["k_idx"]
        p_stk = np.stack([oof_spec_probs[sp][s_i][k_i][::4, ::4].astype(np.float32).ravel() for sp in spec_names], axis=0)
        y_flat = it["y"][::4, ::4].astype(np.float32).ravel() * (1.0 / 255.0)
        ds_P_items.append(p_stk)
        ds_Y_items.append(y_flat)

    s2_w_folds: Dict[int, np.ndarray] = {}
    for f in range(NUM_FOLDS):
        tr_idx = np.flatnonzero(ds_F_items != f)
        P_tr = np.hstack([ds_P_items[idx] for idx in tr_idx])
        Y_tr = np.concatenate([ds_Y_items[idx] for idx in tr_idx])

        def _s2_loss_w(w_raw: np.ndarray) -> float:
            w = np.maximum(w_raw, 0.0)
            w = w / max(float(w.sum()), 1e-8)
            pr = np.clip(w @ P_tr, 1e-6, 1.0 - 1e-6)
            bce = -float(np.mean(Y_tr * np.log(pr) + (1.0 - Y_tr) * np.log(1.0 - pr)))
            inter = float(np.sum(pr * Y_tr))
            dice = 1.0 - (2.0 * inter + 1.0) / (float(np.sum(pr) + np.sum(Y_tr)) + 1.0)
            return bce + 0.65 * dice

        lo_sw = 0.03 if n_spec >= 8 else 0.08
        hi_sw = 0.32 if n_spec >= 8 else 0.45
        res_sw = minimize(
            _s2_loss_w,
            np.ones(n_spec, dtype=np.float64) / float(n_spec),
            method="SLSQP",
            bounds=[(lo_sw, hi_sw)] * n_spec,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}],
        )
        wf = np.maximum(np.asarray(res_sw.x, dtype=np.float32), 0.0)
        wf /= wf.sum()
        s2_w_folds[f] = wf

    s2_w_mean = np.mean(np.stack([s2_w_folds[f] for f in range(NUM_FOLDS)], axis=0), axis=0).astype(np.float32)
    s2_w_mean /= s2_w_mean.sum()
    print(f"[stage2] Mean LOFO 4-Arch Stage-2 weights: {dict(zip(spec_names, [round(float(v), 4) for v in s2_w_mean]))}")

    oof_crop_probs_acc: Dict[str, List[np.ndarray]] = {}
    oof_crop_stds: Dict[str, List[np.ndarray]] = {}
    for s in stems:
        K = len(oof_infer_recs[s])
        wf_s = s2_w_folds[fold_map[s]]
        oof_crop_probs_acc[s] = []
        oof_crop_stds[s] = []
        for k in range(K):
            stk = np.stack([oof_spec_probs[sp][s][k].astype(np.float32) for sp in spec_names], axis=0)
            oof_crop_probs_acc[s].append(np.tensordot(wf_s, stk, axes=(0, 0)).astype(np.float32))
            oof_crop_stds[s].append(np.std(stk, axis=0).astype(np.float16))

    test_crop_probs_acc: Dict[str, List[np.ndarray]] = {}
    test_crop_stds: Dict[str, List[np.ndarray]] = {}
    inv_f = 1.0 / float(NUM_FOLDS)
    for s in test_stems:
        K = len(test_infer_recs[s])
        test_crop_probs_acc[s] = []
        test_crop_stds[s] = []
        for k in range(K):
            means_k = np.stack([test_spec_sum[sp][s][k] * inv_f for sp in spec_names], axis=0)
            fvars_k = [
                np.maximum(test_spec_sq[sp][s][k] * inv_f - (test_spec_sum[sp][s][k] * inv_f) ** 2, 0.0)
                for sp in spec_names
            ]
            raw_v = np.var(means_k, axis=0)
            corr_v = raw_v + ((float(n_spec) - 1.0) / float(n_spec)) * (sum(fvars_k) / float(n_spec))
            test_crop_probs_acc[s].append(np.tensordot(s2_w_mean, means_k, axes=(0, 0)).astype(np.float32))
            test_crop_stds[s].append(np.sqrt(np.maximum(corr_v, 0.0)).astype(np.float16))

    # Evaluate Stage-1 baseline (at Stage-1 LGO best_tau) vs Stage-2 across (alpha_s2, t_pix)
    s1_strict_preds = {
        s: [it["rle"] for it in oof_selected[s] if float(it["u_hat"]) >= stage1_best_tau]
        for s in stems
    }
    s1_base_res = evaluate_pq(s1_strict_preds, gt_cache)
    s1_pool_preds = {s: [it["rle"] for it in oof_selected[s]] for s in stems}
    s1_pool_res = evaluate_pq(s1_pool_preds, gt_cache)
    print(
        f"\n[stage2] Stage-1 Handoff Pool (tau={handoff_tau:.4f}) OOF PQ = {s1_pool_res['pq']:.4f} | "
        f"Stage-1 Strict (tau={stage1_best_tau:.4f}) OOF PQ = {s1_base_res['pq']:.4f}"
    )

    strict_idx_by_stem = {
        s: [k for k, r in enumerate(oof_infer_recs[s]) if float(r["u_hat"]) >= stage1_best_tau]
        for s in stems
    }
    grid_candidates = [
        (alpha, t_p)
        for alpha in (0.85, 0.93, 1.00)
        for t_p in (0.40, 0.43, 0.45, 0.47, 0.49, 0.52)
    ]
    grid_stem_stats: Dict[Tuple[float, float], Dict[str, Tuple[float, int, int, int]]] = {
        cfg: {} for cfg in grid_candidates
    }

    t_grid0 = time.time()

    def _eval_grid_stem(s: str) -> Tuple[str, Dict[Tuple[float, float], Tuple[float, int, int, int]]]:
        gt_recs_s = [
            [{"size": [FULL_SIZE, FULL_SIZE], "counts": g["counts"].encode("ascii")} for g in gt_raw]
            for _, gt_raw in gt_cache["records_by_stem"][s]
        ]
        res_s = {}
        act_idx = strict_idx_by_stem[s]
        for alpha, t_p in grid_candidates:
            preds_rle = reconstruct_disjoint_stage2_image(
                s,
                oof_infer_recs[s],
                oof_crop_probs_acc[s],
                alpha_s2=alpha,
                t_pix=t_p,
                use_join3=True,
                active_indices=act_idx,
                protect_strong_tau=stage1_best_tau,
            )
            preds_b = [{"size": [FULL_SIZE, FULL_SIZE], "counts": r["counts"].encode("ascii")} for r in preds_rle]
            n_pred = len(preds_b)
            iou_s, tp_s, fp_s, fn_s = 0.0, 0, 0, 0
            for gts in gt_recs_s:
                n_gt = len(gts)
                if n_gt == 0:
                    fp_s += n_pred
                    continue
                if n_pred == 0:
                    fn_s += n_gt
                    continue
                iou_m = np.asarray(mask_util.iou(preds_b, gts, [0] * n_gt), dtype=np.float64).T
                hit = iou_m > 0.5
                if hit.any():
                    iou_s += float(iou_m[hit].sum())
                    tp_s += int(hit.sum())
                fp_s += int((hit.sum(axis=0) == 0).sum())
                fn_s += int((hit.sum(axis=1) == 0).sum())
            res_s[(alpha, t_p)] = (iou_s, tp_s, fp_s, fn_s)
        return s, res_s

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for s, res_s in ex.map(_eval_grid_stem, stems):
            for cfg_k, st_v in res_s.items():
                grid_stem_stats[cfg_k][s] = st_v
    print(f"[stage2] Evaluated {len(grid_candidates)} confident (alpha_s2, t_pix) configs across 707 images in {time.time() - t_grid0:.1f}s.")

    def _score_cfg(cfg: Tuple[float, float], sub_s: List[str]) -> float:
        m = grid_stem_stats[cfg]
        iou_t = sum(m[s][0] for s in sub_s)
        tp_t = sum(m[s][1] for s in sub_s)
        fp_t = sum(m[s][2] for s in sub_s)
        fn_t = sum(m[s][3] for s in sub_s)
        den = tp_t + 0.5 * fp_t + 0.5 * fn_t
        return float(iou_t / den) if den > 0 else 0.0

    valid_cfgs = list(grid_stem_stats.keys())
    best_cfg = max(valid_cfgs, key=lambda c: _score_cfg(c, stems))

    nested_iou, nested_tp, nested_fp, nested_fn = 0.0, 0, 0, 0
    for f in range(NUM_FOLDS):
        tr_s = [s for s in stems if fold_map[s] != f]
        va_s = [s for s in stems if fold_map[s] == f]
        f_cfg = max(valid_cfgs, key=lambda c: _score_cfg(c, tr_s))
        m = grid_stem_stats[f_cfg]
        nested_iou += sum(m[s][0] for s in va_s)
        nested_tp += sum(m[s][1] for s in va_s)
        nested_fp += sum(m[s][2] for s in va_s)
        nested_fn += sum(m[s][3] for s in va_s)

    nested_den = nested_tp + 0.5 * nested_fp + 0.5 * nested_fn
    raw_nested_pq = float(nested_iou / nested_den) if nested_den > 0 else 0.0
    best_alpha, best_tpix = best_cfg
    print(
        f"[stage2] Confident 4-Arch Stage-2 @ (alpha={best_alpha:.2f}, t_pix={best_tpix:.2f}): "
        f"OOF PQ = {_score_cfg(best_cfg, stems):.4f} (vs S1 Strict {s1_base_res['pq']:.4f}) | Nested LOFO PQ = {raw_nested_pq:.4f}"
    )

    # Step 2: 19-Feature Post-Refinement Residual Calibrator (Ridge + Shallow LGB) + Survivor-Only Disjoint Reconstruction!
    oof_details_by_stem: Dict[str, List[dict]] = {}
    inst_rows: List[dict] = []
    total_gt_all = 0
    stem_gt_counts: Dict[str, int] = {}
    stem_groups: Dict[str, str] = {}

    for s in stems:
        gt_recs_raw = gt_cache["records_by_stem"][s]
        R_i = len(gt_recs_raw)
        stem_gt_counts[s] = sum(len(g_raw) for _, g_raw in gt_recs_raw)
        total_gt_all += stem_gt_counts[s]
        stem_groups[s] = gt_recs_raw[0][0].split("-")[0][:4] if gt_recs_raw else "0000"

        det_s = reconstruct_disjoint_stage2_image(
            s,
            oof_infer_recs[s],
            oof_crop_probs_acc[s],
            alpha_s2=best_alpha,
            t_pix=best_tpix,
            use_join3=True,
            crop_stds=oof_crop_stds[s],
            return_details=True,
            protect_strong_tau=stage1_best_tau,
        )
        attach_yolo_agreement_to_details(det_s, oof_yolo.get(s, []))
        oof_details_by_stem[s] = det_s

        gt_rles_list = [
            [{"size": [FULL_SIZE, FULL_SIZE], "counts": g["counts"].encode("ascii")} for g in g_raw]
            for _, g_raw in gt_recs_raw
        ]
        for d in det_s:
            rle_b = [{"size": [FULL_SIZE, FULL_SIZE], "counts": d["rle"]["counts"].encode("ascii")}]
            tp_c, fp_c, iou_c = 0, 0, 0.0
            for gts in gt_rles_list:
                if not gts:
                    fp_c += 1
                    continue
                ious_g = np.asarray(mask_util.iou(rle_b, gts, [0] * len(gts)), dtype=np.float64)[0]
                hits = ious_g > 0.5
                if hits.any():
                    tp_c += int(hits.sum())
                    iou_c += float(ious_g[hits].sum())
                else:
                    fp_c += 1
            d["tp"] = tp_c
            d["fp"] = fp_c
            d["iou_sum"] = iou_c
            d["R_i"] = float(R_i)
            d["u_true_s2"] = iou_c / float(R_i)
            d["stem"] = s
            d["fold"] = fold_map[s]
            d["group"] = stem_groups[s]
            d["row_idx"] = len(inst_rows)
            inst_rows.append(d)

    X_res = np.array([[r[c] for c in S2_RESIDUAL_COLS] for r in inst_rows], dtype=np.float64)
    y_res = np.array([r["u_true_s2"] - r["u_s1"] for r in inst_rows], dtype=np.float64)
    w_res = np.array([r["R_i"] for r in inst_rows], dtype=np.float64)
    folds_r = np.array([r["fold"] for r in inst_rows], dtype=np.int32)
    groups_r = np.array([r["group"] for r in inst_rows])
    u_s1_r = np.array([r["u_s1"] for r in inst_rows], dtype=np.float64)
    tp_r = np.array([r["tp"] for r in inst_rows], dtype=np.float64)
    fp_r = np.array([r["fp"] for r in inst_rows], dtype=np.float64)
    iou_r = np.array([r["iou_sum"] for r in inst_rows], dtype=np.float64)

    delta_ridge_oof = np.zeros(len(inst_rows), dtype=np.float64)
    delta_lgb_oof = np.zeros(len(inst_rows), dtype=np.float64)
    fold_res_models = []
    for f in range(NUM_FOLDS):
        tr = folds_r != f
        va = folds_r == f
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_res[tr])
        reg = Ridge(alpha=300.0, solver="svd")
        reg.fit(X_tr_s, y_res[tr], sample_weight=w_res[tr])
        delta_ridge_oof[va] = reg.predict(sc.transform(X_res[va]))

        other_fs = [of for of in range(NUM_FOLDS) if of != f]
        inner_lgbs = []
        va_lgb_acc = np.zeros(int(va.sum()), dtype=np.float64)
        for iv_f in other_fs:
            in_tr = (folds_r != f) & (folds_r != iv_f)
            in_va = folds_r == iv_f
            lgb_m = lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.025,
                num_leaves=7,
                max_depth=3,
                min_child_samples=80,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=3.0,
                reg_lambda=10.0,
                random_state=RANDOM_SEED + f * 17 + iv_f,
                n_jobs=4,
                verbose=-1,
            )
            lgb_m.fit(
                X_res[in_tr],
                y_res[in_tr],
                sample_weight=w_res[in_tr],
                eval_set=[(X_res[in_va], y_res[in_va])],
                eval_sample_weight=[w_res[in_va]],
                callbacks=[lgb.early_stopping(35, verbose=False)],
            )
            va_lgb_acc += lgb_m.predict(X_res[va])
            inner_lgbs.append(lgb_m)
        delta_lgb_oof[va] = va_lgb_acc / float(len(inner_lgbs))
        fold_res_models.append((sc, reg, inner_lgbs))

    w_lgb_grid = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
    betas = (0.0, 0.15, 0.25, 0.35, 0.50)
    taus_s2 = np.round(np.arange(0.1800, 0.3125, 0.0025), 4)

    # Precompute group & fold GT totals for subset PQ tuning
    all_groups = {g: [s for s in stems if stem_groups[s] == g] for g in set(stem_groups.values())}
    big_groups = {g: gs for g, gs in all_groups.items() if len(gs) >= 20}
    group_gt_tot = {g: sum(stem_gt_counts[s] for s in gs) for g, gs in big_groups.items()}
    group_mask_r = {g: (groups_r == g) for g in big_groups}

    fold_gt_tot = {f: sum(stem_gt_counts[s] for s in stems if fold_map[s] == f) for f in range(NUM_FOLDS)}
    fold_mask_r = {f: (folds_r == f) for f in range(NUM_FOLDS)}

    def _subset_pq(keep_m: np.ndarray, sub_mask: np.ndarray, gt_tot_sub: int) -> float:
        m = keep_m & sub_mask
        tp_v = float(tp_r[m].sum())
        fp_v = float(fp_r[m].sum())
        fn_v = float(gt_tot_sub) - tp_v
        den = tp_v + 0.5 * fp_v + 0.5 * fn_v
        return float(iou_r[m].sum() / den) if den > 0 else 0.0

    # Strict Nested LOFO selection over (w_lgb, beta_res, tau_s2) using LGO+Fold-Regret Robust PQ!
    nested_keep = np.zeros(len(inst_rows), dtype=bool)
    u_nested_r = np.zeros(len(inst_rows), dtype=np.float64)
    nested_post_choices = {}
    for f in range(NUM_FOLDS):
        va_m = fold_mask_r[f]
        tr_m = ~va_m
        tr_gt_tot = total_gt_all - fold_gt_tot[f]
        other_fs = [of for of in range(NUM_FOLDS) if of != f]

        w_lgb_f = min(
            w_lgb_grid,
            key=lambda w_l: float(
                np.average(
                    (y_res[tr_m] - ((1.0 - w_l) * delta_ridge_oof[tr_m] + w_l * delta_lgb_oof[tr_m])) ** 2,
                    weights=w_res[tr_m],
                )
            ),
        )
        delta_f = (1.0 - w_lgb_f) * delta_ridge_oof + w_lgb_f * delta_lgb_oof

        tr_scores = {}
        f_scores = {of: {} for of in other_fs}
        g_scores = {}
        tr_big_groups = {}
        for g in big_groups:
            gm_tr = group_mask_r[g] & tr_m
            gt_g_tr = sum(stem_gt_counts[s] for s in big_groups[g] if fold_map[s] != f)
            if gt_g_tr >= 80:
                tr_big_groups[g] = (gm_tr, gt_g_tr)
                g_scores[g] = {}

        for b_v in betas:
            u_cand = u_s1_r + b_v * delta_f
            for t_v in taus_s2:
                key = (float(b_v), float(t_v))
                km = u_cand >= float(t_v)
                tr_scores[key] = _subset_pq(km, tr_m, tr_gt_tot)
                for of in other_fs:
                    f_scores[of][key] = _subset_pq(km, fold_mask_r[of], fold_gt_tot[of])
                for g, (gm_tr, gt_g_tr) in tr_big_groups.items():
                    g_scores[g][key] = _subset_pq(km, gm_tr, gt_g_tr)

        f_opt_tr = {of: max(f_scores[of].values()) for of in other_fs}
        g_opt_tr = {g: max(g_scores[g].values()) for g in tr_big_groups}

        best_key = (0.0, stage1_best_tau)
        best_rob = -1e9
        for key, gpq in tr_scores.items():
            fr = max(f_opt_tr[of] - f_scores[of][key] for of in other_fs)
            gr = max((g_opt_tr[g] - g_scores[g][key] for g in tr_big_groups), default=0.0)
            rob = gpq - 0.06 * gr - 0.10 * fr
            if rob > best_rob:
                best_rob = rob
                best_key = key

        nested_post_choices[f] = (round(float(w_lgb_f), 2), float(best_key[0]), float(best_key[1]))
        u_va = u_s1_r[va_m] + best_key[0] * delta_f[va_m]
        u_nested_r[va_m] = u_va
        nested_keep[va_m] = u_va >= best_key[1]

    # Select full-dataset (best_w_lgb, best_beta, best_tau_s2) with identical LGO+Fold-Regret Robust PQ objective
    best_w_lgb = min(
        w_lgb_grid,
        key=lambda w_l: float(
            np.average(
                (y_res - ((1.0 - w_l) * delta_ridge_oof + w_l * delta_lgb_oof)) ** 2,
                weights=w_res,
            )
        ),
    )
    delta_oof = (1.0 - best_w_lgb) * delta_ridge_oof + best_w_lgb * delta_lgb_oof
    print(
        f"[stage2] Residual OOF wMSE: Ridge={np.average((y_res - delta_ridge_oof) ** 2, weights=w_res):.6f} | "
        f"LGB={np.average((y_res - delta_lgb_oof) ** 2, weights=w_res):.6f} | "
        f"Blend(w_lgb={best_w_lgb:.2f})={np.average((y_res - delta_oof) ** 2, weights=w_res):.6f}"
    )

    all_tr_scores = {}
    all_f_scores = {f: {} for f in range(NUM_FOLDS)}
    all_g_scores = {g: {} for g in big_groups}
    for b_v in betas:
        u_cand = u_s1_r + b_v * delta_oof
        for t_v in taus_s2:
            key = (float(b_v), float(t_v))
            km = u_cand >= float(t_v)
            all_tr_scores[key] = _subset_pq(km, np.ones(len(inst_rows), dtype=bool), total_gt_all)
            for f in range(NUM_FOLDS):
                all_f_scores[f][key] = _subset_pq(km, fold_mask_r[f], fold_gt_tot[f])
            for g in big_groups:
                all_g_scores[g][key] = _subset_pq(km, group_mask_r[g], group_gt_tot[g])

    f_opt_all = {f: max(all_f_scores[f].values()) for f in range(NUM_FOLDS)}
    g_opt_all = {g: max(all_g_scores[g].values()) for g in big_groups}
    best_post_key = max(
        all_tr_scores.keys(),
        key=lambda k: all_tr_scores[k]
        - 0.06 * max(g_opt_all[g] - all_g_scores[g][k] for g in big_groups)
        - 0.10 * max(f_opt_all[f] - all_f_scores[f][k] for f in range(NUM_FOLDS)),
    )

    def _reconstruct_oof_survivors(keep_vec: np.ndarray, u_vec: np.ndarray) -> Dict[str, List[dict]]:
        def _recon_one(s: str) -> Tuple[str, List[dict]]:
            det_s = oof_details_by_stem[s]
            surv_idx = [int(d["cand_idx"]) for d in det_s if bool(keep_vec[d["row_idx"]])]
            if not surv_idx:
                return s, []
            u_tie_s = [float(r["u_hat"]) for r in oof_infer_recs[s]]
            for d in det_s:
                u_tie_s[int(d["cand_idx"])] = float(u_vec[d["row_idx"]])
            rles_s = reconstruct_disjoint_stage2_image(
                s,
                oof_infer_recs[s],
                oof_crop_probs_acc[s],
                alpha_s2=best_alpha,
                t_pix=best_tpix,
                use_join3=True,
                active_indices=surv_idx,
                protect_strong_tau=stage1_best_tau,
                u_tie_scores=u_tie_s,
            )
            return s, rles_s

        out_map: Dict[str, List[dict]] = {}
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            for s, rles_s in ex.map(_recon_one, stems):
                out_map[s] = rles_s
        return out_map

    # Exact survivor-only evaluation of Nested LOFO vs Confident Stage-2 baseline!
    nested_oof_preds = _reconstruct_oof_survivors(nested_keep, u_nested_r)
    post_nested_eval = evaluate_pq(nested_oof_preds, gt_cache)
    post_nested_pq = float(post_nested_eval["pq"])
    post_nested_sq = float(post_nested_eval["sq"])
    post_nested_rq = float(post_nested_eval["rq"])

    if post_nested_pq < raw_nested_pq:
        print(
            f"[stage2] Post-calibration Nested PQ ({post_nested_pq:.4f}) < confident baseline ({raw_nested_pq:.4f}); "
            f"locking onto leak-free Confident Stage-2 threshold ({stage1_best_tau:.4f})."
        )
        best_w_lgb = 0.0
        best_post_key = (0.0, stage1_best_tau)
        nested_pq = raw_nested_pq
        nested_sq = float(nested_iou / max(nested_tp, 1))
        nested_rq = float(nested_tp / max(nested_den, 1e-9))
    else:
        nested_pq, nested_sq, nested_rq = post_nested_pq, post_nested_sq, post_nested_rq

    best_beta, best_tau_s2 = best_post_key
    u_final_r = u_s1_r + best_beta * delta_oof
    final_keep = u_final_r >= best_tau_s2
    final_oof_preds = _reconstruct_oof_survivors(final_keep, u_final_r)
    final_eval = evaluate_pq(final_oof_preds, gt_cache)

    fin_tp = int(final_eval["tp"])
    fin_fp = int(final_eval["fp"])
    fin_fn = int(final_eval["fn"])
    fin_iou = float(final_eval["iou_sum"])
    r_sp = 0.75
    sparse_75_pq = (r_sp * fin_iou) / (r_sp * fin_tp + 0.5 * fin_fp + 0.5 * (1.0 - r_sp) * fin_tp + 0.5 * r_sp * fin_fn)

    final_oof_res = {
        "pq": float(final_eval["pq"]),
        "sq": float(final_eval["sq"]),
        "rq": float(final_eval["rq"]),
        "pq_sparse_r075": float(sparse_75_pq),
        "tp": fin_tp,
        "fp": fin_fp,
        "fn": fin_fn,
        "many_to_one": int(final_eval["many_to_one"]),
        "one_to_many": int(final_eval["one_to_many"]),
        "kept_instances": int(sum(len(v) for v in final_oof_preds.values())),
    }
    print(
        f"\n=== [STAGE-2 4-ARCH ORIENTED REFINER + 19F POST-CALIBRATOR COMPLETE] ===\n"
        f"  Selected (alpha_s2={best_alpha:.2f}, t_pix={best_tpix:.2f}, w_lgb={best_w_lgb:.2f}, beta_res={best_beta:.2f}, tau_s2={best_tau_s2:.4f})\n"
        f"  Final OOF PQ      = {final_oof_res['pq']:.4f} (SQ={final_oof_res['sq']:.4f}, RQ={final_oof_res['rq']:.4f}, TP={fin_tp}, FP={fin_fp}, M2O={final_oof_res['many_to_one']})\n"
        f"  Nested LOFO PQ    = {nested_pq:.4f} (SQ={nested_sq:.4f}, RQ={nested_rq:.4f})\n"
        f"  Sparse PQ(r=0.75) = {sparse_75_pq:.4f} | Nested fold choices = {nested_post_choices}"
    )

    # Generate final survivor-only pixel-disjoint test submission.csv!
    sub_rows = []
    for s in test_stems:
        det_t = reconstruct_disjoint_stage2_image(
            s,
            test_infer_recs[s],
            test_crop_probs_acc[s],
            alpha_s2=best_alpha,
            t_pix=best_tpix,
            use_join3=True,
            crop_stds=test_crop_stds[s],
            return_details=True,
            protect_strong_tau=stage1_best_tau,
        )
        attach_yolo_agreement_to_details(det_t, test_yolo.get(s, []))
        kept_rles = []
        if det_t:
            X_t = np.array([[d[c] for c in S2_RESIDUAL_COLS] for d in det_t], dtype=np.float64)
            delta_ridge_t = np.mean([reg.predict(sc.transform(X_t)) for sc, reg, _ in fold_res_models], axis=0)
            delta_lgb_t = np.mean(
                [lgb_m.predict(X_t) for _, _, lgb_list in fold_res_models for lgb_m in lgb_list],
                axis=0,
            )
            delta_t = (1.0 - best_w_lgb) * delta_ridge_t + best_w_lgb * delta_lgb_t
            surv_idx_t = []
            u_tie_t = [float(r["u_hat"]) for r in test_infer_recs[s]]
            for idx_d, d in enumerate(det_t):
                u_fin_t = float(d["u_s1"]) + best_beta * float(delta_t[idx_d])
                c_idx = int(d["cand_idx"])
                u_tie_t[c_idx] = u_fin_t
                if u_fin_t >= best_tau_s2:
                    surv_idx_t.append(c_idx)
            if surv_idx_t:
                kept_rles = reconstruct_disjoint_stage2_image(
                    s,
                    test_infer_recs[s],
                    test_crop_probs_acc[s],
                    alpha_s2=best_alpha,
                    t_pix=best_tpix,
                    use_join3=True,
                    active_indices=surv_idx_t,
                    protect_strong_tau=stage1_best_tau,
                    u_tie_scores=u_tie_t,
                )
        for k, rle in enumerate(kept_rles):
            sub_rows.append(f"{s}_{k},{rle['counts']}")

    sub_text = "filament_id,segmentation_rle\n" + "\n".join(sub_rows) + "\n"
    final_sub_path = ROOT_DIR / "submission.csv"
    s2_sub_path = SUBMISSIONS_DIR / "sub_stage2_final.csv"
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    final_sub_path.write_text(sub_text, encoding="ascii")
    s2_sub_path.write_text(sub_text, encoding="ascii")

    summary = {
        "stage2_architectures": spec_names,
        "ckpt_subdir": ckpt_subdir,
        "families": families,
        "best_alpha_s2": best_alpha,
        "best_t_pix": best_tpix,
        "best_w_lgb": best_w_lgb,
        "best_beta_res": best_beta,
        "best_tau_s2": best_tau_s2,
        "stage1_pool_oof": s1_pool_res,
        "stage1_strict_oof": s1_base_res,
        "stage2_confident_nested_pq": raw_nested_pq,
        "stage2_oof": final_oof_res,
        "stage2_nested_oof": {"pq": nested_pq, "sq": nested_sq, "rq": nested_rq},
        "submission_path": str(final_sub_path),
        "total_test_rows": len(sub_rows),
        "mean_test_instances_per_img": round(len(sub_rows) / max(len(test_stems), 1), 3),
    }
    (WORK_DIR / "summary_stage2_final.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if save_infer_artifacts:
        with open(WORK_DIR / "stage2_v3_oof_preds.pkl", "wb") as f:
            pickle.dump(final_oof_preds, f, protocol=pickle.HIGHEST_PROTOCOL)
        infer_cfg = {
            "families": families,
            "dense_blend_weights": [float(x) for x in blend_w],
            "gbdt_weights": [float(x) for x in handoff["gbdt_weights"]],
            "stage1_best_tau": float(stage1_best_tau),
            "stage1_handoff_tau": float(handoff_tau),
            "stage2_slots": [
                {
                    "slot_name": sn,
                    "ckpt_subdir": Path(sd_path).name,
                    "base_spec": bs,
                    "weight": float(sw),
                }
                for (sn, sd_path, bs, _), sw in zip(slot_defs, s2_w_mean)
            ],
            "best_alpha_s2": float(best_alpha),
            "best_t_pix": float(best_tpix),
            "best_w_lgb": float(best_w_lgb),
            "best_beta_res": float(best_beta),
            "best_tau_s2": float(best_tau_s2),
            "use_join3": True,
        }
        (WORK_DIR / "stage2_v3_infer_config.json").write_text(json.dumps(infer_cfg, indent=2), encoding="utf-8")
        print(f"[stage2] Saved OOF RLEs -> {WORK_DIR / 'stage2_v3_oof_preds.pkl'} and infer config -> {WORK_DIR / 'stage2_v3_infer_config.json'}")
    print(
        f"[stage2] Wrote {len(sub_rows)} pixel-disjoint test rows "
        f"({len(sub_rows) / max(len(test_stems), 1):.2f}/img) to {final_sub_path}!"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--ckpt-subdir", type=str, default="ckpt_stage2_v3")
    parser.add_argument("--save-infer-artifacts", action="store_true", help="Save exact OOF RLE predictions and Stage-2 inference config.")
    args = parser.parse_args()
    run_stage2_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        ckpt_subdir=args.ckpt_subdir,
        save_infer_artifacts=bool(args.save_infer_artifacts),
    )


if __name__ == "__main__":
    main()
