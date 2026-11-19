import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_util

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CROP_OFFSET,
    CROP_SIZE,
    DISK_META_PATH,
    FOLDS_PATH,
    FULL_SIZE,
    GT_CACHE_PATH,
    NUM_FOLDS,
    PREP_TEST_DIR,
    PREP_TRAIN_DIR,
    RANDOM_SEED,
    TEST_IMG_DIR,
    TRAIN_ANN_PATH,
    TRAIN_IMG_DIR,
    WORK_DIR,
    list_stems_and_paths,
    resolve_image_path,
)
from src.metrics import build_gt_cache


def fit_solar_disk_raycast(img: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit solar disk center (cx, cy) and radius R on a 2048x2048 GONG H-alpha image
    via 720-ray limb gradient detection + 4-step IRLS algebraic circle fit.
    Immune to clouds and intensity vignetting where Otsu thresholding fails.
    """
    h, w = img.shape
    cx0, cy0 = w * 0.5, h * 0.5
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False, dtype=np.float32)
    rs = np.arange(760, 960, dtype=np.float32)

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    xs = np.clip(np.round(cx0 + np.outer(cos_a, rs)).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(cy0 + np.outer(sin_a, rs)).astype(np.int32), 0, h - 1)

    profs = img[ys, xs].astype(np.float32)
    # Smooth radial profile slightly (delta=2) to suppress JPEG noise at the limb
    grad = profs[:, 4:] - profs[:, :-4]
    idx = np.argmin(grad, axis=1) + 2
    r_limb = rs[idx]

    px = cx0 + cos_a * r_limb
    py = cy0 + sin_a * r_limb
    keep = np.ones(len(angles), dtype=bool)
    cx, cy, R = float(cx0), float(cy0), 902.8

    for _ in range(4):
        if int(keep.sum()) < 36:
            break
        p = np.column_stack([px[keep], py[keep]]).astype(np.float64)
        A = np.column_stack([2.0 * p[:, 0], 2.0 * p[:, 1], np.ones(len(p), dtype=np.float64)])
        b = (p ** 2).sum(axis=1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx_new, cy_new = float(sol[0]), float(sol[1])
        R_new = float(math.sqrt(max(sol[2] + cx_new * cx_new + cy_new * cy_new, 1.0)))
        resid = np.abs(np.hypot(px - cx_new, py - cy_new) - R_new)
        med = float(np.median(resid[keep]))
        keep = resid < max(2.5 * med, 1.8)
        cx, cy, R = cx_new, cy_new, R_new

    if not (1005.0 <= cx <= 1042.0 and 1005.0 <= cy <= 1042.0 and 885.0 <= R <= 920.0):
        cx, cy, R = 1023.0, 1024.0, 902.85
        rms = 9.99
    else:
        rms = float(np.sqrt(np.mean((np.hypot(px[keep] - cx, py[keep] - cy) - R) ** 2)))
    return cx, cy, R, rms


def compute_physics_channels(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute 3 isotropic, D4-equivariant astrophysical channels on the native 1:1
    centered 1856x1856 solar disk window [96:1952, 96:1952]:
      Ch 0: Flat-field normalized H-alpha intensity F(x) = 1 + c(x) (CLV + 2-pass cloud flat-fielded)
      Ch 1: Multi-scale chromospheric absorption DoG contrast (highlights 4..32px dark ribbons)
      Ch 2: Solar foreshortening mu(x) = sqrt(max(0, 1 - (r/R)^2))
    Returned as uint8 [1856, 1856, 3] for lossless 0.25% precision & <1ms disk I/O.
    """
    cx, cy, R, rms = fit_solar_disk_raycast(img)

    # Coordinate grid directly on the centered 1856x1856 native window
    coords = np.arange(CROP_OFFSET, CROP_OFFSET + CROP_SIZE, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dx = xx - np.float32(cx)
    dy = yy - np.float32(cy)
    r2 = dx * dx + dy * dy
    inv_R2 = np.float32(1.0 / (R * R))
    u = r2 * inv_R2  # u = (r / R)^2 in [0, ~1.05]

    crop_f = img[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE].astype(np.float32)
    disk_valid = u <= 1.004
    inner_valid = u <= 0.992

    # Step 1: 256 equal-u annuli median Center-to-Limb Variation (CLV) profile L(u)
    n_bins = 256
    u_max = 1.008
    bin_idx = np.clip((u[disk_valid] * (n_bins / u_max)).astype(np.int32), 0, n_bins - 1)
    vals = crop_f[disk_valid]

    order = np.argsort(bin_idx, kind="mergesort")
    sorted_bins = bin_idx[order]
    sorted_vals = vals[order]
    splits = np.flatnonzero(np.diff(sorted_bins)) + 1
    groups = np.split(sorted_vals, splits)
    uniq_bins = sorted_bins[np.r_[0, splits]]

    prof = np.full(n_bins, np.nan, dtype=np.float32)
    for b_id, g_vals in zip(uniq_bins, groups):
        if len(g_vals) >= 16:
            prof[b_id] = float(np.median(g_vals))

    # Interpolate any sparse outer bins and smooth 1D profile mildly
    valid_b = np.flatnonzero(np.isfinite(prof))
    if len(valid_b) >= 4:
        all_b = np.arange(n_bins, dtype=np.float32)
        prof = np.interp(all_b, valid_b.astype(np.float32), prof[valid_b]).astype(np.float32)
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
        prof_pad = np.pad(prof, 2, mode="edge")
        prof = np.convolve(prof_pad, kernel, mode="valid").astype(np.float32)
    else:
        prof[:] = max(float(np.median(vals)), 1.0)

    u_grid_centers = (np.arange(n_bins, dtype=np.float32) + 0.5) * (u_max / n_bins)
    L_map = np.interp(u.ravel(), u_grid_centers, prof).reshape(CROP_SIZE, CROP_SIZE).astype(np.float32)
    I_clv = np.ones((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    I_clv[disk_valid] = crop_f[disk_valid] / np.maximum(L_map[disk_valid], 6.0)

    # Step 2: 2-Pass Feature-Masked 2D Gaussian Background B(x) (removes cloud/vignetting gradients)
    ds = 8
    small_sz = CROP_SIZE // ds  # 232 x 232
    w_mask = inner_valid.astype(np.float32)
    I_small = cv2.resize(I_clv * w_mask, (small_sz, small_sz), interpolation=cv2.INTER_AREA)
    W_small = cv2.resize(w_mask, (small_sz, small_sz), interpolation=cv2.INTER_AREA)
    sigma_s = float(0.10 * R / ds)  # ~11.3 px at 1/8 scale (~90 px on disk)

    num1 = cv2.GaussianBlur(I_small, (0, 0), sigmaX=sigma_s, sigmaY=sigma_s, borderType=cv2.BORDER_REPLICATE)
    den1 = cv2.GaussianBlur(W_small, (0, 0), sigmaX=sigma_s, sigmaY=sigma_s, borderType=cv2.BORDER_REPLICATE)
    B0_small = np.where(den1 > 0.05, num1 / np.maximum(den1, 1e-4), 1.0)

    # Mask strong dark filaments & bright plages before Pass 2
    I_unweighted_s = np.where(W_small > 0.2, I_small / np.maximum(W_small, 1e-4), B0_small)
    dev = np.abs(I_unweighted_s - B0_small)
    in_s = W_small > 0.5
    mad = float(np.median(dev[in_s])) if in_s.any() else 0.02
    quiet_s = (dev <= max(2.5 * 1.4826 * mad, 0.035)) & (W_small > 0.2)
    W2_small = W_small * quiet_s.astype(np.float32)
    I2_small = I_unweighted_s * W2_small

    num2 = cv2.GaussianBlur(I2_small, (0, 0), sigmaX=sigma_s, sigmaY=sigma_s, borderType=cv2.BORDER_REPLICATE)
    den2 = cv2.GaussianBlur(W2_small, (0, 0), sigmaX=sigma_s, sigmaY=sigma_s, borderType=cv2.BORDER_REPLICATE)
    B_small = np.where(den2 > 0.05, num2 / np.maximum(den2, 1e-4), B0_small)
    B_full = cv2.resize(B_small, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)

    # Calibrated physical contrast c(x) in [-0.45, +0.25]
    c = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    c[disk_valid] = np.clip(I_clv[disk_valid] / np.maximum(B_full[disk_valid], 0.25) - 1.0, -0.45, 0.25)

    # Multi-scale isotropic absorption enhancement (positive on dark filaments of width 4..32 px)
    c_blur16 = cv2.GaussianBlur(c, (0, 0), sigmaX=14.0, sigmaY=14.0, borderType=cv2.BORDER_CONSTANT)
    c_blur5 = cv2.GaussianBlur(c, (0, 0), sigmaX=4.5, sigmaY=4.5, borderType=cv2.BORDER_CONSTANT)
    c_fine = cv2.GaussianBlur(c, (0, 0), sigmaX=0.8, sigmaY=0.8, borderType=cv2.BORDER_CONSTANT)
    dog = (c_blur16 - c_fine) + 0.5 * (c_blur5 - c_fine)

    ch0 = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    ch1 = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    ch2 = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)

    ch0[disk_valid] = np.clip(np.round((c[disk_valid] + 0.45) * (255.0 / 0.70)), 0, 255).astype(np.uint8)
    ch1[disk_valid] = np.clip(np.round((dog[disk_valid] + 0.08) * (255.0 / 0.48)), 0, 255).astype(np.uint8)
    mu = np.sqrt(np.maximum(0.0, 1.0 - u))
    ch2[u <= 1.0] = np.clip(np.round(mu[u <= 1.0] * 255.0), 0, 255).astype(np.uint8)

    out = np.stack([ch0, ch1, ch2], axis=-1)
    cloud_grad = float(np.percentile(B_small[in_s], 95) - np.percentile(B_small[in_s], 5)) if in_s.any() else 0.0
    meta = {
        "cx": round(cx, 3),
        "cy": round(cy, 3),
        "R": round(R, 3),
        "rms": round(rms, 3),
        "cloud_grad": round(cloud_grad, 4),
        "disk_median_raw": round(float(np.median(vals)), 2),
    }
    return out, meta


def render_soft_targets_and_consolidated_yolo(
    raw_anns_records: List[Tuple[str, List[dict]]],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Given the R_i annotator records for a physical training observation:
      1. Soft annotator-mean foreground target p(x) in [0, 255] on [1856, 1856]
      2. Soft annotator-mean spine centerline target s(x) in [0, 255] on [1856, 1856]
      3. Cross-annotator IoU-consolidated instance contours in [0, 1] normalized [1856, 1856] coords
         for Family-B YOLO11l-seg training (eliminates duplicate multi-annotator overlapping boxes!).
    """
    R_mult = len(raw_anns_records)
    fg_acc = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    sp_acc = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)

    all_rles = []
    all_rec_idx = []

    for r_idx, (_, anns) in enumerate(raw_anns_records):
        rec_fg = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
        rec_sp = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        for a in anns:
            rle = mask_util.merge(mask_util.frPyObjects(a["segmentation"], FULL_SIZE, FULL_SIZE))
            m = mask_util.decode(rle)
            rec_fg = np.maximum(rec_fg, m)
            all_rles.append(rle)
            all_rec_idx.append(r_idx)

            sp = a.get("spine")
            if sp and len(sp) >= 4:
                pts = np.asarray(sp, dtype=np.float32).reshape(-1, 2) - np.float32(CROP_OFFSET)
                pts_i = np.round(pts).astype(np.int32)
                cv2.polylines(rec_sp, [pts_i], isClosed=False, color=255, thickness=3, lineType=cv2.LINE_AA)

        fg_acc += rec_fg[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE].astype(np.float32)
        sp_acc += (rec_sp.astype(np.float32) / 255.0)

    fg_u8 = np.clip(np.round((fg_acc / max(R_mult, 1)) * 255.0), 0, 255).astype(np.uint8)
    sp_blur = cv2.GaussianBlur(sp_acc / max(R_mult, 1), (0, 0), sigmaX=1.0, sigmaY=1.0)
    sp_u8 = np.clip(np.round(np.minimum(sp_blur * 1.15, 1.0) * 255.0), 0, 255).astype(np.uint8)

    # Consolidate multi-annotator instances for Family-B detector training via IoU >= 0.35 components
    yolo_polys: List[np.ndarray] = []
    n_inst = len(all_rles)
    if n_inst > 0:
        if R_mult == 1:
            clusters = [[k] for k in range(n_inst)]
        else:
            iou_mat = np.asarray(mask_util.iou(all_rles, all_rles, [0] * n_inst), dtype=np.float32)
            visited = [False] * n_inst
            clusters = []
            for k in range(n_inst):
                if visited[k]:
                    continue
                comp = []
                stack = [k]
                visited[k] = True
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    for nb in range(n_inst):
                        if not visited[nb] and all_rec_idx[nb] != all_rec_idx[cur] and iou_mat[cur, nb] >= 0.35:
                            visited[nb] = True
                            stack.append(nb)
                clusters.append(comp)

        for comp in clusters:
            # Keep if single-annotator image, or >=2 annotators agree, or strong 1-of-2 annotator instance
            if R_mult == 3 and len(comp) == 1:
                # Check area; very tiny 1-of-3 noise skipped, substantial filaments kept if area >= 450
                if mask_util.area(all_rles[comp[0]]) < 450:
                    continue
            stack_m = mask_util.decode([all_rles[idx] for idx in comp])  # (2048, 2048, m)
            crop_m = stack_m[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :]
            if len(comp) == 1:
                cons = crop_m[:, :, 0] > 0
            elif len(comp) == 2:
                # Majority (intersection) if IoU high, else union clipped to soft>=0.5
                inter = (crop_m.sum(axis=2) >= 2)
                cons = inter if inter.sum() >= 64 else (crop_m.sum(axis=2) >= 1)
            else:
                cons = crop_m.sum(axis=2) >= 2

            cons_u8 = cons.astype(np.uint8)
            if cons_u8.sum() < 25:
                continue
            contours, _ = cv2.findContours(cons_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            # Take largest external contour (or merge close parts via convex/approx polygon)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            cnt = contours[0]
            if len(cnt) < 3 or cv2.contourArea(cnt) < 16.0:
                continue
            eps = max(0.75, 0.0015 * cv2.arcLength(cnt, True))
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
            if len(approx) >= 3:
                norm_pts = np.clip(approx / float(CROP_SIZE), 0.0, 1.0)
                yolo_polys.append(norm_pts)

    return fg_u8, sp_u8, yolo_polys


def _process_train_worker(args: Tuple[str, Path, List[Tuple[str, List[dict]]], Path]) -> Tuple[str, dict, int]:
    stem, img_path, raw_anns_records, out_dir = args
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read train image: {img_path}")
    ch_u8, meta = compute_physics_channels(img)
    fg_u8, sp_u8, yolo_polys = render_soft_targets_and_consolidated_yolo(raw_anns_records)

    # Save lossless PNG (fast zlib level 1) for fast cv2 decoding + YOLO compatibility, and targets PNG
    png_fast = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    cv2.imwrite(str(out_dir / f"{stem}_x.png"), ch_u8, png_fast)
    target_pack = np.stack([fg_u8, sp_u8, np.zeros_like(fg_u8)], axis=-1)
    cv2.imwrite(str(out_dir / f"{stem}_y.png"), target_pack, png_fast)

    lbl_lines = []
    for poly in yolo_polys:
        coords_str = " ".join(f"{v:.5f}" for v in poly.ravel())
        lbl_lines.append(f"0 {coords_str}")
    (out_dir / f"{stem}.txt").write_text("\n".join(lbl_lines), encoding="ascii")
    return stem, meta, len(yolo_polys)


def _process_test_worker(args: Tuple[str, Path, Path]) -> Tuple[str, dict]:
    stem, img_path, out_dir = args
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read test image: {img_path}")
    ch_u8, meta = compute_physics_channels(img)
    cv2.imwrite(str(out_dir / f"{stem}_x.png"), ch_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return stem, meta


def create_stratified_folds(gt_cache: dict, num_folds: int = NUM_FOLDS, seed: int = RANDOM_SEED) -> Dict[str, int]:
    """
    Create 5 balanced, leak-free observation-grouped folds stratified by:
      (station_code, annotator_multiplicity_R, filament_count_tertile, year_half)
    Guarantees 0 shared physical observations across folds.
    """
    stems = sorted(gt_cache["stems"])
    mean_counts = []
    for s in stems:
        recs = gt_cache["records_by_stem"][s]
        mean_counts.append(float(np.mean([len(rles) for _, rles in recs])))
    q33, q66 = np.percentile(mean_counts, [33.3, 66.7])

    strata = defaultdict(list)
    for s, mc in zip(stems, mean_counts):
        station = s[-2:]
        R_mult = len(gt_cache["records_by_stem"][s])
        cnt_bin = 0 if mc <= q33 else (1 if mc <= q66 else 2)
        yr_bin = 0 if int(s[:4]) <= 2015 else 1
        key = (station, R_mult, cnt_bin, yr_bin)
        strata[key].append(s)

    rng = np.random.default_rng(seed)
    fold_of_stem: Dict[str, int] = {}
    fold_sizes = [0] * num_folds

    # Sort strata largest-first and assign in snake/min-load order for exact fold balance
    for key in sorted(strata.keys(), key=lambda k: (-len(strata[k]), k)):
        group_stems = list(strata[key])
        rng.shuffle(group_stems)
        for s in group_stems:
            best_f = int(np.argmin(fold_sizes))
            fold_of_stem[s] = best_f
            fold_sizes[best_f] += 1

    return fold_of_stem


def run_full_preparation(num_workers: int = 8) -> None:
    t0 = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    PREP_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    PREP_TEST_DIR.mkdir(parents=True, exist_ok=True)

    print("[solar_prep] Building ground-truth RLE cache...")
    gt_cache = build_gt_cache()
    print(
        f"[solar_prep] GT cache ready: {len(gt_cache['stems'])} stems, "
        f"{gt_cache['total_records']} records, {gt_cache['total_annotations']} annotations."
    )

    fold_map = create_stratified_folds(gt_cache)
    with open(FOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(fold_map, f, indent=2, sort_keys=True)

    fold_counts = defaultdict(int)
    for st, fl in fold_map.items():
        fold_counts[fl] += 1
    print(f"[solar_prep] Saved 5-fold stratified split to {FOLDS_PATH}: {dict(sorted(fold_counts.items()))}")

    train_tasks = [
        (stem, resolve_image_path(TRAIN_IMG_DIR, stem), gt_cache["raw_anns_by_stem"][stem], PREP_TRAIN_DIR)
        for stem in gt_cache["stems"]
    ]
    test_tasks = [
        (stem, img_p, PREP_TEST_DIR)
        for stem, img_p in list_stems_and_paths(TEST_IMG_DIR)
    ]

    disk_meta = {"train": {}, "test": {}}
    total_yolo = 0

    print(f"[solar_prep] Processing {len(train_tasks)} train + {len(test_tasks)} test images with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for idx, (stem, meta, n_yolo) in enumerate(ex.map(_process_train_worker, train_tasks, chunksize=8), start=1):
            disk_meta["train"][stem] = meta
            total_yolo += n_yolo
            if idx % 150 == 0 or idx == len(train_tasks):
                print(f"  train prepared: {idx}/{len(train_tasks)} ({time.time() - t0:.1f}s)")

        for idx, (stem, meta) in enumerate(ex.map(_process_test_worker, test_tasks, chunksize=8), start=1):
            disk_meta["test"][stem] = meta
            if idx % 60 == 0 or idx == len(test_tasks):
                print(f"  test prepared:  {idx}/{len(test_tasks)} ({time.time() - t0:.1f}s)")

    with open(DISK_META_PATH, "w", encoding="utf-8") as f:
        json.dump(disk_meta, f, indent=2)

    print(
        f"[solar_prep] DONE in {time.time() - t0:.1f}s! "
        f"Consolidated YOLO instances across 707 train images: {total_yolo} (mean {total_yolo / 707:.2f}/img)."
    )


if __name__ == "__main__":
    workers = min(12, max(4, (os.cpu_count() or 8) - 2))
    run_full_preparation(num_workers=workers)
