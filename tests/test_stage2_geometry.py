import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CROP_OFFSET, CROP_SIZE, FULL_SIZE
from src.metrics import encode_binary_mask
from src.stage2_refiner import (
    S2_H,
    S2_RESIDUAL_COLS,
    S2_W,
    STAGE2_SPECS,
    attach_yolo_agreement_to_details,
    build_stage2_model,
    compute_oriented_crop_affine,
    extract_5ch_crop,
    join3_bridge_fragments,
    reconstruct_disjoint_stage2_image,
)


def test_oriented_affine_roundtrip_and_join3():
    # Create a diagonal 37-degree curved ribbon on 1856x1856
    mask = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    pts = np.array([[600, 700], [660, 745], [730, 795], [800, 845]], dtype=np.int32)
    cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=12)

    M_fwd, M_inv_local, (rx0, ry0, rx1, ry1) = compute_oriented_crop_affine(mask > 0)
    crop = cv2.warpAffine(mask.astype(np.float32), M_fwd, (S2_W, S2_H), flags=cv2.INTER_LINEAR)
    assert crop.shape == (S2_H, S2_W) and crop.max() > 0.95

    x5 = extract_5ch_crop(
        np.full((CROP_SIZE, CROP_SIZE), 0.5, dtype=np.float32),
        mask.astype(np.float32),
        mask.astype(np.float32) * 0.8,
        mask > 0,
        np.zeros((CROP_SIZE, CROP_SIZE), dtype=bool),
        M_fwd,
    )
    assert x5.shape == (5, S2_H, S2_W) and np.isfinite(x5).all() and abs(float(x5[0].max())) < 1e-5

    back_roi = cv2.warpAffine(crop, M_inv_local, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR)
    recon = np.zeros_like(mask)
    recon[ry0:ry1, rx0:rx1] = (back_roi >= 0.5).astype(np.uint8)

    inter = int((mask & recon).sum())
    uni = int((mask | recon).sum())
    iou = inter / max(uni, 1)
    assert iou >= 0.94, f"Affine round-trip IoU too low: {iou:.4f}"

    # Verify join3 bridges a 7-px gap when fg_roi supports the bridge
    broken = recon[ry0:ry1, rx0:rx1].copy()
    mid_y = broken.shape[0] // 2
    broken[mid_y - 3:mid_y + 3, :] = 0  # Cut a 6-px horizontal gap across the diagonal ribbon
    fg_roi = np.where(mask[ry0:ry1, rx0:rx1] > 0, 0.42, 0.01).astype(np.float32)
    bridged = join3_bridge_fragments(broken, fg_roi, mask[ry0:ry1, rx0:rx1] > 0)
    from scipy import ndimage as ndi
    _, n_before = ndi.label(broken)
    _, n_after = ndi.label(bridged)
    assert n_before >= 2 and n_after == 1, f"join3 failed to bridge gap: before={n_before}, after={n_after}"

    # Verify disjoint painting when 2 neighboring crops compete on border pixels
    mask2 = np.roll(mask, 8, axis=0)  # Intentionally overlapping raw region if unguarded!
    m1_full = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
    m2_full = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
    m1_full[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE] = mask
    m2_full[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE] = np.where(mask > 0, 0, mask2)
    M2_fwd, M2_inv, roi2 = compute_oriented_crop_affine(mask2 > 0)
    crop2 = cv2.warpAffine(mask2.astype(np.float32), M2_fwd, (S2_W, S2_H), flags=cv2.INTER_LINEAR)

    recs = [
        {"M_inv_local": M_inv_local, "roi_box": (rx0, ry0, rx1, ry1), "rle_stage1": {"size": [FULL_SIZE, FULL_SIZE], "counts": encode_binary_mask(m1_full)}, "u_hat": 0.7},
        {"M_inv_local": M2_inv, "roi_box": roi2, "rle_stage1": {"size": [FULL_SIZE, FULL_SIZE], "counts": encode_binary_mask(m2_full)}, "u_hat": 0.65},
    ]
    out_rles = reconstruct_disjoint_stage2_image(
        "test", recs, [crop * 0.9, crop2 * 0.85],
        np.maximum(mask, mask2).astype(np.float32) * 0.8,
        np.ones((CROP_SIZE, CROP_SIZE), dtype=bool),
    )
    dec = mask_util.decode([{"size": [FULL_SIZE, FULL_SIZE], "counts": r["counts"].encode("ascii")} for r in out_rles])
    assert int(dec.sum(axis=2).max()) <= 1, "Stage-2 output had overlapping pixels!"

    # Also verify cached ROI path + return_details=True + crop_stds
    for r_idx, r in enumerate(recs):
        bx0, by0, bx1, by1 = r["roi_box"]
        src_m = mask if r_idx == 0 else np.where(mask > 0, 0, mask2)
        r["fg_roi_u16"] = np.clip(np.round(np.maximum(mask, mask2)[by0:by1, bx0:bx1] * 0.8 * 65535.0), 0, 65535).astype(np.uint16)
        r["seed_roi"] = (src_m[by0:by1, bx0:bx1] > 0).copy()
        r["valid_roi"] = np.ones((by1 - by0, bx1 - bx0), dtype=bool)
    det_out = reconstruct_disjoint_stage2_image(
        "test", recs, [crop * 0.9, crop2 * 0.85],
        crop_stds=[np.full_like(crop, 0.04), np.full_like(crop2, 0.05)],
        return_details=True,
    )
    assert len(det_out) == len(out_rles)
    assert all(d["rle"]["counts"] == r["counts"] for d, r in zip(det_out, out_rles))
    assert all(np.isfinite(d["s2_disagree"]) and d["s2_disagree"] > 0 for d in det_out)
    print(f"[OK] Stage-2 affine round-trip IoU={iou:.4f}, join3 MST bridging ({n_before}->1), cached ROI details, and strict disjointness verified!")


def test_survivor_only_painting_and_protect_strong_tau():
    # Candidate 0: Confident real filament [700:724, 600:720] (u_hat=0.72, crop prob=0.80)
    # Candidate 1: Overlapping false-positive [700:724, 660:760] (higher raw crop prob=0.88 on overlap!)
    m0 = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    m0[700:724, 600:720] = 1
    m1_seed = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    m1_seed[700:724, 722:765] = 1  # Disjoint Stage-1 seed right next to m0
    m1_expanded = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    m1_expanded[700:724, 660:765] = 1  # Stage-2 expands m1 over right half of m0!

    M0_fwd, M0_inv, roi0 = compute_oriented_crop_affine(m0 > 0)
    M1_fwd, M1_inv, roi1 = compute_oriented_crop_affine(m1_seed > 0)
    crop0 = cv2.warpAffine(m0.astype(np.float32), M0_fwd, (S2_W, S2_H), flags=cv2.INTER_LINEAR) * 0.80
    crop1 = cv2.warpAffine(m1_expanded.astype(np.float32), M1_fwd, (S2_W, S2_H), flags=cv2.INTER_LINEAR) * 0.88

    fg_full = np.maximum(m0, m1_expanded).astype(np.float32) * 0.78
    ff_u8 = np.where(m0 > 0, 95, 165).astype(np.uint8)
    valid_full = np.ones((CROP_SIZE, CROP_SIZE), dtype=bool)

    def _make_rec(seed_arr: np.ndarray, M_inv: np.ndarray, roi_box: tuple, u_h: float) -> dict:
        bx0, by0, bx1, by1 = roi_box
        f_full = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
        f_full[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE] = seed_arr
        return {
            "M_inv_local": M_inv,
            "roi_box": roi_box,
            "rle_stage1": {"size": [FULL_SIZE, FULL_SIZE], "counts": encode_binary_mask(f_full)},
            "u_hat": u_h,
            "p_hit": min(1.0, u_h / 0.68),
            "m_iou": 0.68,
            "fg_roi_u16": np.clip(np.round(fg_full[by0:by1, bx0:bx1] * 65535.0), 0, 65535).astype(np.uint16),
            "sp_roi_u16": np.clip(np.round(fg_full[by0:by1, bx0:bx1] * 0.85 * 65535.0), 0, 65535).astype(np.uint16),
            "ff_roi_u8": ff_u8[by0:by1, bx0:bx1].copy(),
            "seed_roi": (seed_arr[by0:by1, bx0:bx1] > 0).copy(),
            "valid_roi": valid_full[by0:by1, bx0:bx1].copy(),
        }

    recs_rescue = [
        _make_rec(m0, M0_inv, roi0, 0.72),
        _make_rec(m1_seed, M1_inv, roi1, 0.24),  # Speculative rescue seed < 0.27
    ]

    # 1. Baseline single-survivor area when ONLY cand 0 is active
    only0 = reconstruct_disjoint_stage2_image("test", recs_rescue, [crop0, crop1], active_indices=[0])
    assert len(only0) == 1
    area_0_clean = int(mask_util.area({"size": [FULL_SIZE, FULL_SIZE], "counts": only0[0]["counts"].encode("ascii")}))

    # 2. Unguarded painting (protect_strong_tau=None, active_indices=None): cand 1 steals overlap from cand 0!
    unguarded = reconstruct_disjoint_stage2_image("test", recs_rescue, [crop0, crop1])
    area_0_stolen = int(mask_util.area({"size": [FULL_SIZE, FULL_SIZE], "counts": unguarded[0]["counts"].encode("ascii")}))
    assert area_0_stolen < area_0_clean - 500, f"Expected unguarded overlap theft, got {area_0_stolen} vs {area_0_clean}"

    # 3. With protect_strong_tau=0.27: cand 0 keeps 100% of its pixels AND both remain 100% disjoint, + 19 features finite!
    det_prot = reconstruct_disjoint_stage2_image(
        "test", recs_rescue, [crop0, crop1],
        crop_stds=[np.full_like(crop0, 0.03), np.full_like(crop1, 0.06)],
        return_details=True,
        protect_strong_tau=0.27,
    )
    attach_yolo_agreement_to_details(det_prot, [])
    assert len(det_prot) == 2
    assert len(S2_RESIDUAL_COLS) == 19
    for d in det_prot:
        for col in S2_RESIDUAL_COLS:
            assert col in d and np.isfinite(d[col]), f"Missing or non-finite feature {col}"
    assert det_prot[0]["ff_contrast_drop"] > 0.10, "Expected positive H-alpha dark absorption contrast!"
    area_0_prot = int(mask_util.area({"size": [FULL_SIZE, FULL_SIZE], "counts": det_prot[0]["rle"]["counts"].encode("ascii")}))
    assert area_0_prot == area_0_clean, f"protect_strong_tau failed: {area_0_prot} != {area_0_clean}"
    dec_prot = mask_util.decode([{"size": [FULL_SIZE, FULL_SIZE], "counts": d["rle"]["counts"].encode("ascii")} for d in det_prot])
    assert int(dec_prot.sum(axis=2).max()) <= 1, "Protected Stage-2 output had overlapping pixels!"

    # 4. Even if cand 1 started above protect_strong_tau (u_hat=0.31) and was demoted by Stage-2,
    #    active_indices=[0] survivor-only reconstruction restores 100% of cand 0's pixels!
    recs_both_strong = [
        _make_rec(m0, M0_inv, roi0, 0.72),
        _make_rec(m1_seed, M1_inv, roi1, 0.31),
    ]
    surv_only = reconstruct_disjoint_stage2_image(
        "test", recs_both_strong, [crop0, crop1],
        active_indices=[0], protect_strong_tau=0.27, u_tie_scores=[0.74, 0.21],
    )
    assert len(surv_only) == 1
    area_0_surv = int(mask_util.area({"size": [FULL_SIZE, FULL_SIZE], "counts": surv_only[0]["counts"].encode("ascii")}))
    assert area_0_surv == area_0_clean, f"Survivor-only active_indices failed: {area_0_surv} != {area_0_clean}"
    print(
        f"[OK] Survivor-only painting & protect_strong_tau verified: "
        f"clean={area_0_clean}px vs unguarded_stolen={area_0_stolen}px -> protected={area_0_prot}px, 19 residual features finite!"
    )


if __name__ == "__main__":
    test_oriented_affine_roundtrip_and_join3()
    test_survivor_only_painting_and_protect_strong_tau()
