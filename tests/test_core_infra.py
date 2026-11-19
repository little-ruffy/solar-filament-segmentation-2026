import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def reference_host_pq(gt_df: pd.DataFrame, pred_df: pd.DataFrame, height: int = 2048, width: int = 2048) -> float:
    """Exact copy of Self_Evaluation_Notebook PyTorch PQ@2048 logic for parity verification."""
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0] if len(pred_df) > 0 else pd.Series(dtype=str)
    annotator_images = gt_annotator_image_ids.unique()

    tp_iou_scores = []
    fp_count = 0
    fn_count = 0
    for ann_img in annotator_images:
        image_id = ann_img.split("-", maxsplit=1)[1]
        gt_rles = gt_df.loc[gt_annotator_image_ids == ann_img, "segmentation_rle"].tolist()
        pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist() if len(pred_df) > 0 else []
        n_gt, n_pred = len(gt_rles), len(pred_rles)
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue

        gt_m = mask_util.decode([{"size": [height, width], "counts": r} for r in gt_rles]).transpose(2, 0, 1).astype(np.float32)
        pr_m = mask_util.decode([{"size": [height, width], "counts": r} for r in pred_rles]).transpose(2, 0, 1).astype(np.float32)
        gt_t = torch.from_numpy(gt_m).reshape(n_gt, height * width)
        pr_t = torch.from_numpy(pr_m).reshape(n_pred, height * width)
        inter = torch.matmul(gt_t, pr_t.t())
        union = gt_t.sum(dim=1).view(-1, 1) + pr_t.sum(dim=1).view(1, -1) - inter
        iou_mat = torch.where(union == 0, torch.tensor(0.0), inter / union)
        hit = iou_mat > 0.5
        tp_iou_scores.extend(iou_mat[hit].tolist())
        fp_count += int((hit.sum(dim=0) == 0).sum().item())
        fn_count += int((hit.sum(dim=1) == 0).sum().item())

    tp_count = len(tp_iou_scores)
    denom = tp_count + 0.5 * fp_count + 0.5 * fn_count
    return float(sum(tp_iou_scores) / denom) if denom > 0 else 0.0


def test_metrics_parity_and_human_ceiling():
    from src.metrics import build_gt_cache, encode_binary_mask, evaluate_pq, compute_inter_annotator_pq

    gt_cache = build_gt_cache()
    assert len(gt_cache["stems"]) == 707
    assert gt_cache["total_records"] == 1154
    assert gt_cache["total_annotations"] == 8199

    # Verify inter-annotator human-vs-human PQ matches known ~0.340 benchmark
    human_res = compute_inter_annotator_pq(gt_cache)
    assert 0.335 <= human_res["pq"] <= 0.345, f"Unexpected human PQ: {human_res}"

    # Verify exact parity against reference PyTorch host evaluator on 8 multi-annotator stems
    multi_stems = [s for s in gt_cache["stems"] if len(gt_cache["records_by_stem"][s]) >= 2][:8]
    gt_rows = []
    pred_rows = []
    rng = np.random.default_rng(42)
    for stem in multi_stems:
        recs = gt_cache["records_by_stem"][stem]
        for rec_id, rles in recs:
            for idx, rle in enumerate(rles):
                gt_rows.append({"filament_id": f"{rec_id}_{idx}", "segmentation_rle": rle["counts"]})
        # Construct pixel-disjoint perturbed predictions from first annotator record
        canvas = np.zeros((2048, 2048), dtype=np.int32)
        first_rles = recs[0][1]
        for k, rle in enumerate(first_rles, start=1):
            m = mask_util.decode({"size": [2048, 2048], "counts": rle["counts"].encode("ascii")})
            shift_y = int(rng.integers(-2, 3))
            m_shift = np.roll(m, shift_y, axis=0)
            canvas[(m_shift > 0) & (canvas == 0)] = k
        for k in range(1, len(first_rles) + 1):
            bin_m = (canvas == k).astype(np.uint8)
            if bin_m.sum() > 0:
                pred_rows.append({"filament_id": f"{stem}_{k - 1}", "segmentation_rle": encode_binary_mask(bin_m)})

    gt_df = pd.DataFrame(gt_rows)
    pred_df = pd.DataFrame(pred_rows)
    ref_pq = reference_host_pq(gt_df, pred_df)
    pred_by_stem = {}
    for row in pred_rows:
        st, _ = row["filament_id"].rsplit("_", 1)
        pred_by_stem.setdefault(st, []).append({"size": [2048, 2048], "counts": row["segmentation_rle"]})
    fast_res = evaluate_pq(pred_by_stem, gt_cache, eval_stems=multi_stems)
    assert abs(fast_res["pq"] - ref_pq) < 1e-5, f"Fast PQ {fast_res['pq']} != Ref PQ {ref_pq}"
    print(f"[OK] Metric parity verified: fast={fast_res['pq']:.6f} vs host_ref={ref_pq:.6f}; human PQ={human_res['pq']:.4f}")


def test_validator_catches_overlaps_and_accepts_disjoint():
    from src.config import TEST_IMG_DIR, list_stems_and_paths
    from src.metrics import encode_binary_mask
    from src.validate_submission import validate_submission_file

    test_stems = [stem for stem, _ in list_stems_and_paths(TEST_IMG_DIR)]
    assert len(test_stems) == 180

    with tempfile.TemporaryDirectory() as tmpd:
        good_csv = os.path.join(tmpd, "good.csv")
        bad_overlap_csv = os.path.join(tmpd, "bad_overlap.csv")

        m1 = np.zeros((2048, 2048), dtype=np.uint8)
        m1[500:520, 600:650] = 1
        m2_disjoint = np.zeros((2048, 2048), dtype=np.uint8)
        m2_disjoint[700:720, 800:850] = 1
        m2_overlap = np.zeros((2048, 2048), dtype=np.uint8)
        m2_overlap[510:530, 620:670] = 1

        rle1 = encode_binary_mask(m1)
        rle2_ok = encode_binary_mask(m2_disjoint)
        rle2_bad = encode_binary_mask(m2_overlap)

        good_rows = []
        for s in test_stems:
            good_rows.append({"filament_id": f"{s}_0", "segmentation_rle": rle1})
            good_rows.append({"filament_id": f"{s}_1", "segmentation_rle": rle2_ok})
        pd.DataFrame(good_rows).to_csv(good_csv, index=False)

        res_ok = validate_submission_file(good_csv, expected_stems=set(test_stems))
        assert res_ok["valid"] is True, f"Expected valid submission, got {res_ok}"

        bad_rows = list(good_rows)
        bad_rows[1] = {"filament_id": f"{test_stems[0]}_1", "segmentation_rle": rle2_bad}
        pd.DataFrame(bad_rows).to_csv(bad_overlap_csv, index=False)
        res_bad = validate_submission_file(bad_overlap_csv, expected_stems=set(test_stems))
        assert res_bad["valid"] is False and "overlap" in " ".join(res_bad["errors"]).lower()
        print("[OK] 14-check validator verified: accepts clean disjoint RLEs and rejects overlapping masks.")


def test_solar_prep_physics_and_zero_crop_loss():
    import cv2
    from src.config import TRAIN_IMG_DIR, CROP_OFFSET, CROP_SIZE
    from src.metrics import build_gt_cache
    from src.solar_prep import compute_physics_channels, fit_solar_disk_raycast

    gt_cache = build_gt_cache()
    # Check across 30 images that 100.000% of ground-truth pixels fall inside [96:1952, 96:1952]
    outside_pixels = 0
    total_pixels = 0
    for stem in gt_cache["stems"][::24]:
        for _, rles in gt_cache["records_by_stem"][stem]:
            for rle in rles:
                m = mask_util.decode({"size": [2048, 2048], "counts": rle["counts"].encode("ascii")})
                total_pixels += int(m.sum())
                inside = int(m[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE].sum())
                outside_pixels += int(m.sum()) - inside
    assert outside_pixels == 0 and total_pixels > 0, f"Crop lost {outside_pixels}/{total_pixels} GT pixels!"

    sample_fn = sorted(os.listdir(TRAIN_IMG_DIR))[0]
    img = cv2.imread(os.path.join(TRAIN_IMG_DIR, sample_fn), cv2.IMREAD_GRAYSCALE)
    cx, cy, R, rms = fit_solar_disk_raycast(img)
    assert 1015.0 <= cx <= 1032.0 and 1015.0 <= cy <= 1032.0
    assert 895.0 <= R <= 912.0 and rms < 3.5

    ch_u8, meta = compute_physics_channels(img)
    assert ch_u8.shape == (CROP_SIZE, CROP_SIZE, 3) and ch_u8.dtype == np.uint8
    assert np.isfinite(meta["R"]) and ch_u8[:, :, 0].std() > 5.0
    print(f"[OK] Solar physics channels verified: crop retention=100.000% ({total_pixels} px), R={R:.2f}px, rms={rms:.2f}px")


if __name__ == "__main__":
    test_metrics_parity_and_human_ceiling()
    test_validator_catches_overlaps_and_accepts_disjoint()
    test_solar_prep_physics_and_zero_crop_loss()
    print("ALL CORE INFRASTRUCTURE TESTS PASSED.")
