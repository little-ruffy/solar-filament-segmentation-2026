"""
End-to-End Zero-Retraining Inference Pipeline for Solar Filament Segmentation Challenge 2026 (v3).

Takes raw 2048x2048 GONG H-alpha JPEG observations from `--test-dir` and reproduces the exact
strictly pixel-disjoint 2048x2048 COCO RLE `submission.csv` using ONLY saved model weights:
  1. Astrophysical H-alpha preprocessing (`compute_physics_channels`: 720-ray IRLS disk fit,
     256-annuli equal-u CLV normalization, 2-pass feature-masked cloud flat-fielding, multi-scale DoG).
  2. Stage-1 Native 1856x1856 5-Family Dense Dual-Head Ensemble (25 models x 8-view D4 TTA = 200 views/px)
     with exact Law-of-Total-Variance `fstd` inter-family disagreement restoration.
  3. Stage-1 Orthogonal 5-Fold `YOLO11l-seg` @ 1536 Instance Detector (10 consolidated views/image).
  4. Stage-1.5 16-Level Subset-Monotone Component Tree + 77-Feature 120-Model Hurdle GBDT
     (LightGBM + CatBoost + XGBoost) + Hierarchy-Consistent Two-Tier Antichain DP.
  5. Stage-2 40-Model Multi-Generation PCA-Oriented 384x128 5-Channel Conditional Refiner
     (8 architecture slots x 5 folds x 4-flip TTA = 160 views/filament) + `join3` MST gap bridging
     + survivor-only strictly pixel-disjoint `2048x2048` COCO RLE serialization.
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CROP_OFFSET,
    CROP_SIZE,
    FULL_SIZE,
    NUM_FOLDS,
    TEST_IMG_DIR,
    WORK_DIR,
    list_stems_and_paths,
)
from src.hierarchy import (
    build_subset_monotone_tree,
    extract_tree_features,
    solve_two_tier_antichain_handoff,
)
from src.hurdle_scorer import _predict_hurdle_bundle
from src.solar_prep import compute_physics_channels
from src.stage2_refiner import (
    build_stage2_model,
    compute_oriented_crop_affine,
    extract_5ch_crop,
    predict_crop_4tta,
    reconstruct_disjoint_stage2_image,
)
from src.train_stage1_dense import build_stage1_model, predict_full_disk
from src.train_stage1_yolo import consolidate_view_detections, extract_yolo_view_instances
from src.validate_submission import validate_submission_file


def _quantize_prob_pack(
    fg: np.ndarray,
    sp: np.ndarray,
    std: np.ndarray,
    fstd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match exact uint16 1/1000 probability pack quantization used during training/validation."""
    fg_q = np.clip(np.round(fg * 1000.0), 0, 1000).astype(np.float32) * 0.001
    sp_q = np.clip(np.round(sp * 1000.0), 0, 1000).astype(np.float32) * 0.001
    std_q = np.clip(np.round(std * 1000.0), 0, 1000).astype(np.float32) * 0.001
    fst_q = np.clip(np.round(fstd * 1000.0), 0, 1000).astype(np.float32) * 0.001
    return fg_q, sp_q, std_q, fst_q


def blend_stage1_family_maps(
    fam_packs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Blend M Stage-1 families using exact Law-of-Total-Variance train/test parity:
      Var_corr = Var_m(p_bar_m) + ((M - 1) / M) * mean_m(fstd_m^2).
    """
    w = np.asarray(weights, dtype=np.float32)
    w = w / max(float(w.sum()), 1e-8)

    fgs = [p[0] for p in fam_packs]
    sps = [p[1] for p in fam_packs]
    stds = [p[2] for p in fam_packs]
    fstds = [p[3] for p in fam_packs]

    fg_blend = np.zeros_like(fgs[0], dtype=np.float32)
    sp_blend = np.zeros_like(sps[0], dtype=np.float32)
    std_blend = np.zeros_like(stds[0], dtype=np.float32)
    for wi, fg, sp, st in zip(w, fgs, sps, stds):
        fg_blend += wi * fg
        sp_blend += wi * sp
        std_blend += wi * st

    m_cnt = float(len(fgs))
    if len(fgs) > 1:
        raw_var = np.var(np.stack(fgs, axis=0), axis=0).astype(np.float32)
        mean_fvar = np.mean([fs * fs for fs in fstds], axis=0).astype(np.float32)
        disagree = np.sqrt(np.maximum(raw_var + ((m_cnt - 1.0) / m_cnt) * mean_fvar, 0.0)).astype(np.float32)
    else:
        disagree = np.zeros_like(fg_blend)

    return fg_blend, sp_blend, std_blend, disagree, fgs, fstds


def build_stage2_crops_in_memory(
    x_u8: np.ndarray,
    fg_b: np.ndarray,
    sp_b: np.ndarray,
    sel_list: List[dict],
    stage1_best_tau: float,
) -> List[dict]:
    """Construct rotated 384x128 5-channel Stage-2 crops and local ROIs directly in memory."""
    if not sel_list:
        return []

    ff_u8 = x_u8[:, :, 0]
    ff_norm = ff_u8.astype(np.float32) * (1.0 / 255.0)
    valid_full = x_u8[:, :, 2] > 0

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

    recs: List[dict] = []
    for k in range(len(sel_list)):
        seed_m = dec_1856[:, :, k] > 0
        other_m = (strong_cov - (dec_1856[:, :, k].astype(np.int16) if strong_flags[k] else 0)) > 0
        M_fwd, M_inv_local, roi_box = compute_oriented_crop_affine(seed_m)
        rx0, ry0, rx1, ry1 = roi_box
        x_5ch = extract_5ch_crop(ff_norm, fg_b, sp_b, seed_m, other_m, M_fwd)
        u_k = float(sel_list[k]["u_hat"])
        recs.append({
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
    return recs


def run_end_to_end_inference(
    test_dir: Path = TEST_IMG_DIR,
    weights_dir: Path = WORK_DIR,
    out_csv: Optional[Path] = ROOT_DIR / "submission_reproduced.csv",
    stems_filter: Optional[Sequence[str]] = None,
    device_str: Optional[str] = None,
    return_intermediates: bool = False,
) -> dict:
    """
    Execute the complete v3 zero-retraining inference pipeline on raw GONG H-alpha JPEG images.

    Runs stage-by-stage with strict VRAM cleanup between model families so peak GPU memory
    stays below 8 GB on consumer GPUs (and works on CPU fallback as well).
    """
    t_total0 = time.time()
    weights_dir = Path(weights_dir).resolve()
    test_dir = Path(test_dir).resolve()

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.88)

    infer_cfg = json.loads((weights_dir / "stage2_v3_infer_config.json").read_text(encoding="utf-8"))
    with open(weights_dir / "hurdle_v3_models.pkl", "rb") as f:
        hurdle_pack = pickle.load(f)

    families: List[str] = list(infer_cfg["families"])
    dense_weights = np.asarray(infer_cfg["dense_blend_weights"], dtype=np.float32)
    gbdt_weights = np.asarray(infer_cfg["gbdt_weights"], dtype=np.float32)
    strict_tau = float(infer_cfg["stage1_best_tau"])
    handoff_tau = float(infer_cfg["stage1_handoff_tau"])
    stage2_slots = list(infer_cfg["stage2_slots"])
    alpha_s2 = float(infer_cfg["best_alpha_s2"])
    t_pix = float(infer_cfg["best_t_pix"])
    tau_s2 = float(infer_cfg["best_tau_s2"])
    use_join3 = bool(infer_cfg.get("use_join3", True))

    all_pairs = list_stems_and_paths(test_dir)
    if stems_filter is not None:
        wanted = set(stems_filter)
        all_pairs = [(s, p) for s, p in all_pairs if s in wanted]
    if not all_pairs:
        raise RuntimeError(f"No valid H-alpha images found in {test_dir} (filter={stems_filter}).")

    stems = [s for s, _ in all_pairs]
    print(f"[infer] Running complete v3 pipeline on {len(stems)} raw H-alpha image(s) using device={device}...")

    # 1. Astrophysical Preprocessing (pure per-image physics)
    t0 = time.time()
    prep_x: Dict[str, np.ndarray] = {}
    disk_meta: Dict[str, dict] = {}
    for s, p in all_pairs:
        img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise RuntimeError(f"Failed to read H-alpha image: {p}")
        ch_u8, meta = compute_physics_channels(img_gray)
        prep_x[s] = ch_u8
        disk_meta[s] = meta
    print(f"[infer] Stage 0 (Astrophysical CLV + 2-Pass Flat-Fielding): {len(stems)} images in {time.time() - t0:.1f}s")

    # 2. Stage-1 5-Family Dense Whole-Disk Ensemble (25 models x 8-view D4 TTA = 200 views/pixel)
    fam_outputs: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {s: [] for s in stems}
    for fam in families:
        tf0 = time.time()
        models = []
        for fold in range(NUM_FOLDS):
            m, _ = build_stage1_model(fam, pretrained=False)
            ckpt_p = weights_dir / f"ckpt_{fam}" / f"best_fold{fold}.pt"
            st = torch.load(ckpt_p, map_location=device, weights_only=False)
            m.load_state_dict(st["model"])
            m = m.to(device=device)
            if device.type == "cuda":
                m = m.to(memory_format=torch.channels_last)
            m.eval()
            models.append(m)

        for s in stems:
            x_u8 = prep_x[s]
            fgs, sps, stds = [], [], []
            for m in models:
                fg_f, sp_f, std_f = predict_full_disk(m, x_u8, device, use_d4_tta=True)
                fgs.append(fg_f)
                sps.append(sp_f)
                stds.append(std_f)
            fg_ens = np.mean(fgs, axis=0)
            sp_ens = np.mean(sps, axis=0)
            std_ens = np.mean(stds, axis=0)
            fold_std = np.std(fgs, axis=0)
            fam_outputs[s].append(_quantize_prob_pack(fg_ens, sp_ens, std_ens, fold_std))

        del models
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [infer] Stage-1 Dense `{fam}` (5 folds x 8-view D4) finished in {time.time() - tf0:.1f}s")

    # 3. Stage-1 Orthogonal 5-Fold YOLO11l-seg Instance Detector (10 consolidated views/image)
    ty0 = time.time()
    yolo_models = [YOLO(str(weights_dir / "yolo_runs" / f"fold{f}" / "weights" / "best.pt")) for f in range(NUM_FOLDS)]
    yolo_dets_by_stem: Dict[str, List[dict]] = {}
    for s in stems:
        raw_all = []
        for ym in yolo_models:
            raw_all.extend(extract_yolo_view_instances(ym, prep_x[s], imgsz=1536, conf=0.06))
        yolo_dets_by_stem[s] = consolidate_view_detections(raw_all, num_views=2 * NUM_FOLDS, iou_cluster=0.42)
    del yolo_models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[infer] Stage-1 YOLO11l-seg (5 folds x 2-view @1536) finished in {time.time() - ty0:.1f}s")

    # 4. Stage-1.5 Component Tree + 77-Feature 120-Model Hurdle GBDT + Two-Tier Antichain DP
    th0 = time.time()
    bundles = hurdle_pack["bundles"]
    n_bundles = float(len(bundles))
    stage1_selected: Dict[str, List[dict]] = {}
    stage2_recs: Dict[str, List[dict]] = {}
    blended_cache: Dict[str, dict] = {}

    for s in stems:
        fg_b, sp_b, tta_s, dis_s, fam_fgs, fam_fstds = blend_stage1_family_maps(fam_outputs[s], dense_weights)
        nodes = build_subset_monotone_tree(fg_b, min_area=55)
        if not nodes:
            stage1_selected[s] = []
            stage2_recs[s] = []
            continue

        X = extract_tree_features(
            stem=s,
            nodes=nodes,
            x_u8=prep_x[s],
            fg_blend=fg_b,
            sp_blend=sp_b,
            tta_std=tta_s,
            disagree=dis_s,
            fam_fgs=fam_fgs,
            disk_meta_entry=disk_meta[s],
            yolo_dets=yolo_dets_by_stem[s],
            fam_fstds=fam_fstds,
        )

        acc_lgb = np.zeros(len(nodes), dtype=np.float64)
        acc_cb = np.zeros(len(nodes), dtype=np.float64)
        acc_xgb = np.zeros(len(nodes), dtype=np.float64)
        acc_ph = np.zeros(len(nodes), dtype=np.float64)
        acc_mi = np.zeros(len(nodes), dtype=np.float64)
        for b in bundles:
            pr = _predict_hurdle_bundle(b, X)
            acc_lgb += pr["lgb"]
            acc_cb += pr["cb"]
            acc_xgb += pr["xgb"]
            acc_ph += pr["p_hit"]
            acc_mi += pr["m_iou"]

        u_hat = (
            gbdt_weights[0] * (acc_lgb / n_bundles)
            + gbdt_weights[1] * (acc_cb / n_bundles)
            + gbdt_weights[2] * (acc_xgb / n_bundles)
        ).astype(np.float32)
        p_hit = (acc_ph / n_bundles).astype(np.float32)
        m_iou = (acc_mi / n_bundles).astype(np.float32)

        sel_ho = solve_two_tier_antichain_handoff(nodes, u_hat, strict_tau=strict_tau, rescue_tau=handoff_tau)
        sel_list = [
            {
                "rle": nodes[idx]["rle"],
                "u_hat": float(u_hat[idx]),
                "p_hit": float(p_hit[idx]),
                "m_iou": float(m_iou[idx]),
                "area": int(nodes[idx]["area"]),
            }
            for idx in sel_ho
        ]
        stage1_selected[s] = sel_list
        stage2_recs[s] = build_stage2_crops_in_memory(prep_x[s], fg_b, sp_b, sel_list, stage1_best_tau=strict_tau)
        if return_intermediates:
            blended_cache[s] = {
                "x_u8": prep_x[s],
                "fg_blend": fg_b,
                "sp_blend": sp_b,
                "disagree": dis_s,
                "n_tree_nodes": len(nodes),
                "yolo_count": len(yolo_dets_by_stem[s]),
                "stage1_selected": sel_list,
            }
    del fam_outputs
    gc.collect()
    print(
        f"[infer] Stage-1.5 (Component Tree + 120-GBDT Hurdle + Two-Tier DP): "
        f"{sum(len(v) for v in stage2_recs.values())} candidate crops in {time.time() - th0:.1f}s"
    )

    # 5. Stage-2 40-Model Multi-Generation PCA-Oriented 384x128 5-Channel Conditional Refiner
    ts2_0 = time.time()
    crop_probs_blend: Dict[str, List[np.ndarray]] = {
        s: [np.zeros((128, 384), dtype=np.float32) for _ in stage2_recs[s]]
        for s in stems
    }
    inv_f = 1.0 / float(NUM_FOLDS)
    for slot in stage2_slots:
        w_slot = float(slot["weight"])
        base_spec = str(slot["base_spec"])
        ckpt_sub = str(slot["ckpt_subdir"])
        slot_sum: Dict[str, List[np.ndarray]] = {
            s: [np.zeros((128, 384), dtype=np.float32) for _ in stage2_recs[s]]
            for s in stems
        }
        for fold in range(NUM_FOLDS):
            ckpt_p = weights_dir / ckpt_sub / f"{base_spec}_fold{fold}.pt"
            m = build_stage2_model(base_spec, pretrained=False).to(device).eval()
            st = torch.load(ckpt_p, map_location=device, weights_only=False)
            m.load_state_dict(st["model"])
            for s in stems:
                recs = stage2_recs[s]
                if not recs:
                    continue
                xb = torch.from_numpy(np.stack([r["x"].astype(np.float32) for r in recs], axis=0))
                preds = predict_crop_4tta(m, xb, device)
                for k in range(len(recs)):
                    slot_sum[s][k] += preds[k]
            del m
        for s in stems:
            for k in range(len(stage2_recs[s])):
                crop_probs_blend[s][k] += w_slot * (slot_sum[s][k] * inv_f)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[infer] Stage-2 40-Model Oriented Refiner (160 views/filament) finished in {time.time() - ts2_0:.1f}s")

    # 6. Survivor-Only Strictly Pixel-Disjoint Reconstruction & Canonical COCO RLE Export
    pred_rles_by_stem: Dict[str, List[dict]] = {}
    sub_rows: List[str] = []
    for s in stems:
        recs = stage2_recs[s]
        surv_idx = [k for k, r in enumerate(recs) if float(r["u_hat"]) >= tau_s2]
        if surv_idx:
            kept_rles = reconstruct_disjoint_stage2_image(
                s,
                recs,
                crop_probs_blend[s],
                alpha_s2=alpha_s2,
                t_pix=t_pix,
                use_join3=use_join3,
                active_indices=surv_idx,
                protect_strong_tau=strict_tau,
                u_tie_scores=[float(r["u_hat"]) for r in recs],
            )
        else:
            kept_rles = []
        pred_rles_by_stem[s] = kept_rles
        for k, rle in enumerate(kept_rles):
            sub_rows.append(f"{s}_{k},{rle['counts']}")

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("filament_id,segmentation_rle\n" + "\n".join(sub_rows) + "\n", encoding="ascii")
        print(
            f"[infer] Wrote {len(sub_rows)} pixel-disjoint masks across {len(stems)} image(s) "
            f"to {out_csv} in {time.time() - t_total0:.1f}s total."
        )

    result = {
        "stems": stems,
        "total_rows": len(sub_rows),
        "pred_rles_by_stem": pred_rles_by_stem,
        "out_csv": str(out_csv) if out_csv is not None else None,
        "elapsed_sec": round(time.time() - t_total0, 2),
    }
    if return_intermediates:
        result["intermediates"] = blended_cache
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-retraining end-to-end v3 inference for Solar Filament Segmentation 2026.")
    parser.add_argument("--test-dir", type=str, default=str(TEST_IMG_DIR))
    parser.add_argument("--weights-dir", type=str, default=str(WORK_DIR))
    parser.add_argument("--out-csv", type=str, default=str(ROOT_DIR / "submission_reproduced.csv"))
    parser.add_argument("--stems", nargs="*", default=None, help="Optional subset of image stems for fast demo inference.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto-detected if omitted).")
    parser.add_argument("--validate", action="store_true", help="Run 14-check submission validator after inference (full test set).")
    args = parser.parse_args()

    res = run_end_to_end_inference(
        test_dir=Path(args.test_dir),
        weights_dir=Path(args.weights_dir),
        out_csv=Path(args.out_csv),
        stems_filter=args.stems,
        device_str=args.device,
    )
    if args.validate and args.stems is None:
        val_res = validate_submission_file(Path(args.out_csv), test_dir=Path(args.test_dir))
        print(json.dumps(val_res, indent=2))
        if not val_res["valid"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
