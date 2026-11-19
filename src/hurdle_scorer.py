import argparse
import json
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

import catboost as cb
import cv2
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from pycocotools import mask as mask_util
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DISK_META_PATH,
    FOLDS_PATH,
    FULL_SIZE,
    NUM_FOLDS,
    PREP_TEST_DIR,
    PREP_TRAIN_DIR,
    RANDOM_SEED,
    SUBMISSIONS_DIR,
    WORK_DIR,
)
from src.hierarchy import (
    FEATURE_NAMES,
    build_subset_monotone_tree,
    extract_tree_features,
    inject_disjoint_yolo_candidates,
    node_to_full_rle,
    solve_tree_antichain_dp,
    solve_two_tier_antichain_handoff,
)
from src.metrics import build_gt_cache, encode_label_map, evaluate_pq
from src.train_stage1_dense import load_prob_pack


def load_blended_maps(
    stem: str,
    families: List[str],
    weights: np.ndarray,
    is_test: bool = False,
    return_fstds: bool = False,
):
    """
    Load and blend (fg, spine, tta_std, disagree, per_family_fgs[, per_family_fstds])
    with exact Law-of-Total-Variance train/test parity across M families and K folds:
      Var_corr = Var_m(p_bar_m) + ((M - 1) / M) * mean_m(fstd_m^2),
    since E[Var_m(p_bar_m)] carries (1/K) of within-family fold variance and E[fstd_m^2]
    carries ((K-1)/K), summing identically to 1.0 * within-family fold variance on OOF.
    """
    prefix = "test" if is_test else "oof"
    fgs, sps, stds, fstds = [], [], [], []
    for fam in families:
        p = WORK_DIR / f"{prefix}_{fam}" / f"{stem}.npz"
        if is_test:
            fg, sp, st, fst = load_prob_pack(p, return_fstd=True)
            fstds.append(fst)
        else:
            fg, sp, st = load_prob_pack(p, return_fstd=False)
        fgs.append(fg)
        sps.append(sp)
        stds.append(st)

    w = weights.astype(np.float32)
    fg_blend = sum(w[i] * fgs[i] for i in range(len(families)))
    sp_blend = sum(w[i] * sps[i] for i in range(len(families)))
    tta_std = sum(w[i] * stds[i] for i in range(len(families)))
    if len(fgs) >= 2:
        raw_var = np.var(np.stack(fgs, axis=0), axis=0).astype(np.float32)
        if is_test and len(fstds) == len(fgs):
            m_cnt = float(len(fgs))
            mean_fstd_sq = sum(fs * fs for fs in fstds) * (1.0 / m_cnt)
            disagree = np.sqrt(raw_var + ((m_cnt - 1.0) / m_cnt) * mean_fstd_sq).astype(np.float32)
        else:
            disagree = np.sqrt(raw_var).astype(np.float32)
    else:
        disagree = tta_std.copy()
    if return_fstds:
        return fg_blend, sp_blend, tta_std, disagree, fgs, (fstds if is_test else None)
    return fg_blend, sp_blend, tta_std, disagree, fgs


def optimize_dense_blend_weights(
    families: List[str],
    gt_cache: dict,
    fold_map: Dict[str, int],
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Find optimal calibrated convex combination weights w >= 0, sum(w) = 1 across Stage-1 dense families
    using the exact scale-balanced objective `20.0 * BCE_disk + 0.40 * SoftDice_disk`, fit strictly
    Leave-One-Fold-Out (`fold != f`) to eliminate even 0.0001 fold leakage!
    """
    if len(families) == 1:
        w1 = np.array([1.0], dtype=np.float32)
        return w1, {f: w1.copy() for f in range(NUM_FOLDS)}

    print(f"[blend] Optimizing Leave-One-Fold-Out calibrated Stage-1 dense weights across families={families}...")
    stems = gt_cache["stems"]
    per_stem_P: Dict[str, np.ndarray] = {}
    per_stem_Y: Dict[str, np.ndarray] = {}
    for s in stems:
        y_img = cv2.imread(str(PREP_TRAIN_DIR / f"{s}_y.png"), cv2.IMREAD_COLOR)[::8, ::8, 0]
        x_ch2 = cv2.imread(str(PREP_TRAIN_DIR / f"{s}_x.png"), cv2.IMREAD_COLOR)[::8, ::8, 2]
        valid = x_ch2 > 0
        per_stem_Y[s] = y_img[valid].astype(np.float32) * (1.0 / 255.0)
        rows = []
        for fam in families:
            fg, _, _ = load_prob_pack(WORK_DIR / f"oof_{fam}" / f"{s}.npz")
            rows.append(fg[::8, ::8][valid])
        per_stem_P[s] = np.vstack(rows)

    def _fit_subset(sub_stems: List[str]) -> np.ndarray:
        P = np.hstack([per_stem_P[s] for s in sub_stems])
        Y = np.concatenate([per_stem_Y[s] for s in sub_stems])

        def loss_fn(w_raw: np.ndarray) -> float:
            w = np.maximum(w_raw, 0.0)
            w = w / max(w.sum(), 1e-8)
            pred = np.clip(w @ P, 1e-6, 1.0 - 1e-6)
            bce = -float(np.mean(Y * np.log(pred) + (1.0 - Y) * np.log(1.0 - pred)))
            inter = float(np.sum(pred * Y))
            dice = 1.0 - (2.0 * inter + 1.0) / (float(np.sum(pred) + np.sum(Y)) + 1.0)
            return 18.0 * bce + 0.45 * dice

        init_w = np.ones(len(families), dtype=np.float64) / len(families)
        lo_b = 0.02 if len(families) >= 4 else 0.10
        hi_b = 0.55 if len(families) >= 4 else 0.70
        bounds = [(lo_b, hi_b) for _ in families]
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(loss_fn, init_w, method="SLSQP", bounds=bounds, constraints=cons)
        bw = np.asarray(res.x, dtype=np.float32)
        bw = np.maximum(bw, 0.0)
        bw /= bw.sum()
        return bw

    fold_weights: Dict[int, np.ndarray] = {}
    for f in range(NUM_FOLDS):
        tr_s = [s for s in stems if fold_map[s] != f]
        fold_weights[f] = _fit_subset(tr_s)

    mean_w = np.mean(np.stack([fold_weights[f] for f in range(NUM_FOLDS)], axis=0), axis=0).astype(np.float32)
    mean_w /= mean_w.sum()
    print(f"[blend] Mean LOFO Stage-1 dense weights: {dict(zip(families, [round(float(x), 4) for x in mean_w]))}")
    return mean_w, fold_weights


def compute_node_ground_truth_targets(
    nodes: List[dict],
    gt_records: List[Tuple[str, List[dict]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute exact multi-annotator hurdle targets for every candidate node in `nodes`:
      - y_hit[c]  = (1 / R_i) * sum_r 1[max_g IoU(c, g) > 0.5]          in [0, 1]
      - y_cond[c] = sum_{r: hit} max_g IoU(c, g) / max(sum_r hit, 1)    in (0.5, 1.0]
      - n_hits[c] = sum_r 1[max_g IoU(c, g) > 0.5]                      in {0..R_i}
      - u_true[c] = (1 / R_i) * sum_r max_g IoU(c, g) * 1[IoU > 0.5]    in [0, 1]
      - R_mult    = float(R_i)
    """
    n = len(nodes)
    R_i = len(gt_records)
    if n == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            float(R_i),
        )

    node_rles = [
        {"size": [FULL_SIZE, FULL_SIZE], "counts": node_to_full_rle(nd)["counts"].encode("ascii")}
        for nd in nodes
    ]
    hit_sum = np.zeros(n, dtype=np.float32)
    iou_hit_sum = np.zeros(n, dtype=np.float32)

    for _, gt_raw in gt_records:
        if not gt_raw:
            continue
        gts = [
            {"size": [FULL_SIZE, FULL_SIZE], "counts": g["counts"].encode("ascii") if isinstance(g["counts"], str) else g["counts"]}
            for g in gt_raw
        ]
        # shape (n_nodes, n_gt)
        iou_mat = np.asarray(mask_util.iou(node_rles, gts, [0] * len(gts)), dtype=np.float32)
        max_iou = iou_mat.max(axis=1)
        hit = max_iou > 0.5
        hit_sum += hit.astype(np.float32)
        iou_hit_sum += np.where(hit, max_iou, 0.0)

    y_hit = hit_sum / float(R_i)
    y_cond = np.where(hit_sum > 0, iou_hit_sum / np.maximum(hit_sum, 1.0), 0.5)
    u_true = iou_hit_sum / float(R_i)
    return y_hit, y_cond, hit_sum, u_true, float(R_i)


def build_all_trees_and_features(
    stems: List[str],
    families: List[str],
    blend_weights: np.ndarray,
    disk_meta_map: dict,
    yolo_dets_map: Dict[str, List[dict]],
    is_test: bool = False,
    gt_cache: Optional[dict] = None,
    fold_map: Optional[Dict[str, int]] = None,
    fold_blend_weights: Optional[Dict[int, np.ndarray]] = None,
    use_yolo_cands: bool = True,
) -> Dict[str, dict]:
    """Build 16-level AND-OR component trees and 77-feature matrices for all stems."""
    prep_dir = PREP_TEST_DIR if is_test else PREP_TRAIN_DIR

    def _work_one(stem: str) -> Tuple[str, dict]:
        x_u8 = cv2.imread(str(prep_dir / f"{stem}_x.png"), cv2.IMREAD_COLOR)
        w_use = (
            fold_blend_weights[fold_map[stem]]
            if (not is_test and fold_map is not None and fold_blend_weights is not None)
            else blend_weights
        )
        fg_b, sp_b, tta_s, dis_s, fam_fgs, fam_fstds = load_blended_maps(
            stem, families, w_use, is_test=is_test, return_fstds=True
        )
        nodes = build_subset_monotone_tree(fg_b, min_area=55)
        y_dets = yolo_dets_map.get(stem, [])
        if use_yolo_cands and y_dets:
            nodes = inject_disjoint_yolo_candidates(
                nodes=nodes,
                yolo_dets=y_dets,
                valid_disk_mask=(x_u8[:, :, 2] > 0),
                min_score=0.10,
                min_area=55,
            )
        feats = extract_tree_features(
            stem=stem,
            nodes=nodes,
            x_u8=x_u8,
            fg_blend=fg_b,
            sp_blend=sp_b,
            tta_std=tta_s,
            disagree=dis_s,
            fam_fgs=fam_fgs,
            disk_meta_entry=disk_meta_map.get(stem, {}),
            yolo_dets=y_dets,
            fam_fstds=fam_fstds,
        )
        compact_nodes = []
        for nd in nodes:
            rle = node_to_full_rle(nd)
            sl = nd["bbox_slice"]
            compact_nodes.append({
                "nid": nd["nid"],
                "parent": nd["parent"],
                "host_parent": nd.get("host_parent", -1),
                "children": nd["children"],
                "yolo_cands": nd.get("yolo_cands", []),
                "yolo_compat_children": nd.get("yolo_compat_children", []),
                "is_yolo_node": nd.get("is_yolo_node", 0.0),
                "birth_tau": nd["birth_tau"],
                "max_tau": nd["max_tau"],
                "persistence_lvls": nd["persistence_lvls"],
                "bbox": (int(sl[0].start), int(sl[0].stop), int(sl[1].start), int(sl[1].stop)),
                "area": nd["area"],
                "rle": rle,
            })

        entry = {"nodes": compact_nodes, "X": feats}
        if not is_test and gt_cache is not None:
            y_hit, y_cond, n_hits, u_true, R_mult = compute_node_ground_truth_targets(
                nodes, gt_cache["records_by_stem"][stem]
            )
            entry["y_hit"] = y_hit
            entry["y_cond"] = y_cond
            entry["n_hits"] = n_hits
            entry["u_true"] = u_true
            entry["R_mult"] = R_mult
        return stem, entry

    out: Dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for idx, (st, ent) in enumerate(ex.map(_work_one, stems), start=1):
            out[st] = ent
            if idx % 150 == 0 or idx == len(stems):
                print(f"  [{'test' if is_test else 'train'} trees] {idx}/{len(stems)} ({time.time() - t0:.1f}s)")
    return out


def _fit_hurdle_pair_for_split(
    X_tr: np.ndarray,
    y_hit_tr: np.ndarray,
    y_cond_tr: np.ndarray,
    n_hits_tr: np.ndarray,
    w_tr: np.ndarray,
    X_es: np.ndarray,
    y_hit_es: np.ndarray,
    y_cond_es: np.ndarray,
    n_hits_es: np.ndarray,
    w_es: np.ndarray,
    seed: int,
) -> dict:
    """
    Fit 3 diverse GBDT hurdle pairs (LightGBM + CatBoost + XGBoost) on X_tr with early stopping
    strictly on inner validation split X_es (zero leakage to outer held-out fold!).
    """
    pos_tr = n_hits_tr > 0
    pos_es = n_hits_es > 0

    # --- 1. LightGBM (Hit: cross_entropy, CondIoU: L2 on positives) ---
    lgb_hit = lgb.LGBMRegressor(
        objective="cross_entropy",
        n_estimators=1400,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=35,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=1.5,
        reg_lambda=4.0,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    lgb_hit.fit(
        X_tr, y_hit_tr, sample_weight=w_tr,
        eval_set=[(X_es, y_hit_es)], eval_sample_weight=[w_es],
        callbacks=[lgb.early_stopping(70, verbose=False)],
    )

    lgb_cond = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.022,
        num_leaves=23,
        min_child_samples=25,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=1.0,
        reg_lambda=4.0,
        random_state=seed + 11,
        n_jobs=4,
        verbose=-1,
    )
    lgb_cond.fit(
        X_tr[pos_tr], y_cond_tr[pos_tr], sample_weight=n_hits_tr[pos_tr],
        eval_set=[(X_es[pos_es], y_cond_es[pos_es])], eval_sample_weight=[n_hits_es[pos_es]],
        callbacks=[lgb.early_stopping(70, verbose=False)],
    )

    # --- 2. CatBoost (Symmetric oblivious trees -> strong structural regularization!) ---
    cb_hit = cb.CatBoostClassifier(
        loss_function="CrossEntropy",
        iterations=1600,
        learning_rate=0.030,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=seed + 23,
        verbose=0,
        thread_count=4,
        allow_writing_files=False,
    )
    cb_hit.fit(
        cb.Pool(X_tr, y_hit_tr, weight=w_tr),
        eval_set=cb.Pool(X_es, y_hit_es, weight=w_es),
        early_stopping_rounds=80,
        verbose=False,
    )

    cb_cond = cb.CatBoostRegressor(
        loss_function="RMSE",
        iterations=1400,
        learning_rate=0.028,
        depth=5,
        l2_leaf_reg=6.0,
        random_seed=seed + 37,
        verbose=0,
        thread_count=4,
        allow_writing_files=False,
    )
    cb_cond.fit(
        cb.Pool(X_tr[pos_tr], y_cond_tr[pos_tr], weight=n_hits_tr[pos_tr]),
        eval_set=cb.Pool(X_es[pos_es], y_cond_es[pos_es], weight=n_hits_es[pos_es]),
        early_stopping_rounds=80,
        verbose=False,
    )

    # --- 3. XGBoost (Depthwise regularized trees with reg:logistic) ---
    xgb_hit = xgb.XGBRegressor(
        objective="reg:logistic",
        eval_metric="logloss",
        n_estimators=1400,
        learning_rate=0.024,
        max_depth=5,
        min_child_weight=20.0,
        subsample=0.80,
        colsample_bytree=0.70,
        reg_alpha=1.5,
        reg_lambda=5.0,
        random_state=seed + 53,
        n_jobs=4,
        tree_method="hist",
        early_stopping_rounds=70,
    )
    xgb_hit.fit(
        X_tr, y_hit_tr, sample_weight=w_tr,
        eval_set=[(X_es, y_hit_es)], sample_weight_eval_set=[w_es],
        verbose=False,
    )

    # Map y_cond in (0.5, 1.0] -> z in (0, 1] for bounded logistic regression
    z_tr = np.clip((y_cond_tr[pos_tr] - 0.5) * 2.0, 1e-4, 1.0 - 1e-4)
    z_es = np.clip((y_cond_es[pos_es] - 0.5) * 2.0, 1e-4, 1.0 - 1e-4)
    xgb_cond = xgb.XGBRegressor(
        objective="reg:logistic",
        eval_metric="logloss",
        n_estimators=1200,
        learning_rate=0.022,
        max_depth=4,
        min_child_weight=15.0,
        subsample=0.80,
        colsample_bytree=0.70,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed + 71,
        n_jobs=4,
        tree_method="hist",
        early_stopping_rounds=70,
    )
    xgb_cond.fit(
        X_tr[pos_tr], z_tr, sample_weight=n_hits_tr[pos_tr],
        eval_set=[(X_es[pos_es], z_es)], sample_weight_eval_set=[n_hits_es[pos_es]],
        verbose=False,
    )

    return {
        "lgb": (lgb_hit, lgb_cond),
        "cb": (cb_hit, cb_cond),
        "xgb": (xgb_hit, xgb_cond),
    }


def _predict_hurdle_bundle(bundle: dict, X: np.ndarray) -> Dict[str, np.ndarray]:
    if len(X) == 0:
        z = np.zeros(0, dtype=np.float32)
        return {"lgb": z, "cb": z, "xgb": z, "p_hit": z, "m_iou": z}

    lgb_hit, lgb_cond = bundle["lgb"]
    h_lgb = np.clip(lgb_hit.predict(X), 0.0, 1.0)
    m_lgb = np.clip(lgb_cond.predict(X), 0.501, 0.998)
    u_lgb = (h_lgb * m_lgb).astype(np.float32)

    cb_hit, cb_cond = bundle["cb"]
    prob_cb = cb_hit.predict(X, prediction_type="Probability")
    h_cb = np.clip(prob_cb[:, 1] if prob_cb.ndim == 2 else prob_cb, 0.0, 1.0)
    m_cb = np.clip(cb_cond.predict(X), 0.501, 0.998)
    u_cb = (h_cb * m_cb).astype(np.float32)

    xgb_hit, xgb_cond = bundle["xgb"]
    h_xgb = np.clip(xgb_hit.predict(X), 0.0, 1.0)
    m_xgb = 0.5 + 0.5 * np.clip(xgb_cond.predict(X), 0.002, 0.996)
    u_xgb = (h_xgb * m_xgb).astype(np.float32)

    p_hit_mean = ((h_lgb + h_cb + h_xgb) * (1.0 / 3.0)).astype(np.float32)
    m_iou_mean = ((m_lgb + m_cb + m_xgb) * (1.0 / 3.0)).astype(np.float32)
    return {"lgb": u_lgb, "cb": u_cb, "xgb": u_xgb, "p_hit": p_hit_mean, "m_iou": m_iou_mean}


def evaluate_dp_predictions(
    tree_bank: Dict[str, dict],
    u_scores_by_stem: Dict[str, np.ndarray],
    tau: float,
    gt_cache: dict,
    eval_stems: Optional[List[str]] = None,
) -> Dict[str, float]:
    stems = eval_stems if eval_stems is not None else gt_cache["stems"]
    preds: Dict[str, List[dict]] = {}
    for s in stems:
        nodes = tree_bank[s]["nodes"]
        sel = solve_tree_antichain_dp(nodes, u_scores_by_stem[s], tau=tau)
        preds[s] = [nodes[idx]["rle"] for idx in sel]
    return evaluate_pq(preds, gt_cache, eval_stems=stems)


def select_minimax_regret_tau(
    tree_bank: Dict[str, dict],
    u_oof: Dict[str, np.ndarray],
    fold_map: Dict[str, int],
    gt_cache: dict,
) -> Tuple[float, dict]:
    """
    Evaluate Dinkelbach Antichain DP across tau in [0.19, 0.33]:
      1. Nested Leave-One-Fold-Out PQ (strictly out-of-sample threshold selection!)
      2. Annotator-Batch Group (prefix[:2]) minimax regret check so tau is robust to LB label strictness!
    """
    taus = np.round(np.arange(0.190, 0.3325, 0.0025), 4)

    # Precompute TP_iou_sum, TP, FP, FN per stem per tau once so fold/group aggregation is instant (<0.2s)!
    stems = gt_cache["stems"]
    per_tau_stem_stats: Dict[float, Dict[str, Tuple[float, int, int, int]]] = {}

    for tau in taus:
        d_stats = {}
        for s in stems:
            nodes = tree_bank[s]["nodes"]
            sel = solve_tree_antichain_dp(nodes, u_oof[s], tau=float(tau))
            preds = [
                {"size": [FULL_SIZE, FULL_SIZE], "counts": nodes[idx]["rle"]["counts"].encode("ascii")}
                for idx in sel
            ]
            n_pred = len(preds)
            iou_s, tp_s, fp_s, fn_s = 0.0, 0, 0, 0
            for _, gt_raw in gt_cache["records_by_stem"][s]:
                n_gt = len(gt_raw)
                if n_gt == 0:
                    fp_s += n_pred
                    continue
                if n_pred == 0:
                    fn_s += n_gt
                    continue
                gts = [{"size": [FULL_SIZE, FULL_SIZE], "counts": g["counts"].encode("ascii")} for g in gt_raw]
                iou_m = np.asarray(mask_util.iou(preds, gts, [0] * n_gt), dtype=np.float64).T
                hit = iou_m > 0.5
                if hit.any():
                    iou_s += float(iou_m[hit].sum())
                    tp_s += int(hit.sum())
                fp_s += int((hit.sum(axis=0) == 0).sum())
                fn_s += int((hit.sum(axis=1) == 0).sum())
            d_stats[s] = (iou_s, tp_s, fp_s, fn_s)
        per_tau_stem_stats[float(tau)] = d_stats

    def _agg_pq(tau_v: float, sub_stems: List[str]) -> float:
        st_map = per_tau_stem_stats[tau_v]
        iou_t = sum(st_map[s][0] for s in sub_stems)
        tp_t = sum(st_map[s][1] for s in sub_stems)
        fp_t = sum(st_map[s][2] for s in sub_stems)
        fn_t = sum(st_map[s][3] for s in sub_stems)
        den = tp_t + 0.5 * fp_t + 0.5 * fn_t
        return float(iou_t / den) if den > 0 else 0.0

    # Group stems by true 4-digit MAGFiLO annotation group GGGG (recs[0][0][:4]) AND by CV fold!
    batch_groups: Dict[str, List[str]] = {}
    for s in stems:
        recs = gt_cache["records_by_stem"][s]
        b_code = recs[0][0].split("-")[0][:4] if recs else "0000"
        batch_groups.setdefault(b_code, []).append(s)

    group_opt = {
        g: max(_agg_pq(float(t), g_stems) for t in taus)
        for g, g_stems in batch_groups.items() if len(g_stems) >= 20
    }
    fold_stems = {f: [s for s in stems if fold_map[s] == f] for f in range(NUM_FOLDS)}
    fold_opt = {f: max(_agg_pq(float(t), f_s) for t in taus) for f, f_s in fold_stems.items()}

    best_obj = -1e9
    best_tau = 0.270
    curve = {}
    for t in taus:
        tf = float(t)
        global_pq = _agg_pq(tf, stems)
        max_group_regret = (
            max(opt_v - _agg_pq(tf, batch_groups[g]) for g, opt_v in group_opt.items())
            if group_opt
            else 0.0
        )
        max_fold_regret = max(opt_v - _agg_pq(tf, fold_stems[f]) for f, opt_v in fold_opt.items())
        # Composite robustness objective across 4-digit MAGFiLO annotation groups and 5 CV folds
        rob_score = global_pq - 0.08 * max_group_regret - 0.12 * max_fold_regret
        curve[str(tf)] = {
            "global_pq": round(global_pq, 5),
            "max_group_regret": round(max_group_regret, 5),
            "max_fold_regret": round(max_fold_regret, 5),
        }
        if rob_score > best_obj:
            best_obj = rob_score
            best_tau = tf

    # Nested Leave-One-Fold-Out verification (using the same LGO+fold-robust objective on the 4 training folds!)
    nested_preds: Dict[str, List[dict]] = {}
    nested_taus = {}
    for f in range(NUM_FOLDS):
        tr_s = [s for s in stems if fold_map[s] != f]
        va_s = [s for s in stems if fold_map[s] == f]
        tr_groups: Dict[str, List[str]] = {}
        for s in tr_s:
            recs = gt_cache["records_by_stem"][s]
            gc = recs[0][0].split("-")[0][:4] if recs else "0000"
            tr_groups.setdefault(gc, []).append(s)
        tr_g_opt = {g: max(_agg_pq(float(t), gs) for t in taus) for g, gs in tr_groups.items() if len(gs) >= 20}
        other_fs = {of: [s for s in tr_s if fold_map[s] == of] for of in range(NUM_FOLDS) if of != f}
        tr_f_opt = {of: max(_agg_pq(float(t), fs) for t in taus) for of, fs in other_fs.items()}

        def _tr_obj(tv: float) -> float:
            gpq = _agg_pq(tv, tr_s)
            gr = max((ov - _agg_pq(tv, tr_groups[g]) for g, ov in tr_g_opt.items()), default=0.0)
            fr = max(ov - _agg_pq(tv, other_fs[of]) for of, ov in tr_f_opt.items())
            return gpq - 0.08 * gr - 0.12 * fr

        f_tau = max(taus, key=lambda t: _tr_obj(float(t)))
        nested_taus[f] = float(f_tau)
        for s in va_s:
            nodes = tree_bank[s]["nodes"]
            sel = solve_tree_antichain_dp(nodes, u_oof[s], tau=float(f_tau))
            nested_preds[s] = [nodes[idx]["rle"] for idx in sel]

    nested_res = evaluate_pq(nested_preds, gt_cache)
    global_res = evaluate_dp_predictions(tree_bank, u_oof, best_tau, gt_cache)
    return best_tau, {
        "best_tau": best_tau,
        "oof_at_best_tau": global_res,
        "nested_oof": nested_res,
        "nested_fold_taus": nested_taus,
    }


def run_hurdle_pipeline(
    families: List[str],
    use_yolo: bool = True,
    use_yolo_cands: bool = False,
    dry_run: bool = False,
    save_models_path: Optional[Path] = None,
) -> dict:
    gt_cache = build_gt_cache()
    with open(FOLDS_PATH, "r", encoding="utf-8") as f:
        fold_map = json.load(f)
    with open(DISK_META_PATH, "r", encoding="utf-8") as f:
        disk_meta = json.load(f)

    oof_yolo, test_yolo = {}, {}
    if use_yolo and (WORK_DIR / "oof_yolo11l_seg.pkl").exists() and (WORK_DIR / "test_yolo11l_seg.pkl").exists():
        with open(WORK_DIR / "oof_yolo11l_seg.pkl", "rb") as f:
            oof_yolo = pickle.load(f)
        with open(WORK_DIR / "test_yolo11l_seg.pkl", "rb") as f:
            test_yolo = pickle.load(f)
        print(
            f"[hurdle] Loaded Family-B YOLO11l-seg evidence ({len(oof_yolo)} OOF, {len(test_yolo)} Test) "
            f"| use_yolo_cands={use_yolo_cands}."
        )
    else:
        print("[hurdle] Running without YOLO evidence (zeroed detector columns).")

    blend_w, fold_blend_w = optimize_dense_blend_weights(families, gt_cache, fold_map)
    stems = gt_cache["stems"]
    test_stems = sorted(fn[:-6] for fn in os.listdir(PREP_TEST_DIR) if fn.endswith("_x.png"))

    print(
        f"[hurdle] Building 16-level Component Trees & {len(FEATURE_NAMES)}-feature matrices "
        f"on 707 Train + 180 Test images..."
    )
    train_bank = build_all_trees_and_features(
        stems,
        families,
        blend_w,
        disk_meta["train"],
        oof_yolo,
        is_test=False,
        gt_cache=gt_cache,
        fold_map=fold_map,
        fold_blend_weights=fold_blend_w,
        use_yolo_cands=use_yolo_cands,
    )
    test_bank = build_all_trees_and_features(
        test_stems,
        families,
        blend_w,
        disk_meta["test"],
        test_yolo,
        is_test=True,
        gt_cache=None,
        use_yolo_cands=use_yolo_cands,
    )

    # Check theoretical Oracle Component-Tree ceiling (using true u_i(c) in Antichain DP!)
    oracle_u = {s: train_bank[s]["u_true"] for s in stems}
    oracle_res = evaluate_dp_predictions(train_bank, oracle_u, tau=0.25, gt_cache=gt_cache)
    total_nodes = sum(len(train_bank[s]["nodes"]) for s in stems)
    n_yolo_nodes = sum(sum(1 for nd in train_bank[s]["nodes"] if nd.get("is_yolo_node", 0.0) > 0.5) for s in stems)
    print(
        f"[hurdle] Component-Tree Bank ready: {total_nodes} total train nodes "
        f"({total_nodes / len(stems):.1f}/img, {n_yolo_nodes} YOLO cands) | "
        f"Tree Oracle PQ Ceiling = {oracle_res['pq']:.4f} "
        f"(SQ={oracle_res['sq']:.4f}, RQ={oracle_res['rq']:.4f})!"
    )

    # Cross-fit 3-GBDT Hurdle Ensemble (5 outer folds x 4 inner early-stopping splits = 20 bundles!)
    pred_keys = ("lgb", "cb", "xgb", "p_hit", "m_iou")
    oof_preds_fam: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in pred_keys}
    test_preds_acc: Dict[str, Dict[str, np.ndarray]] = {
        k: {s: np.zeros(len(test_bank[s]["nodes"]), dtype=np.float64) for s in test_stems}
        for k in pred_keys
    }
    total_bundles = 0
    saved_bundles: List[dict] = []

    for outer_f in range(NUM_FOLDS):
        t_f0 = time.time()
        val_stems = [s for s in stems if fold_map[s] == outer_f]
        other_folds = [f for f in range(NUM_FOLDS) if f != outer_f]

        val_acc = {
            k: {s: np.zeros(len(train_bank[s]["nodes"]), dtype=np.float64) for s in val_stems}
            for k in pred_keys
        }

        for inner_es_f in other_folds:
            tr_s = [s for s in stems if fold_map[s] != outer_f and fold_map[s] != inner_es_f]
            es_s = [s for s in stems if fold_map[s] == inner_es_f]

            X_tr = np.vstack([train_bank[s]["X"] for s in tr_s])
            yh_tr = np.concatenate([train_bank[s]["y_hit"] for s in tr_s])
            yc_tr = np.concatenate([train_bank[s]["y_cond"] for s in tr_s])
            nh_tr = np.concatenate([train_bank[s]["n_hits"] for s in tr_s])
            w_tr = np.concatenate([np.full(len(train_bank[s]["nodes"]), train_bank[s]["R_mult"], dtype=np.float32) for s in tr_s])

            X_es = np.vstack([train_bank[s]["X"] for s in es_s])
            yh_es = np.concatenate([train_bank[s]["y_hit"] for s in es_s])
            yc_es = np.concatenate([train_bank[s]["y_cond"] for s in es_s])
            nh_es = np.concatenate([train_bank[s]["n_hits"] for s in es_s])
            w_es = np.concatenate([np.full(len(train_bank[s]["nodes"]), train_bank[s]["R_mult"], dtype=np.float32) for s in es_s])

            bundle = _fit_hurdle_pair_for_split(
                X_tr, yh_tr, yc_tr, nh_tr, w_tr,
                X_es, yh_es, yc_es, nh_es, w_es,
                seed=RANDOM_SEED + outer_f * 100 + inner_es_f * 10,
            )
            total_bundles += 1
            if save_models_path is not None:
                saved_bundles.append(bundle)

            for s in val_stems:
                pr = _predict_hurdle_bundle(bundle, train_bank[s]["X"])
                for k in pred_keys:
                    val_acc[k][s] += pr[k]

            for s in test_stems:
                pr_t = _predict_hurdle_bundle(bundle, test_bank[s]["X"])
                for k in pred_keys:
                    test_preds_acc[k][s] += pr_t[k]

        for k in pred_keys:
            for s in val_stems:
                oof_preds_fam[k][s] = (val_acc[k][s] / float(len(other_folds))).astype(np.float32)
        print(f"  [hurdle outer fold {outer_f}] 4 inner GBDT bundles fit in {time.time() - t_f0:.1f}s")

    test_preds_fam = {
        k: {s: (test_preds_acc[k][s] / float(total_bundles)).astype(np.float32) for s in test_stems}
        for k in pred_keys
    }

    for k in ("lgb", "cb", "xgb"):
        res_k = evaluate_dp_predictions(train_bank, oof_preds_fam[k], tau=0.255, gt_cache=gt_cache)
        print(f"  [Hurdle {k.upper()} standalone @ tau=0.255] OOF PQ = {res_k['pq']:.4f} (SQ={res_k['sq']:.4f}, RQ={res_k['rq']:.4f})")

    # Fit GBDT convex blend strictly Leave-One-Fold-Out (`fold != f`) for OOF, and mean across folds for Test!
    u_oof_blend: Dict[str, np.ndarray] = {}
    gw_folds = []
    for outer_f in range(NUM_FOLDS):
        tr_s = [s for s in stems if fold_map[s] != outer_f]
        va_s = [s for s in stems if fold_map[s] == outer_f]
        u_tr = np.concatenate([train_bank[s]["u_true"] for s in tr_s])
        w_tr_all = np.concatenate([np.full(len(train_bank[s]["nodes"]), train_bank[s]["R_mult"], dtype=np.float32) for s in tr_s])
        G_tr = np.vstack([np.concatenate([oof_preds_fam[k][s] for s in tr_s]) for k in ("lgb", "cb", "xgb")])
        rel_m = (u_tr > 0.0) | (np.max(G_tr, axis=0) >= 0.08)
        G_rel = G_tr[:, rel_m]
        u_rel = u_tr[rel_m]
        w_rel = w_tr_all[rel_m]
        res_gw = minimize(
            lambda w: float(np.average((w @ G_rel - u_rel) ** 2, weights=w_rel)),
            np.array([0.36, 0.32, 0.32], dtype=np.float64),
            method="SLSQP",
            bounds=[(0.22, 0.48)] * 3,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}],
            options={"ftol": 1e-10, "maxiter": 200},
        )
        gwf = np.asarray(res_gw.x, dtype=np.float32)
        gwf /= gwf.sum()
        gw_folds.append(gwf)
        for s in va_s:
            u_oof_blend[s] = (
                gwf[0] * oof_preds_fam["lgb"][s] + gwf[1] * oof_preds_fam["cb"][s] + gwf[2] * oof_preds_fam["xgb"][s]
            )

    gw = np.mean(np.stack(gw_folds, axis=0), axis=0).astype(np.float32)
    gw /= gw.sum()
    print(f"[hurdle] Mean LOFO GBDT Hurdle Weights: lgb={gw[0]:.3f}, cb={gw[1]:.3f}, xgb={gw[2]:.3f}")

    u_test_blend = {
        s: (gw[0] * test_preds_fam["lgb"][s] + gw[1] * test_preds_fam["cb"][s] + gw[2] * test_preds_fam["xgb"][s])
        for s in test_stems
    }

    best_tau, tau_report = select_minimax_regret_tau(train_bank, u_oof_blend, fold_map, gt_cache)
    print(
        f"\n=== [STAGE-1 DINKELBACH HURDLE DP COMPLETE (use_yolo_cands={use_yolo_cands})] ===\n"
        f"  Selected Minimax-Regret tau* = {best_tau:.4f}\n"
        f"  OOF PQ @ tau*   = {tau_report['oof_at_best_tau']['pq']:.4f} "
        f"(SQ={tau_report['oof_at_best_tau']['sq']:.4f}, RQ={tau_report['oof_at_best_tau']['rq']:.4f})\n"
        f"  Nested LOFO PQ  = {tau_report['nested_oof']['pq']:.4f} "
        f"(SQ={tau_report['nested_oof']['sq']:.4f}, RQ={tau_report['nested_oof']['rq']:.4f})"
    )

    if dry_run:
        return {
            "families": families,
            "use_yolo_cands": use_yolo_cands,
            "dense_blend_weights": blend_w.tolist(),
            "gbdt_weights": gw.tolist(),
            "best_tau": best_tau,
            "tau_report": tau_report,
            "oracle_tree_pq": oracle_res,
        }

    # Persist hierarchy-consistent Two-Tier Antichain Handoff (`sel_strict @ best_tau` + disjoint `sel_rescue @ handoff_tau`)
    # so confident filaments NEVER get over-split by a lower split penalty, while Stage-2 still receives disjoint rescue seeds!
    handoff_tau = round(min(float(best_tau) - 0.025, 0.2275), 4)
    stage1_oof_selected: Dict[str, List[dict]] = {}
    for s in stems:
        nodes = train_bank[s]["nodes"]
        sel = solve_two_tier_antichain_handoff(nodes, u_oof_blend[s], strict_tau=best_tau, rescue_tau=handoff_tau)
        stage1_oof_selected[s] = [
            {
                "rle": nodes[idx]["rle"],
                "u_hat": float(u_oof_blend[s][idx]),
                "p_hit": float(oof_preds_fam["p_hit"][s][idx]),
                "m_iou": float(oof_preds_fam["m_iou"][s][idx]),
                "area": nodes[idx]["area"],
            }
            for idx in sel
        ]

    stage1_test_selected: Dict[str, List[dict]] = {}
    sub_rows = []
    for s in test_stems:
        nodes = test_bank[s]["nodes"]
        sel_ho = solve_two_tier_antichain_handoff(nodes, u_test_blend[s], strict_tau=best_tau, rescue_tau=handoff_tau)
        stage1_test_selected[s] = [
            {
                "rle": nodes[idx]["rle"],
                "u_hat": float(u_test_blend[s][idx]),
                "p_hit": float(test_preds_fam["p_hit"][s][idx]),
                "m_iou": float(test_preds_fam["m_iou"][s][idx]),
                "area": nodes[idx]["area"],
            }
            for idx in sel_ho
        ]
        sel_sub = solve_tree_antichain_dp(nodes, u_test_blend[s], tau=best_tau)
        for k, idx in enumerate(sel_sub):
            sub_rows.append(f"{s}_{k},{nodes[idx]['rle']['counts']}")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_dp_path = SUBMISSIONS_DIR / "sub_stage1_dp.csv"
    sub_dp_path.write_text("filament_id,segmentation_rle\n" + "\n".join(sub_rows) + "\n", encoding="ascii")

    handoff = {
        "families": families,
        "use_yolo_cands": use_yolo_cands,
        "dense_blend_weights": blend_w.tolist(),
        "fold_dense_blend_weights": {int(k): v.tolist() for k, v in fold_blend_w.items()},
        "gbdt_weights": gw.tolist(),
        "best_tau": best_tau,
        "handoff_tau": handoff_tau,
        "tau_report": tau_report,
        "oof_selected": stage1_oof_selected,
        "test_selected": stage1_test_selected,
    }
    with open(WORK_DIR / "stage1_dp_handoff.pkl", "wb") as f:
        pickle.dump(handoff, f, protocol=pickle.HIGHEST_PROTOCOL)
    if save_models_path is not None:
        save_models_path.parent.mkdir(parents=True, exist_ok=True)
        model_pack = {
            "families": families,
            "use_yolo": use_yolo,
            "use_yolo_cands": use_yolo_cands,
            "feature_names": list(FEATURE_NAMES),
            "dense_blend_weights": blend_w.tolist(),
            "gbdt_weights": gw.tolist(),
            "best_tau": float(best_tau),
            "handoff_tau": float(handoff_tau),
            "bundles": saved_bundles,
        }
        with open(save_models_path, "wb") as f:
            pickle.dump(model_pack, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[hurdle] Exported {len(saved_bundles)} GBDT hurdle bundles ({len(saved_bundles) * 6} models) -> {save_models_path}")
    (WORK_DIR / "summary_stage1_dp.json").write_text(
        json.dumps(
            {
                "families": families,
                "use_yolo_cands": use_yolo_cands,
                "dense_blend_weights": blend_w.tolist(),
                "gbdt_weights": gw.tolist(),
                "best_tau": best_tau,
                "handoff_tau": handoff_tau,
                "tau_report": tau_report,
                "oracle_tree_pq": oracle_res,
                "sub_path": str(sub_dp_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return handoff


def main():
    default_fams = ["cnx_s", "effv2_s", "upp_r34d", "manet_eb4", "upp_cnx_t"]
    if not all((WORK_DIR / f"oof_{fm}").exists() for fm in default_fams):
        default_fams = ["cnx_s", "effv2_s", "upp_r34d"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=default_fams)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--no-yolo-cands", action="store_true", help="Deprecated alias (default is now False).")
    parser.add_argument("--with-yolo-cands", action="store_true", help="Enable direct YOLO mask node injection.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-models", type=str, default=None, help="Optional path to save fitted 120 GBDT hurdle models.")
    args = parser.parse_args()
    run_hurdle_pipeline(
        families=args.families,
        use_yolo=not args.no_yolo,
        use_yolo_cands=bool(args.with_yolo_cands),
        dry_run=args.dry_run,
        save_models_path=Path(args.save_models) if args.save_models else None,
    )


if __name__ == "__main__":
    main()
