"""
Compute paired observation-level bootstrap Delta-PQ 95% CIs across all pipeline stages,
exact binary semantic IoU & Dice distributions, multi-annotator subset head-to-head vs.
human pairwise ceiling, and render upgraded high-clarity ACM figures (`fig2` with 1:1
native Stage-1 vs Stage-2 zoomed morphology insets + real test prediction; `fig3` with
Semantic Dice + Semantic IoU + Matched TP IoU + Human Baseline comparison).
"""

import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

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
    TEST_IMG_DIR,
    TRAIN_IMG_DIR,
    WORK_DIR,
    resolve_image_path,
)
from src.hurdle_scorer import load_blended_maps
from src.metrics import _to_pycoco_rles, build_gt_cache, compute_inter_annotator_pq, evaluate_pq
from src.train_stage1_dense import decode_simple_cc_rles, load_prob_pack
from src.train_stage1_yolo import detections_to_disjoint_rles

FIG_DIR = ROOT_DIR / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_NAVY = "#1B365D"
C_TEAL = "#007A87"
C_AMBER = "#D97706"
C_RUST = "#B91C1C"
C_SLATE = "#475569"
C_GREEN = "#15803D"
C_BLUE = "#2563EB"
C_CYAN = "#06B6D4"


def _merged_rle_and_area(rles: List[dict]) -> Tuple[dict, int]:
    if not rles:
        empty = mask_util.frPyObjects([{"counts": [FULL_SIZE * FULL_SIZE], "size": [FULL_SIZE, FULL_SIZE]}], FULL_SIZE, FULL_SIZE)[0]
        return empty, 0
    pb = _to_pycoco_rles(rles)
    merged = mask_util.merge(pb, intersect=False)
    return merged, int(mask_util.area(merged))


def per_stem_pq_tuples(pred_rles_by_stem: Dict[str, List[dict]], gt_cache: dict) -> np.ndarray:
    """Return array of shape (707, 4): [iou_sum, tp, fp, fn] aligned with gt_cache['stems']."""
    stems = gt_cache["stems"]
    out = np.zeros((len(stems), 4), dtype=np.float64)
    for i, s in enumerate(stems):
        preds = _to_pycoco_rles(pred_rles_by_stem.get(s, []))
        n_pred = len(preds)
        iou_s, tp, fp, fn = 0.0, 0, 0, 0
        for _, g_raw in gt_cache["records_by_stem"][s]:
            gts = _to_pycoco_rles(g_raw)
            n_gt = len(gts)
            if n_gt == 0:
                fp += n_pred
                continue
            if n_pred == 0:
                fn += n_gt
                continue
            m = np.asarray(mask_util.iou(preds, gts, [0] * n_gt), dtype=np.float64).T
            h = m > 0.5
            if h.any():
                iou_s += float(m[h].sum())
                tp += int(h.sum())
            fp += int((h.sum(axis=0) == 0).sum())
            fn += int((h.sum(axis=1) == 0).sum())
        out[i] = [iou_s, tp, fp, fn]
    return out


def paired_bootstrap_delta(arr_a: np.ndarray, arr_b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """Compute paired observation-level bootstrap Delta PQ = PQ(B) - PQ(A) and 95% CI."""
    def _pq(x):
        iou_s, tp, fp, fn = x.sum(axis=0)
        den = tp + 0.5 * fp + 0.5 * fn
        return float(iou_s / den) if den > 0 else 0.0

    pqa = _pq(arr_a)
    pqb = _pq(arr_b)
    diff_pt = pqb - pqa
    rng = np.random.default_rng(seed)
    n = len(arr_a)
    deltas = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[k] = _pq(arr_b[idx]) - _pq(arr_a[idx])
    return {
        "pq_a": pqa,
        "pq_b": pqb,
        "delta": diff_pt,
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_gt_zero": float(np.mean(deltas > 0.0)),
    }


def compute_semantic_iou_and_multi_annot_stats(pred_rles_by_stem: Dict[str, List[dict]], gt_cache: dict) -> dict:
    """Compute exact per-record binary semantic IoU & Dice + multi-annotator R>=2 subset PQ."""
    rec_ious, rec_dices = [], []
    tot_inter, tot_union, tot_p, tot_g = 0, 0, 0, 0
    r23_stems = [s for s in gt_cache["stems"] if len(gt_cache["records_by_stem"][s]) >= 2]
    r23_eval = evaluate_pq(pred_rles_by_stem, gt_cache, eval_stems=r23_stems)

    for s in gt_cache["stems"]:
        pm, ppx = _merged_rle_and_area(pred_rles_by_stem.get(s, []))
        for _, g_raw in gt_cache["records_by_stem"][s]:
            gm, gpx = _merged_rle_and_area(g_raw)
            inter = int(mask_util.area(mask_util.merge([pm, gm], intersect=True))) if (ppx > 0 and gpx > 0) else 0
            union = ppx + gpx - inter
            rec_ious.append(float(inter / union) if union > 0 else 1.0)
            rec_dices.append(float(2.0 * inter / (ppx + gpx)) if (ppx + gpx) > 0 else 1.0)
            tot_inter += inter
            tot_union += union
            tot_p += ppx
            tot_g += gpx

    ri = np.asarray(rec_ious, dtype=np.float64)
    rd = np.asarray(rec_dices, dtype=np.float64)
    return {
        "semantic_micro_iou": float(tot_inter / max(tot_union, 1)),
        "semantic_micro_dice": float(2.0 * tot_inter / max(tot_p + tot_g, 1)),
        "per_record_semantic_iou": {
            "mean": float(ri.mean()), "std": float(ri.std()),
            "q25": float(np.percentile(ri, 25)), "median": float(np.median(ri)), "q75": float(np.percentile(ri, 75)),
        },
        "per_record_semantic_dice": {
            "mean": float(rd.mean()), "std": float(rd.std()),
            "q25": float(np.percentile(rd, 25)), "median": float(np.median(rd)), "q75": float(np.percentile(rd, 75)),
        },
        "multi_annotator_r23_model": {
            "n_stems": len(r23_stems),
            "pq": float(r23_eval["pq"]),
            "sq": float(r23_eval["sq"]),
            "rq": float(r23_eval["rq"]),
        },
        "_rec_ious": ri,
        "_rec_dices": rd,
    }


def find_best_stage2_zoom_examples(
    s1_strict_preds: Dict[str, List[dict]],
    s2_oof_preds: Dict[str, List[dict]],
    gt_cache: dict,
) -> List[dict]:
    """Find real OOF filaments where Stage-2 significantly improved IoU over Stage-1 DP seed."""
    gains = []
    for s in gt_cache["stems"]:
        p1 = _to_pycoco_rles(s1_strict_preds.get(s, []))
        p2 = _to_pycoco_rles(s2_oof_preds.get(s, []))
        if not p1 or not p2:
            continue
        # Use first annotator record for clean visual comparison
        _, g_raw = gt_cache["records_by_stem"][s][0]
        gs = _to_pycoco_rles(g_raw)
        if not gs:
            continue
        m1 = np.asarray(mask_util.iou(p1, gs, [0] * len(gs)), dtype=np.float64)  # (n1, ng)
        m2 = np.asarray(mask_util.iou(p2, gs, [0] * len(gs)), dtype=np.float64)  # (n2, ng)
        for gi in range(len(gs)):
            area_g = int(mask_util.area(gs[gi]))
            if not (650 <= area_g <= 5000):
                continue
            i1 = int(np.argmax(m1[:, gi]))
            i2 = int(np.argmax(m2[:, gi]))
            iou1 = float(m1[i1, gi])
            iou2 = float(m2[i2, gi])
            if 0.35 <= iou1 <= 0.72 and iou2 >= 0.62 and (iou2 - iou1) >= 0.06:
                gains.append({
                    "stem": s,
                    "g_rle": gs[gi],
                    "p1_rle": p1[i1],
                    "p2_rle": p2[i2],
                    "iou1": iou1,
                    "iou2": iou2,
                    "gain": iou2 - iou1,
                    "area": area_g,
                })
    gains.sort(key=lambda x: x["gain"], reverse=True)
    # Pick 2 distinct stems with high gain and clean geometry
    picked = []
    seen_s = set()
    for g in gains:
        if g["stem"] not in seen_s:
            picked.append(g)
            seen_s.add(g["stem"])
            if len(picked) == 2:
                break
    return picked


def make_upgraded_fig2(
    s1_strict_preds: Dict[str, List[dict]],
    s2_oof_preds: Dict[str, List[dict]],
    gt_cache: dict,
    handoff: dict,
    zoom_examples: List[dict],
    out_pdf: Path,
    out_png: Path,
) -> None:
    """
    Upgraded 2-row x 4-column high-clarity qualitative figure:
      Row 0 (Full Disk, 1856^2 -> 512^2):
        (a) Raw GONG H-alpha | (b) Physics Flat-Field (Ch0) | (c) 5-Fam fg+Spine | (d) Stage-2 Disjoint Pred vs GT
      Row 1 (Native 1:1 Zoomed 256x256 Morphology Crops — zero downsampling blur!):
        (e) Stage-1 -> Stage-2 Refinement #1 (exact IoU gain)
        (f) Stage-1 -> Stage-2 Refinement #2 (exact IoU gain)
        (g) Multi-Annotator Dense Region (R=3 consensus vs sibling separation)
        (h) Real Test-Set Prediction from submitted submission.csv (disjoint multi-filament crop)
    """
    families = handoff["families"]
    blend_w = np.asarray(handoff["dense_blend_weights"], dtype=np.float32)

    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0), gridspec_kw={"wspace": 0.10, "hspace": 0.22})

    # --- ROW 0: Full Disk walkthrough on high-activity 20160317232434Lh (PQ=0.74, Dice=0.85) ---
    full_stem = "20160317232434Lh"
    raw_gray = cv2.imread(str(resolve_image_path(TRAIN_IMG_DIR, full_stem)), cv2.IMREAD_GRAYSCALE)
    x_u8 = cv2.imread(str(PREP_TRAIN_DIR / f"{full_stem}_x.png"), cv2.IMREAD_COLOR)
    fg_b, sp_b, _, _, _ = load_blended_maps(full_stem, families, blend_w, is_test=False)

    raw_512 = cv2.resize(raw_gray[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE], (512, 512), interpolation=cv2.INTER_AREA)
    ff_512 = cv2.resize(x_u8[:, :, 0], (512, 512), interpolation=cv2.INTER_AREA)
    fg_512 = cv2.resize(fg_b, (512, 512), interpolation=cv2.INTER_AREA)
    sp_512 = cv2.resize(sp_b, (512, 512), interpolation=cv2.INTER_AREA)

    prob_rgb = np.repeat((ff_512[:, :, None].astype(np.float32) * 0.55), 3, axis=2)
    prob_rgb[:, :, 0] = np.clip(prob_rgb[:, :, 0] + fg_512 * 220.0, 0, 255)
    prob_rgb[:, :, 1] = np.clip(prob_rgb[:, :, 1] + fg_512 * 90.0 + sp_512 * 165.0, 0, 255)
    prob_rgb[:, :, 2] = np.clip(prob_rgb[:, :, 2] + sp_512 * 230.0, 0, 255)

    # Full-disk Stage-2 overlay
    preds_b = _to_pycoco_rles(s2_oof_preds[full_stem])
    recs = gt_cache["records_by_stem"][full_stem]
    tp_flags = [False] * len(preds_b)
    gt_u = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
    for _, g_raw in recs:
        gb = _to_pycoco_rles(g_raw)
        if gb:
            gt_u = np.maximum(gt_u, mask_util.decode(mask_util.merge(gb, intersect=False)))
            if preds_b:
                im = np.asarray(mask_util.iou(preds_b, gb, [0] * len(gb)), dtype=np.float64)
                for pi in range(len(preds_b)):
                    if im[pi].max() > 0.5:
                        tp_flags[pi] = True

    bg_rgb = np.repeat(x_u8[:, :, 0:1], 3, axis=2).astype(np.float32)
    gt_crop = gt_u[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE]
    gt_cnts, _ = cv2.findContours((gt_crop > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dec_p = mask_util.decode(preds_b)
    if dec_p.ndim == 2:
        dec_p = dec_p[:, :, None]
    dec_p = dec_p[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :]
    for pi in range(dec_p.shape[2]):
        pm = dec_p[:, :, pi] > 0
        col = np.array([16.0, 185.0, 129.0], dtype=np.float32) if tp_flags[pi] else np.array([239.0, 68.0, 68.0], dtype=np.float32)
        bg_rgb[pm] = 0.38 * bg_rgb[pm] + 0.62 * col
    ov_u8 = np.clip(bg_rgb, 0, 255).astype(np.uint8)
    cv2.drawContours(ov_u8, gt_cnts, -1, (255, 196, 0), thickness=5)
    ov_512 = cv2.resize(ov_u8, (512, 512), interpolation=cv2.INTER_AREA)

    row0 = [
        (raw_512, "(a) Raw GONG H-α", "gray"),
        (ff_512, "(b) Flat-Field (Ch0)", "gray"),
        (prob_rgb.astype(np.uint8), "(c) 5-Fam fg + Spine", None),
        (ov_512, "(d) Stage-2 vs. GT", None),
    ]
    for c_idx, (im, title, cmap) in enumerate(row0):
        ax = axes[0, c_idx]
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=8.8, fontweight="bold", pad=5)
        ax.set_xticks([]); ax.set_yticks([])

    # --- ROW 1: 4 Native 1:1 Zoomed Crops (260x260 native pixels!) ---
    def render_zoom_refinement(ax, zitem, title):
        st = zitem["stem"]
        x_f = cv2.imread(str(PREP_TRAIN_DIR / f"{st}_x.png"), cv2.IMREAD_COLOR)[:, :, 0]
        mg = mask_util.decode(zitem["g_rle"])[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE] > 0
        m1 = mask_util.decode(zitem["p1_rle"])[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE] > 0
        m2 = mask_util.decode(zitem["p2_rle"])[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE] > 0
        union_m = mg | m1 | m2
        ys, xs = np.nonzero(union_m)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        half = max(130, int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.65))
        y0 = max(0, min(CROP_SIZE - 2 * half, cy - half))
        x0 = max(0, min(CROP_SIZE - 2 * half, cx - half))
        y1, x1 = min(CROP_SIZE, y0 + 2 * half), min(CROP_SIZE, x0 + 2 * half)

        patch = np.repeat(x_f[y0:y1, x0:x1, None], 3, axis=2).astype(np.float32)
        m2_c = m2[y0:y1, x0:x1]
        patch[m2_c] = 0.45 * patch[m2_c] + 0.55 * np.array([16.0, 185.0, 129.0], dtype=np.float32)
        patch_u8 = np.clip(patch, 0, 255).astype(np.uint8)

        c1, _ = cv2.findContours(m1[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cg, _ = cv2.findContours(mg[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(patch_u8, c1, -1, (56, 189, 248), thickness=2)  # Stage-1 cyan contour
        cv2.drawContours(patch_u8, cg, -1, (255, 196, 0), thickness=2)   # GT gold contour
        ax.imshow(patch_u8)
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=5)
        ax.set_xticks([]); ax.set_yticks([])

    render_zoom_refinement(
        axes[1, 0], zoom_examples[0],
        f"(e) S1→S2: {zoom_examples[0]['iou1']:.2f}→{zoom_examples[0]['iou2']:.2f}",
    )
    render_zoom_refinement(
        axes[1, 1], zoom_examples[1],
        f"(f) S1→S2: {zoom_examples[1]['iou1']:.2f}→{zoom_examples[1]['iou2']:.2f}",
    )

    # Panel (g): Dense Multi-Annotator (R=3) Zoomed Crop showing clean disjoint separation
    st_r3 = "20140413195914Mh"
    x_r3 = cv2.imread(str(PREP_TRAIN_DIR / f"{st_r3}_x.png"), cv2.IMREAD_COLOR)[:, :, 0]
    p_r3 = _to_pycoco_rles(s2_oof_preds[st_r3])
    dec_r3 = mask_util.decode(p_r3)[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :]
    # Pick centroid around the dense active cluster (top-left active region)
    ys, xs = np.nonzero(dec_r3.sum(axis=2) > 0)
    cy, cx = int(np.percentile(ys, 45)), int(np.percentile(xs, 35))
    half = 260
    y0, x0 = max(0, min(CROP_SIZE - 2 * half, cy - half)), max(0, min(CROP_SIZE - 2 * half, cx - half))
    y1, x1 = y0 + 2 * half, x0 + 2 * half
    patch_r3 = np.repeat(x_r3[y0:y1, x0:x1, None], 3, axis=2).astype(np.float32)
    tab_cols = [(16, 185, 129), (59, 130, 246), (168, 85, 247), (236, 72, 153), (20, 184, 166), (249, 115, 22)]
    for pi in range(dec_r3.shape[2]):
        pm = dec_r3[y0:y1, x0:x1, pi] > 0
        if pm.any():
            col = np.array(tab_cols[pi % len(tab_cols)], dtype=np.float32)
            patch_r3[pm] = 0.40 * patch_r3[pm] + 0.60 * col
    p_r3_u8 = np.clip(patch_r3, 0, 255).astype(np.uint8)
    for _, g_raw in gt_cache["records_by_stem"][st_r3]:
        gb = _to_pycoco_rles(g_raw)
        if gb:
            gm = mask_util.decode(mask_util.merge(gb, intersect=False))[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE]
            cg, _ = cv2.findContours(gm[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(p_r3_u8, cg, -1, (255, 196, 0), thickness=2)
    axes[1, 2].imshow(p_r3_u8)
    axes[1, 2].set_title("(g) Active Loop (OOF)", fontsize=8.5, fontweight="bold", pad=5)
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])

    # Panel (h): Real Test-Set Prediction directly from submitted submission.csv!
    df_sub = pd.read_csv(ROOT_DIR / "release_artifacts" / "submission.csv")
    df_sub["stem"] = df_sub["filament_id"].str.rsplit("_", n=1).str[0]
    counts_t = df_sub["stem"].value_counts()
    test_stem = counts_t.index[0]  # richest active test disk in submission.csv!
    x_t = cv2.imread(str(PREP_TEST_DIR / f"{test_stem}_x.png"), cv2.IMREAD_COLOR)[:, :, 0]
    t_rles = [{"size": [FULL_SIZE, FULL_SIZE], "counts": c.encode("ascii")} for c in df_sub[df_sub["stem"] == test_stem]["segmentation_rle"]]
    dec_t = mask_util.decode(t_rles)[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :]
    ys, xs = np.nonzero(dec_t.sum(axis=2) > 0)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    half = 280
    y0, x0 = max(0, min(CROP_SIZE - 2 * half, cy - half)), max(0, min(CROP_SIZE - 2 * half, cx - half))
    y1, x1 = y0 + 2 * half, x0 + 2 * half
    patch_t = np.repeat(x_t[y0:y1, x0:x1, None], 3, axis=2).astype(np.float32)
    for pi in range(dec_t.shape[2]):
        pm = dec_t[y0:y1, x0:x1, pi] > 0
        if pm.any():
            col = np.array(tab_cols[pi % len(tab_cols)], dtype=np.float32)
            patch_t[pm] = 0.38 * patch_t[pm] + 0.62 * col
    axes[1, 3].imshow(np.clip(patch_t, 0, 255).astype(np.uint8))
    axes[1, 3].set_title(f"(h) Test CSV (N={len(t_rles)})", fontsize=8.5, fontweight="bold", pad=5)
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])

    for r in range(2):
        for c in range(4):
            for spn in axes[r, c].spines.values():
                spn.set_visible(True); spn.set_linewidth(0.6); spn.set_color("#CBD5E1")

    handles = [
        patches.Patch(facecolor="#10B981", edgecolor="none", label="Stage-2 Refined Mask (TP / Instance)"),
        patches.Patch(facecolor="none", edgecolor="#38BDF8", linewidth=1.8, label="Stage-1 Seed Contour (Before S2)"),
        patches.Patch(facecolor="none", edgecolor="#FFC400", linewidth=1.8, label="Ground-Truth Polygon Contour"),
        patches.Patch(facecolor="#EF4444", edgecolor="none", label="Unmatched Pred (FP)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.50, -0.02), fontsize=7.8)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)


def make_upgraded_fig3(
    rm: dict,
    rec_ious: np.ndarray,
    rec_dices: np.ndarray,
    matched_ious: np.ndarray,
    out_pdf: Path,
    out_png: Path,
) -> None:
    """Upgraded 3-panel evaluation figure including Semantic IoU + Semantic Dice + Matched TP IoU + Human Baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.05), constrained_layout=True)

    # Panel (a): Per-Record Semantic Dice, Per-Record Semantic IoU, & Matched TP IoU Distributions
    ax = axes[0]
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(rec_ious, bins=bins, density=True, alpha=0.42, color=C_BLUE, edgecolor="white", linewidth=0.35, label=f"Sem. IoU (μ={rec_ious.mean():.2f})")
    ax.hist(rec_dices, bins=bins, density=True, alpha=0.48, color=C_TEAL, edgecolor="white", linewidth=0.35, label=f"Sem. Dice (μ={rec_dices.mean():.2f})")
    ax.hist(matched_ious, bins=bins, density=True, alpha=0.58, color=C_AMBER, edgecolor="white", linewidth=0.35, label=f"TP IoU (μ={matched_ious.mean():.2f})")
    ax.axvline(0.5, color=C_RUST, linestyle="--", linewidth=1.1, label="PQ Thr. (0.50)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Score Value", fontsize=9.0)
    ax.set_ylabel("Density", fontsize=9.0)
    ax.set_title("(a) Semantic Dice/IoU & TP IoU", fontsize=9.4, pad=6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.9, handlelength=1.2)

    # Panel (b): Stage-by-Stage Progression with 5-Fold SD Error Bars
    ax = axes[1]
    short_map = {
        "Human Pairwise": "Human Pairwise",
        "UNet Cnx-S (1-Fam)": "UNet Cnx-S",
        "UNet++ Cnx-T (Best)": "UNet++ Cnx-T",
        "YOLO11l-seg @1536": "YOLO11l @1536",
        "Stage-1.5 Hurdle DP": "Stage-1.5 DP",
        "Stage-2 v3 Final": "Stage-2 Final",
    }
    prog = rm["stage_progression"]
    labels_b = [short_map.get(x["stage"], x["stage"]) for x in prog]
    means_b = [x["pq"] for x in prog]
    stds_b = [x["fold_std"] for x in prog]
    colors_b = [C_SLATE, "#64748B", C_AMBER, C_BLUE, C_TEAL, C_GREEN]
    y_pos = np.arange(len(labels_b))
    bars = ax.barh(y_pos, means_b, xerr=stds_b, height=0.58, color=colors_b, alpha=0.92,
                   error_kw={"ecolor": "#0F172A", "elinewidth": 1.0, "capsize": 2.6})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_b, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(0.30, 0.548)
    ax.set_xlabel("5-Fold OOF PQ (±1 SD)", fontsize=9.0)
    ax.set_title("(b) Stage-by-Stage Ablation Lift", fontsize=9.4, pad=6)
    for bar, val, sd in zip(bars, means_b, stds_b):
        ax.text(val + sd + 0.0045, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                va="center", ha="left", fontsize=7.8, fontweight="bold", color="#0F172A")

    # Panel (c): Cross-Observatory & Multiplicity Stability vs. Human Pairwise Ceiling
    ax = axes[2]
    st_keys = sorted(rm["per_station"].keys())
    r_keys = sorted(rm["per_R"].keys(), key=int)
    x_labels = [f"{k}" for k in st_keys] + [f"R={r}" for r in r_keys]
    vals = [rm["per_station"][k]["pq"] for k in st_keys] + [rm["per_R"][r]["pq"] for r in r_keys]
    ci_lo = [vals[i] - (rm["per_station"][k]["pq_ci95"][0] if i < len(st_keys) else rm["per_R"][r_keys[i - len(st_keys)]]["pq_ci95"][0])
             for i, k in enumerate(st_keys + r_keys)]
    ci_hi = [(rm["per_station"][k]["pq_ci95"][1] if i < len(st_keys) else rm["per_R"][r_keys[i - len(st_keys)]]["pq_ci95"][1]) - vals[i]
             for i, k in enumerate(st_keys + r_keys)]
    x_pos = np.arange(len(x_labels))
    bar_cols = [C_NAVY] * len(st_keys) + [C_TEAL] * len(r_keys)
    ax.errorbar(x_pos, vals, yerr=[ci_lo, ci_hi], fmt="o", markersize=4.6,
                ecolor=C_SLATE, elinewidth=1.1, capsize=2.6, color=C_NAVY, zorder=4)
    for xp, yv, col in zip(x_pos, vals, bar_cols):
        ax.plot(xp, yv, "o", markersize=5.0, color=col, zorder=5)
    ax.axvline(len(st_keys) - 0.5, color="#94A3B8", linestyle=":", linewidth=1.1)
    ax.axhline(rm["global"]["pq"], color=C_GREEN, linestyle="--", linewidth=1.05, alpha=0.85, label=f"Global ({rm['global']['pq']:.4f})")
    ax.axhline(rm["human_inter_annotator"]["pq"], color=C_RUST, linestyle="-.", linewidth=1.0, alpha=0.85, label=f"Human ({rm['human_inter_annotator']['pq']:.4f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=7.9, rotation=28, ha="right")
    ax.set_ylim(0.32, 0.53)
    ax.set_ylabel("OOF PQ (95% Boot. CI)", fontsize=9.0)
    ax.set_title("(c) Station & Multiplicity", fontsize=9.4, pad=6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.0)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)


def main() -> None:
    gt_cache = build_gt_cache()
    with open(WORK_DIR / "stage1_dp_handoff.pkl", "rb") as f:
        handoff = pickle.load(f)
    with open(WORK_DIR / "stage2_v3_oof_preds.pkl", "rb") as f:
        s2_oof = pickle.load(f)
    rm = json.loads((WORK_DIR / "report_metrics.json").read_text(encoding="utf-8"))

    # Reconstruct Stage-1 predictions for exact paired bootstrap deltas
    s_upp = json.loads((WORK_DIR / "summary_upp_cnx_t.json").read_text(encoding="utf-8"))
    s_cnx = json.loads((WORK_DIR / "summary_cnx_s.json").read_text(encoding="utf-8"))
    s_yolo = json.loads((WORK_DIR / "summary_yolo11l_seg.json").read_text(encoding="utf-8"))

    def _dec_fam(fam: str, op: list) -> Dict[str, List[dict]]:
        def _one(s: str):
            fg, _, _ = load_prob_pack(WORK_DIR / f"oof_{fam}" / f"{s}.npz")
            return s, decode_simple_cc_rles(fg, t_lo=op[0], min_mean_p=op[1], min_area=op[2])
        out = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for s, r in ex.map(_one, gt_cache["stems"]):
                out[s] = r
        return out

    preds_cnx = _dec_fam("cnx_s", s_cnx["best_cc_op"])
    preds_upp = _dec_fam("upp_cnx_t", s_upp["best_cc_op"])
    with open(WORK_DIR / "oof_yolo11l_seg.pkl", "rb") as f:
        oof_yolo = pickle.load(f)
    preds_yolo = {s: detections_to_disjoint_rles(oof_yolo[s], score_thr=float(s_yolo["best_thr"])) for s in gt_cache["stems"]}
    strict_tau = float(handoff["best_tau"])
    preds_s1dp = {s: [r["rle"] for r in handoff["oof_selected"][s] if float(r["u_hat"]) >= strict_tau] for s in gt_cache["stems"]}

    t_cnx = per_stem_pq_tuples(preds_cnx, gt_cache)
    t_upp = per_stem_pq_tuples(preds_upp, gt_cache)
    t_yolo = per_stem_pq_tuples(preds_yolo, gt_cache)
    t_s1dp = per_stem_pq_tuples(preds_s1dp, gt_cache)
    t_s2 = per_stem_pq_tuples(s2_oof, gt_cache)

    paired = {
        "cnx_to_upp": paired_bootstrap_delta(t_cnx, t_upp),
        "upp_to_yolo": paired_bootstrap_delta(t_upp, t_yolo),
        "upp_to_s1dp": paired_bootstrap_delta(t_upp, t_s1dp),
        "yolo_to_s1dp": paired_bootstrap_delta(t_yolo, t_s1dp),
        "s1dp_to_s2": paired_bootstrap_delta(t_s1dp, t_s2),
    }

    sem_stats = compute_semantic_iou_and_multi_annot_stats(s2_oof, gt_cache)
    # Recompute matched_ious cleanly for fig3
    matched_ious = []
    for s in gt_cache["stems"]:
        pb = _to_pycoco_rles(s2_oof.get(s, []))
        if not pb:
            continue
        for _, g_raw in gt_cache["records_by_stem"][s]:
            gb = _to_pycoco_rles(g_raw)
            if not gb:
                continue
            m = np.asarray(mask_util.iou(pb, gb, [0] * len(gb)), dtype=np.float64)
            h = m > 0.5
            if h.any():
                matched_ious.extend(m[h].tolist())
    matched_ious_arr = np.asarray(matched_ious, dtype=np.float64)

    zooms = find_best_stage2_zoom_examples(preds_s1dp, s2_oof, gt_cache)
    out_extra = {
        "paired_bootstrap": paired,
        "semantic_micro_iou": sem_stats["semantic_micro_iou"],
        "semantic_micro_dice": sem_stats["semantic_micro_dice"],
        "per_record_semantic_iou": sem_stats["per_record_semantic_iou"],
        "per_record_semantic_dice": sem_stats["per_record_semantic_dice"],
        "multi_annotator_r23_model": sem_stats["multi_annotator_r23_model"],
        "zoom_examples": [{"stem": z["stem"], "iou1": z["iou1"], "iou2": z["iou2"], "gain": z["gain"], "area": z["area"]} for z in zooms],
    }
    (WORK_DIR / "report_upgraded_stats.json").write_text(json.dumps(out_extra, indent=2), encoding="utf-8")

    make_upgraded_fig2(
        preds_s1dp, s2_oof, gt_cache, handoff, zooms,
        FIG_DIR / "fig2_qualitative_examples.pdf", FIG_DIR / "fig2_qualitative_examples.png",
    )
    make_upgraded_fig3(
        rm, sem_stats["_rec_ious"], sem_stats["_rec_dices"], matched_ious_arr,
        FIG_DIR / "fig3_evaluation_distributions.pdf", FIG_DIR / "fig3_evaluation_distributions.png",
    )
    print(json.dumps(out_extra, indent=2))


if __name__ == "__main__":
    main()
