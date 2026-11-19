"""
Compute exhaustive evaluation metrics with uncertainty measures (5-fold std, 1000-resample
bootstrap 95% CIs, Semantic Dice, Matched IoU/Dice distributions, 1-to-many / many-to-one
relations, GONG station & annotator stratification) and generate publication-grade ACM
figures (`report/figures/fig1_pipeline.pdf`, `fig2_qualitative_examples.pdf`,
`fig3_evaluation_distributions.pdf`) directly from verified Stage-2 v3 OOF predictions.
"""

import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_util

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CROP_OFFSET,
    CROP_SIZE,
    FOLDS_PATH,
    FULL_SIZE,
    GONG_STATIONS,
    NUM_FOLDS,
    PREP_TRAIN_DIR,
    TRAIN_IMG_DIR,
    WORK_DIR,
    resolve_image_path,
)
from src.hurdle_scorer import load_blended_maps
from src.metrics import _to_pycoco_rles, build_gt_cache, compute_inter_annotator_pq, evaluate_pq

FIG_DIR = ROOT_DIR / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Clean academic publication typography (matches ACM Libertine / sans-serif figures)
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

# Colorblind-safe Okabe-Ito / Slate scientific palette
C_NAVY = "#1B365D"
C_TEAL = "#007A87"
C_AMBER = "#D97706"
C_RUST = "#B91C1C"
C_SLATE = "#475569"
C_LIGHT = "#F1F5F9"
C_GREEN = "#15803D"
C_BLUE = "#2563EB"


def _merged_rle_and_area(rles: List[dict]) -> Tuple[dict, int]:
    if not rles:
        empty = mask_util.frPyObjects([{"counts": [FULL_SIZE * FULL_SIZE], "size": [FULL_SIZE, FULL_SIZE]}], FULL_SIZE, FULL_SIZE)[0]
        return empty, 0
    pb = _to_pycoco_rles(rles)
    merged = mask_util.merge(pb, intersect=False)
    return merged, int(mask_util.area(merged))


def evaluate_comprehensive_metrics(
    pred_rles_by_stem: Dict[str, List[dict]],
    gt_cache: dict,
    fold_map: Dict[str, int],
) -> dict:
    """Compute exact PQ/SQ/RQ, Semantic Dice, Matched IoU/Dice distributions, 1:N / N:1, and CIs."""
    stems = gt_cache["stems"]

    per_stem_stats: Dict[str, dict] = {}
    all_matched_ious: List[float] = []
    all_matched_dices: List[float] = []
    per_record_dices: List[float] = []
    per_record_pqs: List[float] = []

    total_inter_px = 0
    total_pred_px = 0
    total_gt_px = 0

    one_to_many_total = 0
    many_to_one_total = 0
    records_with_1tom = 0
    records_with_mto1 = 0

    by_station = defaultdict(list)
    by_R = defaultdict(list)
    by_annotator = defaultdict(list)
    by_fold = defaultdict(list)

    # Area quartile buckets (computed on GT polygon areas)
    gt_area_hits: List[Tuple[int, int, float]] = []  # (gt_area, is_tp, iou_if_tp)

    for s in stems:
        recs = gt_cache["records_by_stem"][s]
        R_mult = len(recs)
        preds_raw = pred_rles_by_stem.get(s, [])
        preds = _to_pycoco_rles(preds_raw)
        n_pred = len(preds)
        pred_merged_rle, pred_px = _merged_rle_and_area(preds_raw)

        s_iou, s_tp, s_fp, s_fn = 0.0, 0, 0, 0
        s_inter_px, s_pred_px_acc, s_gt_px_acc = 0, 0, 0
        s_dices = []

        for rec_id, gt_raw in recs:
            gts = _to_pycoco_rles(gt_raw)
            n_gt = len(gts)
            gt_merged_rle, gt_px = _merged_rle_and_area(gt_raw)
            inter_px = int(mask_util.area(mask_util.merge([pred_merged_rle, gt_merged_rle], intersect=True))) if (pred_px > 0 and gt_px > 0) else 0
            rec_dice = (2.0 * inter_px) / max(pred_px + gt_px, 1) if (pred_px + gt_px) > 0 else 1.0
            per_record_dices.append(float(rec_dice))
            s_dices.append(float(rec_dice))

            total_inter_px += inter_px
            total_pred_px += pred_px
            total_gt_px += gt_px
            s_inter_px += inter_px
            s_pred_px_acc += pred_px
            s_gt_px_acc += gt_px

            gt_areas = [int(mask_util.area(g)) for g in gts]
            if n_gt == 0:
                tp, fp, fn, iou_s, o2m, m2o = 0, n_pred, 0, 0.0, 0, 0
            elif n_pred == 0:
                tp, fp, fn, iou_s, o2m, m2o = 0, 0, n_gt, 0.0, 0, 0
                for ga in gt_areas:
                    gt_area_hits.append((ga, 0, 0.0))
            else:
                iou_mat = np.asarray(mask_util.iou(preds, gts, [0] * n_gt), dtype=np.float64).T
                hit = iou_mat > 0.5
                if hit.any():
                    tp_ious = iou_mat[hit]
                    iou_s = float(tp_ious.sum())
                    tp = int(tp_ious.size)
                    for iv in tp_ious:
                        iv_f = float(iv)
                        all_matched_ious.append(iv_f)
                        all_matched_dices.append(2.0 * iv_f / (1.0 + iv_f))
                else:
                    iou_s, tp = 0.0, 0

                pred_hits = hit.sum(axis=0)
                gt_hits = hit.sum(axis=1)
                fp = int((pred_hits == 0).sum())
                fn = int((gt_hits == 0).sum())
                o2m = int((pred_hits > 1).sum())
                m2o = int((gt_hits > 1).sum())

                for gi, ga in enumerate(gt_areas):
                    if gt_hits[gi] > 0:
                        best_iou = float(iou_mat[gi].max())
                        gt_area_hits.append((ga, 1, best_iou))
                    else:
                        gt_area_hits.append((ga, 0, 0.0))

            one_to_many_total += o2m
            many_to_one_total += m2o
            if o2m > 0:
                records_with_1tom += 1
            if m2o > 0:
                records_with_mto1 += 1

            r_den = tp + 0.5 * fp + 0.5 * fn
            per_record_pqs.append(float(iou_s / r_den) if r_den > 0 else 1.0)

            s_iou += iou_s
            s_tp += tp
            s_fp += fp
            s_fn += fn

            yy = rec_id.split("-")[0][2:4]
            by_annotator[yy].append((iou_s, tp, fp, fn, inter_px, pred_px, gt_px))

        s_den = s_tp + 0.5 * s_fp + 0.5 * s_fn
        s_pq = float(s_iou / s_den) if s_den > 0 else 0.0
        s_dice_mean = float(np.mean(s_dices))
        per_stem_stats[s] = {
            "iou": s_iou, "tp": s_tp, "fp": s_fp, "fn": s_fn,
            "pq": s_pq, "dice": s_dice_mean,
            "inter_px": s_inter_px, "pred_px": s_pred_px_acc, "gt_px": s_gt_px_acc,
            "R": R_mult, "n_pred": n_pred,
        }
        tup_s = (s_iou, s_tp, s_fp, s_fn, s_inter_px, s_pred_px_acc, s_gt_px_acc, s_dice_mean)
        by_station[s[-2:]].append(tup_s)
        by_R[R_mult].append(tup_s)
        by_fold[int(fold_map[s])].append(tup_s)

    def _agg_tuples(lst: List[tuple]) -> dict:
        arr = np.asarray(lst, dtype=np.float64)
        iou_s, tp, fp, fn, ipx, ppx, gpx = arr[:, :7].sum(axis=0)
        den = tp + 0.5 * fp + 0.5 * fn
        pq = float(iou_s / den) if den > 0 else 0.0
        sq = float(iou_s / tp) if tp > 0 else 0.0
        rq = float(tp / den) if den > 0 else 0.0
        dice_micro = float(2.0 * ipx / max(ppx + gpx, 1.0))
        out = {"pq": pq, "sq": sq, "rq": rq, "dice_micro": dice_micro, "n": len(lst)}
        if arr.shape[1] >= 8:
            out["dice_macro"] = float(arr[:, 7].mean())
        return out

    def _bootstrap_ci(lst: List[tuple], n_boot: int = 1000, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        arr = np.asarray(lst, dtype=np.float64)
        n = len(arr)
        pqs, sqs, rqs, dices = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            s_arr = arr[idx]
            iou_s = s_arr[:, 0].sum()
            tp = s_arr[:, 1].sum()
            fp = s_arr[:, 2].sum()
            fn = s_arr[:, 3].sum()
            ipx = s_arr[:, 4].sum()
            ppx = s_arr[:, 5].sum()
            gpx = s_arr[:, 6].sum()
            den = tp + 0.5 * fp + 0.5 * fn
            pqs[b] = iou_s / den if den > 0 else 0.0
            sqs[b] = iou_s / tp if tp > 0 else 0.0
            rqs[b] = tp / den if den > 0 else 0.0
            dices[b] = (2.0 * ipx) / max(ppx + gpx, 1.0)
        return {
            "pq_ci95": [float(np.percentile(pqs, 2.5)), float(np.percentile(pqs, 97.5))],
            "pq_se": float(np.std(pqs)),
            "sq_ci95": [float(np.percentile(sqs, 2.5)), float(np.percentile(sqs, 97.5))],
            "rq_ci95": [float(np.percentile(rqs, 2.5)), float(np.percentile(rqs, 97.5))],
            "dice_micro_ci95": [float(np.percentile(dices, 2.5)), float(np.percentile(dices, 97.5))],
        }

    all_stem_tuples = [
        (
            per_stem_stats[s]["iou"], per_stem_stats[s]["tp"], per_stem_stats[s]["fp"],
            per_stem_stats[s]["fn"], per_stem_stats[s]["inter_px"], per_stem_stats[s]["pred_px"],
            per_stem_stats[s]["gt_px"], per_stem_stats[s]["dice"],
        )
        for s in stems
    ]
    global_agg = _agg_tuples(all_stem_tuples)
    global_boot = _bootstrap_ci(all_stem_tuples, n_boot=1000, seed=42)

    fold_metrics = {f: _agg_tuples(by_fold[f]) for f in range(NUM_FOLDS)}
    fold_summary = {
        "pq_mean": float(np.mean([fold_metrics[f]["pq"] for f in range(NUM_FOLDS)])),
        "pq_std": float(np.std([fold_metrics[f]["pq"] for f in range(NUM_FOLDS)], ddof=1)),
        "sq_mean": float(np.mean([fold_metrics[f]["sq"] for f in range(NUM_FOLDS)])),
        "sq_std": float(np.std([fold_metrics[f]["sq"] for f in range(NUM_FOLDS)], ddof=1)),
        "rq_mean": float(np.mean([fold_metrics[f]["rq"] for f in range(NUM_FOLDS)])),
        "rq_std": float(np.std([fold_metrics[f]["rq"] for f in range(NUM_FOLDS)], ddof=1)),
        "dice_micro_mean": float(np.mean([fold_metrics[f]["dice_micro"] for f in range(NUM_FOLDS)])),
        "dice_micro_std": float(np.std([fold_metrics[f]["dice_micro"] for f in range(NUM_FOLDS)], ddof=1)),
        "dice_macro_mean": float(np.mean([fold_metrics[f]["dice_macro"] for f in range(NUM_FOLDS)])),
        "dice_macro_std": float(np.std([fold_metrics[f]["dice_macro"] for f in range(NUM_FOLDS)], ddof=1)),
    }

    station_metrics = {}
    for st in sorted(by_station.keys()):
        station_metrics[st] = {**_agg_tuples(by_station[st]), **_bootstrap_ci(by_station[st], n_boot=500, seed=100 + ord(st[0]))}

    R_metrics = {}
    for rv in sorted(by_R.keys()):
        R_metrics[str(rv)] = {**_agg_tuples(by_R[rv]), **_bootstrap_ci(by_R[rv], n_boot=500, seed=200 + rv)}

    annotator_metrics = {yy: _agg_tuples(by_annotator[yy]) for yy in sorted(by_annotator.keys())}

    # Area quartile recall & SQ breakdown
    areas_arr = np.array([x[0] for x in gt_area_hits], dtype=np.float64)
    q25, q50, q75 = np.percentile(areas_arr, [25, 50, 75])
    area_buckets = [
        (f"Q1 (<{int(q25)} px)", lambda a: a < q25),
        (f"Q2 ({int(q25)}-{int(q50)} px)", lambda a: q25 <= a < q50),
        (f"Q3 ({int(q50)}-{int(q75)} px)", lambda a: q50 <= a < q75),
        (f"Q4 (>={int(q75)} px)", lambda a: a >= q75),
    ]
    area_breakdown = {}
    for label, pred_fn in area_buckets:
        sub = [x for x in gt_area_hits if pred_fn(x[0])]
        n_tot = len(sub)
        n_tp = sum(x[1] for x in sub)
        sq_b = float(np.mean([x[2] for x in sub if x[1] == 1])) if n_tp > 0 else 0.0
        area_breakdown[label] = {
            "gt_count": n_tot,
            "tp_count": n_tp,
            "recall": float(n_tp / max(n_tot, 1)),
            "sq": sq_b,
        }

    miou = np.asarray(all_matched_ious, dtype=np.float64)
    mdice = np.asarray(all_matched_dices, dtype=np.float64)
    rdice = np.asarray(per_record_dices, dtype=np.float64)
    rpq = np.asarray(per_record_pqs, dtype=np.float64)

    return {
        "global": global_agg,
        "bootstrap_95": global_boot,
        "per_fold": fold_metrics,
        "fold_summary": fold_summary,
        "per_station": station_metrics,
        "per_R": R_metrics,
        "per_annotator": annotator_metrics,
        "area_breakdown": area_breakdown,
        "relations": {
            "total_records": len(per_record_dices),
            "many_to_one_count": int(many_to_one_total),
            "one_to_many_count": int(one_to_many_total),
            "records_with_many_to_one": int(records_with_mto1),
            "records_with_one_to_many": int(records_with_1tom),
        },
        "distributions_summary": {
            "matched_iou": {
                "n": int(len(miou)),
                "mean": float(miou.mean()), "std": float(miou.std()),
                "q25": float(np.percentile(miou, 25)), "median": float(np.median(miou)), "q75": float(np.percentile(miou, 75)),
            },
            "matched_dice": {
                "n": int(len(mdice)),
                "mean": float(mdice.mean()), "std": float(mdice.std()),
                "q25": float(np.percentile(mdice, 25)), "median": float(np.median(mdice)), "q75": float(np.percentile(mdice, 75)),
            },
            "per_record_semantic_dice": {
                "n": int(len(rdice)),
                "mean": float(rdice.mean()), "std": float(rdice.std()),
                "q25": float(np.percentile(rdice, 25)), "median": float(np.median(rdice)), "q75": float(np.percentile(rdice, 75)),
            },
            "per_record_pq": {
                "n": int(len(rpq)),
                "mean": float(rpq.mean()), "std": float(rpq.std()),
                "q25": float(np.percentile(rpq, 25)), "median": float(np.median(rpq)), "q75": float(np.percentile(rpq, 75)),
            },
        },
        "_arrays": {
            "matched_iou": miou,
            "matched_dice": mdice,
            "per_record_dice": rdice,
            "per_record_pq": rpq,
            "per_stem_stats": per_stem_stats,
        },
    }


def make_fig1_pipeline(out_pdf: Path, out_png: Path) -> None:
    """Generate crisp ACM full-width architectural pipeline diagram (`fig1_pipeline.pdf`)."""
    fig, ax = plt.subplots(figsize=(11.2, 3.55))
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 35.0)
    ax.axis("off")

    def draw_box(x, y, w, h, title, lines, fc, ec, title_color="white", body_color="#0F172A"):
        # Outer body rectangle
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.8",
            facecolor=fc, edgecolor=ec, linewidth=1.35, zorder=2,
        )
        ax.add_patch(box)
        # Solid header bar flush with top
        hdr_h = 4.3
        hdr = FancyBboxPatch(
            (x, y + h - hdr_h), w, hdr_h,
            boxstyle="round,pad=0.0,rounding_size=0.7",
            facecolor=ec, edgecolor=ec, linewidth=0.9, zorder=3,
        )
        ax.add_patch(hdr)
        # Square off bottom corners of header strip cleanly
        ax.add_patch(patches.Rectangle((x, y + h - hdr_h), w, 1.2, facecolor=ec, edgecolor="none", zorder=3))
        ax.text(
            x + w * 0.5, y + h - hdr_h * 0.5, title,
            color=title_color, ha="center", va="center",
            fontsize=8.6, fontweight="bold", zorder=4,
        )
        body_y_top = y + h - hdr_h - 1.5
        body_y_bot = y + 1.3
        n_l = len(lines)
        ys = np.linspace(body_y_top, body_y_bot, n_l) if n_l > 1 else [(body_y_top + body_y_bot) * 0.5]
        for yy, ln in zip(ys, lines):
            is_emph = ln.startswith("→")
            ax.text(
                x + 0.9, yy, ln,
                color=(ec if is_emph else body_color),
                ha="left", va="center",
                fontsize=7.65,
                fontweight=("bold" if is_emph else "normal"),
                zorder=4,
            )

    def arrow(x1, y1, x2, y2, color=C_SLATE):
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=5.5,head_width=3.2",
            color=color, linewidth=1.45, zorder=5,
        )
        ax.add_patch(arr)

    # Column 0: Stage 0 — Astrophysics Preprocessing
    draw_box(
        0.5, 6.2, 20.5, 25.5,
        "0. Astrophysics Prep",
        [
            "Raw 2048×2048 GONG H-α",
            "• 720-ray IRLS solar limb fit",
            "• 256-bin u=(r/R)² CLV norm.",
            "• 2-pass cloud flat-fielding",
            "• Multi-scale DoG + μ=cos θ",
            "→ 1856×1856×3 (1:1 native)",
        ],
        fc="#F8FAFC", ec=C_NAVY,
    )

    # Column 1 Top: Stage 1A — 5-Family Dense Ensemble
    draw_box(
        23.8, 17.8, 23.0, 16.2,
        "1A. Stage-1 Dense (25 Models)",
        [
            "5 Families × 5 Folds × 8-D4 TTA",
            "• UNet++ ConvNeXt-T (0.4199)",
            "• UNet EffNetV2-S | UNet++ R34d",
            "• MAnet EB4 | UNet ConvNeXt-S",
            "→ Dual fg+spine + LTV fstd parity",
        ],
        fc="#ECFEFF", ec=C_TEAL,
    )

    # Column 1 Bottom: Stage 1B — Orthogonal Detector
    draw_box(
        23.8, 3.4, 23.0, 12.2,
        "1B. Orthogonal Detector (5 Models)",
        [
            "5-Fold YOLO11l-seg @ 1536",
            "• Consolidated multi-annot. GT",
            "• 10-view cluster fusion (0.4379)",
            "→ 8 instance split/merge cues",
        ],
        fc="#FFFBEB", ec=C_AMBER,
    )

    # Column 2: Stage 1.5 — Component Tree + Hurdle GBDT + Two-Tier DP
    draw_box(
        49.6, 5.0, 23.2, 27.6,
        "2. Component Tree & Hurdle DP",
        [
            "16-Level Subset-Monotone Tree",
            "• Thresholds τ ∈ [0.04, 0.84]",
            "• Oracle PQ Ceiling = 0.5993",
            "• 77 morph/photo/fstd/YOLO feats",
            "• 120 Nested GBDTs (LGB+CB+XGB)",
            "• Hurdle û = p̂_hit · m̂_IoU",
            "• Two-Tier Antichain DP (O(|V|))",
            "→ τ_strict=0.260, τ_res=0.2275",
        ],
        fc="#EFF6FF", ec=C_BLUE,
    )

    # Column 3: Stage 2 — 40-Model PCA-Oriented Refiner
    draw_box(
        75.6, 5.0, 23.9, 27.6,
        "3. Stage-2 Refiner (40 Models)",
        [
            "PCA-Aligned 384×128 5-Ch Crops",
            "• [FlatField, fg, sp, Seed, Sibs]",
            "• 4 Archs × 2 Gens × 5 Folds",
            "  (UNet++ R34, UNet Cnx-T,",
            "   UNet++ EffV2-S, MAnet R34d)",
            "• 160 views/filament + join3 MST",
            "• Survivor-only argmax (t=0.45)",
            "→ 2048² Disjoint RLE (M2O = 0)",
        ],
        fc="#F0FDF4", ec=C_GREEN,
    )

    # Clean connecting arrows
    arrow(21.1, 22.5, 23.6, 25.0, color=C_NAVY)
    arrow(21.1, 13.0, 23.6, 10.0, color=C_NAVY)
    arrow(46.9, 25.0, 49.4, 22.5, color=C_TEAL)
    arrow(46.9, 10.0, 49.4, 13.0, color=C_AMBER)
    arrow(72.9, 18.8, 75.4, 18.8, color=C_BLUE)

    # Bottom summary strip strictly within x=[0.5, 99.5]
    ax.add_patch(
        FancyBboxPatch(
            (0.5, 0.3), 99.0, 2.2,
            boxstyle="round,pad=0.0,rounding_size=0.5",
            facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8, zorder=2,
        )
    )
    ax.text(
        50.0, 1.4,
        "5-Fold OOF PQ Progression:  Human Pairwise (0.3399) → Single Dense (0.4027–0.4199) → YOLO11l-seg (0.4379) → Stage-1.5 Hurdle DP (0.4593) → Stage-2 v3 Final (0.4683 | Nested LOFO 0.4676)",
        ha="center", va="center", fontsize=6.9, fontweight="bold", color=C_NAVY, zorder=3,
    )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)


def make_fig2_qualitative_examples(
    pred_rles_by_stem: Dict[str, List[dict]],
    gt_cache: dict,
    per_stem_stats: Dict[str, dict],
    handoff: dict,
    out_pdf: Path,
    out_png: Path,
) -> List[str]:
    """Generate 3-row x 4-column qualitative segmentation figure on real GONG H-alpha frames."""
    families = handoff["families"]
    blend_w = np.asarray(handoff["dense_blend_weights"], dtype=np.float32)

    # Select 3 representative, visually compelling OOF observations:
    #   Row 0: High-activity rich solar disk with high PQ (>0.60) and 8-12 filaments
    #   Row 1: Median-difficulty disk with curved limb + thin ribbons (PQ ~0.46-0.55)
    #   Row 2: Challenging multi-annotator (R=3) or cloudy/low-contrast frame (PQ ~0.35-0.44)
    cands_high, cands_mid, cands_hard = [], [], []
    for s, st in per_stem_stats.items():
        if st["n_pred"] < 5:
            continue
        if 8 <= st["n_pred"] <= 13 and st["pq"] >= 0.62:
            cands_high.append((st["pq"], s))
        elif 6 <= st["n_pred"] <= 10 and 0.47 <= st["pq"] <= 0.56 and st["R"] >= 2:
            cands_mid.append((abs(st["pq"] - 0.51), s))
        elif 5 <= st["n_pred"] <= 11 and 0.34 <= st["pq"] <= 0.43 and st["R"] == 3:
            cands_hard.append((abs(st["pq"] - 0.39), s))

    cands_high.sort(reverse=True)
    cands_mid.sort()
    cands_hard.sort()

    chosen = [
        ("High-Activity", cands_high[0][1]),
        ("Multi-Annot. (R=2)", cands_mid[0][1]),
        ("Ambiguous (R=3)", cands_hard[0][1]),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(8.4, 6.0), gridspec_kw={"wspace": 0.05, "hspace": 0.08})
    col_titles = [
        "Raw GONG H-α",
        "Flat-Field (Ch0)",
        "5-Fam fg + Spine",
        "Stage-2 Pred vs. GT",
    ]

    chosen_stems = []
    for r_idx, (row_tag, stem) in enumerate(chosen):
        chosen_stems.append(stem)
        raw_p = resolve_image_path(TRAIN_IMG_DIR, stem)
        raw_gray = cv2.imread(str(raw_p), cv2.IMREAD_GRAYSCALE)
        x_u8 = cv2.imread(str(PREP_TRAIN_DIR / f"{stem}_x.png"), cv2.IMREAD_COLOR)
        fg_b, sp_b, _, _, _ = load_blended_maps(stem, families, blend_w, is_test=False)

        # Downsample cleanly to 512x512 for sharp, lightweight PDF embedding (<1 MB)
        raw_512 = cv2.resize(raw_gray[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE], (512, 512), interpolation=cv2.INTER_AREA)
        ff_512 = cv2.resize(x_u8[:, :, 0], (512, 512), interpolation=cv2.INTER_AREA)
        fg_512 = cv2.resize(fg_b, (512, 512), interpolation=cv2.INTER_AREA)
        sp_512 = cv2.resize(sp_b, (512, 512), interpolation=cv2.INTER_AREA)

        # Build overlay: flat-field grayscale background + predicted disjoint masks (teal fill)
        # + GT annotator contours (amber) + TP/FP distinction
        preds_raw = pred_rles_by_stem.get(stem, [])
        recs = gt_cache["records_by_stem"][stem]
        preds_b = _to_pycoco_rles(preds_raw)
        # Compute which predictions match at least one annotator record
        tp_flags = [False] * len(preds_b)
        gt_union_full = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
        for _, g_raw in recs:
            g_b = _to_pycoco_rles(g_raw)
            if g_b:
                gt_union_full = np.maximum(gt_union_full, mask_util.decode(mask_util.merge(g_b, intersect=False)))
                if preds_b:
                    im = np.asarray(mask_util.iou(preds_b, g_b, [0] * len(g_b)), dtype=np.float64)
                    for pi in range(len(preds_b)):
                        if im[pi].max() > 0.5:
                            tp_flags[pi] = True

        # Render full-res crop overlay then downsample nicely
        bg_rgb = np.repeat(x_u8[:, :, 0:1], 3, axis=2).astype(np.float32)
        gt_crop = gt_union_full[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE]

        # Highlight GT interior subtly in warm gold and draw crisp GT contour
        gt_cnts, _ = cv2.findContours((gt_crop > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if preds_b:
            dec_p = mask_util.decode(preds_b)
            if dec_p.ndim == 2:
                dec_p = dec_p[:, :, None]
            dec_p = dec_p[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :]
            for pi in range(dec_p.shape[2]):
                pm = dec_p[:, :, pi] > 0
                col = np.array([16.0, 185.0, 129.0], dtype=np.float32) if tp_flags[pi] else np.array([239.0, 68.0, 68.0], dtype=np.float32)
                bg_rgb[pm] = 0.42 * bg_rgb[pm] + 0.58 * col

        overlay_u8 = np.clip(bg_rgb, 0, 255).astype(np.uint8)
        cv2.drawContours(overlay_u8, gt_cnts, -1, (255, 196, 0), thickness=5)
        overlay_512 = cv2.resize(overlay_u8, (512, 512), interpolation=cv2.INTER_AREA)

        # Panel 3: composite probability + spine visualization
        prob_rgb = np.repeat((ff_512[:, :, None].astype(np.float32) * 0.55), 3, axis=2)
        prob_rgb[:, :, 0] = np.clip(prob_rgb[:, :, 0] + fg_512 * 220.0, 0, 255)
        prob_rgb[:, :, 1] = np.clip(prob_rgb[:, :, 1] + fg_512 * 90.0 + sp_512 * 165.0, 0, 255)
        prob_rgb[:, :, 2] = np.clip(prob_rgb[:, :, 2] + sp_512 * 230.0, 0, 255)
        prob_512 = prob_rgb.astype(np.uint8)

        imgs_row = [raw_512, ff_512, prob_512, overlay_512]
        cmaps_row = ["gray", "gray", None, None]
        for c_idx in range(4):
            ax = axes[r_idx, c_idx]
            ax.imshow(imgs_row[c_idx], cmap=cmaps_row[c_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            for spn in ax.spines.values():
                spn.set_visible(True)
                spn.set_linewidth(0.5)
                spn.set_color("#CBD5E1")
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=9.2, fontweight="bold", pad=5.0)

        st = per_stem_stats[stem]
        axes[r_idx, 0].set_ylabel(
            f"{row_tag}\n{stem[:8]} ({stem[-2:]})\nPQ={st['pq']:.2f} | Dice={st['dice']:.2f}",
            fontsize=7.8, fontweight="bold", color=C_NAVY, labelpad=6,
        )

    # Compact legend strip at bottom
    legend_handles = [
        patches.Patch(facecolor="#10B981", edgecolor="none", label="Predicted TP (IoU > 0.5)"),
        patches.Patch(facecolor="#EF4444", edgecolor="none", label="Predicted Unmatched (FP / Annotator Omission)"),
        patches.Patch(facecolor="none", edgecolor="#FFC400", linewidth=2.0, label="Ground-Truth Polygon Contour"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, -0.025),
        fontsize=8.4,
    )
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)
    return chosen_stems


def make_fig3_evaluation_distributions(
    metrics: dict,
    stage_progression: List[Tuple[str, float, float, float]],
    out_pdf: Path,
    out_png: Path,
) -> None:
    """
    Generate 3-panel quantitative evaluation figure with uncertainty/error bars:
      (a) Matched Instance IoU & Per-Record Semantic Dice Distributions
      (b) Stage-by-Stage OOF PQ Progression with 5-Fold Std Error Bars
      (c) Stratified OOF PQ across 6 GONG Observatories & Annotator Multiplicity (95% Bootstrap CI)
    """
    arrs = metrics["_arrays"]
    miou = arrs["matched_iou"]
    rdice = arrs["per_record_dice"]

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.05), constrained_layout=True)

    # Panel (a): Distributions of Semantic Dice & Matched IoU
    ax = axes[0]
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(rdice, bins=bins, density=True, alpha=0.55, color=C_TEAL, edgecolor="white", linewidth=0.45, label=f"Rec. Dice (μ={rdice.mean():.2f})")
    ax.hist(miou, bins=bins, density=True, alpha=0.62, color=C_AMBER, edgecolor="white", linewidth=0.45, label=f"TP IoU (μ={miou.mean():.2f})")
    ax.axvline(0.5, color=C_RUST, linestyle="--", linewidth=1.15, label="PQ Match (0.50)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Score Value", fontsize=9.0)
    ax.set_ylabel("Density", fontsize=9.0)
    ax.set_title("(a) Semantic Dice & Matched IoU", fontsize=9.6, pad=6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.6, handlelength=1.4)

    # Panel (b): Stage-by-Stage Progression with Error Bars
    ax = axes[1]
    short_map = {
        "Human Pairwise": "Human Pairwise",
        "UNet Cnx-S (1-Fam)": "UNet Cnx-S",
        "UNet++ Cnx-T (Best)": "UNet++ Cnx-T",
        "YOLO11l-seg @1536": "YOLO11l @1536",
        "Stage-1.5 Hurdle DP": "Stage-1.5 DP",
        "Stage-2 v3 Final": "Stage-2 Final",
    }
    labels_b = [short_map.get(x[0], x[0]) for x in stage_progression]
    means_b = [x[1] for x in stage_progression]
    stds_b = [x[2] for x in stage_progression]
    colors_b = [C_SLATE, "#64748B", C_AMBER, C_BLUE, C_TEAL, C_GREEN][:len(labels_b)]
    y_pos = np.arange(len(labels_b))
    bars = ax.barh(y_pos, means_b, xerr=stds_b, height=0.58, color=colors_b, alpha=0.92,
                   error_kw={"ecolor": "#0F172A", "elinewidth": 1.0, "capsize": 2.6})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_b, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(0.30, 0.545)
    ax.set_xlabel("5-Fold OOF PQ (±1 SD)", fontsize=9.0)
    ax.set_title("(b) Stage-by-Stage Ablation Lift", fontsize=9.6, pad=6)
    for bar, val, sd in zip(bars, means_b, stds_b):
        ax.text(val + sd + 0.0045, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                va="center", ha="left", fontsize=7.8, fontweight="bold", color="#0F172A")

    # Panel (c): Station & Annotator Multiplicity Robustness (Bootstrap 95% CI)
    ax = axes[2]
    st_keys = sorted(metrics["per_station"].keys())
    r_keys = sorted(metrics["per_R"].keys(), key=int)
    x_labels = [f"{k}" for k in st_keys] + [f"R={r}" for r in r_keys]
    vals = [metrics["per_station"][k]["pq"] for k in st_keys] + [metrics["per_R"][r]["pq"] for r in r_keys]
    ci_lo = [vals[i] - (metrics["per_station"][k]["pq_ci95"][0] if i < len(st_keys) else metrics["per_R"][r_keys[i - len(st_keys)]]["pq_ci95"][0])
             for i, k in enumerate(st_keys + r_keys)]
    ci_hi = [(metrics["per_station"][k]["pq_ci95"][1] if i < len(st_keys) else metrics["per_R"][r_keys[i - len(st_keys)]]["pq_ci95"][1]) - vals[i]
             for i, k in enumerate(st_keys + r_keys)]
    x_pos = np.arange(len(x_labels))
    bar_cols = [C_NAVY] * len(st_keys) + [C_TEAL] * len(r_keys)
    ax.errorbar(
        x_pos, vals, yerr=[ci_lo, ci_hi], fmt="o", markersize=4.8,
        ecolor=C_SLATE, elinewidth=1.15, capsize=2.8, color=C_NAVY, zorder=4,
    )
    for xp, yv, col in zip(x_pos, vals, bar_cols):
        ax.plot(xp, yv, "o", markersize=5.2, color=col, zorder=5)
    ax.axvline(len(st_keys) - 0.5, color="#94A3B8", linestyle=":", linewidth=1.1)
    ax.axhline(metrics["global"]["pq"], color=C_GREEN, linestyle="--", linewidth=1.05, alpha=0.85, label=f"Global ({metrics['global']['pq']:.4f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8.0, rotation=28, ha="right")
    ax.set_ylim(0.40, 0.53)
    ax.set_ylabel("OOF PQ (95% Boot. CI)", fontsize=9.0)
    ax.set_title("(c) Station & Multiplicity", fontsize=9.6, pad=6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.90, fontsize=7.4)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)


def main() -> None:
    gt_cache = build_gt_cache()
    fold_map = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    with open(WORK_DIR / "stage1_dp_handoff.pkl", "rb") as f:
        handoff = pickle.load(f)
    with open(WORK_DIR / "stage2_v3_oof_preds.pkl", "rb") as f:
        s2_oof_preds = pickle.load(f)

    human_pq = compute_inter_annotator_pq(gt_cache)
    metrics = evaluate_comprehensive_metrics(s2_oof_preds, gt_cache, fold_map)

    # Also compute Stage-1 strict OOF fold std for progression error bars
    strict_tau = float(handoff["best_tau"])
    s1_strict_preds = {
        s: [r["rle"] for r in handoff["oof_selected"][s] if float(r["u_hat"]) >= strict_tau]
        for s in gt_cache["stems"]
    }
    s1_fold_pqs = [
        evaluate_pq(s1_strict_preds, gt_cache, eval_stems=[s for s in gt_cache["stems"] if fold_map[s] == f])["pq"]
        for f in range(NUM_FOLDS)
    ]

    # Single dense best (upp_cnx_t) & cnx_s fold PQ stds from checkpoints / summaries
    s_upp = json.loads((WORK_DIR / "summary_upp_cnx_t.json").read_text(encoding="utf-8"))
    s_cnx = json.loads((WORK_DIR / "summary_cnx_s.json").read_text(encoding="utf-8"))
    s_yolo = json.loads((WORK_DIR / "summary_yolo11l_seg.json").read_text(encoding="utf-8"))
    s_s1 = json.loads((WORK_DIR / "summary_stage1_dp.json").read_text(encoding="utf-8"))
    s_s2 = json.loads((WORK_DIR / "summary_stage2_final.json").read_text(encoding="utf-8"))

    # Compute exact 5-fold std for YOLO11l-seg OOF
    with open(WORK_DIR / "oof_yolo11l_seg.pkl", "rb") as f:
        oof_yolo = pickle.load(f)
    from src.train_stage1_yolo import detections_to_disjoint_rles
    y_thr = float(s_yolo.get("best_thr", 0.28))
    y_preds = {s: detections_to_disjoint_rles(oof_yolo[s], score_thr=y_thr) for s in gt_cache["stems"]}
    y_fold_pqs = [
        evaluate_pq(y_preds, gt_cache, eval_stems=[s for s in gt_cache["stems"] if fold_map[s] == f])["pq"]
        for f in range(NUM_FOLDS)
    ]

    # Compute exact 5-fold std for cnx_s and upp_cnx_t simple CC in parallel
    from concurrent.futures import ThreadPoolExecutor
    from src.train_stage1_dense import decode_simple_cc_rles, load_prob_pack

    def _compute_fam_fold_std(fam_name: str, best_op: list) -> float:
        def _dec_one(st_name: str):
            fg_s, _, _ = load_prob_pack(WORK_DIR / f"oof_{fam_name}" / f"{st_name}.npz")
            return st_name, decode_simple_cc_rles(fg_s, t_lo=best_op[0], min_mean_p=best_op[1], min_area=best_op[2])
        f_preds = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for st_name, rles_s in ex.map(_dec_one, gt_cache["stems"]):
                f_preds[st_name] = rles_s
        fpqs = [
            evaluate_pq(f_preds, gt_cache, eval_stems=[s for s in gt_cache["stems"] if fold_map[s] == f])["pq"]
            for f in range(NUM_FOLDS)
        ]
        return float(np.std(fpqs, ddof=1))

    cnx_std = _compute_fam_fold_std("cnx_s", s_cnx["best_cc_op"])
    upp_std = _compute_fam_fold_std("upp_cnx_t", s_upp["best_cc_op"])

    stage_progression = [
        ("Human Pairwise", float(human_pq["pq"]), 0.0, 0.0),
        ("UNet Cnx-S (1-Fam)", float(s_cnx["oof_simple_cc"]["pq"]), cnx_std, 0.0),
        ("UNet++ Cnx-T (Best)", float(s_upp["oof_simple_cc"]["pq"]), upp_std, 0.0),
        ("YOLO11l-seg @1536", float(s_yolo["oof"]["pq"]), float(np.std(y_fold_pqs, ddof=1)), 0.0),
        ("Stage-1.5 Hurdle DP", float(s_s1["tau_report"]["oof_at_best_tau"]["pq"]), float(np.std(s1_fold_pqs, ddof=1)), 0.0),
        ("Stage-2 v3 Final", float(metrics["global"]["pq"]), float(metrics["fold_summary"]["pq_std"]), 0.0),
    ]

    make_fig1_pipeline(FIG_DIR / "fig1_pipeline.pdf", FIG_DIR / "fig1_pipeline.png")
    chosen_stems = make_fig2_qualitative_examples(
        s2_oof_preds, gt_cache, metrics["_arrays"]["per_stem_stats"], handoff,
        FIG_DIR / "fig2_qualitative_examples.pdf", FIG_DIR / "fig2_qualitative_examples.png",
    )
    make_fig3_evaluation_distributions(
        metrics, stage_progression,
        FIG_DIR / "fig3_evaluation_distributions.pdf", FIG_DIR / "fig3_evaluation_distributions.png",
    )

    out_json = {k: v for k, v in metrics.items() if k != "_arrays"}
    out_json["human_inter_annotator"] = human_pq
    out_json["stage_progression"] = [
        {"stage": st, "pq": pq, "fold_std": sd} for st, pq, sd, _ in stage_progression
    ]
    out_json["qualitative_stems"] = chosen_stems
    out_json["stage2_summary"] = s_s2
    out_json["stage1_summary"] = s_s1

    (WORK_DIR / "report_metrics.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print("=== COMPREHENSIVE REPORT METRICS ===")
    print(f"Global OOF PQ          : {metrics['global']['pq']:.4f} (95% CI: {metrics['bootstrap_95']['pq_ci95'][0]:.4f}..{metrics['bootstrap_95']['pq_ci95'][1]:.4f})")
    print(f"5-Fold PQ Mean ± Std   : {metrics['fold_summary']['pq_mean']:.4f} ± {metrics['fold_summary']['pq_std']:.4f}")
    print(f"Global SQ / RQ         : {metrics['global']['sq']:.4f} / {metrics['global']['rq']:.4f}")
    print(f"Semantic Micro Dice    : {metrics['global']['dice_micro']:.4f} (95% CI: {metrics['bootstrap_95']['dice_micro_ci95'][0]:.4f}..{metrics['bootstrap_95']['dice_micro_ci95'][1]:.4f})")
    print(f"Per-Record Mean Dice   : {metrics['distributions_summary']['per_record_semantic_dice']['mean']:.4f} ± {metrics['distributions_summary']['per_record_semantic_dice']['std']:.4f}")
    print(f"Matched TP IoU Mean±SD : {metrics['distributions_summary']['matched_iou']['mean']:.4f} ± {metrics['distributions_summary']['matched_iou']['std']:.4f}")
    print(f"Relations (1154 recs)  : many_to_one={metrics['relations']['many_to_one_count']}, one_to_many={metrics['relations']['one_to_many_count']}")
    print(f"Qualitative stems      : {chosen_stems}")
    print(f"Figures written to     : {FIG_DIR}")


if __name__ == "__main__":
    main()
