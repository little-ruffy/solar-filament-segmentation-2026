import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from pycocotools import mask as mask_util

from src.config import FULL_SIZE, GT_CACHE_PATH, TRAIN_ANN_PATH, WORK_DIR


def polygon_to_rle(segmentation: List[List[float]], height: int = FULL_SIZE, width: int = FULL_SIZE) -> dict:
    """Rasterize a COCO polygon list into canonical COCO RLE matching Self_Evaluation_Notebook."""
    rle = mask_util.merge(mask_util.frPyObjects(segmentation, height, width))
    counts = rle["counts"]
    return {
        "size": [height, width],
        "counts": counts.decode("ascii") if isinstance(counts, bytes) else str(counts),
    }


def encode_binary_mask(mask: np.ndarray) -> str:
    """Encode a 2D binary uint8/bool mask (H, W) into column-major ASCII COCO RLE counts."""
    arr = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(arr)
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def encode_label_map(label_map: np.ndarray) -> List[dict]:
    """Encode a pixel-disjoint integer label map (H, W) with labels 1..K into RLE dicts."""
    max_lbl = int(label_map.max())
    if max_lbl <= 0:
        return []
    uniq = np.unique(label_map)
    uniq = uniq[uniq > 0]
    if len(uniq) == 0:
        return []
    h, w = label_map.shape
    stack = np.zeros((h, w, len(uniq)), dtype=np.uint8, order="F")
    for idx, k in enumerate(uniq):
        stack[:, :, idx] = (label_map == k).astype(np.uint8)
    rles = mask_util.encode(stack)
    out = []
    for rle in rles:
        c = rle["counts"]
        out.append({
            "size": [h, w],
            "counts": c.decode("ascii") if isinstance(c, bytes) else str(c),
        })
    return out


def _to_pycoco_rles(rle_list: List[dict], height: int = FULL_SIZE, width: int = FULL_SIZE) -> List[dict]:
    out = []
    for r in rle_list:
        c = r["counts"] if isinstance(r, dict) else r
        out.append({
            "size": [height, width],
            "counts": c if isinstance(c, bytes) else str(c).encode("ascii"),
        })
    return out


def build_gt_cache(ann_path: Path = TRAIN_ANN_PATH, cache_path: Path = GT_CACHE_PATH, force: bool = False) -> dict:
    """Build or load cached ground-truth COCO RLE structures indexed by physical observation stem."""
    if cache_path.exists() and not force:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_rec = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_rec[str(ann["image_id"])].append(ann)

    records_by_stem: Dict[str, List[Tuple[str, List[dict]]]] = defaultdict(list)
    raw_anns_by_stem: Dict[str, List[Tuple[str, List[dict]]]] = defaultdict(list)
    total_anns = 0

    for im in coco["images"]:
        rec_id = str(im["id"])
        stem = os.path.splitext(im["file_name"])[0]
        rec_anns = anns_by_rec[rec_id]
        rles = [polygon_to_rle(a["segmentation"], FULL_SIZE, FULL_SIZE) for a in rec_anns]
        total_anns += len(rles)
        records_by_stem[stem].append((rec_id, rles))
        raw_anns_by_stem[stem].append((rec_id, rec_anns))

    stems = sorted(records_by_stem.keys())
    cache = {
        "stems": stems,
        "records_by_stem": dict(records_by_stem),
        "raw_anns_by_stem": dict(raw_anns_by_stem),
        "total_records": sum(len(v) for v in records_by_stem.values()),
        "total_annotations": total_anns,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def evaluate_pq(
    pred_rles_by_stem: Dict[str, List[dict]],
    gt_cache: dict,
    eval_stems: Optional[Iterable[str]] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Exact, C-accelerated replica of Self_Evaluation_Notebook global micro-pooled Panoptic Quality.
    Uses pycocotools.mask.iou directly on RLEs (verified < 1e-6 parity with PyTorch tensor reference).
    """
    stems = list(eval_stems) if eval_stems is not None else gt_cache["stems"]
    tp_iou_sum = 0.0
    tp_count = 0
    fp_count = 0
    fn_count = 0
    many_to_one = 0
    one_to_many = 0

    for stem in stems:
        recs = gt_cache["records_by_stem"][stem]
        preds_raw = pred_rles_by_stem.get(stem, [])
        preds = _to_pycoco_rles(preds_raw)
        n_pred = len(preds)

        for _, gt_raw in recs:
            gts = _to_pycoco_rles(gt_raw)
            n_gt = len(gts)
            if n_gt == 0:
                fp_count += n_pred
                continue
            if n_pred == 0:
                fn_count += n_gt
                continue

            # mask_util.iou(dt, gt, pyiscrowd) -> shape (n_pred, n_gt); transpose -> (n_gt, n_pred)
            iou_mat = np.asarray(mask_util.iou(preds, gts, [0] * n_gt), dtype=np.float64).T
            hit = iou_mat > iou_threshold

            if hit.any():
                tp_ious = iou_mat[hit]
                tp_iou_sum += float(tp_ious.sum())
                tp_count += int(tp_ious.size)

            pred_hits = hit.sum(axis=0)
            gt_hits = hit.sum(axis=1)
            fp_count += int((pred_hits == 0).sum())
            fn_count += int((gt_hits == 0).sum())
            one_to_many += int((pred_hits > 1).sum())
            many_to_one += int((gt_hits > 1).sum())

    denom = tp_count + 0.5 * fp_count + 0.5 * fn_count
    pq = float(tp_iou_sum / denom) if denom > 0 else 0.0
    sq = float(tp_iou_sum / tp_count) if tp_count > 0 else 0.0
    rq = float(tp_count / denom) if denom > 0 else 0.0
    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "tp": int(tp_count),
        "fp": int(fp_count),
        "fn": int(fn_count),
        "iou_sum": float(tp_iou_sum),
        "many_to_one": int(many_to_one),
        "one_to_many": int(one_to_many),
    }


def compute_inter_annotator_pq(gt_cache: dict, iou_threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate human-vs-human Panoptic Quality across all ordered pairs of multi-annotator records."""
    tp_iou_sum = 0.0
    tp_count = 0
    fp_count = 0
    fn_count = 0
    n_pairs = 0

    for stem in gt_cache["stems"]:
        recs = gt_cache["records_by_stem"][stem]
        if len(recs) < 2:
            continue
        encoded_recs = [_to_pycoco_rles(rles) for _, rles in recs]
        for i in range(len(encoded_recs)):
            for j in range(len(encoded_recs)):
                if i == j:
                    continue
                n_pairs += 1
                gts = encoded_recs[i]
                preds = encoded_recs[j]
                if len(gts) == 0:
                    fp_count += len(preds)
                    continue
                if len(preds) == 0:
                    fn_count += len(gts)
                    continue
                iou_mat = np.asarray(mask_util.iou(preds, gts, [0] * len(gts)), dtype=np.float64).T
                hit = iou_mat > iou_threshold
                if hit.any():
                    tp_ious = iou_mat[hit]
                    tp_iou_sum += float(tp_ious.sum())
                    tp_count += int(tp_ious.size)
                fp_count += int((hit.sum(axis=0) == 0).sum())
                fn_count += int((hit.sum(axis=1) == 0).sum())

    denom = tp_count + 0.5 * fp_count + 0.5 * fn_count
    return {
        "pq": float(tp_iou_sum / denom) if denom > 0 else 0.0,
        "sq": float(tp_iou_sum / tp_count) if tp_count > 0 else 0.0,
        "rq": float(tp_count / denom) if denom > 0 else 0.0,
        "pairs": n_pairs,
    }
