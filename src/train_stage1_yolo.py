import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
    FOLDS_PATH,
    FULL_SIZE,
    NUM_FOLDS,
    PREP_TEST_DIR,
    PREP_TRAIN_DIR,
    RANDOM_SEED,
    SUBMISSIONS_DIR,
    WORK_DIR,
)
from src.metrics import build_gt_cache, encode_binary_mask, encode_label_map, evaluate_pq


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_yolo_workspace(stems: List[str], fold_map: Dict[str, int], ds_root: Path) -> List[Path]:
    img_dir = ds_root / "images"
    lbl_dir = ds_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (ds_root / "labels.cache").unlink(missing_ok=True)

    for s in stems:
        _hardlink_or_copy(PREP_TRAIN_DIR / f"{s}_x.png", img_dir / f"{s}.png")
        _hardlink_or_copy(PREP_TRAIN_DIR / f"{s}.txt", lbl_dir / f"{s}.txt")

    yaml_paths = []
    for f in range(NUM_FOLDS):
        tr_list = [str((img_dir / f"{s}.png").resolve()).replace("\\", "/") for s in stems if fold_map[s] != f]
        va_list = [str((img_dir / f"{s}.png").resolve()).replace("\\", "/") for s in stems if fold_map[s] == f]
        tr_txt = ds_root / f"train_fold{f}.txt"
        va_txt = ds_root / f"val_fold{f}.txt"
        tr_txt.write_text("\n".join(tr_list) + "\n", encoding="utf-8")
        va_txt.write_text("\n".join(va_list) + "\n", encoding="utf-8")

        yaml_p = ds_root / f"solar_fold{f}.yaml"
        yaml_content = (
            f"path: {str(ds_root.resolve()).replace(chr(92), '/')}\n"
            f"train: {tr_txt.name}\n"
            f"val: {va_txt.name}\n"
            f"names:\n"
            f"  0: filament\n"
        )
        yaml_p.write_text(yaml_content, encoding="utf-8")
        yaml_paths.append(yaml_p)
    return yaml_paths


def extract_yolo_view_instances(model: YOLO, img_bgr: np.ndarray, imgsz: int = 1536, conf: float = 0.06) -> List[dict]:
    """Run 2-view flip TTA (identity + horizontal flip) with retina_masks=True at native 1856x1856 output."""
    raw_dets = []
    for flip in (False, True):
        inp = cv2.flip(img_bgr, 1) if flip else img_bgr
        res = model.predict(
            source=inp,
            imgsz=imgsz,
            conf=conf,
            iou=0.55,
            max_det=65,
            retina_masks=True,
            verbose=False,
            device=0 if torch.cuda.is_available() else "cpu",
        )[0]
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            continue
        masks_t = res.masks.data  # (N, H_m, W_m)
        confs = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
        for k in range(len(confs)):
            m_np = (masks_t[k] > 0.5).detach().cpu().numpy().astype(np.uint8)
            if m_np.shape != (CROP_SIZE, CROP_SIZE):
                m_np = cv2.resize(m_np, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
            if flip:
                m_np = np.ascontiguousarray(m_np[:, ::-1])
            area = int(m_np.sum())
            if area < 45:
                continue
            full_m = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
            full_m[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE] = m_np
            rle = mask_util.encode(np.asfortranarray(full_m))
            c_str = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
            raw_dets.append({
                "rle": {"size": [FULL_SIZE, FULL_SIZE], "counts": c_str},
                "score": float(confs[k]),
                "area": area,
            })
    return raw_dets


def consolidate_view_detections(raw_dets: List[dict], num_views: int, iou_cluster: float = 0.42) -> List[dict]:
    """
    Cluster multi-view / multi-fold instance detections via greedy IoU clustering and normalize
    total score by `num_views` so 2-view OOF and 10-view Test scores share the exact [0, 1] expectation!
    """
    if not raw_dets:
        return []
    dets = sorted(raw_dets, key=lambda d: d["score"], reverse=True)
    rles_bytes = [{"size": [FULL_SIZE, FULL_SIZE], "counts": d["rle"]["counts"].encode("ascii")} for d in dets]
    n = len(dets)
    iou_mat = np.asarray(mask_util.iou(rles_bytes, rles_bytes, [0] * n), dtype=np.float32)

    used = [False] * n
    consolidated = []
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        members = [i]
        for j in range(i + 1, n):
            if not used[j] and iou_mat[i, j] >= iou_cluster:
                used[j] = True
                members.append(j)

        score_norm = float(sum(dets[m]["score"] for m in members) / float(num_views))
        if score_norm < 0.04:
            continue

        if len(members) == 1:
            best_rle = dets[i]["rle"]
            area = dets[i]["area"]
        else:
            # Weighted pixel consensus across matching TTA/fold views
            dec = mask_util.decode([rles_bytes[m] for m in members])  # (2048, 2048, M)
            w = np.array([dets[m]["score"] for m in members], dtype=np.float32)
            w_sum = float(w.sum())
            crop_dec = dec[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :]
            vote = np.tensordot(crop_dec.astype(np.float32), w, axes=([2], [0])) / max(w_sum, 1e-6)
            cons_crop = (vote >= 0.45).astype(np.uint8)
            area = int(cons_crop.sum())
            if area < 45:
                continue
            full_m = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
            full_m[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE] = cons_crop
            best_rle = {"size": [FULL_SIZE, FULL_SIZE], "counts": encode_binary_mask(full_m)}

        consolidated.append({
            "rle": best_rle,
            "score": round(score_norm, 5),
            "area": int(area),
            "support_views": len(members),
        })

    consolidated.sort(key=lambda d: d["score"], reverse=True)
    return consolidated


def detections_to_disjoint_rles(dets: List[dict], score_thr: float = 0.28, min_area: int = 120) -> List[dict]:
    """Convert consolidated YOLO detections into 100% pixel-disjoint COCO RLEs."""
    keep = [d for d in dets if d["score"] >= score_thr and d["area"] >= min_area]
    if not keep:
        return []
    canvas = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.int32)
    cur_id = 0
    for d in keep:
        m = mask_util.decode({"size": [FULL_SIZE, FULL_SIZE], "counts": d["rle"]["counts"].encode("ascii")})
        avail = (m > 0) & (canvas == 0)
        if int(avail.sum()) < min_area:
            continue
        cur_id += 1
        canvas[avail] = cur_id
    return encode_label_map(canvas)


def run_yolo_pipeline(
    model_weights: str = str(WORK_DIR / "yolo11l-seg.pt"),
    imgsz: int = 1536,
    epochs: int = 32,
    batch: int = 4,
    folds_to_run: Optional[List[int]] = None,
    smoke_test: bool = False,
) -> dict:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.88)
    gt_cache = build_gt_cache()
    with open(FOLDS_PATH, "r", encoding="utf-8") as f:
        fold_map = json.load(f)

    ds_root = WORK_DIR / "yolo_ds"
    runs_dir = WORK_DIR / "yolo_runs"
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    yaml_paths = prepare_yolo_workspace(gt_cache["stems"], fold_map, ds_root)

    if smoke_test:
        print("[yolo smoke] Testing model load + 1-epoch mini-run on 8 images...")
        mini_root = WORK_DIR / "yolo_smoke"
        mini_root.mkdir(parents=True, exist_ok=True)
        mini_list = [str((ds_root / "images" / f"{s}.png").resolve()).replace("\\", "/") for s in gt_cache["stems"][:8]]
        (mini_root / "mini.txt").write_text("\n".join(mini_list) + "\n", encoding="utf-8")
        mini_yaml = mini_root / "mini.yaml"
        mini_yaml.write_text(
            f"path: {str(mini_root.resolve()).replace(chr(92), '/')}\ntrain: mini.txt\nval: mini.txt\nnames:\n  0: filament\n",
            encoding="utf-8",
        )
        m = YOLO(model_weights)
        m.train(
            data=str(mini_yaml),
            epochs=1,
            imgsz=640,
            batch=2,
            project=str(mini_root),
            name="run",
            exist_ok=True,
            verbose=False,
            workers=0,
        )
        sample_bgr = cv2.imread(str(PREP_TRAIN_DIR / f"{gt_cache['stems'][0]}_x.png"), cv2.IMREAD_COLOR)
        dets = extract_yolo_view_instances(m, sample_bgr, imgsz=640, conf=0.05)
        shutil.rmtree(mini_root, ignore_errors=True)
        print(f"[yolo smoke] OK! Extracted {len(dets)} raw detections.")
        return {"status": "smoke_ok"}

    target_folds = folds_to_run if folds_to_run is not None else list(range(NUM_FOLDS))
    oof_pkl = WORK_DIR / "oof_yolo11l_seg.pkl"
    oof_dets: Dict[str, List[dict]] = {}
    if oof_pkl.exists():
        with open(oof_pkl, "rb") as f:
            oof_dets = pickle.load(f)

    for fold in target_folds:
        val_stems = [s for s in gt_cache["stems"] if fold_map[s] == fold]
        fold_dir = runs_dir / f"fold{fold}"
        best_pt = fold_dir / "weights" / "best.pt"
        done_flag = fold_dir / "train_complete.flag"
        if not (done_flag.exists() and best_pt.exists()):
            if fold_dir.exists():
                shutil.rmtree(fold_dir, ignore_errors=True)
            t0 = time.time()
            print(f"\n[YOLO11l-seg | Fold {fold}/{NUM_FOLDS - 1}] Training {epochs} epochs @ imgsz={imgsz}, batch={batch}...")
            model = YOLO(model_weights)
            model.train(
                data=str(yaml_paths[fold]),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=0 if torch.cuda.is_available() else "cpu",
                project=str(runs_dir),
                name=f"fold{fold}",
                exist_ok=True,
                seed=RANDOM_SEED + fold,
                patience=12,
                cos_lr=True,
                lr0=0.0012,
                lrf=0.02,
                optimizer="AdamW",
                weight_decay=0.0005,
                degrees=180.0,
                flipud=0.5,
                fliplr=0.5,
                scale=0.20,
                mosaic=0.25,
                close_mosaic=8,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.10,
                overlap_mask=False,
                mask_ratio=2,  # High-resolution 1/2 stride mask training!
                workers=0,
                verbose=False,
            )
            done_flag.write_text("ok\n", encoding="utf-8")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[YOLO11l-seg | Fold {fold}] Trained in {(time.time() - t0)/60:.1f}m -> {best_pt}")

        if not all(s in oof_dets for s in val_stems):
            best_m = YOLO(str(best_pt))
            for s in val_stems:
                img_bgr = cv2.imread(str(PREP_TRAIN_DIR / f"{s}_x.png"), cv2.IMREAD_COLOR)
                raw = extract_yolo_view_instances(best_m, img_bgr, imgsz=imgsz, conf=0.06)
                oof_dets[s] = consolidate_view_detections(raw, num_views=2, iou_cluster=0.42)
            del best_m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with open(oof_pkl, "wb") as f:
                pickle.dump(oof_dets, f, protocol=pickle.HIGHEST_PROTOCOL)
            fold_eval = evaluate_pq(
                {s: detections_to_disjoint_rles(oof_dets[s], score_thr=0.28) for s in val_stems},
                gt_cache,
                eval_stems=val_stems,
            )
            print(f"[YOLO11l-seg | Fold {fold}] Standalone Disjoint OOF PQ = {fold_eval['pq']:.4f}")

    all_ready = all((runs_dir / f"fold{f}" / "weights" / "best.pt").exists() for f in range(NUM_FOLDS))
    if all_ready and len(oof_dets) == len(gt_cache["stems"]):
        # Sweep optimal standalone score threshold on OOF
        best_thr, best_oof_res = 0.28, {"pq": -1.0}
        for thr in np.linspace(0.16, 0.46, 16):
            res = evaluate_pq(
                {s: detections_to_disjoint_rles(oof_dets[s], score_thr=float(thr)) for s in gt_cache["stems"]},
                gt_cache,
            )
            if res["pq"] > best_oof_res["pq"]:
                best_thr, best_oof_res = float(thr), res
        print(
            f"\n[YOLO11l-seg FULL 707 OOF] Best thr={best_thr:.3f} -> "
            f"OOF PQ = {best_oof_res['pq']:.4f} (SQ={best_oof_res['sq']:.4f}, RQ={best_oof_res['rq']:.4f})"
        )

        # Infer 5-fold x 2-view = 10-view consolidated Test predictions
        test_pkl = WORK_DIR / "test_yolo11l_seg.pkl"
        test_stems = sorted(fn[:-6] for fn in os.listdir(PREP_TEST_DIR) if fn.endswith("_x.png"))
        if not test_pkl.exists():
            fold_models = [YOLO(str(runs_dir / f"fold{f}" / "weights" / "best.pt")) for f in range(NUM_FOLDS)]
            test_dets: Dict[str, List[dict]] = {}
            t0 = time.time()
            for idx, s in enumerate(test_stems, start=1):
                img_bgr = cv2.imread(str(PREP_TEST_DIR / f"{s}_x.png"), cv2.IMREAD_COLOR)
                raw_all = []
                for fm in fold_models:
                    raw_all.extend(extract_yolo_view_instances(fm, img_bgr, imgsz=imgsz, conf=0.06))
                test_dets[s] = consolidate_view_detections(raw_all, num_views=2 * NUM_FOLDS, iou_cluster=0.42)
                if idx % 60 == 0 or idx == len(test_stems):
                    print(f"  [YOLO11l-seg] test inferred: {idx}/{len(test_stems)} ({time.time() - t0:.1f}s)")
            with open(test_pkl, "wb") as f:
                pickle.dump(test_dets, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(test_pkl, "rb") as f:
                test_dets = pickle.load(f)

        sub_rows = []
        for s in test_stems:
            rles = detections_to_disjoint_rles(test_dets[s], score_thr=best_thr)
            for k, rle in enumerate(rles):
                sub_rows.append(f"{s}_{k},{rle['counts']}")
        sub_path = SUBMISSIONS_DIR / "sub_yolo11l_seg.csv"
        sub_path.write_text("filament_id,segmentation_rle\n" + "\n".join(sub_rows) + "\n", encoding="ascii")
        summary = {"family": "yolo11l_seg", "best_thr": best_thr, "oof": best_oof_res, "sub_path": str(sub_path)}
        (WORK_DIR / "summary_yolo11l_seg.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    return {"status": "partial"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(WORK_DIR / "yolo11l-seg.pt"))
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_yolo_pipeline(
        model_weights=args.weights,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        folds_to_run=args.folds,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
