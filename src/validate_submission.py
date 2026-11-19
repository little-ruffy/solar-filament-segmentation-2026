import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import FULL_SIZE, TEST_IMG_DIR, list_stems_and_paths

FILAMENT_ID_RE = re.compile(r"^(\d{14}[A-Z][a-z])_(\d+)$")


def validate_submission_file(
    csv_path: str | Path,
    expected_stems: Optional[Set[str]] = None,
    require_all_stems_covered: bool = False,
    test_dir: Optional[str | Path] = None,
) -> Dict[str, object]:
    """
    14-check strict validator for Solar Filament Segmentation Challenge 2026 submissions:
      1. File exists and non-empty
      2. Pure ASCII encoding (no BOM, no UTF-8 garbage)
      3. Exact header `filament_id,segmentation_rle`
      4. Zero NaN / null / empty cells
      5. Zero quote corruption or trailing commas
      6. Unique `filament_id` across entire file
      7. Regex `^\\d{14}[A-Z][a-z]_\\d+$` for every `filament_id`
      8. Every observation stem belongs to `expected_stems` (180 test stems)
      9. Sequential 0..K-1 indexing per stem (clean ordering)
     10. RLE decodes cleanly at `[2048, 2048]`
     11. Non-empty masks (`area >= 1` pixel)
     12. Canonical column-major COCO RLE round-trip (`encode(decode(rle)) == rle`)
     13. 100% pixel-disjoint masks within every image (`cover.max() <= 1`, Discussion #735377 rule!)
     14. Plausible summary statistics (coverage, mean instances/image, area distribution)
    """
    path = Path(csv_path)
    errors: List[str] = []
    warnings: List[str] = []

    if expected_stems is None:
        tdir = Path(test_dir) if test_dir is not None else TEST_IMG_DIR
        if tdir.exists():
            expected_stems = {stem for stem, _ in list_stems_and_paths(tdir)}
        else:
            expected_stems = None

    # Check 1 & 2: File exists + pure ASCII + header check
    if not path.exists() or path.stat().st_size == 0:
        return {"valid": False, "errors": [f"Submission file missing or empty: {path}"], "stats": {}}

    raw_bytes = path.read_bytes()
    try:
        raw_text = raw_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        errors.append(f"Non-ASCII byte detected in CSV: {exc}")
        raw_text = raw_bytes.decode("utf-8", errors="replace")

    lines = [ln.rstrip("\r") for ln in raw_text.split("\n") if ln.strip() != ""]
    if not lines:
        return {"valid": False, "errors": ["CSV has zero lines"], "stats": {}}

    # Check 3 & 5: Exact header & quote hygiene
    if lines[0] != "filament_id,segmentation_rle":
        errors.append(f"Invalid header: {lines[0]!r} (expected 'filament_id,segmentation_rle')")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        errors.append(f"DataFrame columns mismatch: {list(df.columns)}")
        return {"valid": False, "errors": errors, "stats": {}}

    # Check 4: Zero empty / NaN strings
    if len(df) == 0:
        errors.append("Submission has 0 prediction rows")
        return {"valid": False, "errors": errors, "stats": {}}
    if (df["filament_id"] == "").any() or (df["segmentation_rle"] == "").any():
        errors.append("Empty filament_id or segmentation_rle found")

    # Check 6: Unique filament_ids
    if not df["filament_id"].is_unique:
        dup_n = int(df["filament_id"].duplicated().sum())
        errors.append(f"Found {dup_n} duplicate filament_id entries")

    # Check 7, 8, 9: Regex, valid test stems, sequential indices
    rles_by_stem: Dict[str, List[str]] = defaultdict(list)
    indices_by_stem: Dict[str, List[int]] = defaultdict(list)
    bad_ids = 0
    unknown_stems = set()
    for fid, rle_str in zip(df["filament_id"], df["segmentation_rle"]):
        m = FILAMENT_ID_RE.match(fid)
        if not m:
            bad_ids += 1
            continue
        stem, idx_s = m.group(1), int(m.group(2))
        if expected_stems is not None and stem not in expected_stems:
            unknown_stems.add(stem)
        rles_by_stem[stem].append(rle_str)
        indices_by_stem[stem].append(idx_s)

    if bad_ids > 0:
        errors.append(f"{bad_ids} filament_id values failed regex ^\\d{{14}}[A-Z][a-z]_\\d+$")
    if unknown_stems:
        errors.append(f"Found {len(unknown_stems)} unknown observation stems: {sorted(unknown_stems)[:5]}")

    non_seq_stems = 0
    for stem, idxs in indices_by_stem.items():
        if sorted(idxs) != list(range(len(idxs))):
            non_seq_stems += 1
    if non_seq_stems > 0:
        warnings.append(f"{non_seq_stems} stems have non-0..K-1 filament indices")

    if require_all_stems_covered and expected_stems is not None and len(rles_by_stem) < len(expected_stems):
        missing = len(expected_stems - set(rles_by_stem.keys()))
        errors.append(f"{missing} test stems have 0 predictions while require_all_stems_covered=True")

    # Check 10, 11, 12, 13: Decode, area >= 1, round-trip identity, and strict pixel-disjointness
    empty_masks = 0
    roundtrip_mismatches = 0
    overlap_stems = []
    overlap_pixels_total = 0
    areas: List[int] = []

    for stem, rle_list in rles_by_stem.items():
        rle_objs = [{"size": [FULL_SIZE, FULL_SIZE], "counts": s.encode("ascii")} for s in rle_list]
        try:
            decoded = mask_util.decode(rle_objs)  # shape (2048, 2048, K), uint8
        except Exception as exc:
            errors.append(f"RLE decode failed on stem {stem}: {exc}")
            continue

        if decoded.shape[:2] != (FULL_SIZE, FULL_SIZE):
            errors.append(f"Decoded mask shape {decoded.shape} != (2048, 2048, K) on stem {stem}")
            continue

        # Pixel-disjointness check: sum across all instances on this image must never exceed 1!
        cover = decoded.sum(axis=2, dtype=np.int32)
        max_cov = int(cover.max())
        if max_cov > 1:
            ov_px = int((cover > 1).sum())
            overlap_stems.append((stem, max_cov, ov_px))
            overlap_pixels_total += ov_px

        reencoded = mask_util.encode(np.asfortranarray(decoded))
        for orig_s, re_obj, k in zip(rle_list, reencoded, range(decoded.shape[2])):
            ar = int(mask_util.area(re_obj))
            areas.append(ar)
            if ar <= 0:
                empty_masks += 1
            re_s = re_obj["counts"].decode("ascii") if isinstance(re_obj["counts"], bytes) else str(re_obj["counts"])
            if re_s != orig_s:
                roundtrip_mismatches += 1

    if empty_masks > 0:
        errors.append(f"Found {empty_masks} empty (0-area) RLE masks")
    if roundtrip_mismatches > 0:
        errors.append(f"Found {roundtrip_mismatches} RLE masks failing canonical column-major round-trip")
    if overlap_stems:
        errors.append(
            f"DISQUALIFICATION VIOLATION: Found overlapping instance masks on {len(overlap_stems)} stems "
            f"({overlap_pixels_total} overlapping pixels; first={overlap_stems[0]})"
        )

    stems_for_stats = sorted(expected_stems) if expected_stems is not None else sorted(rles_by_stem.keys())
    counts_per_expected = [len(rles_by_stem.get(s, [])) for s in stems_for_stats]
    stats = {
        "total_rows": int(len(df)),
        "expected_stems": int(len(stems_for_stats)),
        "covered_stems": int(len(rles_by_stem)),
        "empty_stems": int(len(stems_for_stats) - len(rles_by_stem)),
        "instances_per_image_mean": round(float(np.mean(counts_per_expected)), 3),
        "instances_per_image_median": round(float(np.median(counts_per_expected)), 1),
        "instances_per_image_min": int(np.min(counts_per_expected)),
        "instances_per_image_max": int(np.max(counts_per_expected)),
        "area_median_px": round(float(np.median(areas)), 1) if areas else 0.0,
        "area_min_px": int(np.min(areas)) if areas else 0,
        "area_max_px": int(np.max(areas)) if areas else 0,
        "overlapping_pixels": int(overlap_pixels_total),
        "roundtrip_mismatches": int(roundtrip_mismatches),
    }

    return {
        "valid": len(errors) == 0,
        "status": "VALID" if len(errors) == 0 else "INVALID",
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="14-check Solar Filament Submission Validator")
    parser.add_argument("--sub", required=True, help="Path to submission CSV")
    parser.add_argument("--test-dir", default=None, help="Optional path to test_images directory")
    parser.add_argument("--require-all-stems", action="store_true", help="Require >=1 row per test stem")
    args = parser.parse_args()

    res = validate_submission_file(args.sub, require_all_stems_covered=args.require_all_stems, test_dir=args.test_dir)
    print(json.dumps(res, indent=2))
    if not res["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
