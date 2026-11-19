"""
Download all pre-trained v3 model weights and inference configs from Hugging Face Hub.

Public repository (no authentication required):
    https://huggingface.co/CToraq/solar-filament-segmentation-2026

Weights layout restored under ./work/ (or custom --weights-dir):
  - Stage-1 Dense Whole-Disk 5-Family Ensemble (25 checkpoints, 5 families x 5 folds)
  - Stage-1 Orthogonal Instance Detector (5-fold YOLO11l-seg checkpoints)
  - Stage-1.5 Hurdle GBDT Meta-Scorer (120 cross-fitted LightGBM/CatBoost/XGBoost models)
  - Stage-2 PCA-Oriented 5-Channel Conditional Refiner (40 checkpoints, 8 slots x 5 folds)
  - Stage-2 Inference & Blending Config (`stage2_v3_infer_config.json`)
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "CToraq/solar-filament-segmentation-2026"
ROOT_DIR = Path(__file__).resolve().parent.parent

STAGE1_FAMILIES = ("cnx_s", "effv2_s", "upp_r34d", "manet_eb4", "upp_cnx_t")
STAGE2_SPECS = ("s2_upp_r34", "s2_unet_cnx", "s2_upp_effv2s", "s2_manet_r34d")
STAGE2_DIRS = ("ckpt_stage2_v3", "ckpt_stage2")


def expected_weight_files(weights_dir: Path) -> list[Path]:
    files: list[Path] = []
    for fam in STAGE1_FAMILIES:
        for fold in range(5):
            files.append(weights_dir / f"ckpt_{fam}" / f"best_fold{fold}.pt")
    for fold in range(5):
        files.append(weights_dir / "yolo_runs" / f"fold{fold}" / "weights" / "best.pt")
    for sdir in STAGE2_DIRS:
        for spec in STAGE2_SPECS:
            for fold in range(5):
                files.append(weights_dir / sdir / f"{spec}_fold{fold}.pt")
    files.append(weights_dir / "hurdle_v3_models.pkl")
    files.append(weights_dir / "stage2_v3_infer_config.json")
    return files


def verify_weights(weights_dir: Path) -> tuple[int, list[Path]]:
    expected = expected_weight_files(weights_dir)
    missing = [p for p in expected if not p.exists() or p.stat().st_size == 0]
    return len(expected), missing


def download_weights(
    repo_id: str = DEFAULT_REPO_ID,
    weights_dir: Path = ROOT_DIR / "work",
    revision: str = "main",
    force: bool = False,
) -> Path:
    weights_dir = Path(weights_dir).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    total, missing = verify_weights(weights_dir)
    if not missing and not force:
        print(f"[weights] All {total}/{total} weight & config files already present in {weights_dir}.")
        return weights_dir

    print(
        f"[weights] Downloading public model weights from https://huggingface.co/{repo_id} "
        f"into {weights_dir} ({len(missing)}/{total} files to fetch)..."
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=str(weights_dir),
        allow_patterns=[
            "ckpt_*/best_fold*.pt",
            "ckpt_stage2/*.pt",
            "ckpt_stage2_v3/*.pt",
            "yolo_runs/fold*/weights/best.pt",
            "hurdle_v3_models.pkl",
            "stage2_v3_infer_config.json",
            "summary_*.json",
        ],
    )

    total, missing = verify_weights(weights_dir)
    if missing:
        missing_str = "\n  - ".join(str(p) for p in missing[:10])
        raise RuntimeError(
            f"Incomplete weight download: {len(missing)}/{total} files missing:\n  - {missing_str}"
        )
    print(f"[weights] Verified all {total}/{total} model checkpoints and inference configs in {weights_dir}.")
    return weights_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pre-trained Solar Filament Segmentation 2026 v3 weights from Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, default=os.environ.get("FILSEG_HF_REPO", DEFAULT_REPO_ID))
    parser.add_argument("--weights-dir", type=str, default=str(ROOT_DIR / "work"))
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    wdir = Path(args.weights_dir)
    if args.verify_only:
        total, missing = verify_weights(wdir)
        print(f"[weights] Verified {total - len(missing)}/{total} files in {wdir}.")
        if missing:
            for p in missing[:20]:
                print(f"  MISSING: {p}")
            sys.exit(1)
        return

    download_weights(repo_id=args.repo_id, weights_dir=wdir, revision=args.revision, force=args.force)


if __name__ == "__main__":
    main()
