import os
from pathlib import Path

IS_KAGGLE = os.name == "posix" and os.path.exists("/kaggle/working")

if IS_KAGGLE:
    PROJECT_ROOT = Path("/kaggle/working")
    DATA_ROOT = Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    if not DATA_ROOT.exists():
        candidates = list(Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
        if candidates:
            DATA_ROOT = candidates[0].parent.parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "MAGFiLO_1.0_Kaggle_2026"

TRAIN_IMG_DIR = DATA_ROOT / "train" / "train_images"
TEST_IMG_DIR = DATA_ROOT / "test" / "test_images"
TRAIN_ANN_PATH = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

WORK_DIR = PROJECT_ROOT / "work"
PREP_TRAIN_DIR = WORK_DIR / "prep_train"
PREP_TEST_DIR = WORK_DIR / "prep_test"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
FOLDS_PATH = WORK_DIR / "folds.json"
GT_CACHE_PATH = WORK_DIR / "gt_rle_cache.pkl"
DISK_META_PATH = WORK_DIR / "disk_meta.json"

FULL_SIZE = 2048
CROP_OFFSET = 96
CROP_SIZE = 1856  # [96:1952, 96:1952], 29 * 64, retains 100.000% of solar disk & GT filaments
NUM_FOLDS = 5
RANDOM_SEED = 2026

GONG_STATIONS = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")
IMAGE_EXTS = (".jpeg", ".jpg", ".png")


def list_stems_and_paths(img_dir: Path) -> list[tuple[str, Path]]:
    items = []
    for fn in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in IMAGE_EXTS:
            items.append((stem, img_dir / fn))
    return items


def resolve_image_path(img_dir: Path, stem: str) -> Path:
    for ext in (".jpeg", ".jpg", ".JPEG", ".JPG"):
        cand = img_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No image file found for stem={stem} in {img_dir}")

