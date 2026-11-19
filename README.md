# Solar Filament Segmentation Challenge 2026 — Top-12 Private LB Solution (Team `MXlop`)

[![Competition](https://img.shields.io/badge/Kaggle-Solar%20Filament%20Segmentation%202026-20BEFF.svg)](https://www.kaggle.com/competitions/filament-segmentation-2026)
[![Weights](https://img.shields.io/badge/HuggingFace-Pre--Trained%20Weights%20(190%20Models)-FFD21E.svg)](https://huggingface.co/CToraq/solar-filament-segmentation-2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official source code, reproducible zero-retraining inference pipeline, interactive demonstration notebook, and technical report for **Team `MXlop`** (Timur Shakurov) in the **Solar Filament Segmentation Challenge 2026** (ESAIR Lab / IEEE BigData Cup 2026).

| Metric / Split | Value | Notes |
|---|---:|---|
| **Private Leaderboard PQ** | **`0.39`** | **Rank #12** (post-cleanup final Private LB, `ref: 58314458`) |
| **Public Leaderboard PQ** | **`0.41`** | 2-decimal rounded Public LB (`#1` honest pixel-disjoint tier) |
| **5-Fold OOF PQ (707 disks, 1,154 records)** | **`0.4683`** | `SQ = 0.6883`, `RQ = 0.6804`, `TP = 5,336`, `FP = 2,150` |
| **Strict Nested LOFO PQ** | **`0.4676`** | Zero hyperparameter leakage across outer folds |
| **Human Inter-Annotator Baseline PQ** | `0.3399` | `SQ = 0.6337`, `RQ = 0.5364` across 1,196 ordered human pairs |
| **Pixel Overlap / Many-to-One Violations** | **`0`** | Guaranteed `cover.max() <= 1` (`0` overlapping pixels across all 1,196 test masks) |

---

## 1. Architecture Overview

Our pipeline operates strictly on individual raw `2048x2048` GONG H-alpha observations (zero external test metadata or cross-frame information) across four modular stages:

1. **Stage 0 — Astrophysical GONG H-alpha Calibration (`src/solar_prep.py`)**:
   - **720-Ray IRLS Disk Fit**: Estimates exact solar center `(cx, cy)` and limb radius `R` (`RMS ~ 1.01 px`), immune to clouds/vignetting where Otsu fails.
   - **256-Annuli Equal-$u=(r/R)^2$ CLV Normalization & 2-Pass Feature-Masked Cloud Flat-Fielding**: Removes radial limb darkening and atmospheric illumination gradients while masking dark filaments and bright plages on Pass 2.
   - **Native `1:1` Centered `1856x1856` Crop (`[96:1952, 96:1952]`)**: Retains `100.000%` of the solar disk and `100.000%` of all `8,199` training ground-truth vertices without downsampling blur on thin (`4-11 px` wide) filament ribbons, encoding `[FlatField, Multi-Scale DoG, Heliocentric mu]`.
2. **Stage 1 — Multi-Paradigm Native Ensemble (`src/train_stage1_dense.py`, `src/train_stage1_yolo.py`)**:
   - **5 Calibrated Whole-Disk Dual-Head Dense Families** (`25` models = `5` families x `5` observation-grouped folds x `8`-view `D4` TTA = `200` views/pixel on test): `UNet++ ConvNeXt-Tiny` (`0.4199` standalone OOF PQ), `UNet EfficientNetV2-S` (`0.4132`), `UNet++ ResNet34d` (`0.4119`), `MAnet EfficientNet-B4` (`0.4040`), and `UNet ConvNeXt-Small` (`0.4027`), trained with gradient-balanced whole-disk loss (`20*BCE + 0.40*SoftDice`) on both soft annotator-mean foreground `fg` and `spine` centerlines.
   - **Exact Law-of-Total-Variance (`fstd`) Train/Test Parity**: Restores exact single-fold disagreement variance on 5-fold averaged test probability maps via $\mathrm{Var}_{\text{corr}} = \mathrm{Var}_m(\bar{p}_m) + \frac{M-1}{M}\overline{\mathrm{fstd}_m^2}$, eliminating the `-18.8%` test disagreement shrinkage.
   - **Orthogonal Instance Evidence**: 5-fold `YOLO11l-seg` (`1536`, `mask_ratio=2`, `0.4379` standalone OOF PQ) trained on cross-annotator IoU-consolidated masks supplies 8 instance-level detector agreement & split cues.
3. **Stage 1.5 — Subset-Monotone Component Tree, 120-GBDT Hurdle Scorer & Two-Tier Antichain DP (`src/hierarchy.py`, `src/hurdle_scorer.py`)**:
   - Builds a 16-level strictly subset-monotone, sibling-disjoint Component Tree (`Oracle PQ Ceiling = 0.5993`).
   - Extracts **77** morphological, photometric, epistemic uncertainty, tree-topology, and YOLO-agreement descriptors per node and cross-fits `120` Hurdle GBDTs (`LightGBM + CatBoost + XGBoost` x `20` nested splits) predicting expected matched IoU $\hat{u} = \hat{p}_{\text{hit}} \cdot \hat{m}_{\text{IoU}}$.
   - Solves exact $O(|V|)$ Dinkelbach Antichain DP via **`solve_two_tier_antichain_handoff`** (`strict_tau = 0.2600`, disjoint `rescue_tau = 0.2275`, `0.4593` OOF PQ / `0.4587` Nested LOFO PQ) with zero parent-to-child over-splitting.
4. **Stage 2 — 40-Model PCA-Oriented Conditional Refiner & Disjoint Serialization (`src/stage2_refiner.py`)**:
   - Rotates each candidate along its principal axis into a super-resolved `384x128` 5-channel crop (`[FlatField, fg, spine, Target_Seed, Competing_Sibling_Seeds]`).
   - Blends `40` checkpoints (`4` architectures [`UNet++ ResNet34`, `UNet ConvNeXt-Tiny`, `UNet++ EfficientNetV2-S`, `MAnet ResNet34d`] x `2` generations x `5` folds x `4`-flip TTA = `160` views/filament), applies `join3` MST gap bridging (`< 4 px`), and paints strictly survivor-only (`active_indices`, `protect_strong_tau = 0.2600`) pixel-disjoint `2048x2048` masks into canonical column-major COCO RLE (`0.4683` OOF PQ / `0.4676` Nested LOFO PQ).

---

## 2. Quickstart: Zero-Retraining Reproduction

### Step 1 — Environment Setup
```bash
git clone https://github.com/little-ruffy/solar-filament-segmentation-2026.git
cd solar-filament-segmentation-2026
pip install -r requirements.txt
```
*(Python `3.11.9`, tested on NVIDIA RTX 3090 with CUDA 12.6; CPU fallback is supported automatically.)*

### Step 2 — Download Pre-Trained Weights (Public Hugging Face Hub, No Login Required)
All **72 weight & configuration files (190 trained models)** are hosted publicly at:
**[https://huggingface.co/CToraq/solar-filament-segmentation-2026](https://huggingface.co/CToraq/solar-filament-segmentation-2026)**

```bash
python scripts/download_weights.py
```

### Step 3 — Run End-to-End Inference or Interactive Demo Notebook
- **Interactive Notebook (Recommended for Committee Review):**
  Open and run **[`notebooks/demo_inference_pipeline.ipynb`](notebooks/demo_inference_pipeline.ipynb)** — downloads/verifies weights automatically, visualizes Stage 0 -> Stage 1 -> Stage 1.5 -> Stage 2 on raw GONG H-alpha test images, and validates `submission.csv` with all 14 checks.
- **Command-Line Full Test-Set Reproduction:**
  ```bash
  python src/infer_pipeline.py \
      --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images \
      --weights-dir work \
      --out-csv submission_reproduced.csv \
      --validate
  ```
- **Validate the Exact Submitted File Included in Repo:**
  ```bash
  python src/validate_submission.py --sub submission/submission.csv
  ```

---

## 3. Full Training From Scratch & Unit Tests

Place the official competition dataset under `MAGFiLO_1.0_Kaggle_2026/` (`train/train_images/`, `test/test_images/`, `MAGFiLO_1.0_Annotations_kaggle2026_train.json`) and run:

```bash
# 1. Astrophysical preprocessing + 5-fold observation-grouped split
python src/solar_prep.py

# 2. Stage-1 5-family native 1856x1856 dense ensemble (25 checkpoints)
for fam in cnx_s effv2_s upp_r34d manet_eb4 upp_cnx_t; do
    python src/train_stage1_dense.py --family "$fam"
done

# 3. Stage-1 orthogonal 5-fold YOLO11l-seg instance detector
python src/train_stage1_yolo.py

# 4. Stage-1.5 16-level Component Tree + 120-GBDT Hurdle Scorer + Two-Tier Antichain DP
python src/hurdle_scorer.py --save-models work/hurdle_v3_models.pkl

# 5. Stage-2 40-model PCA-oriented refiner & final pixel-disjoint submission.csv
python src/stage2_refiner.py --ckpt-subdir ckpt_stage2_v3 --save-infer-artifacts
```

Run the automated verification test suite (exact metric parity, subset-monotone component tree, LTV `fstd` parity, two-tier DP, and Stage-2 affine geometry):
```bash
python tests/test_core_infra.py
python tests/test_hierarchy_dp.py
python tests/test_stage2_geometry.py
```

---

## 4. Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── demo_inference_pipeline.ipynb     # Complete end-to-end zero-retraining demo notebook
├── scripts/
│   ├── download_weights.py               # Public Hugging Face Hub weights downloader + verifier
│   └── compute_report_metrics_and_figures.py
├── src/
│   ├── config.py                         # Paths, constants, and GONG station metadata
│   ├── metrics.py                        # Exact C-accelerated Panoptic Quality evaluator
│   ├── solar_prep.py                     # 720-ray IRLS disk fit, 256-u CLV, 2-pass flat-fielding
│   ├── train_stage1_dense.py             # Stage-1 5-family native 1856x1856 dense training & D4 TTA
│   ├── train_stage1_yolo.py              # Stage-1 5-fold YOLO11l-seg detector & consolidation
│   ├── hierarchy.py                      # 16-level subset-monotone tree, 77 features & Two-Tier DP
│   ├── hurdle_scorer.py                  # 120-model nested Hurdle GBDT (LGB+CB+XGB) & LOFO blending
│   ├── stage2_refiner.py                 # 40-model PCA-oriented 5-ch refiner & join3 MST bridging
│   ├── infer_pipeline.py                 # Standalone zero-retraining end-to-end inference engine
│   └── validate_submission.py            # 14-check strict disjointness & RLE round-trip validator
├── tests/
│   ├── test_core_infra.py
│   ├── test_hierarchy_dp.py
│   └── test_stage2_geometry.py
├── submission/
│   └── submission.csv                    # Exact Kaggle submission file (ref: 58314458, 1,196 rows)
└── report/
    ├── main.tex                          # Official ACM sigconf technical report source
    ├── preamble.tex
    ├── main.bib
    ├── figures/                          # High-DPI vector/raster figures
    └── technical_report_MXlop.pdf        # Compiled 4-page (+ Ack/Refs) technical report PDF
```
