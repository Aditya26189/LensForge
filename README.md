# Gravitational Lensing Modeling Suite - Notebook Guide

This repository contains three core project notebooks:

1. `d8-task1.ipynb`
2. `d8-task2.ipynb`
3. `d8-task3.ipynb`

Architecture documents:

- `ARCHITECTURE_d8-task1.md`
- `ARCHITECTURE_d8-task2.md`
- `ARCHITECTURE_d8-task3.md`

## 1) Notebook purposes

### A) `d8-task1.ipynb` (Task 1 classification)
Verified from notebook code and markdown:

- 3-class lensing classification (`no`, `sphere`, `vort`)
- C8-equivariant model (`C8LensNet`) vs `ResNet18Baseline`
- robust percentile normalization and deterministic split logic
- warmup + cosine scheduling via `LinearLR` + `CosineAnnealingLR` + `SequentialLR`
- post-hoc vort calibration analysis (`vort_calibration.png`)

Notebook-reported headline results in markdown/text output:

- C8 macro AUC around `0.9715`
- ResNet-18 macro AUC around `0.979x`

### B) `d8-task2.ipynb` (Physics-guided classification)
Verified from notebook code and markdown:

- stratified `80/10/10` split using `StratifiedShuffleSplit`
- physics-guided stack with D4-equivariant components
- main model class `D4LensPINN`
- baseline class `ResNet18Baseline`
- training scheduler stack again uses `LinearLR` + `CosineAnnealingLR` + `SequentialLR`
- TTA inference path exists (`predict_with_tta`) and is explicitly D4-TTA (8 transforms)
- Optuna search section is present (largely commented in this snapshot)

### C) `d8-task3.ipynb` (DiT + OT-CFM run-after-train evaluation)
Verified from notebook code and markdown:

- pseudo-labeling with `MiniBatchKMeans`
- core classes present: `DiT`, `DiTBlock`, `EMA`
- inference sweep controls present: `CFG_SWEEP_VALUES`, `TEMP_SWEEP_VALUES`, `N_FID_IMAGES`, `USE_HEUN_SAMPLER`
- sweep generation function present: `flow_sample_cfg_temp(...)`
- FID/ISC calculation uses `torch_fidelity.calculate_metrics(...)`
- realism evaluation computes `AUC_SCORE` with `roc_auc_score`
- physics evaluation artifact path includes `physics_eval.png`
- training-related blocks exist but are commented in this run-after-train notebook form (`RUN_FULL_TRAINING` references appear in commented sections)

## 2) Environment assumptions

All notebooks assume Kaggle-style GPU execution.

Core dependencies used across notebooks:

- Python 3.x
- PyTorch + CUDA
- numpy
- torchvision
- scikit-learn
- matplotlib
- Pillow
- gdown

Task-specific additions:

- Task 1/2: `escnn`
- Task 3: `torch-fidelity`, `scipy`

## 3) Output artifacts (as implemented)

### Task 1 typical outputs

- `c8_lensnet.pth` (workspace may contain renamed copy `c8_lensnet (1).pth`)
- `resnet18_baseline.pth`
- `roc_curves.png`
- `confusion_matrices.png`
- `efficiency_plot.png`
- `loss_curves.png`
- `vort_calibration.png`

### Task 2 typical outputs

- phase best checkpoints saved with `{label}_best.pth` pattern
- workspace output folder includes:
  - `d4_phase1_best.pth`
  - `d4_phase2_best.pth`
  - `resnet18_baseline_best.pth`
- notebook also contains save call for `resnet18_baseline_final.pth`

### Task 3 typical outputs

- generated image grids and evaluation figures (for example `physics_eval.png`)
- generated/real folders for FID computations
- checkpoint paths for `dit_otfm_best.pth` and `dit_otfm_latest.pth` are configured in code

## 4) Evidence-based verification status

What is verified in this workspace:

1. The current canonical notebook filenames are the three files listed at the top of this README.
2. Each notebook contains the core classes/functions described above.
3. README and architecture filenames now match existing workspace files.

What is not guaranteed by static verification alone:

1. End-to-end runtime success in every fresh environment.
2. Identical numeric metric values across different sessions/hardware.
3. Availability of all external datasets/checkpoints at execution time.

Recommended full-proof validation:

1. Run each notebook from top in a clean GPU session.
2. Confirm dataset and checkpoint paths resolve before long runs.
3. Validate generation outputs exist before AUC/physics metric cells.
4. Archive final metrics and produced artifacts per notebook.
