# DeepLense GSoC 2026 — Notebook Guide (Task 1 + Task 2)

This README documents **only** the following two notebooks:

1. `submittable/d8-task1-final-11.ipynb`  
2. `d8-task2 (6).ipynb`

---

## 1) What each notebook is for

### A. `submittable/d8-task1-final-11.ipynb` (Common Test)
This notebook is the Common Test baseline submission for 3-class strong lensing classification:
- `no_sub` (no substructure)
- `sphere` (CDM subhalo)
- `vort` (axion vortex)

It compares:
- **C8LensNet** (equivariant CNN, ~673k params)
- **ResNet-18** baseline (~11.2M params)

Primary reported result in notebook text:
- C8LensNet Macro AUC: **0.9715**
- ResNet-18 Macro AUC: **0.9798**

---

### B. `d8-task2 (6).ipynb` (Specific Test VII: Physics-Guided ML)
This notebook is the Physics-Guided ML attempt using a PINN-style architecture (D4-LensPINN):
- Physics-aware preprocessing
- D4-equivariant U-Net to infer convergence map `kappa`
- Differentiable lensing physics block (Poisson/deflection/inverse lens)
- EfficientNetV2 classifier head on `[I, kappa, S_hat, R]`

Current version includes Priority 0 + 1 stabilization changes:
- Reduced overfitting risk in EfficientNet fine-tuning policy
- Spectral-consistent Poisson residual in physics loss
- Physics loss ramping for phase-2 training
- Train/validation objective consistency across warmup/full phases
- Single-seed workflow (`SEED=42`) for time-constrained submission

---

## 2) Environment assumptions (both notebooks)

Both notebooks are written for Kaggle-style GPU execution.

### Required setup
- Python 3.x
- PyTorch with CUDA
- `numpy==1.26.4`
- `escnn`
- `gdown`
- `torchvision`, `scikit-learn`, `matplotlib`, `Pillow`

### Dataset expectation
- Dataset zip downloaded via Google Drive link inside notebooks
- Expected class folder structure under extracted dataset root
- Images are `.npy` grayscale arrays

---

## 3) `d8-task1-final-11.ipynb` — Detailed structure

### Cell-by-cell flow
1. **Cell 1 (markdown):** task statement, architecture summary, headline result.
2. **Cell 2:** dependency install + optional kernel restart for numpy pinning.
3. **Cell 3:** imports, seed setup, GPU placement (`cuda:0`/`cuda:1`).
4. **Cell 4:** robust dataset download/unzip checks.
5. **Cell 5:** root/class discovery + counts.
6. **Cell 6 (markdown):** preprocessing strategy and augmentation rationale.
7. **Cell 7:** dataset class, robust percentile scaling, 80/10/10 split, dataloaders.
8. **Cell 8 (markdown):** C8 architecture explanation.
9. **Cell 9:** C8LensNet implementation + sanity check.
10. **Cell 10 (markdown):** ResNet baseline description.
11. **Cell 11:** ResNet-18 baseline implementation + parameter comparison.
12. **Cell 12 (markdown):** training setup summary.
13. **Cell 13:** training infrastructure (optimizer/scheduler/train loop).
14. **Cell 14:** sample visualization.
15. **Cell 15 (markdown):** sample interpretation.
16. **Cell 16:** debug overfit gate (300 samples).
17. **Cell 17:** full C8 training.
18. **Cell 18:** full ResNet-18 training.
19. **Cell 19:** calibration-focused analysis cell.
20. **Cell 20 (markdown):** evaluation section intro.
21. **Cell 21:** full evaluation (ROC, confusion matrix, efficiency, curves).
22. **Cell 22 (markdown):** long-form scientific discussion + references.

### Core modeling design
- C8 equivariance (`rot2dOnR2(N=8)`) encodes rotational symmetry directly.
- Group pooling creates rotation-invariant features.
- Three-branch invariant/orientation-statistics head for classification.

### Main outputs/artifacts
- `c8_lensnet.pth`
- `resnet18_baseline.pth`
- `roc_curves.png`
- `confusion_matrices.png`
- `efficiency_plot.png`
- `loss_curves.png`
- plus optional calibration plot(s)

---

## 4) `d8-task2 (6).ipynb` — Detailed structure

### Cell-by-cell flow
1. **Cell 1 (markdown):** D4-LensPINN architecture overview.
2. **Cell 2:** dependency setup and numpy pinning.
3. **Cell 3:** imports + globals + deterministic setup.
4. **Cell 4:** GPU assignment logic.
5. **Cell 5:** robust dataset download/unzip.
6. **Cell 6:** root discovery, class mapping, loader/split setup.
7. **Cell 7:** physics preprocessing (log-intensity + Sobel cross-gradient).
8. **Cell 8:** Poisson solver + deflection + inverse lensing block.
9. **Cell 9:** D4-equivariant U-Net (`kappa` prediction).
10. **Cell 10:** EfficientNetV2 classifier head (late-block fine-tuning policy).
11. **Cell 11:** full D4-LensPINN model assembly.
12. **Cell 12:** ResNet-18 baseline model.
13. **Cell 13:** training framework with spectral physics loss + ramp.
14. **Cell 14:** TTA/no-TTA prediction helpers.
15. **Cell 15:** sample visualization.
16. **Cell 16:** optional sanity gate (commented).
17. **Cell 17:** optional Optuna block (commented) + best constants.
18. **Cell 18:** dataloader recreation / worker reset safeguards.
19. **Cell 19:** two-phase PINN training (warmup then physics-enabled).
20. **Cell 20:** ResNet baseline training.
21. **Cell 21:** evaluation, ROC/AUC table, confusion matrices, curves, efficiency plot.
22. **Cell 22 (markdown):** short discussion and fallbacks.
23. **Cells 23–24:** email helper cells (artifact sending).

### Current Task-2 architecture (implemented)
- **Stage 0:** Physics preprocess on image `I`
- **Stage 1:** D4-equivariant U-Net predicts `kappa >= 0`
- **Stage 2:** Poisson solve -> potential `psi` -> deflection `alpha` -> reconstructed source `S_hat` and residual `R`
- **Stage 3:** EfficientNetV2 head classifies concatenated physics-informed representation `[I, kappa, S_hat, R]`

### Priority changes now active in Task 2
- EfficientNet head fine-tuning constrained to late blocks (`features.6`, `features.7`) to reduce overfit risk.
- Physics loss uses **spectral residual** aligned with FFT Poisson solver domain.
- Physics loss weight is **ramped** during phase-2 instead of hard-on from epoch start.
- Validation objective now respects the same phase mode (`use_phys_loss`) as training.
- Training cell wires `lambda_poisson` explicitly per phase:
  - Phase 1: `0.0`
  - Phase 2: `BEST_LAMBDA`

### Main outputs/artifacts
- `d4_lenspinn_final.pth`
- `resnet18_baseline_final.pth`
- `roc_curves.png`
- `confusion_matrices.png`
- `training_curves.png`
- `efficiency_scatter.png`

---

## 5) Recommended execution order

### For Task 1 notebook
Run sequentially from Cell 2 onward.  
Keep Cell 16 (debug gate) before full training cells.

### For Task 2 notebook
Run sequentially from Cell 2 onward.  
Suggested minimal path under deadline:
- Run setup/data/model cells
- Skip heavy optional Optuna rerun if constants already fixed
- Run Cell 19 (PINN) then Cell 20 (ResNet) then Cell 21 (evaluation)

---

## 6) Practical cautions

1. **GPU memory:** if OOM, reduce `BATCH` first.
2. **Kernel restarts:** after package pinning, rerun from imports cell.
3. **Worker issues:** use the dataloader recreation cell before long training phases.
4. **Runtime planning:** Task 2 is multi-phase and can take several hours.
5. **Email cells:** last two cells include credential-based sending utility; keep secrets in environment variables and avoid hardcoding credentials.

---

## 7) Scope statement

This README intentionally covers **only**:
- `submittable/d8-task1-final-11.ipynb`
- `d8-task2 (6).ipynb`

No other scripts or notebooks are documented here by design.
