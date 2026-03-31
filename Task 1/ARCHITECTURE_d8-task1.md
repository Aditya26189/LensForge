# Architecture Overview: d8-task1.ipynb

## 1) Goal and scope
This notebook implements a 3-class gravitational lensing classifier for classes:
- no
- sphere
- vort

It compares:
- C8LensNet: a C8-equivariant steerable CNN (escnn)
- ResNet-18 baseline: standard convolutional baseline

Primary comparison targets are discrimination quality (per-class AUC, macro AUC) and parameter efficiency.

## 2) High-level pipeline
1. Environment and dependency setup
2. Dataset discovery and integrity checks
3. Deterministic split and dataloader creation
4. Two model definitions (equivariant and baseline)
5. Shared training loop and optimizer/scheduler strategy
6. Test-set evaluation (ROC, confusion matrices, class accuracy)
7. Efficiency and calibration analysis
8. Save trained weights and figures

## 3) Data and preprocessing architecture
Core preprocessing blocks:
- Dataset root auto-discovery from zip/folder candidates
- Robust image intensity scaling using percentile normalization:
  - function: robust_percentile_scale(img, low=1.0, high=99.0, eps=1e-6)
- Torchvision transform pipeline adapted to model input needs
- Dataset wrapper:
  - class: LensingDataset

Data loading setup:
- Train, validation, and test loaders are instantiated explicitly.
- Batch configuration in notebook code: BATCH = 128
- Validation/test use larger effective batch (BATCH*2).

## 4) Model architecture details

### 4.1 C8LensNet
Key idea:
- Encode rotational symmetry with C8 group equivariance, reducing the need for brute-force rotation augmentation.

Main components:
- Equivariant residual block:
  - class: EquivResBlock
- Full network:
  - class: C8LensNet

Behavior:
- Accepts image tensor input and processes via escnn geometric tensors.
- Uses group-aware feature representations for rotationally consistent features.

### 4.2 ResNet-18 baseline
Main components:
- class: ResNet18Baseline

Behavior:
- Standard ResNet-18 adapted to 3 output classes.
- Serves as performance and parameter-count baseline.

## 5) Training architecture
Shared strategy for both models:
- Mixed precision training (GradScaler/autocast)
- Gradient clipping (max_norm=1.0)
- Optimizer: AdamW
- LR schedule: Linear warmup + cosine annealing
  - via LinearLR + CosineAnnealingLR wrapped in SequentialLR

Main training/evaluation functions:
- get_optimizer_and_scheduler(...)
- evaluate_loss(...)
- train_model(...)
- get_probs(...)
- get_probs_labels(...)

Typical notebook defaults in training function signature:
- epochs=80
- warmup_epochs=3
- patience=20
- lr=3e-4
- wd=1e-4

## 6) Evaluation architecture
Evaluation is organized around one-vs-rest class diagnostics:
- ROC and AUC computation:
  - function: compute_roc(probs, labels)
- Confusion matrix rendering:
  - function: plot_cm(...)
- Macro AUC and per-class AUC reporting
- Parameter-vs-AUC efficiency scatter
- Vort confidence scaling analysis (post-hoc class-specific calibration sweep)

## 7) Saved outputs produced by this notebook
Expected output directory in this workspace:
- d8-task1-output_files

Artifacts:
- c8_lensnet (1).pth: trained C8LensNet weights
- resnet18_baseline.pth: trained ResNet-18 baseline weights
- sample_images.png: sample class visualization grid
- roc_curves.png: per-class ROC comparison plot
- confusion_matrices.png: confusion matrices for both models
- efficiency_plot.png: AUC vs parameter count comparison
- loss_curves.png: training/validation learning curves
- vort_calibration.png: vort confidence multiplier sweep vs validation accuracy

## 8) Design rationale summary
- C8 equivariance targets rotational structure in lensed arcs/rings directly in architecture.
- Baseline ResNet-18 provides a conventional reference point.
- Shared optimization stack keeps comparison fair.
- Evaluation outputs emphasize both ranking quality (AUC) and operational errors (confusion matrices), plus model-size tradeoff.
