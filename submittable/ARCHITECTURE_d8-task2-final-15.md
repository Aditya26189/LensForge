# Architecture Document: d8-task2-final-15.ipynb

## 1) Goal and scope
This notebook implements a physics-informed lensing classifier for the same 3 classes:
- no
- sphere
- vort

Compared systems:
- D4-LensPINN (physics-informed, D4-equivariant hybrid)
- D4-LensPINN + TTA (8-way D4 test-time averaging)
- ResNet-18 baseline

The architecture couples learned feature extraction with differentiable gravitational-lensing operators.

## 2) High-level pipeline
1. Environment setup and reproducibility controls
2. Data loading and stratified split
3. Physics-aware preprocessing
4. Differentiable physics engine
5. D4-equivariant convergence estimator
6. EfficientNetV2 classification head
7. Full model assembly (D4LensPINN)
8. Two-phase training (warmup then physics-regularized)
9. Baseline ResNet-18 training
10. Test-time evaluation with and without D4-TTA
11. Save checkpoints and figures

## 3) Data architecture
Main data components:
- Dataset loading from npy-backed folders
- StratifiedShuffleSplit for class-balanced partitioning:
  - first split with test_size=0.2
  - second split with test_size=0.5 on held-out subset
  - net result: 80/10/10 train/val/test

Data wrappers and helpers:
- npy_loader(path)
- DataLoader construction with reproducible seed control

## 4) Core model architecture

### 4.1 Physics preprocessing
Component:
- class: PhysicsPreprocess

Role:
- Builds physics-motivated input channels from raw lensing images.
- Explicitly encodes structures useful for downstream physical inversion and classification.

### 4.2 Differentiable physics engine
Components:
- PoissonSolverFFT: spectral solution of Poisson equation
- DeflectionField: spatial gradient to get deflection field
- InverseLensLayer: map image-plane to source-plane relation

Role:
- Implements lensing-inspired operators as differentiable modules.
- Supplies physically constrained intermediate quantities used by loss and representation learning.

### 4.3 D4-equivariant convergence estimator
Component:
- class: EfficientD4UNet

Role:
- U-Net-like encoder-decoder under D4 symmetry constraints.
- Parameter-efficient intermediate estimator (noted in notebook as ~0.46M params for this block).

### 4.4 Classifier backbone
Component:
- class: EfficientNetV2Head

Role:
- Pretrained EfficientNet-V2-S based head for final class logits.
- Uses channel projection and ImageNet-pretrained features.

### 4.5 Full assembly
Component:
- class: D4LensPINN

Role:
- End-to-end composition of:
  - preprocess
  - physics engine
  - D4 UNet estimator
  - EfficientNetV2 classifier head

### 4.6 Baseline branch
Component:
- class: ResNet18Baseline

Role:
- Standard deep CNN baseline trained on same task for comparison.

## 5) Losses and training architecture

### 5.1 Physics-aware objective
Component:
- class: PhysicsLoss

Role:
- Adds physical regularization terms to complement classification objective.

### 5.2 Optimization stack
- Optimizer: AdamW
- Scheduler: warmup + cosine annealing
  - LinearLR + CosineAnnealingLR + SequentialLR
- Gradient clipping and mixed precision are integrated in training loop.

### 5.3 Two-phase training strategy
- Phase 1 (15 epochs): CE pre-training / warmup, no physics penalty
- Phase 2 (40 epochs): physics-regularized fine-tuning

Hyperparameter search:
- Optuna section included (15 trials, TPE + MedianPruner)
- Best values are then fixed for the final phase runs.

## 6) Inference architecture
Prediction modes:
- predict_no_tta(...)
- predict_with_tta(...)

TTA strategy:
- D4 group averaging over 8 transforms
  - 4 rotations x with/without reflection

Reported systems at test time:
- D4-LensPINN
- D4-LensPINN + TTA
- ResNet-18

## 7) Evaluation architecture
Outputs include:
- Per-class ROC curves and AUC table
- Confusion matrices for all compared systems
- Training/validation curves
- Parameter-efficiency scatter (trainable params vs mean AUC)

## 8) Saved outputs produced by this notebook
Expected output directory in this submission:
- d8-task2-output_files

Artifacts:
- d4_phase1_best.pth: best checkpoint from phase 1
- d4_phase2_best.pth: best checkpoint from phase 2
- resnet18_baseline_best.pth: best baseline checkpoint
- sample_grid.png: class-wise sample image grid
- roc_curves.png: per-class ROC comparison figure
- confusion_matrices.png: confusion matrix panel
- training_curves.png: train/val loss and accuracy curves
- efficiency_scatter.png: parameter efficiency scatter

## 9) Design rationale summary
- D4 equivariance explicitly targets dihedral image symmetries common in lensing morphology.
- Physics modules inject inductive bias from lensing equations instead of relying only on purely data-driven features.
- Two-phase optimization stabilizes training by first learning coarse class separation, then applying physical regularization.
- TTA improves inference robustness under group-consistent transforms.
