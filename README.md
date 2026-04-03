# LensForge 🔭
### Equivariant Physics-Informed Deep Learning for Gravitational Lensing

[![DeepLense](https://img.shields.io/badge/ML4SCI-DeepLense-blue)](https://ml4sci.org/)
[![GSoC](https://img.shields.io/badge/GSoC-2026-orange)](https://summerofcode.withgoogle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)

> Three-task pipeline for dark matter substructure classification and Einstein ring synthesis — combining group-equivariant networks, differentiable physics engines, and flow-based generative models on the ML4SCI DeepLense benchmark.

---

## Overview

Gravitational lensing imprints Einstein rings on detector focal planes at **arbitrary orientations** — orientation carries zero physical information about dark matter class. LensForge exploits this symmetry at the architecture level rather than augmenting it away.

Three progressively deeper approaches:

| Task | Method | Key Result |
|------|---------|------------|
| Task 1 | C8-Equivariant CNN | Macro AUC **0.9715** at 673K params (16.6× fewer than ResNet-18) |
| Task 2 | D4-PINN + Differentiable Lensing Engine | Macro AUC **0.9809** (+6.27 pp over ResNet-18) |
| Task 3 | DiT-L/4 + OT-CFM Generative Model | PSD Pearson-R = **0.9886**, ring ΔR = **1 px** |

Dataset: [ML4SCI DeepLense](https://ml4sci.org/) — 30,000 gravitational lensing images, 3-class dark matter substructure (smooth / CDM spherical subhalo / axion vortex condensate).

---

## Task 1 — C8-Equivariant Steerable CNN

**Model:** `C8LensNet` via [`escnn`](https://github.com/QUVA-Lab/escnn) (rot2dOnR2, N=8)

Einstein rings are invariant under 8-fold discrete rotation (C8 group). Encoding this algebraically eliminates rotation augmentation entirely and preserves the subtle azimuthal fringe patterns of the vortex class that bilinear interpolation smears.

**Architecture highlights:**
- 5× `EquivResBlock` stages (C8 steerable filters, `InnerBatchNorm`)
- `GroupPooling` → rotation-invariant scalar features
- **Orientation statistics fusion:** group mean + std + max concatenated into 144-d head — captures both invariant and orientation-discriminative features simultaneously
- `PointwiseDropout(p=0.30)` on equivariant feature maps (escnn-native)

**Results:**

| Model | Macro AUC | Params | AUC/param ratio |
|-------|-----------|--------|-----------------|
| C8LensNet | **0.9715** | 673K | **1.44×10⁻⁶** |
| ResNet-18 baseline | 0.9793 | 11.2M | 8.76×10⁻⁸ |

- Post-hoc vortex calibration: column-multiplier grid search (200 steps, val set) recovers **+26.3 pp vortex accuracy (67.9% → 94.2%)** with zero AUC cost — confirms representation quality, not capacity, was the bottleneck

---

## Task 2 — D4-PINN with Differentiable Lensing Engine

**Model:** `D4LensPINN` — end-to-end differentiable gravitational lensing physics as PyTorch modules

The lensing equation β = θ − α(θ) is parity-symmetric (D4 ⊃ C8). Embedding the full lensing physics pipeline as differentiable modules allows the model to estimate the convergence map κ̂, reconstruct the source plane, and compute a lensing residual — all differentiably and all used as classifier inputs.

**Differentiable physics pipeline:**
```
Input I → PoissonSolverFFT (∇²Ψ = 2κ̂, 256×256 zero-padded)
        → DeflectionField (α = ∇Ψ, central finite differences)
        → InverseLensLayer (Ŝ = β = θ − α(θ), bilinear grid_sample)
        → Residual R = I − Ŝ
Classifier input: [I, κ̂, Ŝ, R]  →  EfficientNetV2-S head  →  logits
```

**Physics loss (4 terms):**

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_{TV} \cdot TV(\hat{\kappa}) + \lambda_{L1} \cdot \mathbb{E}[\hat{\kappa}] + \lambda_{ctr} \cdot \mathbb{E}[\hat{\kappa} \cdot r^2] + \lambda_{poisson} \cdot \mathbb{E}[(\nabla^2\Psi - 2\hat{\kappa})^2]$$

λ_poisson Optuna-tuned (TPE sampler, 15 trials, MedianPruner).

**Results:**

| Model | Macro AUC | Sphere AUC | Vortex AUC |
|-------|-----------|------------|------------|
| D4LensPINN + TTA | **0.9809** | **0.9734** | **0.9845** |
| D4LensPINN (no TTA) | 0.9786 | 0.9695 | 0.9814 |
| ResNet-18 baseline | 0.9182 | 0.8738 | 0.9151 |

- CDM sphere class: **+9.57 pp AUC** — Poisson-resolved sub-pixel deflection anomalies
- D4 TTA trade-off documented: +0.0023 AUC but −5.5 pp accuracy due to Sobel feature sign flip under horizontal reflection

---

## Task 3 — DiT-L/4 + OT-CFM Generative Model

**Model:** 129M-parameter class-conditional Diffusion Transformer with Optimal Transport Conditional Flow Matching

Replaces standard Gaussian noise paths with **straight OT-coupled transport paths** (Hungarian algorithm mini-batch pairing), reducing path curvature and enabling high-fidelity generation in far fewer steps than DDPM.

**Key design choices:**
- **Self-conditioning:** 25% of training steps reuse prior denoised estimate (1→2 input channels via patch conv)
- **Logit-normal time curriculum:** Logistic-Normal(μ=0, σ=1.0) → σ×1.6 after epoch 120 — concentrates training on high-curvature mid-trajectory
- **Heun 2nd-order ODE:** predictor-corrector, 100 steps (2× NFE/step)
- **CFG:** 15% null-class dropout, inference scale w=2.5
- **EMA:** decay=0.9952, all evaluation on shadow weights only
- **Checkpoint averaging:** last 4 EMA checkpoints post-hoc averaged

**Physics-motivated evaluation (not just FID):**

| Metric | Value | Meaning |
|--------|-------|---------|
| PSD Pearson-R | **0.9886** | Radial power spectrum fidelity |
| Ring ΔR | **1 px** | Einstein ring centre offset |
| Pixel KL divergence | **0.258** | Intensity distribution match |
| Bootstrap Z | **+2.59** (P > 0.999) | Epoch 175→335 improvement significance |

---

## Repository Structure

```
LensForge/
├── Task 1/
│   ├── d8-task1.ipynb          # C8LensNet training + evaluation
│   ├── ARCHITECTURE_d8-task1.md
│   └── d8-task1-output_files/  # Checkpoints, ROC curves, calibration plots
├── Task 2/
│   ├── d8-task2.ipynb          # D4-PINN training + Optuna HPO
│   ├── ARCHITECTURE_d8-task2.md
│   └── d8-task2-output_files/  # Phase 1/2 checkpoints, confusion matrices
└── Task 3/
    ├── d8-task3.ipynb          # DiT+OT-CFM inference + physics eval
    ├── ARCHITECTURE_d8-task3.md
    └── (checkpoints loaded from Kaggle dataset attachment)
```

---

## Setup

```bash
git clone https://github.com/Aditya26189/LensForge
cd LensForge
pip install torch torchvision escnn timm optuna torch_fidelity scipy h5py gdown
```

Notebooks are self-contained and designed for **Kaggle T4 GPU** (2× for Tasks 1–2, single for Task 3). All dataset downloads handled via `gdown` inside notebooks.

---

## Tech Stack

`PyTorch 2.x` · `escnn` · `EfficientNetV2` · `Optuna` · `torch_fidelity` · `scipy` · `CUDA`

**Techniques:** C8/D4 Equivariant Networks · Differentiable Physics (PINN) · OT-CFM · DiT · AdaLN-Zero · Self-Conditioning · CFG · Heun ODE · Bootstrap CI

---

## Citation

If you use this work, please cite the ML4SCI DeepLense project:

```bibtex
@misc{lensforge2026,
  author = {Aditya Singh},
  title  = {LensForge: Equivariant Physics-Informed Deep Learning for Gravitational Lensing},
  year   = {2026},
  url    = {https://github.com/Aditya26189/LensForge}
}
```

---

*Developed as part of ML4SCI DeepLense GSoC 2026 test tasks · IIT Kharagpur*
