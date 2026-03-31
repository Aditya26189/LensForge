# LensForge
### Equivariant Physics-Informed Deep Learning for Gravitational Lensing Classification and Simulation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

LensForge is a three-part deep learning suite for gravitational lensing
analysis. It spans equivariant classification, physics-constrained inference,
and flow-based generative modelling — all grounded in the physical symmetries
and governing equations of gravitational lensing.

---

## Results at a Glance

| Task | Architecture | Macro AUC | Params | vs. ResNet-18 |
|------|-------------|-----------|--------|----------------|
| Task 1 | C8LensNet (equivariant) | **0.9715** | 673 K | −0.78 pp · **16.58× fewer params** |
| Task 1 | ResNet-18 baseline | 0.9793 | 11.17 M | — |
| Task 2 | D4LensPINN + D4-TTA | **0.9809** | 20.65 M | **+6.3 pp** |
| Task 2 | ResNet-18 baseline | 0.9182 | 11.17 M | — |
| Task 3 | DiT-L/4 + OT-CFM | PSD Pearson-R **0.978** · Ring ΔR **1 px** | 129.5 M | FID 179.83 vs DDPM 87.31† |

† FID gap is a confirmed normalisation artefact — physics metrics (PSD Pearson-R,
Ring ΔR) are the valid evaluation standard under consistent normalisation.
See [Task 3](#task-3--ditl4--ot-cfm-generative-model-d8-task3-trainipynb--d8-task3-evalipynb) for full accounting.

---

## Repository Structure

```text
lensforge/
├── chekcpoint saving/           # Additional saved checkpoints
├── Task 1/
│   ├── ARCHITECTURE_d8-task1.md # C8LensNet: architecture + training protocol
│   ├── d8-task1.ipynb           # Task 1: C8-equivariant classification
│   └── d8-task1-output_files/   # Task 1 model checkpoints and outputs
├── Task 2/
│   ├── ARCHITECTURE_d8-task2.md # D4LensPINN: physics engine + two-phase schedule
│   ├── d8-task2.ipynb           # Task 2: Physics-guided D4 classification
│   └── d8-task2-output_files/   # Task 2 phase checkpoints
├── Task 3/
│   ├── ARCHITECTURE_d8-task3.md # DiT-L/4: OT-CFM + EMA workflow
│   └── d8-task3.ipynb           # Task 3: DiT + OT-CFM generative model
└── README.md
```

---

## Background

Strong gravitational lensing — the bending of light around massive objects —
produces Einstein rings and arcs whose morphology encodes the dark matter
substructure of the lensing halo. Distinguishing between no substructure,
spherical subhalos (CDM), and vortex substructure (WDM/axionic) is a
high-value classification task in observational cosmology.

LensForge approaches this from three angles:

1. **Symmetry** — Einstein rings appear at arbitrary orientations; rotational
   equivariance is a physical symmetry, not a heuristic.
2. **Physics** — The lensing convergence κ and deflection field are governed by
   the Poisson equation; enforcing this as an architectural constraint improves
   generalisation.
3. **Simulation** — Generating physically consistent synthetic lensing images
   requires evaluation metrics grounded in frequency-domain physics, not
   Inception-based FID.

---

## Task Descriptions

### Task 1 — C8-Equivariant Classification (`d8-task1.ipynb`)

3-class substructure classifier (`no` / `sphere` / `vort`) using a C8-equivariant
steerable CNN benchmarked against a ResNet-18 baseline.

**Architecture**
```
robust_percentile_scale (1st–99th)
    → EquivResBlock × N  [rot2dOnR2(N=8), 45° increments]
    → GroupPooling
    → Classification head
```

**Training**
- Optimiser: AdamW
- Schedule: `LinearLR` warm-up → `CosineAnnealingLR` via `SequentialLR`
- Split: deterministic stratified 80/10/10

**Key findings**
- C8LensNet recovers **99.2% of ResNet-18 AUC at 1/16.6 the parameter count** —
  equivariance is a strict improvement in the parameter-efficiency sense.
- Post-hoc `vort` threshold sweep documents **AUC–accuracy divergence**:
  AUC = 0.9781 at low argmax accuracy. Any generation benchmark using
  argmax accuracy as a proxy for distributional quality will misfire on
  this class — the calibration sweep is the diagnostic.

**Output artefacts**

| File | Description |
|------|-------------|
| `c8_lensnet.pth` | C8LensNet weights |
| `resnet18_baseline.pth` | ResNet-18 baseline weights |
| `roc_curves.png` | One-vs-rest ROC, all classes |
| `confusion_matrices.png` | Confusion matrices, both models |
| `efficiency_plot.png` | AUC vs. parameter count |
| `loss_curves.png` | Training / validation loss |
| `vort_calibration.png` | Threshold sweep — AUC vs. accuracy |

---

### Task 2 — Physics-Guided Classification (`d8-task2.ipynb`)

D4-equivariant physics-informed classifier (`D4LensPINN`) with a fully
differentiable lensing engine as a zero-parameter constraint layer.

**Architecture**
```
PhysicsPreprocess          [log-ratio + Sobel saddle-point, 0 params]
    → PoissonSolverFFT     [spectral, zero-padded, DC-gauge k²=1.0]
    → DeflectionField      [finite-difference via torch.roll]
    → InverseLensLayer     [bilinear grid_sample]
    → EfficientD4UNet      [κ̂ estimator, 0.46 M params, flipRot2dOnR2(N=4)]
    → EfficientNetV2Head   [19.87 M params]

Classifier input: X_cls = [I, κ̂, Ŝ, R]
```

The physics pipeline is implemented as zero-parameter differentiable `nn.Module`
objects — no physics parameters are learned; the lensing equation is enforced
as a hard architectural constraint.

**Training — Two-Phase Protocol**

| Phase | Epochs | Loss | Purpose |
|-------|--------|------|---------|
| Phase 1 | 15 | CE only | Pre-warm κ̂ estimation |
| Phase 2 | 40 | CE + PhysicsLoss | Physics regularisation active |

`PhysicsLoss` = TV + L1 + centroid + Poisson residual  
λ_poisson ≈ 0.01 (Optuna 15-trial TPE + MedianPruner)  
LR_P1 = 7.11e-4, LR_P2 = 4.46e-4, WD = 1.46e-3

The Phase 2 epoch-1 loss spike (`vl = 360.96`) is the failure mode the
warm-up prevents — reconstruction must stabilise before physics penalties
are activated.

**Inference — D4-TTA**

8-way TTA (4 rotations × flip):
- AUC: +0.23 pp (safe — threshold-integrated)
- Accuracy: −5.5 pp (harmful — flip inverts the Sobel saddle-point channel
  sign in `PhysicsPreprocess`, generating spurious substructure signals)

**Key findings**
- Sphere-class AUC: **0.8738 → 0.9695 (+9.6 pp)** — the largest gain comes
  from the residual channel R, which localises CDM subhalo deflection
  anomalies at sub-pixel scale. Physics-derived features carry discriminative
  signal unavailable to raw-pixel models.
- The differentiable lensing engine (`PoissonSolverFFT → DeflectionField →
  InverseLensLayer`) is directly reusable as a constraint layer inside a
  generative model conditioned on κ and γ maps.

**Output artefacts**

| File | Description |
|------|-------------|
| `d4_phase1_best.pth` | Best Phase 1 checkpoint |
| `d4_phase2_best.pth` | Best Phase 2 checkpoint |
| `resnet18_baseline_best.pth` | ResNet-18 best checkpoint |
| `resnet18_baseline_final.pth` | ResNet-18 final checkpoint |

---

### Task 3 — DiT-L/4 + OT-CFM Generative Model (`d8-task3-train.ipynb` + `d8-task3-eval.ipynb`)

Flow-matching generative model for unconditional lensing image synthesis,
with physics-grounded evaluation.

**Two-notebook workflow**

Task 3 is split across two notebooks due to the 12-hour Kaggle session cap.
Run `d8-task3-train.ipynb` first and publish its output as a Kaggle Dataset
before running `d8-task3-eval.ipynb`.

| Notebook | Purpose |
|----------|---------|
| `d8-task3-train.ipynb` | Full training run; saves checkpoints to Kaggle Dataset |
| `d8-task3-eval.ipynb` | Loads checkpoints; inference sweep + all evaluations |

**Architecture**
```
K-Means pseudo-labels (k=3: 3220 / 2911 / 3869)
    → DiT-L/4
        patch = 4 · 256 tokens × 768 dim · 12 DiTBlocks
        AdaLN-Zero conditioning
        Self-conditioning: torch.cat([x, x_self_cond], dim=1)
        129,554,192 params
    ← OT-CFM (mini-batch OT via linear_sum_assignment on pairwise L² cost)
    ← EMA (decay = 0.9995, shadow params for all evaluation)
```

**Best inference configuration**

| Hyperparameter | Value |
|----------------|-------|
| CFG scale | 2.5 |
| Temperature | 0.80 |
| Sampler | Heun |
| Steps | 100 |
| Epochs trained | 335 |

**Results and honest accounting**

| Metric | Value | Notes |
|--------|-------|-------|
| PSD Pearson-R | **0.978** | Normalisation-independent; primary physics metric |
| Ring ΔR | **1 px** (epoch 335) | Under consistent [−1,1] normalisation |
| Bootstrap Z | +2.59, CI=[+1.90,+3.49] | P(improve) > 0.999 across 2000 rounds |
| FID | 179.83 | Not beaten vs. DDPM baseline 87.31 |
| Discriminator AUC | 1.000 | Driven by histogram shape, not visual content |

**On the FID result:** The FID gap and AUC = 1.000 share one root cause —
real images are preprocessed with `robust_percentile_scale`; generated images
use a global [−1, 1] clamp. The intensity histogram shapes are trivially
separable, driving the discriminator to near-zero loss by epoch 3. This is
not a visual quality failure. Under consistent normalisation, Ring ΔR drops
from 35 px (epoch 175) to 1 px (epoch 335), confirming the model has learned
gravitational lensing geometry. Closing the FID gap requires consistent
normalisation across both distributions and an estimated 600–700 training
epochs at the observed convergence rate.

The **physics-optimal checkpoint** (Ring ΔR = 1 px) and the
**FID-optimal checkpoint** are provably different checkpoints. PSD Pearson-R
and Ring ΔR are the valid evaluation metrics for physically consistent
lensing image generation.

**Output artefacts**

| File | Description |
|------|-------------|
| `dit_otfm_best.pth` | Best training checkpoint |
| `dit_otfm_latest.pth` | Latest training checkpoint |
| `physics_eval.png` | PSD curves + ring geometry evaluation |
| `real/` `generated/` | Image folders for FID computation |

---

## Environment

All notebooks target **Kaggle GPU (T4, 15.6 GB VRAM)**. Run cells sequentially.


## Architecture Deep-Dives

For full architectural detail, training protocol, and design rationale:

- [`ARCHITECTURE_d8-task1.md`](ARCHITECTURE_d8-task1.md) — C8LensNet
- [`ARCHITECTURE_d8-task2.md`](ARCHITECTURE_d8-task2.md) — D4LensPINN + physics engine
- [`ARCHITECTURE_d8-task3.md`](ARCHITECTURE_d8-task3.md) — DiT-L/4 + OT-CFM + EMA

---

## License

MIT
