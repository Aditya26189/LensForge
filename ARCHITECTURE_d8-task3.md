# Architecture Overview: d8-task3.ipynb

## 1) Goal and scope
This notebook implements a run-after-train generative pipeline for DeepLense Task 3 using a DiT + OT-CFM design with EMA inference.

Primary goals:
- generate class-conditional lensing-like images
- optimize inference-time settings (CFG, temperature)
- evaluate generation quality with FID/ISC, discriminator AUC, and physics metrics

Workflow mode:
- inference/evaluation-first
- training blocks are present but commented in this notebook revision
- checkpoint loading from external Kaggle dataset paths is expected

## 2) High-level pipeline
1. Environment setup (CUDA/NCCL stability flags, package install)
2. Imports, global config, dataset extraction, pseudo-label generation
3. Dataset/DataLoader sanity checks
4. DiT model definition
5. EMA, OT pairing, and checkpoint utility setup
6. Checkpoint comparison (epoch 175 vs 335) for trajectory diagnostics
7. Bootstrap confidence analysis on physics deltas
8. Sample grid generation
9. CFG/temperature sweep with FID/ISC scoring
10. Real-vs-fake discriminator AUC
11. Physics metrics (PSD correlation, ring offset, KL)
12. Final summary and self-audit

## 3) Data architecture
Inputs:
- Task 8 npy dataset (generation and conditioning source)
- Task 1 npy dataset (reference set paths available in config)

Preprocessing and dataset blocks:
- to_grayscale_2d(img_np)
- percentile_norm(img_np, p_low=1.0, p_high=99.0)
- FlowLensingDataset (returns tensor shape (1, 64, 64) and class label)

Labeling strategy:
- MiniBatchKMeans pseudo-labels over normalized Task 8 images
- labels persisted to task8_pseudo_labels.json for reuse

## 4) Core generative architecture

### 4.1 Backbone (DiT)
Main classes:
- SinusoidalPosEmb2D
- DiTBlock
- DiT

Design features:
- patch embedding with patch size 4 on 64x64 images
- AdaLN-Zero style modulation in transformer blocks
- class conditioning via embedding table including NULL_CLASS
- time embedding MLP
- self-conditioning input path (x plus x_self_cond)

### 4.2 Flow objective and coupling
Training-side objective (implemented in commented training block):
- conditional flow matching target v_target = x1_paired - x0
- optional OT pairing via linear_sum_assignment
- optional global OT path for distributed setting

### 4.3 EMA inference path
Main class:
- EMA

Behavior:
- maintains shadow parameters for evaluation
- FID sweep, sampling, and comparison use EMA shadow weights

## 5) Sampling architecture
Sampling functions:
- euler_sample(...)
- flow_sample_cfg_temp(...)
- _sample_with_same_noise(...)

Inference controls:
- CFG scale
- temperature scaling of initial noise
- number of integration steps (EULER_STEPS)
- Heun toggle (USE_HEUN_SAMPLER)

Sweep logic:
- loops over CFG_SWEEP_VALUES and TEMP_SWEEP_VALUES
- generates per-combination folders
- computes FID and ISC
- selects best FID run and copies outputs into fid_fake_rgb

## 6) Evaluation architecture

### 6.1 Image-distribution metrics
- FID and ISC via torch_fidelity.calculate_metrics
- real set prepared in fid_real_rgb
- generated set prepared in fid_fake_rgb or per-sweep directories

### 6.2 Discriminator realism test
- ResNet-18 binary classifier (real vs fake)
- output metric: AUC_SCORE

### 6.3 Physics-oriented metrics
Functions:
- radial_power_spectrum(images_np)
- ring_radial_profile(images_np)

Reported metrics:
- PEARSON_R (PSD correlation)
- RING_DR (ring radius/position offset proxy)
- KL_DIV (histogram divergence)

## 7) Robustness and audit architecture
Safety checks and diagnostics:
- shape and dtype assertions in early cells
- checkpoint path checks before load
- min sample-count guards before AUC/physics evaluation
- self-audit cell validating config symbols, artifacts, and metric presence

Known runtime failure mode:
- np.stack on empty fake image list if generation cell is skipped
- mitigation: run FID/generation sweep before AUC/physics cells

## 8) Key artifacts produced
Generated folders:
- checkpoints_task8
- fid_real_rgb
- fid_fake_rgb
- fid_fake_rgb_cfg_*_temp_*

Typical outputs:
- generated_grid.png
- generated_unconditional.png
- compare_epoch175_vs_335.png
- physics_eval.png
- real_vs_generated.png
- training_history_task8.json (only if training is enabled elsewhere)

## 9) Design rationale summary
- DiT provides scalable transformer-based image modeling with conditioning support.
- OT-CFM framing targets smoother transport paths versus vanilla diffusion trajectories.
- EMA improves inference stability and metric quality versus raw training weights.
- CFG/temperature sweep separates model quality from inference hyperparameter effects.
- Physics metrics complement FID/ISC and expose domain-relevant structural behavior.

## 10) Scope and limits
This architecture document describes the notebook as implemented in its current run-after-train style revision.
It does not claim end-to-end runtime correctness in all environments without execution, dataset availability, and checkpoint availability.