# CLAUDE.md — DIANE (Dynamic Inference Adaptation with Neural Elasticity)

## Project Overview

DIANE is a research prototype implementing **NP-TTA (Neural-Path Test-Time Adaptation)** on CIFAR-10 / CIFAR-10-C. It investigates how neural networks adapt at inference time under distribution shift using teacher-student distillation, dynamic neuron routing, sequential hypothesis testing, and continual memory-based adaptation.

**Author:** Rosario Patanè (2025)

## Architecture Summary

Single-file Python/PyTorch implementation (`main.py`, ~1435 LOC) with these components:

- **ResNet18WithAdapter**: ResNet18 backbone + lightweight adapter (bottleneck MLP) + depthwise `in_adapter` on input. Adapter is identity-initialized (zeros on last layer) so the model starts as vanilla ResNet18.
- **Teacher EMA**: Two separate teacher networks (main + blur) updated via exponential moving average of student weights.
- **RoTTA (Robust Test-Time Adaptation)**: RobustBN2d replacing standard BatchNorm + CSTU memory bank for continual-stable updates.
- **SPRT (Sequential Probability Ratio Test)**: Gating mechanism for detecting blur-type distribution shifts.
- **BN Banks**: Separate BatchNorm running stats for blur vs. non-blur segments to prevent cross-contamination.
- **Multi-view teacher aggregation**: Multiple augmented views averaged to produce robust pseudo-labels and disagreement statistics.

## File Structure

```
DIANE/
├── main.py          # Complete implementation (all modules, training, eval, sweep)
├── README.md        # Project description and citation
├── CLAUDE.md        # This file
├── .git/            # Git repository
├── data/            # Auto-created: CIFAR-10 and CIFAR-10-C datasets
└── checkpoints/     # Auto-created: model checkpoints and sweep results
```

## Dependencies

- Python 3.8+
- PyTorch (with CUDA recommended)
- torchvision
- NumPy
- tqdm
- Standard library: argparse, copy, csv, math, os, random, tarfile, urllib, pathlib, dataclasses, typing

No `requirements.txt` exists. Install manually:
```bash
pip install torch torchvision numpy tqdm
```

## Entry Points and CLI

All execution is via `python main.py [flags]`:

| Mode | Command | Description |
|------|---------|-------------|
| **Train** | `python main.py --train` | Train source ResNet18+adapter on CIFAR-10 |
| **Stream eval** | `python main.py --eval` | Run baseline + NP-TTA on 6-segment corruption stream |
| **C10-C eval** | `python main.py --eval_c10c` | Per-corruption evaluation on CIFAR-10-C |
| **Sweep** | `python main.py --sweep` | Random hyperparameter search |

### Key CLI Flags (grouped)

**Training:** `--epochs`, `--batch_size`, `--lr`, `--seed`, `--ckpt`, `--amp`

**Stream config:** `--stream_len` (default 20000), `--adapter_rank` (default 128)

**NP-TTA core:** `--entropy_thresh`, `--tta_lr`, `--disagree_thresh`, `--teacher_momentum`, `--confidence_min`, `--conf_frac_min`, `--adapt_steps`, `--aug_views`

**Blur branch:** `--blur_loss {distill,im,kld}`, `--blur_distill_temp`, `--blur_distill_w`, `--blur_lr_scale`, `--blur_phys_w`, `--in_adapter_lr_scale`, `--no_blur_bn_bank`, `--blur_update_teacher`

**SPRT:** `--use_sprt_for_blur`, `--sprt_alpha`, `--sprt_beta`, `--sprt_mu0`, `--sprt_mu1`, `--sprt_sigma`

**Regularization:** `--prox_w`, `--prox_w_id`, `--adapter_decay_when_clean`

**RoTTA:** `--rotta_enable`, `--rotta_memory_size`, `--rotta_update_frequency`, `--rotta_batch_size`, `--rotta_steps`, `--rotta_alpha`, `--rotta_robustbn_mom`

## Stream Schedule

The default evaluation stream (`--eval`) has 6 segments:
```
clean → gauss(sev=3) → clean → blur(sev=3) → clean → gauss(sev=3, revisit)
```
Each segment is `stream_len / (6 * batch_size)` batches.

## Code Architecture (main.py sections)

| Lines | Section | Key symbols |
|-------|---------|-------------|
| 25–110 | **Utilities** | `set_seed`, `accuracy`, `entropy_from_probs`, `ema_update`, `freeze_bn_stats`, `bn_buffers_state`, `load_bn_buffers`, `unsharp_mask` |
| 113–270 | **RoTTA** | `RobustBN2d`, `replace_bn_with_robustbn`, `robustbn_update_from_input`, `MemItem`, `CSTUMemory` |
| 272–321 | **Corruptions** | `apply_corruption`, `CIFAR10C`, `c10c_kind_tag` |
| 323–375 | **Model** | `ResNet18WithAdapter`, `freeze_all_except_adapter`, `adapter_decay_step`, `stochastic_restore_backbone` |
| 377–431 | **Teacher views** | `make_teacher_views_blur`, `make_teacher_views_general`, `teacher_aggregate` |
| 434–524 | **Config** | `NPTTAConfig` dataclass, `build_cfg_from_args` |
| 526–564 | **Training** | `train_source` (SGD + cosine schedule) |
| 567–610 | **Stream helpers + Baseline** | `build_schedule`, `eval_stream_baseline` |
| 612–1165 | **NP-TTA eval loop** | `eval_stream_nptta` — the core adaptation loop |
| 1167–1259 | **Sweep** | `run_one_tta_eval`, `sweep` |
| 1263–1435 | **CLI + main** | `parse_args`, `main` |

## Adaptation Loop Flow (eval_stream_nptta)

```
For each segment (kind, severity, num_batches):
  1. BN bank switch (if entering/leaving blur segment)
  2. For each batch:
     a. Apply corruption
     b. RobustBN update on current batch (if rotta_enable and not clean)
     c. Source model confidence check (cheap gating)
     d. Teacher multi-view aggregation → p_t, conf, ent, disagree
     e. RoTTA memory add (pseudo-label + uncertainty)
     f. RoTTA periodic memory update (KL loss on sampled memory batch)
     g. Gating decision: do_adapt based on entropy/disagreement/SPRT
     h. If adapting: compute loss (KLD + consistency + physics + prox)
     i. Safety: NaN/Inf rollback, loss hard cap, grad clipping
     j. Teacher EMA update
     k. Evaluate with teacher predictions
  3. Save/restore BN bank state
```

## Numerical Safety Rules

- **Loss hard cap**: `LOSS_HARD_CAP = 50.0` — skip step if exceeded
- **NaN/Inf rollback**: restore adapter to segment anchor checkpoint
- **Gradient clipping**: norm clipped to 1.0 (or 0.5 when high disagreement)
- **EPS clamping**: `1e-12` on probabilities before log
- **Scaler**: AMP GradScaler with proper unscale/update sequencing

## Key Design Decisions and Conventions

1. **Monolithic single-file**: Designed for easy Colab portability. All classes/functions in `main.py`.
2. **Adapter-only fine-tuning**: Only adapter + in_adapter weights have `requires_grad=True` during TTA. Backbone and classifier are frozen.
3. **Dual teacher/BN bank**: Blur and non-blur segments each maintain their own teacher EMA, adapter state, and BN running stats.
4. **Teacher for evaluation**: Final accuracy is computed using teacher (EMA) predictions, not the student.
5. **Clean protection**: `do_adapt = False` during clean segments. Soft restore toward identity adapter on clean batches.
6. **No external config files**: All configuration via CLI argparse with sane defaults.
7. **Comments mix Italian and English**: Some inline comments are in Italian (e.g., "momenti", "stampa", "ricalibrare").

## Development Guidelines

### Making Changes
- **Small, isolable patches**: Prefer minimal targeted fixes with ablation flags and sanity prints.
- **No new heavy dependencies**: Stick to PyTorch + torchvision + numpy + tqdm.
- **CLI backwards-compatible**: Add new flags with safe defaults; don't break existing CLI invocations.
- **Seed reproducibility**: All changes must be reproducible with `--seed`.

### Testing a Change
```bash
# Train source model (if no checkpoint exists)
python main.py --train --epochs 3 --amp

# Quick stream eval (primary benchmark)
python main.py --eval --amp

# Per-corruption eval on C10-C
python main.py --eval_c10c --amp --c10c_corruptions gaussian_noise,gaussian_blur --c10c_severities 3

# Hyperparameter sweep
python main.py --sweep --sweep_trials 10 --amp
```

### What to Check in Logs
- **Gating stats**: `gate=X/N` — how many batches triggered adaptation
- **Confidence**: `conf_ok=X/N`, `conf_m`, `conf_f` — teacher confidence levels
- **Loss scale**: `loss_kld` should be O(0.01–1.0), not O(10–100)
- **Gradient norm**: `grad` should be < 1.0 after clipping
- **Mask coverage**: `mask` — fraction of samples passing confidence threshold
- **Memory**: `[SANITY][Mem]` lines for RoTTA memory state
- **VRAM**: `peakVRAM` in MB
- **Clean accuracy**: `avg_clean` should not drop > 0.002 from baseline

### Known Pitfalls
1. **RobustBN contamination**: `robustbn_update_from_input` must NOT run during clean segments (guarded by `kind != "clean"` check).
2. **Memory time stepping**: `step_time` is called once per batch (not per sample) to avoid age explosion. The `add()` call uses `step_time=False` after the first sample in a batch.
3. **BN bank device mismatch**: After `replace_bn_with_robustbn`, must call `.to(device)` on backbone.
4. **Teacher update during blur**: Controlled by `blur_update_teacher` flag. Default is False to prevent blur-contaminated teacher from hurting non-blur segments.
5. **Distillation temperature**: `blur_distill_temp` < 1.0 sharpens distributions. With low T, KLD magnitude scales by T^2.

## No Tests / No CI

This is a research prototype. There is no test suite, no CI/CD pipeline, and no linter configuration. Validation is done by running `--eval` and checking accuracy metrics against baseline.

## Metrics of Interest

| Metric | Description | Target |
|--------|-------------|--------|
| `avg_clean` | Mean accuracy across clean segments | No drop from baseline (|delta| <= 0.002) |
| `blur3` | Accuracy on blur severity=3 segment | Improve over baseline |
| `gauss3 revisit` | Accuracy on second gauss severity=3 segment | Improve over baseline |
| `peakVRAM` | Peak GPU memory in MB | Reasonable (no doubling from bugs) |
| `ms/batch` | Latency per batch | No degradation from adaptation overhead |
