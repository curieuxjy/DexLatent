# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DexLatent is the official implementation of **XL-VLA (CVPR 2026)** — a cross-embodiment hand latent learning and retargeting system. It trains VAE-style autoencoders that map multiple dexterous hand embodiments (xhand, ability, inspire, paxini, allegro) into a shared latent space, enabling trajectory transfer between different hand morphologies on XArm7 arms.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Train from scratch (wandb logging enabled automatically)
uv run -m HandLatent.train
uv run -m HandLatent.train --num_steps 1000 --batch_size 1024 --learning_rate 0.002

# Resume training with saved config
uv run -m HandLatent.train --config Checkpoints/<timestamp>/train_cfg.json

# Run inference/visualization (uses pretrained checkpoint by default)
uv run -m HandLatent.infer
uv run -m HandLatent.infer --ckpt <path> --data <path> --side right

# Evaluate (self-reconstruction, cross-embodiment pinch, fingertip position errors)
uv run -m HandLatent.evaluate --ckpt <path> --side right

# View URDF in Rerun (any hand, optional animation)
uv run -m HandLatent.view_urdf
uv run -m HandLatent.view_urdf --hand xarm7_allegro_right --animate
```

There are no tests or linting configured.

## Architecture

### Module Dependency Graph

```
train.py ────> model.py ──> kinematics.py
infer.py ────> model.py
             ──> ik.py ────> kinematics.py
             ──> visualize.py ──> kinematics.py
evaluate.py ─> model.py
             ──> ik.py
view_urdf.py ─> kinematics.py (Rerun URDF viewer)
```

### Core Design

**CrossEmbodimentTrainer** (`model.py`) is the central class. It manages five separate `HandAutoencoder` instances (one per hand embodiment) that share a common latent space (dim=32). The arm (7 DOF) passes through unchanged; only the hand configuration is encoded/decoded.

**Training pipeline:**
1. Pre-generates pinch template poses via IK (`pinch_template_count=2048`)
2. Samples mixed batches: 50% uniform random + 50% pinch-centric poses
3. Loss = reconstruction MSE + cross-embodiment pinch loss (distance + direction) + KL regularization

**Inference pipeline:**
1. Encode source trajectory → hand latent + arm pass-through
2. Decode to each target hand using latent + cached IK arm seed (`TrainerCacheState`)
3. Visualize with Rerun (URDF-based 3D rendering)

### Key Classes

- `HandAutoencoder` (`model.py`): VAE encoder/decoder MLPs per hand type
- `MultiHandDifferentiableFK` (`kinematics.py`): Differentiable forward kinematics, parses URDFs from `Assets/`
- `pink_align_arm` (`ik.py`): Gradient-based IK via Pink/Pinocchio for arm pose solving
- `TrainingConfig` (`model.py`): All hyperparameters as a dataclass

### 4-Finger Hands

Paxini and Allegro have 4 fingers (no pinky). Their `tip_links` in `HAND_CONFIGS` are padded to 5 entries by duplicating the ring finger tip (`link_11.0_tip`). This enables consistent 5-way fingertip comparison across all hands but means the "pinky" slot for these hands is actually the ring fingertip again.

### URDF Notes

Rerun's URDF loader requires materials referenced by name (e.g., `<material name="black" />`) to have global definitions under the `<robot>` tag. Inline-only material definitions cause `KeyError`.

### Data Layout

- `Assets/xarm7_*/` — URDF models for each hand embodiment (left/right)
- `Dataset/demo.npz` — Demo trajectory data
- `Checkpoints/<timestamp>/checkpoint_epoch_XXXX.pt` — Saved autoencoder state dicts
- `Checkpoints/<timestamp>/train_cfg.json` — Training parameters snapshot (for reproducibility via `--config`)

### Training Logging

`train.py` integrates wandb (project: `DexLatent`). Logged per step: `loss_total`, `loss_rec_total`, `loss_rec_hand`, `loss_pinch_dis`, `loss_pinch_dir`, `loss_kl`, `exp_dis`, `grad_norm`, `lr`, plus per-hand breakdowns under `rec/{hand}` and `kl/{hand}`. Training parameters are saved as `train_cfg.json` in the checkpoint directory. `_step_callback` on `CrossEmbodimentTrainer` is the hook used by `train.py` for wandb logging.

### Hardware

Auto-detects CUDA; falls back to CPU. Training and IK solving benefit from GPU.
