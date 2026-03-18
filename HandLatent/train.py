"""Training entry point for minimal hand latent project."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import wandb

from HandLatent.model import CrossEmbodimentTrainer, TrainingConfig

def main() -> None:
    """Run latent training with default parameters matching the reference repo.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Trains the model and writes checkpoints.
    """

    parser = argparse.ArgumentParser(
        description="Train minimal cross-embodiment hand latent model."
    )
    parser.add_argument("--num_steps", type=int, default=1_000)
    parser.add_argument("--checkpoint_interval", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--pinch_template_count", type=int, default=2048)
    parser.add_argument("--pinch_template_iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to train_cfg.json to load parameters from.",
    )
    args = parser.parse_args()

    # If --config is given, load parameters from JSON (CLI args still override)
    if args.config is not None:
        cfg_path = Path(args.config).expanduser().resolve()
        with open(cfg_path) as f:
            saved_cfg = json.load(f)
        defaults = {k: saved_cfg[k] for k in saved_cfg if k in vars(args)}
        parser.set_defaults(**defaults)
        args = parser.parse_args()

    torch.manual_seed(args.seed)
    hand_names = [
        "xarm7_xhand_right",
        "xarm7_ability_right",
        "xarm7_inspire_right",
        "xarm7_paxini_right",
        "xarm7_allegro_right",
    ]

    train_params = {
        "num_steps": args.num_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "pinch_template_count": args.pinch_template_count,
        "pinch_template_iterations": args.pinch_template_iterations,
        "seed": args.seed,
    }

    config = TrainingConfig(
        num_steps=args.num_steps,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pinch_template_count=args.pinch_template_count,
        pinch_template_iterations=args.pinch_template_iterations,
    )

    os.environ["WANDB_SILENT"] = "true"
    wandb.init(
        project="DexLatent",
        config=train_params,
        settings=wandb.Settings(disable_git=True),
        mode="online",
    )

    trainer = CrossEmbodimentTrainer(hand_names, config)
    trainer._step_callback = lambda step, m: wandb.log(m, step=step)

    # Save train_cfg.json to checkpoint dir right after it's created
    def _save_cfg_on_first_step(step: int, metrics: dict) -> None:
        cfg_path = Path(trainer.checkpoint_dir) / "train_cfg.json"
        if not cfg_path.exists():
            with open(cfg_path, "w") as f:
                json.dump(train_params, f, indent=2)
            print(f"Saved training config to {cfg_path}")
        wandb.log(metrics, step=step)

    trainer._step_callback = _save_cfg_on_first_step
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
