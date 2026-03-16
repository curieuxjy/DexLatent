"""Numerical evaluation for cross-embodiment hand retargeting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from HandLatent.infer import (
    EvaluationConfig,
    decode_hand_sequence_eepose,
    encode_hand_sequence_eepose,
)
from HandLatent.model import (
    CrossEmbodimentTrainer,
    TrainingConfig,
    compute_pinch_loss,
)


def evaluate_self_reconstruction(
    trainer: CrossEmbodimentTrainer,
    hand_name: str,
    qpos_norm: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate encode→decode on the same hand (self-reconstruction).

    Returns hand qpos MSE.
    """
    device = trainer.config.device
    qpos = qpos_norm.to(device=device)

    with torch.no_grad():
        _, hand_gt = trainer._split_qpos(hand_name, qpos)
        _, mean_hand, _ = trainer.autoencoders[hand_name].encode(qpos)
        _, qpos_hand_pred = trainer.autoencoders[hand_name].decode_from_latents(
            qpos[:, : trainer.arm_dof], mean_hand
        )
        mse = torch.nn.functional.mse_loss(qpos_hand_pred, hand_gt).item()

    return {"hand_qpos_mse": mse}


def evaluate_cross_embodiment_pinch(
    trainer: CrossEmbodimentTrainer,
    source_hand: str,
    target_hand: str,
    qpos_norm: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate cross-embodiment pinch loss (distance + direction)."""
    device = trainer.config.device
    qpos = qpos_norm.to(device=device)
    pinch_pairs = list(trainer.shared_pinch_pairs(source_hand, target_hand))

    if len(pinch_pairs) == 0:
        return {"pinch_distance": 0.0, "pinch_direction": 0.0}

    with torch.no_grad():
        source_tips = trainer.hand_models[source_hand].forward(qpos)

        _, mean_hand, _ = trainer.autoencoders[source_hand].encode(qpos)
        latent_arm = qpos[:, : trainer.arm_dof]
        _, target_hand_pred = trainer.autoencoders[target_hand].decode_from_latents(
            latent_arm, mean_hand
        )
        target_qpos = trainer._merge_qpos(latent_arm, target_hand_pred)
        target_tips = trainer.hand_models[target_hand].forward(target_qpos)

        dist_term, dir_term, weights = compute_pinch_loss(
            source_tips, target_tips, pinch_pairs, trainer.config.lambda_dis_exp
        )

    return {
        "pinch_distance": (dist_term * weights).mean().item(),
        "pinch_direction": (dir_term * weights).mean().item(),
    }


def evaluate_fingertip_position(
    trainer: CrossEmbodimentTrainer,
    source_hand: str,
    target_hand: str,
    source_qpos_norm: torch.Tensor,
    decoded_qpos: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate fingertip L2 distance between source and decoded target.

    Compares shared fingertips (min of source/target tip count).
    """
    with torch.no_grad():
        source_tips = trainer.hand_models[source_hand].forward(
            source_qpos_norm.to(device=trainer.config.device)
        )
        target_tips = trainer.hand_models[target_hand].forward(
            decoded_qpos.to(device=trainer.config.device)
        )

    n_shared = min(source_tips.shape[1], target_tips.shape[1])
    source_shared = source_tips[:, :n_shared, :]
    target_shared = target_tips[:, :n_shared, :]

    per_tip_l2 = torch.norm(source_shared - target_shared, dim=-1)
    mean_l2 = per_tip_l2.mean().item()
    per_finger_l2 = per_tip_l2.mean(dim=0).tolist()

    finger_names = ["thumb", "index", "middle", "ring", "pinky"][:n_shared]
    result = {"mean_fingertip_l2": mean_l2}
    for name, val in zip(finger_names, per_finger_l2):
        result[f"fingertip_l2_{name}"] = val

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerical evaluation of retargeting quality.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="Checkpoints/20260311_225425/checkpoint_epoch_1000.pt",
    )
    parser.add_argument("--data", type=str, default="Dataset/demo.npz")
    parser.add_argument(
        "--side", type=str, default="right", choices=("right", "left")
    )
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Random sample count for evaluation (0 = use demo data only)")
    args = parser.parse_args()

    side = args.side
    hand_names = [
        f"xarm7_xhand_{side}",
        f"xarm7_ability_{side}",
        f"xarm7_inspire_{side}",
        f"xarm7_paxini_{side}",
        f"xarm7_allegro_{side}",
    ]

    config = TrainingConfig()
    trainer = CrossEmbodimentTrainer(hand_names, config)
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    payload = torch.load(ckpt_path, map_location="cpu")
    trainer.load_autoencoders_from_payload(payload)

    source_hand = f"xarm7_inspire_{side}"

    # --- Load or generate evaluation data ---
    if args.num_samples > 0:
        n = args.num_samples
        dof = trainer.dof_per_hand[source_hand]
        qpos_norm = torch.empty(n, dof).uniform_(-1.0, 1.0).to(device=config.device)
        print(f"Using {n} random samples for evaluation\n")
    else:
        data_path = Path(args.data).expanduser().resolve()
        with np.load(data_path) as dataset:
            source_qpos_raw = torch.as_tensor(dataset[f"{side}_qpos"], dtype=torch.float32)
        qpos_norm = trainer.normalized_qpos(source_hand, source_qpos_raw).to(device=config.device)
        print(f"Using demo data ({qpos_norm.shape[0]} frames) for evaluation\n")

    # === 1. Self-reconstruction ===
    print("=" * 60)
    print("1. Self-Reconstruction Error (encode→decode same hand)")
    print("=" * 60)
    n_eval = args.num_samples if args.num_samples > 0 else qpos_norm.shape[0]
    for hand in hand_names:
        dof = trainer.dof_per_hand[hand]
        if hand == source_hand and args.num_samples == 0:
            hand_qpos = qpos_norm
        else:
            hand_qpos = torch.empty(n_eval, dof).uniform_(-1.0, 1.0).to(device=config.device)
        metrics = evaluate_self_reconstruction(trainer, hand, hand_qpos)
        data_src = "demo" if (hand == source_hand and args.num_samples == 0) else "random"
        print(f"  {hand:30s}  hand_qpos_mse = {metrics['hand_qpos_mse']:.6f}  ({data_src})")
    print()

    # === 2. Cross-embodiment pinch loss ===
    print("=" * 60)
    print("2. Cross-Embodiment Pinch Loss (source→target)")
    print("=" * 60)
    for target in hand_names:
        if target == source_hand:
            continue
        metrics = evaluate_cross_embodiment_pinch(trainer, source_hand, target, qpos_norm)
        print(f"  {source_hand} → {target}")
        print(f"    pinch_distance  = {metrics['pinch_distance']:.6f}")
        print(f"    pinch_direction = {metrics['pinch_direction']:.6f}")
    print()

    # === 3. Fingertip position error (full pipeline with IK) ===
    print("=" * 60)
    print("3. Fingertip Position Error (full encode→decode with IK)")
    print("=" * 60)

    with torch.no_grad():
        latents = encode_hand_sequence_eepose(trainer, source_hand, qpos_norm)

    for target in hand_names:
        with torch.no_grad():
            decoded = decode_hand_sequence_eepose(
                trainer, target, latents, EvaluationConfig()
            ).cpu()

        metrics = evaluate_fingertip_position(
            trainer, source_hand, target, qpos_norm.cpu(), decoded
        )
        print(f"  {source_hand} → {target}")
        print(f"    mean_fingertip_l2 = {metrics['mean_fingertip_l2']:.6f} m")
        for key, val in metrics.items():
            if key.startswith("fingertip_l2_"):
                finger = key.replace("fingertip_l2_", "")
                print(f"    {finger:10s} = {val:.6f} m")
    print()


if __name__ == "__main__":
    main()
