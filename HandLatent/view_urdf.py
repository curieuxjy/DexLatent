"""Rerun으로 URDF 로봇을 띄워서 관절을 슬라이더로 조작해보는 스크립트.

Usage:
    # allegro right (기본)
    uv run scripts/view_urdf.py

    # 다른 핸드
    uv run scripts/view_urdf.py --hand xarm7_inspire_right

    # 랜덤 포즈 애니메이션
    uv run scripts/view_urdf.py --animate
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation
from urdf_parser_py import urdf as urdf_parser

from HandLatent.kinematics import HAND_CONFIGS, load_urdf_silent
from HandLatent.visualize import (
    discover_revolute_joints,
    discover_mimic_joints,
    resolve_urdf_path,
)


def set_joint_angles(recording, joints, mimic_joints, angles, step=0):
    """관절 각도를 Rerun에 로깅."""
    recording.set_time("step", duration=step * 0.02)
    evaluated = {}
    for joint, angle in zip(joints, angles):
        evaluated[joint.name] = float(angle)
        quat = (
            Rotation.from_quat(joint.origin_quaternion)
            * Rotation.from_rotvec(joint.axis * float(angle))
        ).as_quat()
        recording.log(
            joint.link_path,
            rr.Transform3D.from_fields(
                translation=joint.origin_translation,
                quaternion=quat,
            ),
        )
    for mj in mimic_joints:
        angle = mj.angle_from_reference(evaluated[mj.reference])
        evaluated[mj.name] = angle
        quat = (
            Rotation.from_quat(mj.origin_quaternion)
            * Rotation.from_rotvec(mj.axis * angle)
        ).as_quat()
        recording.log(
            mj.link_path,
            rr.Transform3D.from_fields(
                translation=mj.origin_translation,
                quaternion=quat,
            ),
        )


def main():
    parser = argparse.ArgumentParser(description="Rerun URDF viewer")
    parser.add_argument(
        "--hand", default="xarm7_allegro_right",
        help=f"핸드 이름. 선택지: {', '.join(HAND_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="랜덤 포즈 사이를 보간하며 애니메이션",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="애니메이션 프레임 수 (기본: 200)",
    )
    args = parser.parse_args()

    hand_name = args.hand
    urdf_path = resolve_urdf_path(hand_name)
    urdf = load_urdf_silent(urdf_path)
    prefix = urdf.get_root()

    rr.init(f"view_{hand_name}", spawn=True)
    rec = rr.get_global_data_recording()

    from rerun_loader_urdf import URDFLogger
    logger = URDFLogger(urdf_path, prefix)
    rec.set_time("step", duration=0.0)
    logger.log(rec)

    joints = discover_revolute_joints(urdf, logger)
    mimic_joints = discover_mimic_joints(urdf, logger)
    n_joints = len(joints)

    print(f"Hand: {hand_name}")
    print(f"URDF: {urdf_path}")
    print(f"Actuated joints: {n_joints}")
    for j in joints:
        print(f"  {j.name}: [{j.lower:.3f}, {j.upper:.3f}]")
    print()

    # 초기 포즈: 모든 관절 0
    zero_angles = np.zeros(n_joints, dtype=np.float32)
    set_joint_angles(rec, joints, mimic_joints, zero_angles, step=0)

    if args.animate:
        print(f"Animating {args.steps} frames with random poses...")
        # 시작/끝 포즈를 랜덤 생성하고 보간
        lowers = np.array([j.lower for j in joints], dtype=np.float32)
        uppers = np.array([j.upper for j in joints], dtype=np.float32)

        current = zero_angles.copy()
        for seg in range(4):
            target = np.random.uniform(lowers, uppers).astype(np.float32)
            seg_steps = args.steps // 4
            for i in range(seg_steps):
                t = i / seg_steps
                # smooth interpolation (ease in-out)
                t = t * t * (3 - 2 * t)
                angles = current + (target - current) * t
                step = seg * seg_steps + i
                set_joint_angles(rec, joints, mimic_joints, angles, step=step)
            current = target

        print("Done! Rerun viewer에서 타임라인을 드래그해서 재생하세요.")
    else:
        print("Zero pose logged. Rerun viewer에서 확인하세요.")
        print("Tip: --animate 옵션으로 랜덤 포즈 애니메이션을 볼 수 있습니다.")


if __name__ == "__main__":
    main()
