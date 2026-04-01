#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from adapters import _euler_xyz_to_matrix, make_robot_adapter
from safety import matrix_to_rot6d


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug RM pose -> training-state transform")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    adapter = make_robot_adapter(
        cfg["robot_adapter"]["name"],
        cfg["robot_adapter"]["config"],
        dry_run=bool(cfg["robot_adapter"].get("dry_run", False)),
    )

    adapter.connect()
    try:
        if not hasattr(adapter, "_robot") or adapter._robot is None:
            raise RuntimeError("Robot adapter did not create a real robot handle.")

        ret, state = adapter._robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed with code {ret}")

        pose = np.asarray(state.get("pose", [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]), dtype=np.float64)
        pos_world = pose[:3]
        euler_world = pose[3:]
        rot_world = _euler_xyz_to_matrix(*euler_world.tolist())

        ref_pos = adapter._reference_pos_world
        ref_rot = adapter._reference_rot_world
        if ref_pos is None or ref_rot is None:
            raise RuntimeError("Reference pose is not initialized")
        rel_pos = ref_rot.T @ (pos_world - ref_pos)
        rel_rot = ref_rot.T @ rot_world
        pos_norm = (rel_pos - adapter._xyz_mean) / adapter._xyz_std
        rot6d = matrix_to_rot6d(rel_rot)

        obs = adapter.get_observation()

        np.set_printoptions(precision=6, suppress=True)
        print("camera_source =", adapter.camera_source())
        print("raw_pose_world =", pose.tolist())
        print("reference_pos_world =", ref_pos.tolist())
        print("reference_rot_world =")
        print(ref_rot)
        print("xyz_mean =", adapter._xyz_mean.tolist())
        print("xyz_std =", adapter._xyz_std.tolist())
        print("rel_pos_reference =", rel_pos.tolist())
        print("pos_norm =", pos_norm.tolist())
        print("rot6d_reference =", rot6d.tolist())
        print("observation.state =", obs["observation.state"].tolist())
        print("image_mean =", float(obs["observation.image"].mean()))
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
