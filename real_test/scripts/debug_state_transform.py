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

from adapters import make_robot_adapter
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
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to dump debug information as json",
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

        pos_base, rot_base, pose_base = adapter._read_current_pose_base()
        pos_manual, rot_manual = adapter._base_to_manual_pose(pos_base, rot_base)
        rot6d = matrix_to_rot6d(rot_manual)

        obs = adapter.get_observation()
        obs_state = np.asarray(obs["observation.state"], dtype=np.float64).reshape(-1)

        debug_info = {
            "camera_source": adapter.camera_source(),
            "frame_lock": {
                "work": getattr(adapter, "_connected_work_frame_name", None),
                "tool": getattr(adapter, "_connected_tool_frame_name", None),
            },
            "manual_origin_base": adapter._manual_origin_base.tolist(),
            "manual_rotation_base": adapter._manual_rotation_base.tolist(),
            "pose_base_euler": pose_base.tolist(),
            "manual_position_from_pose": pos_manual.tolist(),
            "manual_rot6d_from_pose": rot6d.tolist(),
            "observation_state": obs_state.tolist(),
            "state_minus_manual_pose": (obs_state[:9] - np.concatenate([pos_manual, rot6d], axis=0)).tolist(),
            "image_mean": float(obs["observation.image"].mean()),
        }

        np.set_printoptions(precision=6, suppress=True)
        print("camera_source =", debug_info["camera_source"])
        print("frame_lock =", debug_info["frame_lock"])
        print("pose_base_euler =", debug_info["pose_base_euler"])
        print("manual_origin_base =", debug_info["manual_origin_base"])
        print("manual_rotation_base =")
        print(np.asarray(debug_info["manual_rotation_base"]))
        print("manual_position_from_pose =", debug_info["manual_position_from_pose"])
        print("manual_rot6d_from_pose =", debug_info["manual_rot6d_from_pose"])
        print("observation.state =", debug_info["observation_state"])
        print("state_minus_manual_pose =", debug_info["state_minus_manual_pose"])
        print("image_mean =", debug_info["image_mean"])

        if args.output_json:
            out_path = Path(args.output_json).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2)
            print(f"[OK] wrote debug json: {out_path}")
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
