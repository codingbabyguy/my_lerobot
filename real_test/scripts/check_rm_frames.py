#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print RM65B current pose, work frame, and tool frame")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    robot_cfg = cfg["robot_adapter"]["config"]

    sdk_src = Path(robot_cfg["sdk_src_path"]).expanduser().resolve()
    if str(sdk_src) not in sys.path:
        sys.path.insert(0, str(sdk_src))

    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    host = robot_cfg.get("host", "192.168.1.18")
    port = int(robot_cfg.get("port", 8080))
    level = int(robot_cfg.get("level", 3))
    thread_mode = int(robot_cfg.get("thread_mode", 2))

    robot = RoboticArm(rm_thread_mode_e(thread_mode))
    handle = robot.rm_create_robot_arm(host, port, level)
    if getattr(handle, "id", -1) == -1:
        raise RuntimeError(f"Failed to connect to RM arm at {host}:{port}")

    try:
        ret_state, state = robot.rm_get_current_arm_state()
        ret_work, work = robot.rm_get_current_work_frame()
        ret_tool, tool = robot.rm_get_current_tool_frame()
        work_names = robot.rm_get_total_work_frame()

        print(f"arm_state_ret: {ret_state}")
        print(f"arm_pose: {state.get('pose')}")
        print(f"arm_state_keys: {list(state.keys())}")
        print()
        print(f"current_work_frame_ret: {ret_work}")
        print(json.dumps(work, indent=2, ensure_ascii=False))
        print()
        print(f"current_tool_frame_ret: {ret_tool}")
        print(json.dumps(tool, indent=2, ensure_ascii=False))
        print()
        print("all_work_frames:")
        print(json.dumps(work_names, indent=2, ensure_ascii=False))
        print()
        print("checklist:")
        print("1. If current work frame is not the default/base frame, do not assume state['pose'] is already the training frame.")
        print("2. If tool frame is not the same TCP definition used in data collection, current pose will not match training semantics.")
        print("3. Move the arm by a small +X command and verify the physical motion matches your expected base-frame X direction.")
    finally:
        robot.rm_delete_robot_arm()


if __name__ == "__main__":
    main()
