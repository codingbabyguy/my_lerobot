#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _normalize(name: str | None) -> str:
    if name is None:
        return ""
    return str(name).replace("\x00", "").strip().lower()


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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when current work/tool frame does not match config expected names",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    robot_cfg = cfg["robot_adapter"]["config"]
    lock_frame = bool(robot_cfg.get("lock_work_tool_frame", True))
    strict_expected_names = bool(robot_cfg.get("frame_lock_require_expected_names", True))
    expected_work = [_normalize(x) for x in robot_cfg.get("expected_work_frame_names", []) if str(x).strip()]
    expected_tool = [_normalize(x) for x in robot_cfg.get("expected_tool_frame_names", []) if str(x).strip()]

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
        tool_names = robot.rm_get_total_tool_frame()
        ret_joint, joint_deg = robot.rm_get_joint_degree()
        ret_jmin, joint_min = robot.rm_get_joint_min_pos()
        ret_jmax, joint_max = robot.rm_get_joint_max_pos()
        joint_err = robot.rm_get_joint_err_flag()
        ret_collision, collision_mode = robot.rm_get_collision_detection()
        ret_sing, avoid_sing_mode = robot.rm_get_avoid_singularity_mode()

        current_work_name = _normalize(work.get("name") if isinstance(work, dict) else None)
        current_tool_name = _normalize(tool.get("name") if isinstance(tool, dict) else None)
        work_match = (not expected_work) or (current_work_name in expected_work)
        tool_match = (not expected_tool) or (current_tool_name in expected_tool)

        print(f"arm_state_ret: {ret_state}")
        print(f"arm_pose: {state.get('pose')}")
        print(f"arm_state_keys: {list(state.keys())}")
        print(f"joint_degree_ret: {ret_joint}")
        print(f"joint_degree: {joint_deg}")
        print(f"joint_min_ret: {ret_jmin}, joint_min: {joint_min}")
        print(f"joint_max_ret: {ret_jmax}, joint_max: {joint_max}")
        print(f"joint_err_flag: {json.dumps(joint_err, ensure_ascii=False)}")
        print(f"collision_detection_ret: {ret_collision}, mode: {collision_mode}")
        print(f"avoid_singularity_ret: {ret_sing}, mode: {avoid_sing_mode}")
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
        print("all_tool_frames:")
        print(json.dumps(tool_names, indent=2, ensure_ascii=False))
        print()
        print("frame_lock_config:")
        print(f"  lock_work_tool_frame: {lock_frame}")
        print(f"  frame_lock_require_expected_names: {strict_expected_names}")
        print(f"  expected_work_frame_names: {expected_work}")
        print(f"  expected_tool_frame_names: {expected_tool}")
        print(f"  current_work_frame_name: {current_work_name}")
        print(f"  current_tool_frame_name: {current_tool_name}")
        print(f"  work_match: {work_match}")
        print(f"  tool_match: {tool_match}")
        if lock_frame and strict_expected_names and (not expected_work or not expected_tool):
            print("  status: INVALID (expected frame name list is empty)")
        print()
        print("checklist:")
        print("1. Training and inference should use the same manual_relative_frame.")
        print("2. If current work/tool frame differs from config expected names, pose semantics can drift.")
        print("3. Verify joint_degree is not near joint_min/joint_max before automatic move-to-start.")
        print("4. Move arm by a small +X action in policy frame and verify physical direction against calibration.")
        if lock_frame and strict_expected_names and (not expected_work or not expected_tool):
            print("5. Fill expected_work_frame_names and expected_tool_frame_names before running inference.")
        if args.strict and (not work_match or not tool_match):
            raise RuntimeError(
                "Frame check failed in strict mode: "
                f"work_match={work_match}, tool_match={tool_match}"
            )
        if args.strict and lock_frame and strict_expected_names and (not expected_work or not expected_tool):
            raise RuntimeError("Frame check failed in strict mode: expected frame names are empty in config")
    finally:
        robot.rm_delete_robot_arm()


if __name__ == "__main__":
    main()
