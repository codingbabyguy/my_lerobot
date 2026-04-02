#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from adapters import _matrix_to_euler_xyz, make_robot_adapter
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from safety import matrix_to_rot6d, rot6d_to_matrix


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
        raise ValueError(f"episode_index out of range: {episode_index}")
    ep_from = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    ep_to = int(dataset.meta.episodes["dataset_to_index"][episode_index])
    return ep_from, ep_to


def _matrix_to_rotvec(r: np.ndarray) -> np.ndarray:
    tr = float(np.trace(r))
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-9:
        rx = math.sqrt(max((r[0, 0] + 1.0) / 2.0, 0.0))
        ry = math.sqrt(max((r[1, 1] + 1.0) / 2.0, 0.0))
        rz = math.sqrt(max((r[2, 2] + 1.0) / 2.0, 0.0))
        axis = np.array([rx, ry, rz], dtype=np.float64)
        axis /= np.linalg.norm(axis) + 1e-12
        return axis * theta
    kx = (r[2, 1] - r[1, 2]) / (2.0 * sin_theta)
    ky = (r[0, 2] - r[2, 0]) / (2.0 * sin_theta)
    kz = (r[1, 0] - r[0, 1]) / (2.0 * sin_theta)
    axis = np.array([kx, ky, kz], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    return axis * theta


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)
    k = rotvec / theta
    kx, ky, kz = k
    k_mat = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    ident = np.eye(3, dtype=np.float64)
    return ident + math.sin(theta) * k_mat + (1.0 - math.cos(theta)) * (k_mat @ k_mat)


def _rotation_geodesic_distance(r_a: np.ndarray, r_b: np.ndarray) -> float:
    rel = r_a.T @ r_b
    cos_theta = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def _slerp_matrix(r0: np.ndarray, r1: np.ndarray, alpha: float) -> np.ndarray:
    rel = r0.T @ r1
    rv = _matrix_to_rotvec(rel)
    return r0 @ _rotvec_to_matrix(rv * float(alpha))


def _format_vec(v: np.ndarray) -> str:
    return np.array2string(np.asarray(v, dtype=np.float64), precision=6, suppress_small=False)


def _format_joint(v: np.ndarray) -> str:
    return np.array2string(np.asarray(v, dtype=np.float64), precision=3, suppress_small=False)


def _positive_xyz_from_workspace_bounds(
    workspace_bounds: dict[str, list[float]] | None,
    current_xyz: np.ndarray,
) -> np.ndarray:
    if not isinstance(workspace_bounds, dict):
        raise ValueError("workspace_bounds is required for safe_positive target mode.")

    out = np.asarray(current_xyz, dtype=np.float64).copy()
    for idx, axis in enumerate(("x", "y", "z")):
        bounds = workspace_bounds.get(axis)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"workspace_bounds.{axis} is invalid for safe_positive mode: {bounds}")
        low = float(bounds[0])
        high = float(bounds[1])
        if high <= 0.0:
            raise ValueError(
                f"workspace_bounds.{axis} has no positive range: low={low}, high={high}. "
                "Cannot build xyz>0 startup target."
            )
        low_pos = max(0.0, low)
        span = high - low_pos
        if span <= 1e-8:
            out[idx] = low_pos
        else:
            # Pick a conservative inner point in positive half-space.
            out[idx] = low_pos + 0.35 * span
        out[idx] = float(max(out[idx], 1e-4))
    return out


def _safe_dof(robot: object, fallback: int = 7) -> int:
    dof = int(getattr(robot, "arm_dof", 0) or 0)
    return dof if dof > 0 else int(fallback)


def _read_joint_degree(robot: object, dof: int) -> np.ndarray:
    ret, joint = robot.rm_get_joint_degree()
    if int(ret) != 0:
        raise RuntimeError(f"rm_get_joint_degree failed with code {ret}")
    arr = np.asarray(joint, dtype=np.float64).reshape(-1)
    if arr.shape[0] < dof:
        raise RuntimeError(f"rm_get_joint_degree returned insufficient length: {arr.shape[0]} < dof {dof}")
    return arr[:dof].copy()


def _read_joint_limits(robot: object, dof: int) -> tuple[np.ndarray, np.ndarray]:
    ret_min, jmin = robot.rm_get_joint_min_pos()
    ret_max, jmax = robot.rm_get_joint_max_pos()
    if int(ret_min) != 0 or int(ret_max) != 0:
        raise RuntimeError(f"failed to read joint limits: min_ret={ret_min}, max_ret={ret_max}")
    min_arr = np.asarray(jmin, dtype=np.float64).reshape(-1)
    max_arr = np.asarray(jmax, dtype=np.float64).reshape(-1)
    if min_arr.shape[0] < dof or max_arr.shape[0] < dof:
        raise RuntimeError(
            f"joint limit length mismatch: min={min_arr.shape[0]}, max={max_arr.shape[0]}, dof={dof}"
        )
    return min_arr[:dof].copy(), max_arr[:dof].copy()


def _to_len7_joint(q_deg: np.ndarray, dof: int) -> list[float]:
    out = np.zeros(7, dtype=np.float64)
    n = min(int(dof), 7, int(q_deg.shape[0]))
    out[:n] = q_deg[:n]
    return out.tolist()


def _joint_health_summary(robot: object, dof: int) -> dict:
    if not hasattr(robot, "rm_get_joint_err_flag"):
        return {"available": False}
    try:
        info = robot.rm_get_joint_err_flag()
    except Exception as exc:
        return {"available": True, "error": str(exc)}
    ret = int(info.get("return_code", -999))
    err_flag = np.asarray(info.get("err_flag", []), dtype=np.int64).reshape(-1)
    brake = np.asarray(info.get("brake_state", []), dtype=np.int64).reshape(-1)
    err_flag = err_flag[:dof] if err_flag.size >= dof else err_flag
    brake = brake[:dof] if brake.size >= dof else brake
    return {
        "available": True,
        "return_code": ret,
        "err_flag": err_flag.tolist(),
        "brake_state": brake.tolist(),
        "has_joint_err": bool(ret == 0 and err_flag.size > 0 and np.any(err_flag != 0)),
    }


def _wait_joint_close(
    robot: object,
    target_joint: np.ndarray,
    timeout_s: float,
    tol_deg: float,
    poll_dt_s: float,
) -> tuple[bool, np.ndarray, str]:
    dof = int(target_joint.shape[0])
    t_end = time.perf_counter() + float(max(timeout_s, 0.01))
    last = target_joint.copy()
    while time.perf_counter() < t_end:
        try:
            cur = _read_joint_degree(robot, dof)
        except Exception:
            cur = last
        last = cur
        max_abs = float(np.max(np.abs(cur - target_joint)))
        if max_abs <= float(tol_deg):
            return True, cur, "joint_close"

        health = _joint_health_summary(robot, dof)
        if bool(health.get("has_joint_err", False)):
            return False, cur, f"joint_err_flag={health.get('err_flag')}"

        time.sleep(float(max(poll_dt_s, 0.005)))
    return False, last, "timeout"


def _check_joint_safety(
    robot: object,
    q_deg: np.ndarray,
    q_min_soft: np.ndarray,
    q_max_soft: np.ndarray,
    dof: int,
    *,
    enable_self_collision_check: bool,
    enable_singularity_check: bool,
    require_algo_checks: bool,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    margin_low = q_deg - q_min_soft
    margin_high = q_max_soft - q_deg
    if np.any(margin_low < 0.0) or np.any(margin_high < 0.0):
        bad = np.where((margin_low < 0.0) | (margin_high < 0.0))[0]
        for idx in bad.tolist():
            issues.append(
                f"joint_limit_soft_violation:j{idx+1}:q={q_deg[idx]:.3f},"
                f"min_soft={q_min_soft[idx]:.3f},max_soft={q_max_soft[idx]:.3f}"
            )

    if enable_self_collision_check and hasattr(robot, "rm_algo_safety_robot_self_collision_detection"):
        try:
            col = int(robot.rm_algo_safety_robot_self_collision_detection(_to_len7_joint(q_deg, dof)))
            if col == 1:
                issues.append("self_collision_detected")
            elif col not in (0, 1):
                msg = f"self_collision_check_unexpected_return={col}"
                if require_algo_checks:
                    issues.append(msg)
                else:
                    print(f"[WARN] {msg}")
        except Exception as exc:
            msg = f"self_collision_check_failed={exc}"
            if require_algo_checks:
                issues.append(msg)
            else:
                print(f"[WARN] {msg}")

    if enable_singularity_check and dof == 6 and hasattr(robot, "rm_algo_kin_robot_singularity_analyse"):
        try:
            sing_ret, sing_dist = robot.rm_algo_kin_robot_singularity_analyse(q_deg[:6].tolist())
            sing_ret = int(sing_ret)
            if sing_ret != 0:
                issues.append(f"singularity_detected:code={sing_ret},dist={float(sing_dist):.6f}")
        except Exception as exc:
            msg = f"singularity_check_failed={exc}"
            if require_algo_checks:
                issues.append(msg)
            else:
                print(f"[WARN] {msg}")

    return len(issues) == 0, issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move RM arm slowly to dataset initial pose in manual_relative_frame."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.generated.json",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["safe_positive", "dataset", "config"],
        default="safe_positive",
        help="How to choose startup target state.",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Dataset episode index for target-mode=dataset")
    parser.add_argument("--step", type=int, default=0, help="Step inside episode for target-mode=dataset")
    parser.add_argument(
        "--target-xyz",
        type=float,
        nargs=3,
        default=None,
        help="Startup xyz override in manual_relative_frame (meters).",
    )
    parser.add_argument(
        "--target-gripper",
        type=float,
        default=None,
        help="Startup gripper override in [0,1].",
    )
    parser.add_argument(
        "--keep-current-rotation",
        action="store_true",
        help="Keep current end-effector orientation when building startup target.",
    )
    parser.add_argument(
        "--max-pos-step-m",
        type=float,
        default=0.004,
        help="Max interpolated translation per command in manual frame (m).",
    )
    parser.add_argument(
        "--max-rot-step-rad",
        type=float,
        default=0.03,
        help="Max interpolated rotation per command (rad).",
    )
    parser.add_argument(
        "--wait-timeout-s",
        type=float,
        default=3.0,
        help="Wait timeout for each interpolated command.",
    )
    parser.add_argument(
        "--max-timeout-streak",
        type=int,
        default=2,
        help="Abort when consecutive wait timeouts exceed this value.",
    )
    parser.add_argument(
        "--max-initial-pos-dist-m",
        type=float,
        default=0.35,
        help="Safety gate: abort if initial distance to target exceeds this value (m).",
    )
    parser.add_argument(
        "--max-initial-rot-dist-rad",
        type=float,
        default=1.2,
        help="Safety gate: abort if initial rotation distance exceeds this value (rad).",
    )
    parser.add_argument(
        "--allow-large-move",
        action="store_true",
        help="Override initial distance safety gate.",
    )
    parser.add_argument(
        "--simultaneous-pose-rot",
        action="store_true",
        help="Move translation and rotation simultaneously (default is safer two-stage move).",
    )
    parser.add_argument(
        "--motion-space",
        type=str,
        choices=["joint", "cartesian"],
        default="joint",
        help="Planning space for moving to dataset start pose.",
    )
    parser.add_argument(
        "--max-joint-step-deg",
        type=float,
        default=1.0,
        help="Max per-step interpolation in joint space (deg) when motion-space=joint.",
    )
    parser.add_argument(
        "--joint-speed",
        type=int,
        default=8,
        help="rm_movej speed percent [1,100] when motion-space=joint.",
    )
    parser.add_argument(
        "--joint-tol-deg",
        type=float,
        default=0.8,
        help="Joint close tolerance for each interpolated step (deg).",
    )
    parser.add_argument(
        "--joint-limit-margin-deg",
        type=float,
        default=8.0,
        help="Soft margin away from RM joint hard limits (deg).",
    )
    parser.add_argument(
        "--disable-self-collision-check",
        action="store_true",
        help="Disable SDK self-collision precheck in joint-space planner.",
    )
    parser.add_argument(
        "--disable-singularity-check",
        action="store_true",
        help="Disable SDK singularity precheck in joint-space planner.",
    )
    parser.add_argument(
        "--require-algo-checks",
        action="store_true",
        help="Fail if SDK algorithm checks are unavailable or error.",
    )
    parser.add_argument(
        "--joint-poll-dt-s",
        type=float,
        default=0.02,
        help="Polling interval for joint convergence wait.",
    )
    parser.add_argument("--max-steps", type=int, default=300, help="Max interpolation steps.")
    parser.add_argument(
        "--accept-pos-err-m",
        type=float,
        default=0.01,
        help="Acceptable final position error in manual frame (m).",
    )
    parser.add_argument(
        "--accept-rot-err-rad",
        type=float,
        default=0.12,
        help="Acceptable final rotation geodesic error (rad).",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Optional output report json path.",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    action_names = cfg["action_schema"]["names"]

    if action_names != [
        "x",
        "y",
        "z",
        "rot6d_0",
        "rot6d_1",
        "rot6d_2",
        "rot6d_3",
        "rot6d_4",
        "rot6d_5",
        "gripper",
    ]:
        raise ValueError(
            "Unexpected action schema order. This script expects 10D "
            "[x,y,z,rot6d_0..5,gripper]."
        )

    adapter = make_robot_adapter(
        cfg["robot_adapter"]["name"],
        cfg["robot_adapter"]["config"],
        dry_run=bool(cfg["robot_adapter"].get("dry_run", False)),
    )

    dataset_idx: int | None = None
    row_frame_index: int | None = None

    report: dict = {
        "config": str(Path(args.config).expanduser().resolve()),
        "target_mode": str(args.target_mode),
        "episode_index": int(args.episode_index),
        "step": int(args.step),
    }

    adapter.connect()
    try:
        cur_obs = adapter.get_observation()
        current_state = np.asarray(cur_obs["observation.state"], dtype=np.float64).reshape(-1)
        if current_state.shape[0] != 10:
            raise ValueError(f"Current state must be 10D, got {current_state.shape}")

        startup_cfg = cfg.get("startup_pose", {})
        if not isinstance(startup_cfg, dict):
            startup_cfg = {}

        target_state = current_state.copy()
        target_source = ""
        if str(args.target_mode) == "dataset":
            dataset = LeRobotDataset(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])
            ep_from, ep_to = _resolve_episode_bounds(dataset, int(args.episode_index))
            dataset_idx = ep_from + int(args.step)
            if dataset_idx >= ep_to:
                raise ValueError(f"step out of range for episode {args.episode_index}: {args.step}")
            row = dataset.hf_dataset.with_format(None)[dataset_idx]
            target_state = np.asarray(row["observation.state"], dtype=np.float64).reshape(-1)
            if target_state.shape[0] != 10:
                raise ValueError(f"Target state must be 10D, got {target_state.shape}")
            row_frame_index = int(row["frame_index"])
            target_source = f"dataset(ep={int(args.episode_index)}, step={int(args.step)}, idx={int(dataset_idx)})"
        else:
            if args.target_xyz is not None:
                xyz = np.asarray(args.target_xyz, dtype=np.float64).reshape(3)
                target_source = "cli_target_xyz"
            elif "xyz" in startup_cfg and isinstance(startup_cfg["xyz"], (list, tuple)) and len(startup_cfg["xyz"]) == 3:
                xyz = np.asarray(startup_cfg["xyz"], dtype=np.float64).reshape(3)
                target_source = "config.startup_pose.xyz"
            elif str(args.target_mode) == "safe_positive":
                xyz = _positive_xyz_from_workspace_bounds(cfg["safety"].get("workspace_bounds"), current_state[:3])
                target_source = "safe_positive_from_workspace_bounds"
            else:
                raise ValueError(
                    "target-mode=config requires startup_pose.xyz in config or --target-xyz on CLI."
                )

            keep_rot = bool(args.keep_current_rotation) or bool(startup_cfg.get("keep_current_rotation", True))
            if keep_rot:
                rot6d = current_state[3:9].copy()
            else:
                rot6d_cfg = startup_cfg.get("rot6d")
                if isinstance(rot6d_cfg, (list, tuple)) and len(rot6d_cfg) == 6:
                    rot6d = np.asarray(rot6d_cfg, dtype=np.float64).reshape(6)
                else:
                    rot6d = current_state[3:9].copy()

            if args.target_gripper is not None:
                gripper = float(args.target_gripper)
            elif "gripper" in startup_cfg:
                gripper = float(startup_cfg.get("gripper", current_state[9]))
            else:
                gripper = float(current_state[9])
            gripper = float(np.clip(gripper, 0.0, 1.0))

            target_state = np.concatenate([xyz, rot6d, np.array([gripper], dtype=np.float64)], axis=0)

        if str(args.target_mode) == "safe_positive" and np.any(target_state[:3] < 0.0):
            raise RuntimeError(
                f"safe_positive mode requires xyz >= 0, got target xyz={target_state[:3].tolist()}"
            )

        report["target_source"] = target_source
        report["dataset_index"] = int(dataset_idx) if dataset_idx is not None else None
        report["dataset_frame_index"] = int(row_frame_index) if row_frame_index is not None else None
        report["target_state"] = target_state.tolist()

        pos0 = current_state[:3]
        pos1 = target_state[:3]
        r0 = rot6d_to_matrix(current_state[3:9])
        r1 = rot6d_to_matrix(target_state[3:9])
        g0 = float(current_state[9])
        g1 = float(target_state[9])

        pos_dist = float(np.linalg.norm(pos1 - pos0))
        rot_dist = _rotation_geodesic_distance(r0, r1)
        g_dist = abs(g1 - g0)

        if dataset_idx is not None:
            print(
                f"[INFO] dataset idx={dataset_idx} episode={args.episode_index} "
                f"frame={row_frame_index}"
            )
        print(f"[INFO] target_source={target_source}")
        print(f"[INFO] camera_source={adapter.camera_source()}")
        print(f"[INFO] current xyz={_format_vec(pos0)}")
        print(f"[INFO] target  xyz={_format_vec(pos1)}")
        print(f"[INFO] pos_dist={pos_dist:.6f} m, rot_dist={rot_dist:.6f} rad, gripper_dist={g_dist:.6f}")

        exceed_initial_gap = (
            pos_dist > float(args.max_initial_pos_dist_m) or rot_dist > float(args.max_initial_rot_dist_rad)
        )
        if exceed_initial_gap and (not bool(args.allow_large_move)):
            if str(args.motion_space).lower() == "joint":
                print(
                    "[WARN] Initial Cartesian gap is large, but joint-space planner is enabled. "
                    "Proceeding with strict joint safety checks."
                )
            else:
                raise RuntimeError(
                    "Initial gap is too large for automatic approach: "
                    f"pos_dist={pos_dist:.3f}m (limit={args.max_initial_pos_dist_m}), "
                    f"rot_dist={rot_dist:.3f}rad (limit={args.max_initial_rot_dist_rad}). "
                    "Please jog robot closer manually or rerun with --allow-large-move after safety confirmation."
                )

        ok_count = 0
        timeout_streak = 0
        global_step = 0
        max_steps = int(args.max_steps)
        truncated = False
        total_planned_steps = 0

        if str(args.motion_space).lower() == "joint":
            robot = getattr(adapter, "_robot", None)
            rm_module = getattr(adapter, "_rm_module", None)
            if robot is None or rm_module is None:
                raise RuntimeError("Joint-space planner requires a connected RM robot and SDK module.")

            ret_joint, raw_joint = robot.rm_get_joint_degree()
            if int(ret_joint) != 0:
                raise RuntimeError(f"rm_get_joint_degree failed with code {ret_joint}")
            raw_joint = np.asarray(raw_joint, dtype=np.float64).reshape(-1)
            if raw_joint.size <= 0:
                raise RuntimeError("rm_get_joint_degree returned empty list")
            dof = min(_safe_dof(robot, fallback=int(raw_joint.size)), int(raw_joint.size))
            q0 = raw_joint[:dof].copy()

            q_min, q_max = _read_joint_limits(robot, dof)
            margin = float(max(args.joint_limit_margin_deg, 0.0))
            q_min_soft = q_min + margin
            q_max_soft = q_max - margin
            if np.any(q_min_soft >= q_max_soft):
                print("[WARN] joint_limit_margin_deg too large for hardware limits, fallback to hard limits.")
                q_min_soft = q_min.copy()
                q_max_soft = q_max.copy()

            pos1_base, rot1_base = adapter._manual_to_base_pose(pos1, r1)
            euler1_base = _matrix_to_euler_xyz(rot1_base)
            target_pose_base = [
                float(pos1_base[0]),
                float(pos1_base[1]),
                float(pos1_base[2]),
                float(euler1_base[0]),
                float(euler1_base[1]),
                float(euler1_base[2]),
            ]

            q_in_7 = _to_len7_joint(q0, dof)
            ik_params = rm_module.rm_inverse_kinematics_params_t(
                q_in=q_in_7,
                q_pose=target_pose_base,
                flag=1,
            )
            ik_ret, q_target_raw = robot.rm_algo_inverse_kinematics(ik_params)
            if int(ik_ret) != 0:
                raise RuntimeError(
                    f"rm_algo_inverse_kinematics failed: ret={ik_ret}, "
                    "target pose may be unreachable in current branch."
                )
            q_target_raw = np.asarray(q_target_raw, dtype=np.float64).reshape(-1)
            if q_target_raw.size < dof:
                raise RuntimeError(f"IK output length mismatch: {q_target_raw.size} < dof {dof}")
            q1 = q_target_raw[:dof].copy()

            ok_target, target_issues = _check_joint_safety(
                robot,
                q1,
                q_min_soft,
                q_max_soft,
                dof,
                enable_self_collision_check=not bool(args.disable_self_collision_check),
                enable_singularity_check=not bool(args.disable_singularity_check),
                require_algo_checks=bool(args.require_algo_checks),
            )
            if not ok_target:
                raise RuntimeError(f"Target joint pose failed safety checks: {target_issues}")

            health = _joint_health_summary(robot, dof)
            print(f"[INFO] motion_mode=joint-space dof={dof}")
            print(f"[INFO] current joint(deg)={_format_joint(q0)}")
            print(f"[INFO] target  joint(deg)={_format_joint(q1)}")
            print(f"[INFO] joint hard min(deg)={_format_joint(q_min)}")
            print(f"[INFO] joint hard max(deg)={_format_joint(q_max)}")
            print(f"[INFO] joint soft min(deg)={_format_joint(q_min_soft)}")
            print(f"[INFO] joint soft max(deg)={_format_joint(q_max_soft)}")
            if health.get("available", False):
                print(
                    "[INFO] joint health: "
                    f"ret={health.get('return_code')}, "
                    f"err_flag={health.get('err_flag')}, "
                    f"brake_state={health.get('brake_state')}"
                )

            max_joint_delta = float(np.max(np.abs(q1 - q0)))
            n_steps = int(math.ceil(max_joint_delta / max(float(args.max_joint_step_deg), 1e-3)))
            n_steps = max(1, n_steps)
            total_planned_steps = int(n_steps)
            if n_steps > max_steps:
                truncated = True
                print(f"[WARN] planned steps {n_steps} > max_steps {max_steps}, truncating.")
                n_steps = max_steps

            print(
                f"[INFO] planned_joint_steps={total_planned_steps}, executed_limit={n_steps}, "
                f"max_joint_delta_deg={max_joint_delta:.3f}"
            )

            speed = int(np.clip(int(args.joint_speed), 1, 100))
            connect = 0
            if getattr(adapter, "_trajectory_enum", None) is not None:
                try:
                    connect = int(adapter._trajectory_enum.RM_TRAJECTORY_DISCONNECT_E)
                except Exception:
                    connect = 0

            for i in range(1, n_steps + 1):
                alpha = float(i / n_steps)
                q_i = (1.0 - alpha) * q0 + alpha * q1
                ok_i, issues_i = _check_joint_safety(
                    robot,
                    q_i,
                    q_min_soft,
                    q_max_soft,
                    dof,
                    enable_self_collision_check=not bool(args.disable_self_collision_check),
                    enable_singularity_check=not bool(args.disable_singularity_check),
                    require_algo_checks=bool(args.require_algo_checks),
                )
                if not ok_i:
                    raise RuntimeError(f"joint safety precheck failed at step {i}/{n_steps}: {issues_i}")

                t0 = time.perf_counter()
                ret_cmd = robot.rm_movej(q_i.tolist(), speed, 0, connect, 0)
                if int(ret_cmd) != 0:
                    raise RuntimeError(f"rm_movej failed at step {i}/{n_steps} with code {ret_cmd}")

                done, q_last, reason = _wait_joint_close(
                    robot=robot,
                    target_joint=q_i,
                    timeout_s=float(args.wait_timeout_s),
                    tol_deg=float(args.joint_tol_deg),
                    poll_dt_s=float(args.joint_poll_dt_s),
                )
                dt = (time.perf_counter() - t0) * 1000.0
                global_step += 1
                if done:
                    ok_count += 1
                    timeout_streak = 0
                else:
                    timeout_streak += 1
                    print(
                        f"[WARN] wait timeout at step {i}/{n_steps}, streak={timeout_streak}, "
                        f"reason={reason}, dt={dt:.1f}ms"
                    )
                    if timeout_streak >= int(args.max_timeout_streak):
                        try:
                            adapter.stop_motion()
                        except Exception:
                            pass
                        raise RuntimeError(
                            "Aborting due to consecutive joint-step timeouts. "
                            "Robot may be in protective stop / unreachable state."
                        )

                if i <= 3 or i == n_steps or i % 10 == 0:
                    print(
                        f"[STEP {i:03d}/{n_steps}] done={int(done)} "
                        f"joint={_format_joint(q_i)} "
                        f"joint_err={float(np.max(np.abs(q_last - q_i))):.3f}deg dt={dt:.1f}ms"
                    )
        else:
            if bool(args.simultaneous_pose_rot):
                segments = [("pose+rot", pos0, r0, g0, pos1, r1, g1)]
            else:
                # Safer on real hardware: reduce coupled kinematic stress by translating first, then rotating in place.
                segments = [
                    ("translate", pos0, r0, g0, pos1, r0, g0),
                    ("rotate", pos1, r0, g0, pos1, r1, g1),
                ]

            seg_specs: list[tuple[str, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, int]] = []
            for name, s_pos, s_rot, s_g, e_pos, e_rot, e_g in segments:
                seg_pos_dist = float(np.linalg.norm(e_pos - s_pos))
                seg_rot_dist = _rotation_geodesic_distance(s_rot, e_rot)
                seg_g_dist = abs(float(e_g) - float(s_g))
                seg_n_pos = int(math.ceil(seg_pos_dist / max(float(args.max_pos_step_m), 1e-6)))
                seg_n_rot = int(math.ceil(seg_rot_dist / max(float(args.max_rot_step_rad), 1e-6)))
                seg_n_gri = int(math.ceil(seg_g_dist / 0.02))
                seg_steps = max(1, seg_n_pos, seg_n_rot, seg_n_gri)
                seg_specs.append((name, s_pos, s_rot, float(s_g), e_pos, e_rot, float(e_g), seg_steps))
                total_planned_steps += seg_steps

            print(
                f"[INFO] motion_mode={'simultaneous' if args.simultaneous_pose_rot else 'position-first'} "
                f"segments={len(seg_specs)} planned_steps={total_planned_steps} max_steps={int(args.max_steps)}"
            )

            for seg_idx, (seg_name, s_pos, s_rot, s_g, e_pos, e_rot, e_g, seg_steps) in enumerate(seg_specs, start=1):
                seg_pos_dist = float(np.linalg.norm(e_pos - s_pos))
                seg_rot_dist = _rotation_geodesic_distance(s_rot, e_rot)
                print(
                    f"[SEG {seg_idx}/{len(seg_specs)}] {seg_name} "
                    f"steps={seg_steps} pos_dist={seg_pos_dist:.6f} rot_dist={seg_rot_dist:.6f}"
                )
                for i in range(1, seg_steps + 1):
                    if global_step >= max_steps:
                        truncated = True
                        print(f"[WARN] reached max_steps={max_steps}, stop interpolation early.")
                        break

                    alpha = float(i / seg_steps)
                    pos_i = (1.0 - alpha) * s_pos + alpha * e_pos
                    r_i = _slerp_matrix(s_rot, e_rot, alpha)
                    rot6d_i = matrix_to_rot6d(r_i)
                    g_i = float((1.0 - alpha) * s_g + alpha * e_g)

                    action = {
                        "x": float(pos_i[0]),
                        "y": float(pos_i[1]),
                        "z": float(pos_i[2]),
                        "rot6d_0": float(rot6d_i[0]),
                        "rot6d_1": float(rot6d_i[1]),
                        "rot6d_2": float(rot6d_i[2]),
                        "rot6d_3": float(rot6d_i[3]),
                        "rot6d_4": float(rot6d_i[4]),
                        "rot6d_5": float(rot6d_i[5]),
                        "gripper": float(g_i),
                    }

                    t0 = time.perf_counter()
                    adapter.send_action(action)
                    done = adapter.wait_until_action_complete(float(args.wait_timeout_s))
                    dt = (time.perf_counter() - t0) * 1000.0
                    global_step += 1
                    if done:
                        ok_count += 1
                        timeout_streak = 0
                    else:
                        timeout_streak += 1
                        print(
                            f"[WARN] wait timeout at global_step {global_step}/{max_steps}, "
                            f"segment={seg_name} step={i}/{seg_steps}, streak={timeout_streak}, dt={dt:.1f}ms"
                        )
                        if timeout_streak >= int(args.max_timeout_streak):
                            try:
                                adapter.stop_motion()
                            except Exception:
                                pass
                            raise RuntimeError(
                                "Aborting due to consecutive action timeouts. "
                                "Robot may be in protective stop / unreachable state."
                            )

                    if global_step <= 3 or i == seg_steps or global_step % 10 == 0:
                        print(
                            f"[STEP {global_step:03d}/{max_steps}] done={int(done)} "
                            f"seg={seg_name} xyz={_format_vec(pos_i)} dt={dt:.1f}ms"
                        )

                if truncated:
                    break

        final_obs = adapter.get_observation()
        final_state = np.asarray(final_obs["observation.state"], dtype=np.float64).reshape(-1)
        pos_err = float(np.linalg.norm(final_state[:3] - target_state[:3]))
        rot_err = _rotation_geodesic_distance(
            rot6d_to_matrix(final_state[3:9]),
            rot6d_to_matrix(target_state[3:9]),
        )
        l2_err = float(np.linalg.norm(final_state - target_state))

        passed = (pos_err <= float(args.accept_pos_err_m)) and (rot_err <= float(args.accept_rot_err_rad))
        print(f"[RESULT] final xyz={_format_vec(final_state[:3])}")
        print(f"[RESULT] target xyz={_format_vec(target_state[:3])}")
        print(
            f"[RESULT] pos_err={pos_err:.6f}m rot_err={rot_err:.6f}rad "
            f"l2_err={l2_err:.6f} pass={passed}"
        )
        print(f"[RESULT] wait_success={ok_count}/{global_step}")

        report.update(
            {
                "motion_space": str(args.motion_space),
                "current_state_before": current_state.tolist(),
                "final_state": final_state.tolist(),
                "pos_dist_before": pos_dist,
                "rot_dist_before": rot_dist,
                "planned_steps": int(total_planned_steps),
                "executed_steps": int(global_step),
                "truncated_by_max_steps": bool(truncated),
                "wait_success_count": int(ok_count),
                "joint_limit_margin_deg": float(args.joint_limit_margin_deg),
                "joint_step_deg": float(args.max_joint_step_deg),
                "final_pos_err_m": pos_err,
                "final_rot_err_rad": rot_err,
                "final_l2_err": l2_err,
                "pass": bool(passed),
            }
        )
    finally:
        adapter.disconnect()

    if args.report_json:
        out = Path(args.report_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[DONE] report: {out}")


if __name__ == "__main__":
    main()
