#!/usr/bin/env python3

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device

from adapters import _matrix_to_euler_xyz, make_robot_adapter
from keyboard_control import KeyboardController
from safety import ActionSafetyFilter, EStop, SafetyConfig, matrix_to_rot6d, rot6d_to_matrix
from move_to_dataset_start_pose import (
    _check_joint_safety,
    _positive_xyz_from_workspace_bounds,
    _read_joint_degree,
    _read_joint_limits,
    _safe_dof,
    _to_len7_joint,
    _wait_joint_close,
)


@dataclass
class RuntimeStats:
    step: int
    ts_unix: float
    wait_ms: float
    observe_ms: float
    infer_ms: float
    safety_ms: float
    send_ms: float
    total_ms: float
    sleep_ms: float
    loop_hz: float
    overrun: int
    action_complete: int
    paused: int
    key_event: str
    camera_source: str
    safety_flags: str
    image_mean: float
    image_std: float
    image_delta_mae: float
    image_min: int
    image_max: int
    raw_action_oob_count: int
    raw_action_oob_ratio: float


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv_header(path: str, headers: list[str]) -> None:
    _ensure_parent(path)
    rewrite = False
    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            first_line = f.readline().strip()
        existing = first_line.split(",") if first_line else []
        rewrite = existing != headers
    else:
        rewrite = True

    if rewrite:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def _append_csv(path: str, row: list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _compute_image_stats(curr: np.ndarray, prev: np.ndarray | None) -> tuple[float, float, float, int, int]:
    img = np.asarray(curr)
    if img.size == 0:
        return 0.0, 0.0, 0.0, 0, 0

    img_f32 = img.astype(np.float32, copy=False)
    mean = float(np.mean(img_f32))
    std = float(np.std(img_f32))
    vmin = int(np.min(img_f32))
    vmax = int(np.max(img_f32))

    if prev is None:
        delta_mae = 0.0
    else:
        prev_f32 = np.asarray(prev).astype(np.float32, copy=False)
        if prev_f32.shape != img_f32.shape:
            delta_mae = float("nan")
        else:
            delta_mae = float(np.mean(np.abs(img_f32 - prev_f32)))

    return mean, std, delta_mae, vmin, vmax


def _compute_raw_action_oob(
    raw_action: np.ndarray,
    action_names: list[str],
    action_bounds: dict[str, list[float]],
) -> tuple[int, float, list[str]]:
    oob_names: list[str] = []
    for idx, name in enumerate(action_names):
        bounds = action_bounds.get(name)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        low = float(bounds[0])
        high = float(bounds[1])
        val = float(raw_action[idx])
        if val < low or val > high:
            oob_names.append(name)

    dim = max(len(action_names), 1)
    ratio = float(len(oob_names) / float(dim))
    return len(oob_names), ratio, oob_names


def _extract_anchor_from_state(state10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(state10, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 10:
        raise ValueError(f"anchor state must be 10D, got {arr.shape}")
    pos = arr[:3].copy()
    rot = rot6d_to_matrix(arr[3:9])
    return pos, rot


def _map_state_adapter_to_policy(
    state10: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_rot: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(state10, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 10:
        raise ValueError(f"state must be 10D, got {arr.shape}")
    pos = arr[:3]
    rot = rot6d_to_matrix(arr[3:9])
    gripper = float(arr[9])

    pos_policy = anchor_rot.T @ (pos - anchor_pos)
    rot_policy = anchor_rot.T @ rot
    rot6d_policy = matrix_to_rot6d(rot_policy)
    out = np.zeros(10, dtype=np.float64)
    out[:3] = pos_policy
    out[3:9] = rot6d_policy
    out[9] = gripper
    return out


def _map_state_policy_to_adapter(
    state10: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_rot: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(state10, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 10:
        raise ValueError(f"state must be 10D, got {arr.shape}")
    pos_policy = arr[:3]
    rot_policy = rot6d_to_matrix(arr[3:9])
    gripper = float(arr[9])

    pos = anchor_pos + anchor_rot @ pos_policy
    rot = anchor_rot @ rot_policy
    rot6d = matrix_to_rot6d(rot)
    out = np.zeros(10, dtype=np.float64)
    out[:3] = pos
    out[3:9] = rot6d
    out[9] = gripper
    return out


def _build_startup_target_state(
    cfg: dict,
    current_state: np.ndarray,
) -> tuple[np.ndarray, str]:
    startup_cfg = cfg.get("startup_pose", {})
    if not isinstance(startup_cfg, dict):
        startup_cfg = {}

    mode = str(startup_cfg.get("mode", "safe_positive")).lower()
    target = current_state.astype(np.float64).copy()
    source = "current"

    if mode in {"off", "disabled", "none"}:
        return target, "startup_disabled"

    if "xyz" in startup_cfg and isinstance(startup_cfg["xyz"], (list, tuple)) and len(startup_cfg["xyz"]) == 3:
        xyz = np.asarray(startup_cfg["xyz"], dtype=np.float64).reshape(3)
        source = "startup_pose.xyz"
    elif mode == "safe_positive":
        xyz = _positive_xyz_from_workspace_bounds(cfg["safety"].get("workspace_bounds"), current_state[:3])
        source = "safe_positive_from_workspace_bounds"
    else:
        raise ValueError(
            "startup_pose.mode must be 'safe_positive' (or provide startup_pose.xyz). "
            f"Got mode={mode!r}."
        )
    target[:3] = xyz

    keep_rot = bool(startup_cfg.get("keep_current_rotation", True))
    if not keep_rot:
        rot6d = startup_cfg.get("rot6d")
        if isinstance(rot6d, (list, tuple)) and len(rot6d) == 6:
            target[3:9] = np.asarray(rot6d, dtype=np.float64).reshape(6)

    if "gripper" in startup_cfg:
        target[9] = float(np.clip(float(startup_cfg["gripper"]), 0.0, 1.0))

    return target, source


def _move_to_startup_state_joint(
    adapter,
    cfg: dict,
    current_state: np.ndarray,
    target_state: np.ndarray,
) -> None:
    startup_cfg = cfg.get("startup_pose", {})
    if not isinstance(startup_cfg, dict):
        startup_cfg = {}

    robot = getattr(adapter, "_robot", None)
    rm_module = getattr(adapter, "_rm_module", None)
    if robot is None or rm_module is None:
        raise RuntimeError("startup joint move requires connected RM robot + SDK module")

    raw_joint = np.asarray(robot.rm_get_joint_degree()[1], dtype=np.float64).reshape(-1)
    if raw_joint.size <= 0:
        raise RuntimeError("rm_get_joint_degree returned empty list during startup")
    dof = min(_safe_dof(robot, fallback=int(raw_joint.size)), int(raw_joint.size))
    q0 = _read_joint_degree(robot, dof)

    q_min, q_max = _read_joint_limits(robot, dof)
    limit_margin = float(startup_cfg.get("joint_limit_margin_deg", 10.0))
    q_min_soft = q_min + max(limit_margin, 0.0)
    q_max_soft = q_max - max(limit_margin, 0.0)
    if np.any(q_min_soft >= q_max_soft):
        q_min_soft = q_min.copy()
        q_max_soft = q_max.copy()

    pos1 = target_state[:3]
    rot1 = target_state[3:9]
    pos1_base, rot1_base = adapter._manual_to_base_pose(pos1, rot6d_to_matrix(rot1))
    euler1_base = _matrix_to_euler_xyz(rot1_base)
    target_pose_base = [
        float(pos1_base[0]),
        float(pos1_base[1]),
        float(pos1_base[2]),
        float(euler1_base[0]),
        float(euler1_base[1]),
        float(euler1_base[2]),
    ]

    ik_params = rm_module.rm_inverse_kinematics_params_t(
        q_in=_to_len7_joint(q0, dof),
        q_pose=target_pose_base,
        flag=1,
    )
    ik_ret, q_target_raw = robot.rm_algo_inverse_kinematics(ik_params)
    if int(ik_ret) != 0:
        raise RuntimeError(f"startup IK failed with code {ik_ret}")
    q_target_raw = np.asarray(q_target_raw, dtype=np.float64).reshape(-1)
    if q_target_raw.size < dof:
        raise RuntimeError(f"startup IK output length mismatch: {q_target_raw.size} < dof {dof}")
    q1 = q_target_raw[:dof].copy()

    ok_target, target_issues = _check_joint_safety(
        robot,
        q1,
        q_min_soft,
        q_max_soft,
        dof,
        enable_self_collision_check=not bool(startup_cfg.get("disable_self_collision_check", False)),
        enable_singularity_check=not bool(startup_cfg.get("disable_singularity_check", False)),
        require_algo_checks=bool(startup_cfg.get("require_algo_checks", True)),
    )
    if not ok_target:
        raise RuntimeError(f"startup target joint safety failed: {target_issues}")

    max_joint_step_deg = float(startup_cfg.get("max_joint_step_deg", 0.8))
    max_joint_step_deg = max(max_joint_step_deg, 1e-3)
    max_joint_delta = float(np.max(np.abs(q1 - q0)))
    n_steps = max(1, int(np.ceil(max_joint_delta / max_joint_step_deg)))
    max_steps = int(startup_cfg.get("max_steps", 240))
    if n_steps > max_steps:
        print(f"[WARN] startup steps {n_steps} > max_steps {max_steps}, truncating.")
        n_steps = max_steps

    joint_speed = int(np.clip(int(startup_cfg.get("joint_speed", 6)), 1, 100))
    wait_timeout_s = float(startup_cfg.get("wait_timeout_s", 4.0))
    max_timeout_streak = int(startup_cfg.get("max_timeout_streak", 2))
    joint_tol_deg = float(startup_cfg.get("joint_tol_deg", 0.8))
    poll_dt_s = float(startup_cfg.get("joint_poll_dt_s", 0.02))

    connect = 0
    if getattr(adapter, "_trajectory_enum", None) is not None:
        try:
            connect = int(adapter._trajectory_enum.RM_TRAJECTORY_DISCONNECT_E)
        except Exception:
            connect = 0

    print(
        f"[STARTUP] joint-space move: steps={n_steps}, max_joint_delta={max_joint_delta:.3f}deg, "
        f"speed={joint_speed}, joint_tol={joint_tol_deg}"
    )

    timeout_streak = 0
    for i in range(1, n_steps + 1):
        alpha = float(i / n_steps)
        q_i = (1.0 - alpha) * q0 + alpha * q1
        ok_i, issues_i = _check_joint_safety(
            robot,
            q_i,
            q_min_soft,
            q_max_soft,
            dof,
            enable_self_collision_check=not bool(startup_cfg.get("disable_self_collision_check", False)),
            enable_singularity_check=not bool(startup_cfg.get("disable_singularity_check", False)),
            require_algo_checks=bool(startup_cfg.get("require_algo_checks", True)),
        )
        if not ok_i:
            raise RuntimeError(f"startup step safety failed at {i}/{n_steps}: {issues_i}")

        ret_cmd = robot.rm_movej(q_i.tolist(), joint_speed, 0, connect, 0)
        if int(ret_cmd) != 0:
            raise RuntimeError(f"startup rm_movej failed at step {i}/{n_steps}: {ret_cmd}")

        done, _, reason = _wait_joint_close(
            robot=robot,
            target_joint=q_i,
            timeout_s=wait_timeout_s,
            tol_deg=joint_tol_deg,
            poll_dt_s=poll_dt_s,
        )
        if done:
            timeout_streak = 0
        else:
            timeout_streak += 1
            print(
                f"[WARN] startup wait timeout at step {i}/{n_steps}, "
                f"streak={timeout_streak}, reason={reason}"
            )
            if timeout_streak >= max_timeout_streak:
                try:
                    adapter.stop_motion()
                except Exception:
                    pass
                raise RuntimeError("startup move aborted due to consecutive timeouts")

        if i <= 3 or i == n_steps or i % 10 == 0:
            print(f"[STARTUP step {i:03d}/{n_steps}] done={int(done)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="10Hz real-time inference with safety and latency logging")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Force dummy robot adapter")
    args = parser.parse_args()

    cfg = _load_json(args.config)
    action_names = cfg["action_schema"]["names"]

    target_hz = float(cfg["control"]["target_hz"])
    target_dt = 1.0 / target_hz
    max_steps = int(cfg["control"]["max_steps"])
    execution_mode = str(cfg["control"].get("execution_mode", "serial")).lower()
    action_wait_timeout_s = float(cfg["control"].get("action_wait_timeout_s", 3.0))
    debug_print_steps = int(cfg["control"].get("debug_print_steps", 3))

    latency_csv = cfg["control"]["latency_log_csv"]
    action_csv = cfg["control"]["action_log_csv"]
    observation_csv = cfg["control"].get("observation_log_csv")
    _write_csv_header(
        latency_csv,
        [
            "step",
            "ts_unix",
            "wait_ms",
            "observe_ms",
            "infer_ms",
            "safety_ms",
            "send_ms",
            "total_ms",
            "sleep_ms",
            "loop_hz",
            "overrun",
            "action_complete",
            "paused",
            "key_event",
            "camera_source",
            "safety_flags",
            "image_mean",
            "image_std",
            "image_delta_mae",
            "image_min",
            "image_max",
            "raw_action_oob_count",
            "raw_action_oob_ratio",
        ],
    )
    _write_csv_header(action_csv, ["step", *action_names])
    if observation_csv:
        _write_csv_header(observation_csv, ["step", *action_names])

    safety_cfg = SafetyConfig(
        action_bounds=cfg["safety"]["action_bounds"],
        workspace_bounds=cfg["safety"]["workspace_bounds"],
        max_xyz_speed_mps=float(cfg["safety"]["max_xyz_speed_mps"]),
        max_rot_delta_rad=float(cfg["safety"]["max_rot_delta_rad"]),
        max_gripper_delta_per_step=float(cfg["safety"]["max_gripper_delta_per_step"]),
        clip_workspace_in_action_space=not bool(
            cfg["robot_adapter"]["config"].get("workspace_clip_in_adapter", False)
        ),
    )
    safety_filter = ActionSafetyFilter(safety_cfg, action_names)
    estop = EStop(cfg["estop"]["enabled"], cfg["estop"]["trigger_file"])
    keyboard = KeyboardController(enabled=bool(cfg["control"].get("keyboard_enabled", True)))

    dataset = LeRobotDataset(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])

    pretrained_path = cfg["checkpoint"]["pretrained_model_path"]
    from lerobot.configs.policies import PreTrainedConfig
    # Isolate this script's CLI args from lerobot config parser.
    argv_backup = list(sys.argv)
    try:
        sys.argv = [sys.argv[0]]
        policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    finally:
        sys.argv = argv_backup

    policy_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_cfg.pretrained_path = pretrained_path

    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    robot_adapter_cfg = dict(cfg["robot_adapter"]["config"])
    if robot_adapter_cfg.get("workspace_clip_in_adapter", False):
        has_base_bounds = "workspace_bounds_base" in robot_adapter_cfg or "workspace_bounds_world" in robot_adapter_cfg
        if not has_base_bounds:
            print(
                "[WARN] workspace_clip_in_adapter=true but workspace_bounds_base/workspace_bounds_world is missing. "
                "Disable adapter-side clip and keep safety clip in policy frame."
            )
            robot_adapter_cfg["workspace_clip_in_adapter"] = False

    adapter = make_robot_adapter(
        cfg["robot_adapter"]["name"],
        robot_adapter_cfg,
        dry_run=(args.dry_run or bool(cfg["robot_adapter"].get("dry_run", False))),
    )

    prev_action = None
    prev_image = None
    step = 0
    oob_streak = 0
    use_startup_anchor = False
    policy_anchor_pos = None
    policy_anchor_rot = None

    print(f"[INFO] target_hz={target_hz}, pretrained={pretrained_path}")
    print(f"[INFO] execution_mode={execution_mode}, action_wait_timeout_s={action_wait_timeout_s}")
    print(f"[INFO] estop trigger file: {cfg['estop']['trigger_file']}")
    print("[INFO] keyboard: p=pause c=continue e=estop q=quit")

    adapter.connect()
    keyboard.start()
    pause_applied = False
    startup_cfg = cfg.get("startup_pose", {})
    if not isinstance(startup_cfg, dict):
        startup_cfg = {}
    try:
        use_startup_anchor = bool(startup_cfg.get("map_startup_to_policy_origin", True))
        startup_enabled = bool(startup_cfg.get("enabled", True))
        if startup_enabled:
            startup_obs = adapter.get_observation()
            startup_state = np.asarray(startup_obs.get("observation.state"), dtype=np.float64).reshape(-1)
            if startup_state.shape[0] != len(action_names):
                raise RuntimeError(
                    f"Invalid startup observation.state shape: expected {len(action_names)}, got {startup_state.shape[0]}"
                )
            target_state, target_source = _build_startup_target_state(cfg, startup_state)
            print(f"[STARTUP] source={target_source}")
            print(f"[STARTUP] current xyz={startup_state[:3].tolist()}")
            print(f"[STARTUP] target  xyz={target_state[:3].tolist()}")
            _move_to_startup_state_joint(
                adapter=adapter,
                cfg=cfg,
                current_state=startup_state,
                target_state=target_state,
            )
            startup_obs = adapter.get_observation()
            startup_state = np.asarray(startup_obs.get("observation.state"), dtype=np.float64).reshape(-1)
            if use_startup_anchor:
                policy_anchor_pos, policy_anchor_rot = _extract_anchor_from_state(startup_state)
                prev_action = _map_state_adapter_to_policy(startup_state, policy_anchor_pos, policy_anchor_rot)
                print(
                    f"[STARTUP] policy origin anchored at current startup state; "
                    f"anchor_xyz={policy_anchor_pos.tolist()}"
                )
            else:
                prev_action = startup_state.copy()
            prev_image = np.asarray(startup_obs.get("observation.image")).copy()
            print(f"[STARTUP] completed. xyz={startup_state[:3].tolist()}")
        else:
            seed_obs = adapter.get_observation()
            seed_state = np.asarray(seed_obs.get("observation.state"), dtype=np.float64).reshape(-1)
            if seed_state.shape[0] == len(action_names):
                if use_startup_anchor:
                    policy_anchor_pos, policy_anchor_rot = _extract_anchor_from_state(seed_state)
                    prev_action = _map_state_adapter_to_policy(seed_state, policy_anchor_pos, policy_anchor_rot)
                    print(
                        "[STARTUP] disabled, but policy origin is anchored at current state; "
                        f"anchor_xyz={policy_anchor_pos.tolist()}"
                    )
                else:
                    prev_action = seed_state.copy()
                prev_image = np.asarray(seed_obs.get("observation.image")).copy()
            print("[STARTUP] disabled in config (startup_pose.enabled=false)")

        while True:
            ctrl = keyboard.snapshot()
            if ctrl.estop:
                print("[ESTOP] Keyboard estop received.")
                adapter.emergency_stop()
                break
            if ctrl.quit:
                print("[INFO] Keyboard quit received.")
                break
            if ctrl.paused:
                if not pause_applied:
                    adapter.pause_motion()
                    pause_applied = True
                    print("[CTRL] paused")
                time.sleep(0.05)
                continue
            if pause_applied:
                adapter.continue_motion()
                pause_applied = False
                print("[CTRL] continued")

            if estop.triggered():
                print("[ESTOP] Trigger file detected. Stop inference loop.")
                adapter.emergency_stop()
                break

            if max_steps > 0 and step >= max_steps:
                print("[INFO] Reached max_steps. Stop inference loop.")
                break

            t0 = time.perf_counter()
            ts_unix = time.time()
            wait_ms = 0.0
            action_complete = 1

            if execution_mode == "serial" and step > 0:
                t_wait0 = time.perf_counter()
                action_complete = int(adapter.wait_until_action_complete(action_wait_timeout_s))
                wait_ms = (time.perf_counter() - t_wait0) * 1000.0
                if action_complete == 0:
                    print(f"[WARN] action completion timeout at step={step}")

            t_obs0 = time.perf_counter()
            obs_adapter = adapter.get_observation()
            t_obs1 = time.perf_counter()
            obs_state_adapter = np.asarray(obs_adapter.get("observation.state"), dtype=np.float64).reshape(-1)
            obs_image = np.asarray(obs_adapter.get("observation.image"))
            image_mean, image_std, image_delta_mae, image_min, image_max = _compute_image_stats(obs_image, prev_image)
            prev_image = obs_image.copy()
            if obs_state_adapter.shape[0] != len(action_names):
                raise RuntimeError(
                    "Invalid observation.state shape from adapter: "
                    f"expected {len(action_names)}, got {obs_state_adapter.shape[0]}"
                )
            if use_startup_anchor:
                if policy_anchor_pos is None or policy_anchor_rot is None:
                    raise RuntimeError("startup anchor is enabled but not initialized")
                obs_state_policy = _map_state_adapter_to_policy(
                    obs_state_adapter, policy_anchor_pos, policy_anchor_rot
                )
            else:
                obs_state_policy = obs_state_adapter.copy()
            obs_for_policy = dict(obs_adapter)
            obs_for_policy["observation.state"] = obs_state_policy.astype(np.float32)

            t_inf0 = time.perf_counter()
            action_tensor = predict_action(
                observation=obs_for_policy,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task="real_test",
                robot_type=adapter.robot_type(),
            )
            raw_action_dict = make_robot_action(action_tensor, dataset.features)
            raw_action_policy = np.array([raw_action_dict[n] for n in action_names], dtype=np.float64)
            t_inf1 = time.perf_counter()
            raw_oob_count, raw_oob_ratio, raw_oob_names = _compute_raw_action_oob(
                raw_action=raw_action_policy,
                action_names=action_names,
                action_bounds=cfg["safety"]["action_bounds"],
            )
            if raw_oob_ratio >= 0.5:
                oob_streak += 1
            else:
                oob_streak = 0

            if not np.all(np.isfinite(raw_action_policy)):
                print(f"[ERROR] Non-finite raw_action at step={step}: {raw_action_policy.tolist()}")
                print(f"[ERROR] observation.state(policy)={obs_state_policy.tolist()}")
                print(f"[ERROR] observation.state(adapter)={obs_state_adapter.tolist()}")
                print(f"[ERROR] image_mean={float(np.asarray(obs_adapter.get('observation.image')).mean())}")
                adapter.emergency_stop()
                break

            t_safe0 = time.perf_counter()
            safe_action_policy, flags = safety_filter.apply(raw_action_policy, prev_action, dt_s=target_dt)
            if use_startup_anchor:
                safe_action_adapter = _map_state_policy_to_adapter(
                    safe_action_policy, policy_anchor_pos, policy_anchor_rot
                )
            else:
                safe_action_adapter = safe_action_policy.copy()
            safe_action_dict = {name: float(safe_action_adapter[i]) for i, name in enumerate(action_names)}
            t_safe1 = time.perf_counter()

            t_send0 = time.perf_counter()
            adapter.send_action(safe_action_dict)
            t_send1 = time.perf_counter()

            prev_action = safe_action_policy.copy()

            total = time.perf_counter() - t0
            sleep_s = max(target_dt - total, 0.0)
            overrun = int(total > target_dt)
            if sleep_s > 0:
                time.sleep(sleep_s)

            loop_hz = 1.0 / max(total + sleep_s, 1e-8)

            stat = RuntimeStats(
                step=step,
                ts_unix=ts_unix,
                wait_ms=wait_ms,
                observe_ms=(t_obs1 - t_obs0) * 1000.0,
                infer_ms=(t_inf1 - t_inf0) * 1000.0,
                safety_ms=(t_safe1 - t_safe0) * 1000.0,
                send_ms=(t_send1 - t_send0) * 1000.0,
                total_ms=total * 1000.0,
                sleep_ms=sleep_s * 1000.0,
                loop_hz=loop_hz,
                overrun=overrun,
                action_complete=action_complete,
                paused=int(ctrl.paused),
                key_event=ctrl.last_key,
                camera_source=adapter.camera_source(),
                safety_flags="|".join(flags),
                image_mean=image_mean,
                image_std=image_std,
                image_delta_mae=image_delta_mae,
                image_min=image_min,
                image_max=image_max,
                raw_action_oob_count=raw_oob_count,
                raw_action_oob_ratio=raw_oob_ratio,
            )

            _append_csv(
                latency_csv,
                [
                    stat.step,
                    f"{stat.ts_unix:.6f}",
                    f"{stat.wait_ms:.3f}",
                    f"{stat.observe_ms:.3f}",
                    f"{stat.infer_ms:.3f}",
                    f"{stat.safety_ms:.3f}",
                    f"{stat.send_ms:.3f}",
                    f"{stat.total_ms:.3f}",
                    f"{stat.sleep_ms:.3f}",
                    f"{stat.loop_hz:.3f}",
                    stat.overrun,
                    stat.action_complete,
                    stat.paused,
                    stat.key_event,
                    stat.camera_source,
                    stat.safety_flags,
                    f"{stat.image_mean:.4f}",
                    f"{stat.image_std:.4f}",
                    f"{stat.image_delta_mae:.4f}",
                    stat.image_min,
                    stat.image_max,
                    stat.raw_action_oob_count,
                    f"{stat.raw_action_oob_ratio:.4f}",
                ],
            )
            _append_csv(action_csv, [step, *[f"{x:.8f}" for x in safe_action_policy.tolist()]])
            if observation_csv:
                _append_csv(observation_csv, [step, *[f"{x:.8f}" for x in obs_state_policy.tolist()]])
            keyboard.clear_last_key()

            if step < debug_print_steps:
                raw_action_str = ", ".join(
                    f"{name}={raw_action_policy[i]:.6f}" for i, name in enumerate(action_names)
                )
                safe_action_str = ", ".join(
                    f"{name}={safe_action_policy[i]:.6f}" for i, name in enumerate(action_names)
                )
                print(f"[DEBUG step={step}] raw_action(policy): {raw_action_str}")
                print(f"[DEBUG step={step}] safe_action(policy): {safe_action_str}")
                if use_startup_anchor:
                    send_action_str = ", ".join(
                        f"{name}={safe_action_adapter[i]:.6f}" for i, name in enumerate(action_names)
                    )
                    print(f"[DEBUG step={step}] send_action(adapter): {send_action_str}")
                print(
                    f"[DEBUG step={step}] image_stats: mean={image_mean:.3f}, std={image_std:.3f}, "
                    f"delta_mae={image_delta_mae:.3f}, min={image_min}, max={image_max}"
                )
                print(
                    f"[DEBUG step={step}] raw_action_oob: {raw_oob_count}/{len(action_names)} "
                    f"({raw_oob_ratio:.0%}) dims={raw_oob_names}"
                )

            if image_std < 2.0:
                print(
                    f"[WARN] step={step} image_std={image_std:.3f} is very low. "
                    "Camera may be dark/frozen or transform params are wrong."
                )

            if oob_streak == 3:
                print(
                    "[WARN] Raw action has been out-of-bounds for >=3 consecutive steps on most dims. "
                    "This usually indicates checkpoint/data semantic mismatch "
                    "(e.g. action coordinate frame or normalization mismatch)."
                )

            if step % 10 == 0:
                print(
                    f"[STEP {step}] hz={stat.loop_hz:.2f}, wait={stat.wait_ms:.1f}ms, "
                    f"infer={stat.infer_ms:.2f}ms, complete={stat.action_complete}, "
                    f"overrun={stat.overrun}, cam={stat.camera_source}, "
                    f"img_std={stat.image_std:.2f}, img_delta={stat.image_delta_mae:.2f}, "
                    f"raw_oob={stat.raw_action_oob_count}/{len(action_names)}, "
                    f"flags={stat.safety_flags or 'none'}"
                )

            step += 1

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stop inference loop.")
    finally:
        keyboard.stop()
        adapter.disconnect()


if __name__ == "__main__":
    main()
