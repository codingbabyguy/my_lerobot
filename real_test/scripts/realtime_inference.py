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

from adapters import make_robot_adapter
from keyboard_control import KeyboardController
from safety import ActionSafetyFilter, EStop, SafetyConfig


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
        ],
    )
    _write_csv_header(action_csv, ["step", *action_names])

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
    robot_adapter_cfg.setdefault("workspace_bounds_world", cfg["safety"]["workspace_bounds"])

    adapter = make_robot_adapter(
        cfg["robot_adapter"]["name"],
        robot_adapter_cfg,
        dry_run=(args.dry_run or bool(cfg["robot_adapter"].get("dry_run", False))),
    )

    prev_action = None
    step = 0

    print(f"[INFO] target_hz={target_hz}, pretrained={pretrained_path}")
    print(f"[INFO] execution_mode={execution_mode}, action_wait_timeout_s={action_wait_timeout_s}")
    print(f"[INFO] estop trigger file: {cfg['estop']['trigger_file']}")
    print("[INFO] keyboard: p=pause c=continue e=estop q=quit")

    adapter.connect()
    keyboard.start()
    pause_applied = False
    try:
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
            obs = adapter.get_observation()
            t_obs1 = time.perf_counter()

            t_inf0 = time.perf_counter()
            action_tensor = predict_action(
                observation=obs,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task="real_test",
                robot_type=adapter.robot_type(),
            )
            raw_action_dict = make_robot_action(action_tensor, dataset.features)
            raw_action = np.array([raw_action_dict[n] for n in action_names], dtype=np.float64)
            t_inf1 = time.perf_counter()

            t_safe0 = time.perf_counter()
            safe_action, flags = safety_filter.apply(raw_action, prev_action, dt_s=target_dt)
            safe_action_dict = {name: float(safe_action[i]) for i, name in enumerate(action_names)}
            t_safe1 = time.perf_counter()

            t_send0 = time.perf_counter()
            adapter.send_action(safe_action_dict)
            t_send1 = time.perf_counter()

            prev_action = safe_action.copy()

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
                ],
            )
            _append_csv(action_csv, [step, *[f"{x:.8f}" for x in safe_action.tolist()]])
            keyboard.clear_last_key()

            if step < debug_print_steps:
                raw_action_str = ", ".join(f"{name}={raw_action[i]:.6f}" for i, name in enumerate(action_names))
                safe_action_str = ", ".join(f"{name}={safe_action[i]:.6f}" for i, name in enumerate(action_names))
                print(f"[DEBUG step={step}] raw_action: {raw_action_str}")
                print(f"[DEBUG step={step}] safe_action: {safe_action_str}")

            if step % 10 == 0:
                print(
                    f"[STEP {step}] hz={stat.loop_hz:.2f}, wait={stat.wait_ms:.1f}ms, "
                    f"infer={stat.infer_ms:.2f}ms, complete={stat.action_complete}, "
                    f"overrun={stat.overrun}, cam={stat.camera_source}, "
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
