#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import termios
import time
import tty
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
from safety import ActionSafetyFilter, SafetyConfig, matrix_to_rot6d, rot6d_to_matrix


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _state_to_action_dict(state: np.ndarray, action_names: list[str]) -> dict[str, float]:
    return {name: float(state[i]) for i, name in enumerate(action_names)}


def _format_xyz(v: np.ndarray) -> str:
    return np.array2string(np.asarray(v, dtype=np.float64), precision=6, suppress_small=False)


def _print_help() -> None:
    print("=== Manual Axis Teleop Keys ===")
    print("w/s : +Y / -Y")
    print("a/d : -X / +X")
    print("r/f : +Z / -Z")
    print("u/j : +RX / -RX")
    print("i/k : +RY / -RY")
    print("o/l : +RZ / -RZ")
    print("[/] : gripper -/+ (when enabled)")
    print("1/2 : decrease/increase position step")
    print("3/4 : decrease/increase rotation step")
    print("t   : toggle axis frame manual <-> base")
    print("space: resend current target")
    print("h   : print help")
    print("e   : emergency stop")
    print("q   : quit")
    print("===============================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard axis teleop for RM arm via action space.")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.generated.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--axis-frame", type=str, choices=["manual", "base"], default="manual")
    parser.add_argument("--pos-step-m", type=float, default=0.003)
    parser.add_argument("--rot-step-deg", type=float, default=1.2)
    parser.add_argument("--gripper-step", type=float, default=0.03)
    parser.add_argument("--send-hz", type=float, default=20.0)
    parser.add_argument("--wait-timeout-s", type=float, default=2.0)
    parser.add_argument("--blocking", action="store_true", help="Wait each action until robot reports completion.")
    args = parser.parse_args()

    cfg = _load_json(args.config)
    action_names = cfg["action_schema"]["names"]
    expected_names = [
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
    ]
    if action_names != expected_names:
        raise ValueError(f"Unexpected action schema: {action_names}")

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
    safety = ActionSafetyFilter(safety_cfg, action_names)

    adapter = make_robot_adapter(
        cfg["robot_adapter"]["name"],
        cfg["robot_adapter"]["config"],
        dry_run=(args.dry_run or bool(cfg["robot_adapter"].get("dry_run", False))),
    )

    axis_frame = str(args.axis_frame)
    pos_step = float(args.pos_step_m)
    rot_step = math.radians(float(args.rot_step_deg))
    grip_step = float(args.gripper_step)

    old_term = None
    last_sent = None
    target = None
    dt_target = 1.0 / max(float(args.send_hz), 1e-6)

    adapter.connect()
    try:
        obs = adapter.get_observation()
        state = np.asarray(obs["observation.state"], dtype=np.float64).reshape(-1)
        if state.shape[0] != 10:
            raise RuntimeError(f"Invalid observation.state shape: {state.shape}")

        target = state.copy()
        last_sent = state.copy()
        manual_rot_base = np.asarray(getattr(adapter, "_manual_rotation_base"), dtype=np.float64).reshape(3, 3)

        if not sys.stdin.isatty():
            raise RuntimeError("stdin is not a tty. Please run this script in an interactive terminal.")
        old_term = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

        print(f"[INFO] connected, camera_source={adapter.camera_source()}, axis_frame={axis_frame}")
        print(f"[INFO] initial target xyz(manual)={_format_xyz(target[:3])}")
        _print_help()

        last_loop_t = time.perf_counter()
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            key = ""
            if ready:
                key = sys.stdin.read(1)
            key_l = key.lower().strip()

            moved = False
            flags: list[str] = []

            delta_pos = np.zeros(3, dtype=np.float64)
            delta_rot_axis = np.zeros(3, dtype=np.float64)
            delta_grip = 0.0

            if key_l == "q":
                print("[INFO] quit")
                break
            if key_l == "e":
                print("[ESTOP] emergency stop")
                adapter.emergency_stop()
                break
            if key_l == "h":
                _print_help()
                continue
            if key == " ":
                moved = True
            elif key_l == "t":
                axis_frame = "base" if axis_frame == "manual" else "manual"
                print(f"[INFO] axis_frame -> {axis_frame}")
                continue
            elif key_l == "1":
                pos_step = max(pos_step * 0.5, 0.0005)
                print(f"[INFO] pos_step -> {pos_step:.6f} m")
                continue
            elif key_l == "2":
                pos_step = min(pos_step * 2.0, 0.03)
                print(f"[INFO] pos_step -> {pos_step:.6f} m")
                continue
            elif key_l == "3":
                rot_step = max(rot_step * 0.5, math.radians(0.2))
                print(f"[INFO] rot_step -> {math.degrees(rot_step):.3f} deg")
                continue
            elif key_l == "4":
                rot_step = min(rot_step * 2.0, math.radians(10.0))
                print(f"[INFO] rot_step -> {math.degrees(rot_step):.3f} deg")
                continue
            elif key_l in ("w", "s", "a", "d", "r", "f", "u", "j", "i", "k", "o", "l", "[", "]"):
                moved = True
                if key_l == "w":
                    delta_pos[1] += pos_step
                elif key_l == "s":
                    delta_pos[1] -= pos_step
                elif key_l == "a":
                    delta_pos[0] -= pos_step
                elif key_l == "d":
                    delta_pos[0] += pos_step
                elif key_l == "r":
                    delta_pos[2] += pos_step
                elif key_l == "f":
                    delta_pos[2] -= pos_step
                elif key_l == "u":
                    delta_rot_axis[0] += 1.0
                elif key_l == "j":
                    delta_rot_axis[0] -= 1.0
                elif key_l == "i":
                    delta_rot_axis[1] += 1.0
                elif key_l == "k":
                    delta_rot_axis[1] -= 1.0
                elif key_l == "o":
                    delta_rot_axis[2] += 1.0
                elif key_l == "l":
                    delta_rot_axis[2] -= 1.0
                elif key_l == "[":
                    delta_grip -= grip_step
                elif key_l == "]":
                    delta_grip += grip_step
            else:
                continue

            if moved:
                # Position delta
                if np.linalg.norm(delta_pos) > 0.0:
                    if axis_frame == "base":
                        delta_pos = manual_rot_base.T @ delta_pos
                    target[:3] = target[:3] + delta_pos

                # Rotation delta
                if np.linalg.norm(delta_rot_axis) > 0.0:
                    axis = delta_rot_axis / (np.linalg.norm(delta_rot_axis) + 1e-12)
                    dR = _rotvec_to_matrix(axis * rot_step)
                    R_manual = rot6d_to_matrix(target[3:9])
                    if axis_frame == "manual":
                        R_manual_new = dR @ R_manual
                    else:
                        R_base = manual_rot_base @ R_manual
                        R_base_new = dR @ R_base
                        R_manual_new = manual_rot_base.T @ R_base_new
                    target[3:9] = matrix_to_rot6d(R_manual_new)

                # Gripper
                if abs(delta_grip) > 0.0:
                    target[9] = float(np.clip(target[9] + delta_grip, 0.0, 1.0))

                now = time.perf_counter()
                dt = max(now - last_loop_t, dt_target)
                last_loop_t = now

                safe_action, flags = safety.apply(target, last_sent, dt_s=dt)
                target = safe_action.copy()
                action_dict = _state_to_action_dict(safe_action, action_names)

                t0 = time.perf_counter()
                adapter.send_action(action_dict)
                done = True
                if args.blocking:
                    done = adapter.wait_until_action_complete(float(args.wait_timeout_s))
                dt_ms = (time.perf_counter() - t0) * 1000.0
                last_sent = safe_action.copy()

                print(
                    f"[CMD] key={key_l or 'space'} frame={axis_frame} "
                    f"xyz={_format_xyz(safe_action[:3])} done={int(done)} dt={dt_ms:.1f}ms "
                    f"flags={'|'.join(flags) if flags else 'none'}"
                )
    finally:
        if old_term is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term)
        try:
            adapter.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()

