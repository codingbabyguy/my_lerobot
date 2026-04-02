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

from adapters import make_robot_adapter
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move RM arm slowly to dataset initial pose in manual_relative_frame."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.generated.json",
    )
    parser.add_argument("--episode-index", type=int, required=True, help="Dataset episode index")
    parser.add_argument("--step", type=int, default=0, help="Step inside episode")
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

    dataset = LeRobotDataset(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])
    ep_from, ep_to = _resolve_episode_bounds(dataset, int(args.episode_index))
    dataset_idx = ep_from + int(args.step)
    if dataset_idx >= ep_to:
        raise ValueError(f"step out of range for episode {args.episode_index}: {args.step}")

    row = dataset.hf_dataset.with_format(None)[dataset_idx]
    target_state = np.asarray(row["observation.state"], dtype=np.float64).reshape(-1)
    if target_state.shape[0] != 10:
        raise ValueError(f"Target state must be 10D, got {target_state.shape}")

    adapter = make_robot_adapter(
        cfg["robot_adapter"]["name"],
        cfg["robot_adapter"]["config"],
        dry_run=bool(cfg["robot_adapter"].get("dry_run", False)),
    )

    report: dict = {
        "config": str(Path(args.config).expanduser().resolve()),
        "episode_index": int(args.episode_index),
        "step": int(args.step),
        "dataset_index": int(dataset_idx),
        "target_state": target_state.tolist(),
    }

    adapter.connect()
    try:
        cur_obs = adapter.get_observation()
        current_state = np.asarray(cur_obs["observation.state"], dtype=np.float64).reshape(-1)
        if current_state.shape[0] != 10:
            raise ValueError(f"Current state must be 10D, got {current_state.shape}")

        pos0 = current_state[:3]
        pos1 = target_state[:3]
        r0 = rot6d_to_matrix(current_state[3:9])
        r1 = rot6d_to_matrix(target_state[3:9])
        g0 = float(current_state[9])
        g1 = float(target_state[9])

        pos_dist = float(np.linalg.norm(pos1 - pos0))
        rot_dist = _rotation_geodesic_distance(r0, r1)
        g_dist = abs(g1 - g0)

        n_pos = int(math.ceil(pos_dist / max(float(args.max_pos_step_m), 1e-6)))
        n_rot = int(math.ceil(rot_dist / max(float(args.max_rot_step_rad), 1e-6)))
        n_gri = int(math.ceil(g_dist / 0.02))
        n_steps = max(1, n_pos, n_rot, n_gri)
        if n_steps > int(args.max_steps):
            print(f"[WARN] planned steps {n_steps} > max_steps {args.max_steps}, truncating.")
            n_steps = int(args.max_steps)

        print(f"[INFO] dataset idx={dataset_idx} episode={args.episode_index} frame={int(row['frame_index'])}")
        print(f"[INFO] camera_source={adapter.camera_source()}")
        print(f"[INFO] current xyz={_format_vec(pos0)}")
        print(f"[INFO] target  xyz={_format_vec(pos1)}")
        print(f"[INFO] pos_dist={pos_dist:.6f} m, rot_dist={rot_dist:.6f} rad, gripper_dist={g_dist:.6f}")
        print(f"[INFO] interpolation steps={n_steps}")

        ok_count = 0
        for i in range(1, n_steps + 1):
            alpha = float(i / n_steps)
            pos_i = (1.0 - alpha) * pos0 + alpha * pos1
            r_i = _slerp_matrix(r0, r1, alpha)
            rot6d_i = matrix_to_rot6d(r_i)
            g_i = float((1.0 - alpha) * g0 + alpha * g1)

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
            if done:
                ok_count += 1

            if i <= 3 or i == n_steps or i % 10 == 0:
                print(
                    f"[STEP {i:03d}/{n_steps}] done={int(done)} "
                    f"xyz={_format_vec(pos_i)} dt={dt:.1f}ms"
                )

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
        print(f"[RESULT] wait_success={ok_count}/{n_steps}")

        report.update(
            {
                "current_state_before": current_state.tolist(),
                "final_state": final_state.tolist(),
                "pos_dist_before": pos_dist,
                "rot_dist_before": rot_dist,
                "planned_steps": int(n_steps),
                "wait_success_count": int(ok_count),
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
