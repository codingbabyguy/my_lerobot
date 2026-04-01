#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_latency_csv(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows in latency csv: {path}")

    data = {}
    numeric_cols = [
        "step",
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
    ]
    for col in numeric_cols:
        if col in rows[0]:
            data[col] = np.array([float(r[col]) for r in rows], dtype=np.float64)
        else:
            data[col] = np.zeros(len(rows), dtype=np.float64)

    flags = [r.get("safety_flags", "") for r in rows]
    data["safety_flags"] = np.array(flags, dtype=object)
    data["key_event"] = np.array([r.get("key_event", "") for r in rows], dtype=object)
    data["camera_source"] = np.array([r.get("camera_source", "") for r in rows], dtype=object)
    return data


def _read_action_csv(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(header) < 2:
        raise ValueError(f"Unexpected action csv header: {header}")

    action_names = header[1:]
    steps = np.array([int(r[0]) for r in rows], dtype=np.int64)
    actions = np.array([[float(x) for x in r[1:]] for r in rows], dtype=np.float64)
    return steps, action_names, actions


def _read_observation_csv(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if len(header) < 2:
        raise ValueError(f"Unexpected observation csv header: {header}")
    if not rows:
        raise ValueError(f"No rows in observation csv: {path}")
    names = header[1:]
    steps = np.array([int(r[0]) for r in rows], dtype=np.int64)
    values = np.array([[float(x) for x in r[1:]] for r in rows], dtype=np.float64)
    return steps, names, values


def _count_safety_events(flags: np.ndarray) -> dict[str, int]:
    c = Counter()
    for row in flags:
        row_s = str(row).strip()
        if not row_s:
            continue
        for item in row_s.split("|"):
            if item:
                c[item] += 1
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def _summary(lat: dict[str, np.ndarray], safety_counter: dict[str, int]) -> dict:
    total = lat["total_ms"]
    infer = lat["infer_ms"]
    hz = lat["loop_hz"]
    overrun = lat["overrun"]
    wait_ms = lat["wait_ms"]
    action_complete = lat["action_complete"]
    key_counter = _count_safety_events(lat["key_event"])
    camera_counter = _count_safety_events(lat["camera_source"])

    return {
        "num_steps": int(total.shape[0]),
        "total_ms": {
            "mean": float(np.mean(total)),
            "p95": float(np.percentile(total, 95)),
            "p99": float(np.percentile(total, 99)),
            "max": float(np.max(total)),
        },
        "infer_ms": {
            "mean": float(np.mean(infer)),
            "p95": float(np.percentile(infer, 95)),
            "p99": float(np.percentile(infer, 99)),
            "max": float(np.max(infer)),
        },
        "loop_hz": {
            "mean": float(np.mean(hz)),
            "p05": float(np.percentile(hz, 5)),
            "p95": float(np.percentile(hz, 95)),
            "min": float(np.min(hz)),
        },
        "overrun": {
            "count": int(np.sum(overrun > 0)),
            "ratio": float(np.mean(overrun > 0)),
        },
        "wait_ms": {
            "mean": float(np.mean(wait_ms)),
            "p95": float(np.percentile(wait_ms, 95)),
            "max": float(np.max(wait_ms)),
        },
        "action_completion": {
            "success_ratio": float(np.mean(action_complete > 0)),
            "timeout_count": int(np.sum(action_complete <= 0)),
        },
        "key_events": key_counter,
        "camera_source": camera_counter,
        "safety_events": safety_counter,
    }


def _state_summary(obs_names: list[str], obs_values: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(obs_names):
        col = obs_values[:, i]
        out[name] = {
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
        }
    return out


def _plot_dashboard(
    latency: dict[str, np.ndarray],
    action_steps: np.ndarray,
    action_names: list[str],
    actions: np.ndarray,
    fig_path: Path,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    steps = latency["step"]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, latency["wait_ms"], label="wait_ms", linewidth=1.2)
    ax1.plot(steps, latency["observe_ms"], label="observe_ms", linewidth=1.2)
    ax1.plot(steps, latency["infer_ms"], label="infer_ms", linewidth=1.2)
    ax1.plot(steps, latency["safety_ms"], label="safety_ms", linewidth=1.2)
    ax1.plot(steps, latency["send_ms"], label="send_ms", linewidth=1.2)
    ax1.set_title("Step Latency Breakdown")
    ax1.set_xlabel("step")
    ax1.set_ylabel("ms")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, latency["total_ms"], label="total_ms", linewidth=1.4)
    ax2.axhline(100.0, color="tab:red", linestyle="--", linewidth=1.0, label="10Hz budget")
    ax2.set_title("End-to-End Loop Time")
    ax2.set_xlabel("step")
    ax2.set_ylabel("ms")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, latency["loop_hz"], color="tab:green", linewidth=1.4)
    ax3.axhline(10.0, color="tab:red", linestyle="--", linewidth=1.0)
    ax3.set_title("Loop Frequency")
    ax3.set_xlabel("step")
    ax3.set_ylabel("Hz")
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 1])
    overrun = latency["overrun"] > 0
    completed = latency["action_complete"] > 0
    ax4.plot(steps, overrun.astype(np.float64), linewidth=1.0, label="overrun")
    ax4.plot(steps, completed.astype(np.float64), linewidth=1.0, label="action_complete")
    ax4.set_title("Overrun and Action Completion")
    ax4.set_xlabel("step")
    ax4.set_ylabel("flag")
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(alpha=0.25)

    ax5 = fig.add_subplot(gs[2, :])
    max_dims = min(actions.shape[1], 10)
    for i in range(max_dims):
        ax5.plot(action_steps, actions[:, i], linewidth=1.0, label=action_names[i])
    ax5.set_title("Action Trajectories")
    ax5.set_xlabel("step")
    ax5.set_ylabel("value")
    ax5.legend(loc="upper right", fontsize=8, ncol=5)
    ax5.grid(alpha=0.25)

    fig.tight_layout()
    _ensure_parent(fig_path)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def _plot_state_action_compare(
    state_steps: np.ndarray,
    states: np.ndarray,
    action_steps: np.ndarray,
    actions: np.ndarray,
    fig_path: Path,
) -> dict[str, float]:
    common_steps, state_idx, action_idx = np.intersect1d(state_steps, action_steps, return_indices=True)
    if common_steps.size == 0:
        return {"num_aligned_steps": 0}

    s = states[state_idx]
    a = actions[action_idx]
    if s.shape[1] != a.shape[1]:
        return {"num_aligned_steps": int(common_steps.size), "warning": "state/action dimension mismatch"}

    xyz_err = np.linalg.norm(s[:, :3] - a[:, :3], axis=1)
    rot_err = np.linalg.norm(s[:, 3:9] - a[:, 3:9], axis=1)
    gripper_err = np.abs(s[:, 9] - a[:, 9]) if s.shape[1] > 9 else np.zeros(common_steps.size, dtype=np.float64)

    metrics = {
        "num_aligned_steps": int(common_steps.size),
        "xyz_abs_err_mean": float(np.mean(xyz_err)),
        "xyz_abs_err_p95": float(np.percentile(xyz_err, 95)),
        "rot6d_abs_err_mean": float(np.mean(rot_err)),
        "rot6d_abs_err_p95": float(np.percentile(rot_err, 95)),
        "gripper_abs_err_mean": float(np.mean(gripper_err)),
        "gripper_abs_err_p95": float(np.percentile(gripper_err, 95)),
    }

    if plt is None:
        return metrics

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    for i, axis_name in enumerate(["x", "y", "z"]):
        axes[0].plot(common_steps, s[:, i], linewidth=1.4, label=f"state_{axis_name}")
        axes[0].plot(common_steps, a[:, i], linewidth=1.0, linestyle="--", label=f"action_{axis_name}")
    axes[0].set_ylabel("meter")
    axes[0].set_title("State vs Action (XYZ, manual_relative_frame)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    axes[1].plot(common_steps, xyz_err, linewidth=1.3, label="xyz_abs_err_norm")
    axes[1].plot(common_steps, rot_err, linewidth=1.3, label="rot6d_abs_err_norm")
    axes[1].set_ylabel("norm")
    axes[1].set_title("State-Action Error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    if s.shape[1] > 9:
        axes[2].plot(common_steps, s[:, 9], linewidth=1.4, label="state_gripper")
        axes[2].plot(common_steps, a[:, 9], linewidth=1.0, linestyle="--", label="action_gripper")
        axes[2].set_ylabel("normalized")
        axes[2].set_title("State vs Action (Gripper)")
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="upper right")

    axes[-1].set_xlabel("step")
    fig.tight_layout()
    _ensure_parent(fig_path)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize real-time inference logs")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/results/visualization",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    latency_csv = Path(cfg["control"]["latency_log_csv"])
    action_csv = Path(cfg["control"]["action_log_csv"])
    observation_csv = cfg["control"].get("observation_log_csv")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latency = _read_latency_csv(latency_csv)
    action_steps, action_names, actions = _read_action_csv(action_csv)
    safety_counter = _count_safety_events(latency["safety_flags"])

    summary = _summary(latency, safety_counter)
    if observation_csv:
        obs_path = Path(observation_csv)
        if obs_path.is_file():
            obs_steps, obs_names, obs_values = _read_observation_csv(obs_path)
            summary["observation_state"] = _state_summary(obs_names, obs_values)
            compare_fig_path = out_dir / "state_action_compare.png"
            summary["state_action_alignment"] = _plot_state_action_compare(
                state_steps=obs_steps,
                states=obs_values,
                action_steps=action_steps,
                actions=actions,
                fig_path=compare_fig_path,
            )
            if plt is not None:
                print(f"[OK] state-action compare: {compare_fig_path}")
        else:
            summary["observation_state"] = {"warning": f"missing observation csv: {obs_path}"}

    summary_path = out_dir / "realtime_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig_path = out_dir / "realtime_dashboard.png"
    if plt is None:
        print("[WARN] matplotlib is not installed; skip realtime_dashboard.png generation")
    else:
        _plot_dashboard(latency, action_steps, action_names, actions, fig_path)
        print(f"[OK] dashboard: {fig_path}")

    print(f"[OK] summary: {summary_path}")


if __name__ == "__main__":
    main()
