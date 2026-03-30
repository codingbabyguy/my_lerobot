#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _quantiles(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p50": float(np.quantile(arr, 0.50)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _even_segment_starts(total_steps: int, seg_len: int, num_segments: int) -> list[int]:
    max_start = max(total_steps - seg_len, 0)
    if num_segments <= 1:
        return [0]
    return [int(round(i * max_start / (num_segments - 1))) for i in range(num_segments)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline replay analysis for 20 segments")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    action_names = cfg["action_schema"]["names"]
    safety_bounds = cfg["safety"]["action_bounds"]

    offline_cfg = cfg["offline_replay"]
    parquet_path = offline_cfg["parquet_path"]
    output_dir = Path(offline_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    num_segments = int(offline_cfg["num_segments"])
    seg_len = int(offline_cfg["segment_length"])

    table = pq.read_table(parquet_path, columns=["action"])
    action_col = table["action"]
    actions = np.stack([np.array(x.as_py(), dtype=np.float64) for x in action_col], axis=0)

    if actions.shape[1] != len(action_names):
        raise ValueError(
            f"Action dimension mismatch: parquet has {actions.shape[1]}, config has {len(action_names)}"
        )

    total_steps = actions.shape[0]
    starts = _even_segment_starts(total_steps, seg_len, num_segments)

    dist = {}
    bound_violations = {}
    for i, name in enumerate(action_names):
        arr = actions[:, i]
        dist[name] = _quantiles(arr)
        low, high = safety_bounds[name]
        violations = np.logical_or(arr < low, arr > high)
        bound_violations[name] = {
            "count": int(np.sum(violations)),
            "ratio": float(np.mean(violations)),
        }

    xyz_idx = [action_names.index("x"), action_names.index("y"), action_names.index("z")]
    rot_idx = [action_names.index(f"rot6d_{i}") for i in range(6)]
    g_idx = action_names.index("gripper")

    segment_metrics: list[dict] = []
    for s in starts:
        e = min(s + seg_len, total_steps)
        seg = actions[s:e]
        if len(seg) < 2:
            continue

        dxyz = np.diff(seg[:, xyz_idx], axis=0)
        dgripper = np.diff(seg[:, g_idx], axis=0)
        drot6d = np.diff(seg[:, rot_idx], axis=0)

        segment_metrics.append(
            {
                "start": int(s),
                "end": int(e),
                "length": int(e - s),
                "max_xyz_step_norm": float(np.max(np.linalg.norm(dxyz, axis=1))),
                "max_rot6d_step_norm": float(np.max(np.linalg.norm(drot6d, axis=1))),
                "max_gripper_step_abs": float(np.max(np.abs(dgripper))),
            }
        )

    summary = {
        "total_steps": int(total_steps),
        "action_dim": int(actions.shape[1]),
        "num_segments": int(len(segment_metrics)),
        "segment_length": seg_len,
        "segment_starts": starts,
        "distribution": dist,
        "bound_violations": bound_violations,
        "segment_metrics": segment_metrics,
    }

    with open(output_dir / "offline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "action_distribution.csv", "w", encoding="utf-8") as f:
        headers = ["name", "min", "p01", "p50", "p99", "max", "mean", "std", "bound_violation_ratio"]
        f.write(",".join(headers) + "\n")
        for name in action_names:
            q = dist[name]
            vr = bound_violations[name]["ratio"]
            row = [
                name,
                q["min"],
                q["p01"],
                q["p50"],
                q["p99"],
                q["max"],
                q["mean"],
                q["std"],
                vr,
            ]
            f.write(",".join([str(x) for x in row]) + "\n")

    with open(output_dir / "segment_metrics.csv", "w", encoding="utf-8") as f:
        headers = ["start", "end", "length", "max_xyz_step_norm", "max_rot6d_step_norm", "max_gripper_step_abs"]
        f.write(",".join(headers) + "\n")
        for m in segment_metrics:
            row = [
                m["start"],
                m["end"],
                m["length"],
                m["max_xyz_step_norm"],
                m["max_rot6d_step_norm"],
                m["max_gripper_step_abs"],
            ]
            f.write(",".join([str(x) for x in row]) + "\n")

    print(f"[DONE] Offline replay summary saved to: {output_dir / 'offline_summary.json'}")
    print(f"[DONE] Action distribution csv: {output_dir / 'action_distribution.csv'}")
    print(f"[DONE] Segment metrics csv: {output_dir / 'segment_metrics.csv'}")


if __name__ == "__main__":
    main()
