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
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    ep_from = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    ep_to = int(dataset.meta.episodes["dataset_to_index"][episode_index])
    return ep_from, ep_to


def _format_vec(arr: np.ndarray) -> str:
    return np.array2string(arr, precision=6, suppress_small=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current real robot observation.state with dataset episode start")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--step", type=int, default=0, help="Start step inside the episode for direct comparison")
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="How many steps from the selected step to compare for nearest-state search",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    dataset = LeRobotDataset(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])
    ep_from, ep_to = _resolve_episode_bounds(dataset, args.episode_index)
    base_idx = ep_from + int(args.step)
    if base_idx >= ep_to:
        raise ValueError(f"Requested step {args.step} exceeds episode {args.episode_index} length {ep_to - ep_from}")

    adapter = make_robot_adapter(
        cfg["robot_adapter"]["name"],
        cfg["robot_adapter"]["config"],
        dry_run=bool(cfg["robot_adapter"].get("dry_run", False)),
    )

    adapter.connect()
    try:
        current_obs = adapter.get_observation()
        current_state = np.asarray(current_obs["observation.state"], dtype=np.float64)
        image_mean = float(np.asarray(current_obs["observation.image"]).mean())
        camera_source = adapter.camera_source()
    finally:
        adapter.disconnect()

    hf = dataset.hf_dataset.with_format(None)
    compare_rows: list[tuple[int, np.ndarray, float, float]] = []
    end_idx = min(base_idx + max(args.window, 1), ep_to)
    for idx in range(base_idx, end_idx):
        state = np.asarray(hf[idx]["observation.state"], dtype=np.float64)
        diff = current_state - state
        l2 = float(np.linalg.norm(diff))
        mae = float(np.mean(np.abs(diff)))
        compare_rows.append((idx, state, l2, mae))

    best_idx, best_state, best_l2, best_mae = min(compare_rows, key=lambda x: x[2])
    direct_state = compare_rows[0][1]
    direct_diff = current_state - direct_state

    print(f"camera_source = {camera_source}")
    print(f"image_mean = {image_mean:.6f}")
    print(f"current_state = {_format_vec(current_state)}")
    print("")
    print(f"direct_compare_episode = {args.episode_index}")
    print(f"direct_compare_step = {args.step}")
    print(f"dataset_state = {_format_vec(direct_state)}")
    print(f"state_diff = {_format_vec(direct_diff)}")
    print(f"state_diff_l2 = {float(np.linalg.norm(direct_diff)):.6f}")
    print(f"state_diff_mae = {float(np.mean(np.abs(direct_diff))):.6f}")
    print("")
    print(f"best_match_index = {best_idx}")
    print(f"best_match_episode = {int(hf[best_idx]['episode_index'])}")
    print(f"best_match_frame = {int(hf[best_idx]['frame_index'])}")
    print(f"best_match_state = {_format_vec(best_state)}")
    print(f"best_match_l2 = {best_l2:.6f}")
    print(f"best_match_mae = {best_mae:.6f}")
    print("")
    print("window_summary:")
    for idx, state, l2, mae in compare_rows:
        frame_idx = int(hf[idx]["frame_index"])
        print(f"  idx={idx} frame={frame_idx} l2={l2:.6f} mae={mae:.6f}")


if __name__ == "__main__":
    main()
