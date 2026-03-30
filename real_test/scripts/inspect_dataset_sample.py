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

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_index(dataset: LeRobotDataset, episode_index: int | None, step: int, global_index: int | None) -> int:
    if episode_index is not None and global_index is not None:
        raise ValueError("--episode-index and --global-index cannot be used together")

    if global_index is not None:
        return int(global_index)

    if episode_index is None:
        return int(step)

    ep_from = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    ep_to = int(dataset.meta.episodes["dataset_to_index"][episode_index])
    idx = ep_from + int(step)
    if idx >= ep_to:
        raise ValueError(f"Requested step {step} exceeds episode {episode_index} length {ep_to - ep_from}")
    return idx


def _print_array(name: str, value) -> None:
    if value is None:
        print(f"{name}: <missing>")
        return
    arr = np.asarray(value)
    preview = np.array2string(arr, precision=5, threshold=20)
    print(f"{name}: shape={arr.shape} dtype={arr.dtype}")
    print(preview)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LeRobot dataset sample keys and low-dim semantics")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--global-index", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_json(args.config)
    ds = LeRobotDataset(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])

    print(f"repo_id: {cfg['dataset']['repo_id']}")
    print(f"root: {cfg['dataset']['root']}")
    print(f"total_frames: {len(ds.hf_dataset)}")
    print(f"total_episodes: {ds.meta.total_episodes}")
    print(f"fps: {ds.meta.fps}")
    print(f"dataset keys: {list(ds.hf_dataset.features.keys())}")

    idx = _resolve_index(ds, args.episode_index, args.step, args.global_index)
    row = ds.hf_dataset.with_format(None)[idx]
    print(f"selected_index: {idx}")
    print(f"row keys: {list(row.keys())}")

    _print_array("observation.state", row.get("observation.state"))
    _print_array("action", row.get("action"))

    img = row.get("observation.image")
    if img is None:
        print("observation.image: <missing>")
    elif hasattr(img, "size"):
        print(f"observation.image: PIL size={img.size}")
    else:
        arr = np.asarray(img)
        print(f"observation.image: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
