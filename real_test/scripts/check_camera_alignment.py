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

from adapters import _apply_image_transform, _capture_camera_frame, _configure_camera_for_uvc, _get_cv2_api_id, _open_camera
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_index(dataset: LeRobotDataset, episode_index: int | None, step: int, global_index: int | None) -> int:
    if episode_index is not None and global_index is not None:
        raise ValueError("--episode-index and --dataset-index cannot be used together")

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


def _load_dataset_image(cfg: dict, episode_index: int | None, step: int, global_index: int | None):
    ds = LeRobotDataset(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])
    idx = _resolve_index(ds, episode_index, step, global_index)
    row = ds.hf_dataset.with_format(None)[idx]
    img = row["observation.image"]
    if hasattr(img, "convert"):
        return np.array(img.convert("RGB"), dtype=np.uint8), row, idx
    return np.array(img, dtype=np.uint8), row, idx


def _stack_panels(panels: list[np.ndarray], cv2_module) -> np.ndarray:
    heights = [img.shape[0] for img in panels]
    target_h = max(heights)
    resized = []
    for img in panels:
        h, w = img.shape[:2]
        scale = target_h / float(h)
        target_w = max(1, int(round(w * scale)))
        resized.append(cv2_module.resize(img, (target_w, target_h), interpolation=cv2_module.INTER_AREA))
    return np.concatenate(resized, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one live GoPro frame and compare it with a dataset frame")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/results/camera_check",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    robot_cfg = cfg["robot_adapter"]["config"]

    import cv2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    camera_api = _get_cv2_api_id(cv2, robot_cfg.get("camera_api", "auto"))
    camera_device_path = robot_cfg.get("camera_device_path")
    camera_index = robot_cfg.get("camera_index")
    cap, source_desc = _open_camera(
        cv2_module=cv2,
        camera_api=camera_api,
        camera_device_path=camera_device_path,
        camera_index=camera_index,
    )

    if cap is None or not cap.isOpened():
        raise RuntimeError("Failed to open camera. Check camera_device_path / camera_index and camera_api.")

    _configure_camera_for_uvc(cap, cv2, robot_cfg)

    try:
        frame = None
        for _ in range(15):
            frame = _capture_camera_frame(cap, cv2)
            if frame is not None:
                break
        if frame is None:
            raise RuntimeError("Camera opened but failed to read a frame.")
    finally:
        cap.release()

    live_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_shape = tuple(robot_cfg.get("image_shape", [224, 224, 3]))
    processed = _apply_image_transform(
        frame_bgr=frame,
        cv2_module=cv2,
        output_hw=(int(image_shape[0]), int(image_shape[1])),
        rotation_deg=int(robot_cfg.get("camera_rotation_deg", 0)),
        flip_horizontal=bool(robot_cfg.get("camera_flip_horizontal", False)),
        flip_vertical=bool(robot_cfg.get("camera_flip_vertical", False)),
        trim_left=float(robot_cfg.get("camera_trim_left", 0.0)),
        trim_right=float(robot_cfg.get("camera_trim_right", 0.0)),
        trim_top=float(robot_cfg.get("camera_trim_top", 0.0)),
        trim_bottom=float(robot_cfg.get("camera_trim_bottom", 0.0)),
        crop_ratio=float(robot_cfg.get("camera_crop_ratio", 1.0)),
    )

    dataset_img, row, dataset_idx = _load_dataset_image(cfg, args.episode_index, args.step, args.dataset_index)
    composite = _stack_panels([live_rgb, processed, dataset_img], cv2)

    raw_path = output_dir / "live_raw.png"
    processed_path = output_dir / "live_processed.png"
    dataset_path = output_dir / "dataset_reference.png"
    compare_path = output_dir / "compare_side_by_side.png"

    cv2.imwrite(str(raw_path), cv2.cvtColor(live_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(processed_path), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(dataset_path), cv2.cvtColor(dataset_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(compare_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    print(f"saved: {raw_path}")
    print(f"saved: {processed_path}")
    print(f"saved: {dataset_path}")
    print(f"saved: {compare_path}")
    print(f"camera_source: {source_desc}")
    print(f"dataset_index: {dataset_idx}")
    print(f"dataset_episode_index: {row.get('episode_index')}")
    print(f"dataset_frame_index: {row.get('frame_index')}")
    print("confirmation:")
    print("1. Check whether the processed live image and dataset image have the same up/down and left/right orientation.")
    print("2. Check whether the object scale is close. If live image looks too wide, lower camera_crop_ratio such as 0.8 or 0.65.")
    print("3. Check whether the task workspace appears in the same central region after processing.")


if __name__ == "__main__":
    main()
