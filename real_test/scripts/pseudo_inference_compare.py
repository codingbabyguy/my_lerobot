#!/usr/bin/env python3

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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

from safety import rot6d_to_matrix


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _select_indices(
    dataset: LeRobotDataset,
    episode_index: int | None,
    start_step: int,
    num_chunks: int,
    chunk_size: int,
    global_start: int | None,
    chunk_stride: int,
) -> tuple[list[int], int]:
    if episode_index is not None and global_start is not None:
        raise ValueError("--episode-index 和 --global-start 不能同时设置")

    if episode_index is not None:
        if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
            raise ValueError(f"episode_index 超界: {episode_index}")
        ep_from = int(dataset.meta.episodes["dataset_from_index"][episode_index])
        ep_to = int(dataset.meta.episodes["dataset_to_index"][episode_index])
        start = ep_from + max(start_step, 0)
        end_limit = ep_to
    else:
        start = max(global_start or 0, 0)
        end_limit = len(dataset.hf_dataset)

    if start >= end_limit:
        raise ValueError("选取区间为空，请检查 --start-step/--global-start")

    chunk_size = max(1, int(chunk_size))
    chunk_stride = max(1, int(chunk_stride))
    num_chunks = max(1, int(num_chunks))

    starts: list[int] = []
    cursor = start
    while len(starts) < num_chunks and (cursor + chunk_size) <= end_limit:
        starts.append(cursor)
        cursor += chunk_stride

    if not starts:
        raise ValueError("没有足够数据组成一个 chunk，请减小 chunk_size 或调整起点")

    return starts, end_limit


def _build_observation(row: dict) -> dict[str, np.ndarray]:
    img = row["observation.image"]
    if hasattr(img, "convert"):
        img_np = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
    else:
        img_np = np.array(img, dtype=np.uint8, copy=True)

    state = np.array(row["observation.state"], dtype=np.float32, copy=True)
    return {
        "observation.image": img_np,
        "observation.state": state,
    }


def _arm_points_from_action(action: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    if base is None:
        base = np.zeros(3, dtype=np.float64)

    ee = action[:3].astype(np.float64)
    rot6d = action[3:9].astype(np.float64)

    vec = ee - base
    dist = float(np.linalg.norm(vec))
    u = vec / (dist + 1e-12) if dist > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    r = rot6d_to_matrix(rot6d)
    side = r[:, 0].astype(np.float64)
    side = side - np.dot(side, u) * u
    s_norm = float(np.linalg.norm(side))
    if s_norm < 1e-8:
        side = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        side = side / s_norm

    fracs = np.array([0.0, 0.18, 0.36, 0.55, 0.73, 0.88, 1.0], dtype=np.float64)
    bends = np.array([0.0, 0.02, 0.03, 0.025, 0.016, 0.008, 0.0], dtype=np.float64)

    pts = []
    for f, b in zip(fracs, bends, strict=True):
        p = base + u * (dist * f) + side * b
        pts.append(p)
    pts = np.stack(pts, axis=0)
    pts[-1] = ee
    return pts


def _save_csv(
    path: Path,
    action_names: list[str],
    chunk_ids: list[int],
    step_in_chunk: list[int],
    chunk_starts: list[int],
    step_ids: list[int],
    gt: np.ndarray,
    pred: np.ndarray,
    err: np.ndarray,
) -> None:
    headers = ["chunk_id", "step_in_chunk", "chunk_start", "step"]
    for n in action_names:
        headers += [f"gt_{n}", f"pred_{n}", f"err_{n}"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, sid in enumerate(step_ids):
            row = [chunk_ids[i], step_in_chunk[i], chunk_starts[i], sid]
            for j in range(len(action_names)):
                row += [f"{gt[i, j]:.8f}", f"{pred[i, j]:.8f}", f"{err[i, j]:.8f}"]
            writer.writerow(row)


def _save_summary(
    path: Path,
    action_names: list[str],
    err: np.ndarray,
    num_chunks: int,
    chunk_size: int,
) -> None:
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    max_abs = np.max(np.abs(err), axis=0)

    out = {
        "num_chunks": int(num_chunks),
        "chunk_size": int(chunk_size),
        "num_steps": int(err.shape[0]),
        "action_dim": int(err.shape[1]),
        "overall": {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "max_abs": float(np.max(np.abs(err))),
        },
        "per_dim": {
            action_names[i]: {
                "mae": float(mae[i]),
                "rmse": float(rmse[i]),
                "max_abs": float(max_abs[i]),
            }
            for i in range(len(action_names))
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _plot_action_compare(
    out_path: Path,
    x_steps: np.ndarray,
    action_names: list[str],
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    x_label: str,
) -> None:
    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(18, 14), sharex=True)
    axes = axes.flatten()

    for i, name in enumerate(action_names):
        ax = axes[i]
        ax.plot(x_steps, gt[:, i], label="gt", linewidth=1.6)
        ax.plot(x_steps, pred[:, i], label="pred", linewidth=1.2, alpha=0.9)
        ax.set_title(name)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel(x_label)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_global_branch_compare(
    out_path: Path,
    action_names: list[str],
    step_ids: np.ndarray,
    gt: np.ndarray,
    chunk_starts: list[int],
    chunk_size: int,
    pred_chunk: np.ndarray,
) -> None:
    """
    关键图：
    - GT 为全局连续主线。
    - 在每个 chunk 起点打点。
    - 从起点引出该 chunk 的预测分支线，直观看到同起点下未来 8 帧偏差。
    """
    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(18, 14), sharex=True)
    axes = axes.flatten()

    # 若出现重叠 step（例如 stride < chunk_size），对 GT 先按 step 聚合平均，保证主线连续。
    uniq_steps = np.unique(step_ids)
    gt_unique = np.zeros((len(uniq_steps), gt.shape[1]), dtype=np.float64)
    for k, s in enumerate(uniq_steps):
        mask = step_ids == s
        gt_unique[k] = np.mean(gt[mask], axis=0)

    for i, name in enumerate(action_names):
        ax = axes[i]
        ax.plot(uniq_steps, gt_unique[:, i], color="#1f77b4", linewidth=2.2, label="gt")

        start_vals = []
        for cidx, start in enumerate(chunk_starts):
            x_seg = np.arange(start, start + chunk_size, dtype=np.int64)
            y_seg = pred_chunk[cidx, :, i]
            ax.plot(x_seg, y_seg, color="#ff7f0e", linewidth=1.35, alpha=0.52)

            # 起点点位取 GT 主线上对应值，确保“从 GT 点生长出来”的视觉语义。
            pos = np.where(uniq_steps == start)[0]
            if len(pos) > 0:
                start_vals.append(gt_unique[pos[0], i])

        if len(start_vals) > 0:
            ax.scatter(
                np.asarray(chunk_starts, dtype=np.int64),
                np.asarray(start_vals, dtype=np.float64),
                s=18,
                color="#1f77b4",
                edgecolors="black",
                linewidths=0.35,
                zorder=5,
                label="chunk start",
            )

        ax.set_title(name)
        ax.grid(alpha=0.25)

    axes[0].plot([], [], color="#ff7f0e", linewidth=1.5, alpha=0.7, label="pred branch")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("dataset step")
    fig.suptitle("Pseudo Inference: GT Backbone with Pred Branches from Chunk Starts", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_chunk_action_compare(
    out_path: Path,
    action_names: list[str],
    gt_chunk: np.ndarray,
    pred_chunk: np.ndarray,
) -> None:
    """
    按 chunk 起点对齐后绘制动作对比：
    每个 chunk 的 step=0 被归一化为 0，清晰显示同起点下的发散差异。
    """
    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(18, 14), sharex=True)
    axes = axes.flatten()

    steps = np.arange(gt_chunk.shape[1], dtype=np.int64)
    gt_delta = gt_chunk - gt_chunk[:, :1, :]
    pred_delta = pred_chunk - pred_chunk[:, :1, :]

    for i, name in enumerate(action_names):
        ax = axes[i]
        gt_mean = np.mean(gt_delta[:, :, i], axis=0)
        gt_std = np.std(gt_delta[:, :, i], axis=0)
        pred_mean = np.mean(pred_delta[:, :, i], axis=0)
        pred_std = np.std(pred_delta[:, :, i], axis=0)

        ax.plot(steps, gt_mean, label="gt mean", linewidth=2.0, color="#1f77b4")
        ax.fill_between(steps, gt_mean - gt_std, gt_mean + gt_std, alpha=0.20, color="#1f77b4")

        ax.plot(steps, pred_mean, label="pred mean", linewidth=2.0, color="#ff7f0e")
        ax.fill_between(steps, pred_mean - pred_std, pred_mean + pred_std, alpha=0.20, color="#ff7f0e")

        ax.scatter([0], [0], s=24, color="black", zorder=5)
        ax.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.45, color="black")
        ax.set_title(name)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("step in chunk")
    fig.suptitle("Chunk Start-Aligned Action Compare (mean ± std)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_chunk_pose_compare(
    out_path: Path,
    gt_chunk: np.ndarray,
    pred_chunk: np.ndarray,
) -> None:
    """
    按 chunk 起点对齐后统计姿态维度误差，降低线条重叠造成的杂乱。
    """
    pose_names = ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5"]
    pose_idx = list(range(9))
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    axes = axes.flatten()

    steps = np.arange(gt_chunk.shape[1], dtype=np.int64)
    gt_delta = gt_chunk[:, :, :9] - gt_chunk[:, :1, :9]
    pred_delta = pred_chunk[:, :, :9] - pred_chunk[:, :1, :9]
    abs_err = np.abs(pred_delta - gt_delta)

    for i, (name, idx) in enumerate(zip(pose_names, pose_idx, strict=True)):
        ax = axes[i]
        mae = np.mean(abs_err[:, :, idx], axis=0)
        p90 = np.percentile(abs_err[:, :, idx], 90, axis=0)
        ax.plot(steps, mae, linewidth=2.0, label="MAE", color="#d62728")
        ax.plot(steps, p90, linewidth=1.5, linestyle="--", alpha=0.9, label="P90 |err|", color="#9467bd")
        ax.scatter([0], [0], s=22, color="black", zorder=5)
        ax.set_title(name)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("step in chunk")
    fig.suptitle("Chunk Start-Aligned Pose Error (xyz + rot6d)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_xyz_trajectory(out_path: Path, gt: np.ndarray, pred: np.ndarray) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], label="gt xyz", linewidth=2.0)
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], label="pred xyz", linewidth=1.8)

    ax.scatter([gt[0, 0]], [gt[0, 1]], [gt[0, 2]], c="green", s=35, label="gt start")
    ax.scatter([pred[0, 0]], [pred[0, 1]], [pred[0, 2]], c="purple", s=35, label="pred start")
    ax.scatter([gt[-1, 0]], [gt[-1, 1]], [gt[-1, 2]], c="red", s=45, label="gt end")
    ax.scatter([pred[-1, 0]], [pred[-1, 1]], [pred[-1, 2]], c="orange", s=45, label="pred end")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("End-Effector XYZ Trajectory")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_simple_arm_snapshots(
    out_path: Path,
    step_ids: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    num_snapshots: int,
) -> None:
    num_snapshots = int(max(1, min(num_snapshots, len(step_ids))))
    sel = np.linspace(0, len(step_ids) - 1, num_snapshots).round().astype(int)

    fig = plt.figure(figsize=(4.8 * num_snapshots, 4.6))
    for k, i in enumerate(sel, start=1):
        ax = fig.add_subplot(1, num_snapshots, k, projection="3d")
        arm_gt = _arm_points_from_action(gt[i])
        arm_pr = _arm_points_from_action(pred[i])

        ax.plot(arm_gt[:, 0], arm_gt[:, 1], arm_gt[:, 2], "-o", linewidth=2.0, markersize=3.5, label="gt")
        ax.plot(arm_pr[:, 0], arm_pr[:, 1], arm_pr[:, 2], "-o", linewidth=1.8, markersize=3.2, label="pred")

        ax.set_title(f"step={step_ids[i]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(alpha=0.25)
        if k == 1:
            ax.legend(loc="upper left")

    fig.suptitle("Simple 6-axis Line-arm Snapshot (GT vs Pred)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo inference on dataset and compare predicted actions")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    parser.add_argument("--episode-index", type=int, default=None, help="按 episode 选取输入片段")
    parser.add_argument("--start-step", type=int, default=0, help="episode 内起点步")
    parser.add_argument("--start-offset", type=int, default=None, help="兼容旧参数，等价于 --start-step")
    parser.add_argument("--global-start", type=int, default=None, help="全局数据索引起点")
    parser.add_argument("--num-chunks", type=int, default=25, help="chunk 数量")
    parser.add_argument("--chunk-size", type=int, default=0, help="每个 chunk 的步数，0 表示使用模型 n_action_steps")
    parser.add_argument("--chunk-stride", type=int, default=0, help="chunk 起点步长，0 表示等于 chunk-size")
    parser.add_argument("--num-arm-snapshots", type=int, default=6, help="简化机械臂快照数量")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/results/pseudo_inference",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    action_names = cfg["action_schema"]["names"]

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    dataset = LeRobotDataset(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])
    pretrained_path = cfg["checkpoint"]["pretrained_model_path"]

    from lerobot.configs.policies import PreTrainedConfig

    argv_backup = list(sys.argv)
    try:
        sys.argv = [sys.argv[0]]
        policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    finally:
        sys.argv = argv_backup

    policy_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_cfg.pretrained_path = pretrained_path

    default_chunk_size = int(getattr(policy_cfg, "n_action_steps", 8))
    chunk_size = int(args.chunk_size) if int(args.chunk_size) > 0 else default_chunk_size
    chunk_stride = int(args.chunk_stride) if int(args.chunk_stride) > 0 else chunk_size
    start_step = int(args.start_offset) if args.start_offset is not None else int(args.start_step)

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

    chunk_starts, end_limit = _select_indices(
        dataset=dataset,
        episode_index=args.episode_index,
        start_step=start_step,
        num_chunks=args.num_chunks,
        chunk_size=chunk_size,
        global_start=args.global_start,
        chunk_stride=chunk_stride,
    )

    if len(chunk_starts) < int(args.num_chunks):
        print(f"[WARN] 可用 chunk 不足，请求={args.num_chunks}, 实际={len(chunk_starts)}, end_limit={end_limit}")

    sample_count = len(chunk_starts) * chunk_size
    step_ids: list[int] = []
    chunk_ids: list[int] = []
    step_in_chunk: list[int] = []
    row_chunk_starts: list[int] = []
    gt_actions: list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []

    hf = dataset.hf_dataset.with_format(None)

    print(
        "[INFO] chunk_mode="
        f"true, chunk_size={chunk_size}, chunk_stride={chunk_stride}, "
        f"num_chunks={len(chunk_starts)}, total_steps={sample_count}"
    )

    safe_device = get_safe_torch_device(policy.config.device)
    for cidx, start_idx in enumerate(chunk_starts):
        row0 = hf[start_idx]
        obs0 = _build_observation(row0)

        # 每个 chunk 开始时重置策略缓存，只用 chunk 起点观测做开环预测。
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        pred_chunk: list[np.ndarray] = []
        for inner in range(chunk_size):
            action_tensor = predict_action(
                observation=obs0,
                policy=policy,
                device=safe_device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task="pseudo_inference_chunk",
                robot_type="dataset_replay",
            )
            pred_dict = make_robot_action(action_tensor, dataset.features)
            pred = np.array([pred_dict[n] for n in action_names], dtype=np.float64)
            pred_chunk.append(pred)

        for inner in range(chunk_size):
            idx = start_idx + inner
            gt = np.asarray(hf[idx]["action"], dtype=np.float64)
            pred = pred_chunk[inner]

            step_ids.append(idx)
            chunk_ids.append(cidx)
            step_in_chunk.append(inner)
            row_chunk_starts.append(start_idx)
            gt_actions.append(gt)
            pred_actions.append(pred)

        if cidx % 10 == 0:
            print(f"[CHUNK {cidx:04d}] start={start_idx}")

    gt_arr = np.stack(gt_actions, axis=0)
    pred_arr = np.stack(pred_actions, axis=0)
    err = pred_arr - gt_arr

    _save_csv(
        out_dir / "pseudo_compare.csv",
        action_names,
        chunk_ids,
        step_in_chunk,
        row_chunk_starts,
        step_ids,
        gt_arr,
        pred_arr,
        err,
    )
    _save_summary(
        out_dir / "pseudo_summary.json",
        action_names,
        err,
        num_chunks=len(chunk_starts),
        chunk_size=chunk_size,
    )

    if plt is None:
        print("[WARN] matplotlib not installed, skip png plots")
    else:
        step_arr = np.asarray(step_ids, dtype=np.int64)
        gt_chunk = gt_arr.reshape(len(chunk_starts), chunk_size, -1)
        pred_chunk = pred_arr.reshape(len(chunk_starts), chunk_size, -1)
        _plot_global_branch_compare(
            out_dir / "compare_actions_global.png",
            action_names,
            step_arr,
            gt_arr,
            chunk_starts,
            chunk_size,
            pred_chunk,
        )
        _plot_chunk_action_compare(
            out_dir / "compare_actions_chunked.png",
            action_names,
            gt_chunk,
            pred_chunk,
        )
        _plot_chunk_pose_compare(
            out_dir / "compare_pose_dims_chunked.png",
            gt_chunk,
            pred_chunk,
        )
        _plot_xyz_trajectory(out_dir / "compare_xyz_trajectory.png", gt_arr[:, :3], pred_arr[:, :3])
        _plot_simple_arm_snapshots(
            out_dir / "compare_simple_arm_snapshots.png",
            step_arr,
            gt_arr,
            pred_arr,
            num_snapshots=args.num_arm_snapshots,
        )

    print(f"[DONE] csv: {out_dir / 'pseudo_compare.csv'}")
    print(f"[DONE] summary: {out_dir / 'pseudo_summary.json'}")
    if plt is not None:
        print(f"[DONE] figure: {out_dir / 'compare_actions_global.png'}")
        print(f"[DONE] figure: {out_dir / 'compare_actions_chunked.png'}")
        print(f"[DONE] figure: {out_dir / 'compare_pose_dims_chunked.png'}")
        print(f"[DONE] figure: {out_dir / 'compare_xyz_trajectory.png'}")
        print(f"[DONE] figure: {out_dir / 'compare_simple_arm_snapshots.png'}")


if __name__ == "__main__":
    main()
