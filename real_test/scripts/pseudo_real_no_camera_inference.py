#!/usr/bin/env python3

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from safety import ActionSafetyFilter, SafetyConfig


@dataclass
class PseudoStepStat:
    step: int
    global_index: int
    episode_index: int
    frame_index: int
    segment_id: int
    step_in_segment: int
    observe_source: str
    infer_ms: float
    safety_ms: float
    total_ms: float
    raw_action_oob_count: int
    raw_action_oob_ratio: float
    safety_flags: str


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv_header(path: Path, headers: list[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def _append_csv(path: Path, row: list[Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _row_to_observation(row: dict) -> dict[str, np.ndarray]:
    img = row["observation.image"]
    if hasattr(img, "convert"):
        img_np = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
    else:
        img_np = np.array(img, dtype=np.uint8, copy=True)
    state = np.array(row["observation.state"], dtype=np.float32, copy=True).reshape(-1)
    return {
        "observation.image": img_np,
        "observation.state": state,
    }


def _resolve_episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
        raise ValueError(f"episode_index out of range: {episode_index}")
    ep_from = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    ep_to = int(dataset.meta.episodes["dataset_to_index"][episode_index])
    return ep_from, ep_to


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
    dim = max(1, len(action_names))
    ratio = float(len(oob_names) / float(dim))
    return len(oob_names), ratio, oob_names


def _segment_boundaries(num_steps: int, segment_length: int) -> list[int]:
    starts = []
    for s in range(0, num_steps, segment_length):
        starts.append(s)
    return starts


def _plot_rollout(
    out_path: Path,
    steps: np.ndarray,
    safe_actions: np.ndarray,
    gt_actions: np.ndarray,
    displacement_safe: np.ndarray,
    displacement_gt: np.ndarray,
    segment_starts_local: list[int],
) -> None:
    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax0, ax1, ax2, ax3 = axes.flatten()

    xyz_names = ["x", "y", "z"]
    for i, name in enumerate(xyz_names):
        ax0.plot(steps, gt_actions[:, i], label=f"gt_{name}", linewidth=1.8)
        ax0.plot(steps, safe_actions[:, i], label=f"pred_{name}", linewidth=1.2, alpha=0.9)
    for s in segment_starts_local:
        ax0.axvline(s, color="gray", linewidth=0.7, alpha=0.4)
    ax0.set_title("XYZ Trajectory (GT vs Pseudo-Execution)")
    ax0.set_xlabel("step")
    ax0.set_ylabel("meter")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8, ncol=2)

    ax1.plot(gt_actions[:, 0], gt_actions[:, 1], label="gt_xy", linewidth=2.0)
    ax1.plot(safe_actions[:, 0], safe_actions[:, 1], label="pred_xy", linewidth=1.4)
    ax1.scatter([gt_actions[0, 0]], [gt_actions[0, 1]], marker="o", s=40, label="start")
    ax1.scatter([safe_actions[-1, 0]], [safe_actions[-1, 1]], marker="x", s=50, label="pred_end")
    ax1.set_title("End-Effector XY Path")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(steps, displacement_gt[:, 0], label="gt_dx", linewidth=1.8)
    ax2.plot(steps, displacement_gt[:, 1], label="gt_dy", linewidth=1.8)
    ax2.plot(steps, displacement_gt[:, 2], label="gt_dz", linewidth=1.8)
    ax2.plot(steps, displacement_safe[:, 0], "--", label="pred_dx", linewidth=1.2)
    ax2.plot(steps, displacement_safe[:, 1], "--", label="pred_dy", linewidth=1.2)
    ax2.plot(steps, displacement_safe[:, 2], "--", label="pred_dz", linewidth=1.2)
    for s in segment_starts_local:
        ax2.axvline(s, color="gray", linewidth=0.7, alpha=0.4)
    ax2.set_title("Displacement from Initial Step")
    ax2.set_xlabel("step")
    ax2.set_ylabel("meter")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    norm_gt = np.linalg.norm(displacement_gt, axis=1)
    norm_pred = np.linalg.norm(displacement_safe, axis=1)
    ax3.plot(steps, norm_gt, label="gt_disp_norm", linewidth=1.8)
    ax3.plot(steps, norm_pred, label="pred_disp_norm", linewidth=1.2)
    for s in segment_starts_local:
        ax3.axvline(s, color="gray", linewidth=0.7, alpha=0.4)
    ax3.set_title("Displacement Norm")
    ax3.set_xlabel("step")
    ax3.set_ylabel("meter")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pseudo real inference without camera/pose feedback: GT observation + segmented open-loop rollout."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Dataset episode index to replay")
    parser.add_argument("--start-step", type=int, default=0, help="Start step in selected episode")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="Total pseudo-execution steps. Clipped by episode length.",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=8,
        help="Open-loop segment length. Policy and pre/post processors reset each segment.",
    )
    parser.add_argument(
        "--feedback-mode",
        type=str,
        default="segment_open_loop",
        choices=["segment_open_loop", "gt_every_step"],
        help=(
            "segment_open_loop: each segment starts from GT step then runs without feedback; "
            "gt_every_step: feed GT observation every step (oracle upper bound)."
        ),
    )
    parser.add_argument(
        "--keep-prev-action-across-segments",
        action="store_true",
        help="Keep safety-filter prev_action across segment boundaries (default resets per segment).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/results/pseudo_no_camera",
        help="Output folder for csv/json/png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_json(args.config)
    action_names = cfg["action_schema"]["names"]
    target_hz = float(cfg["control"]["target_hz"])
    target_dt = 1.0 / max(target_hz, 1e-6)

    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_dir(output_dir)
    latency_csv = output_dir / "pseudo_latency.csv"
    safe_action_csv = output_dir / "pseudo_safe_actions.csv"
    raw_action_csv = output_dir / "pseudo_raw_actions.csv"
    gt_action_csv = output_dir / "pseudo_gt_actions.csv"
    obs_state_csv = output_dir / "pseudo_observation_state.csv"
    compare_csv = output_dir / "pseudo_compare.csv"
    summary_json = output_dir / "pseudo_summary.json"
    plot_png = output_dir / "pseudo_rollout.png"

    _write_csv_header(
        latency_csv,
        [
            "step",
            "global_index",
            "episode_index",
            "frame_index",
            "segment_id",
            "step_in_segment",
            "observe_source",
            "infer_ms",
            "safety_ms",
            "total_ms",
            "raw_action_oob_count",
            "raw_action_oob_ratio",
            "safety_flags",
        ],
    )
    _write_csv_header(safe_action_csv, ["step", *action_names])
    _write_csv_header(raw_action_csv, ["step", *action_names])
    _write_csv_header(gt_action_csv, ["step", *action_names])
    _write_csv_header(obs_state_csv, ["step", *action_names])
    _write_csv_header(
        compare_csv,
        [
            "step",
            "global_index",
            "segment_id",
            "step_in_segment",
            "raw_gt_l2",
            "safe_gt_l2",
            "safe_raw_l2",
        ],
    )

    dataset = LeRobotDataset(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])
    ep_from, ep_to = _resolve_episode_bounds(dataset, int(args.episode_index))
    start_idx = ep_from + int(args.start_step)
    if start_idx >= ep_to:
        raise ValueError(f"start step out of range: episode_len={ep_to - ep_from}, start={args.start_step}")

    max_steps = ep_to - start_idx
    run_steps = max(1, min(int(args.num_steps), max_steps))
    segment_length = max(1, int(args.segment_length))

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

    safety_cfg = SafetyConfig(
        action_bounds=cfg["safety"]["action_bounds"],
        workspace_bounds=cfg["safety"]["workspace_bounds"],
        max_xyz_speed_mps=float(cfg["safety"]["max_xyz_speed_mps"]),
        max_rot_delta_rad=float(cfg["safety"]["max_rot_delta_rad"]),
        max_gripper_delta_per_step=float(cfg["safety"]["max_gripper_delta_per_step"]),
        clip_workspace_in_action_space=True,
    )
    safety_filter = ActionSafetyFilter(safety_cfg, action_names)
    safe_device = get_safe_torch_device(policy.config.device)

    hf = dataset.hf_dataset.with_format(None)

    segment_summaries: list[dict[str, Any]] = []
    stats_rows: list[PseudoStepStat] = []
    safe_actions_log: list[np.ndarray] = []
    raw_actions_log: list[np.ndarray] = []
    gt_actions_log: list[np.ndarray] = []
    obs_states_log: list[np.ndarray] = []

    initial_state: np.ndarray | None = None
    prev_action: np.ndarray | None = None
    simulated_state: np.ndarray | None = None
    anchor_observation: dict[str, np.ndarray] | None = None
    anchor_idx = -1
    segment_id = -1
    segment_start_step = 0
    segment_clip_count = 0
    segment_infer_ms_sum = 0.0
    segment_start_state: np.ndarray | None = None

    print(
        "[INFO] pseudo-run "
        f"episode={args.episode_index}, start_step={args.start_step}, run_steps={run_steps}, "
        f"segment_length={segment_length}, feedback_mode={args.feedback_mode}"
    )
    print(f"[INFO] pretrained={pretrained_path}")
    print(f"[INFO] output_dir={output_dir}")

    for local_step in range(run_steps):
        global_idx = start_idx + local_step
        step_in_segment = local_step % segment_length
        start_new_segment = step_in_segment == 0

        if start_new_segment:
            if segment_id >= 0 and simulated_state is not None and segment_start_state is not None:
                seg_disp = simulated_state[:3].astype(np.float64) - segment_start_state[:3].astype(np.float64)
                segment_summaries.append(
                    {
                        "segment_id": segment_id,
                        "global_step_start": int(segment_start_step),
                        "global_step_end": int(local_step - 1),
                        "dataset_index_start": int(anchor_idx),
                        "dataset_index_end": int(global_idx - 1),
                        "steps": int(local_step - segment_start_step),
                        "segment_displacement_xyz": seg_disp.tolist(),
                        "segment_displacement_norm_m": float(np.linalg.norm(seg_disp)),
                        "segment_clip_count": int(segment_clip_count),
                        "segment_mean_infer_ms": float(
                            segment_infer_ms_sum / max(local_step - segment_start_step, 1)
                        ),
                    }
                )

            segment_id += 1
            segment_start_step = local_step
            segment_clip_count = 0
            segment_infer_ms_sum = 0.0
            anchor_idx = global_idx

            row_anchor = hf[anchor_idx]
            anchor_observation = _row_to_observation(row_anchor)
            simulated_state = anchor_observation["observation.state"].astype(np.float64).copy()
            segment_start_state = simulated_state.copy()
            if initial_state is None:
                initial_state = simulated_state.copy()

            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            if not args.keep_prev_action_across_segments:
                prev_action = None

            print(f"[SEG {segment_id:03d}] start_dataset_idx={anchor_idx}")

        assert anchor_observation is not None
        assert simulated_state is not None
        assert segment_start_state is not None

        if args.feedback_mode == "gt_every_step":
            obs_row = hf[global_idx]
            obs = _row_to_observation(obs_row)
            observe_source = "gt_every_step"
        else:
            if step_in_segment == 0:
                obs = {
                    "observation.image": anchor_observation["observation.image"].copy(),
                    "observation.state": simulated_state.astype(np.float32).copy(),
                }
                observe_source = "gt_segment_start"
            else:
                obs = {
                    "observation.image": anchor_observation["observation.image"].copy(),
                    "observation.state": simulated_state.astype(np.float32).copy(),
                }
                observe_source = "sim_open_loop"

        t0 = time.perf_counter()
        t_inf0 = time.perf_counter()
        action_tensor = predict_action(
            observation=obs,
            policy=policy,
            device=safe_device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task="pseudo_real_no_camera",
            robot_type="dataset_replay_no_camera",
        )
        raw_dict = make_robot_action(action_tensor, dataset.features)
        raw_action = np.array([raw_dict[n] for n in action_names], dtype=np.float64)
        if raw_action.shape[0] != len(action_names):
            raise RuntimeError(
                f"Invalid raw action shape: expected {len(action_names)}, got {raw_action.shape[0]}"
            )
        t_inf1 = time.perf_counter()

        raw_oob_count, raw_oob_ratio, _ = _compute_raw_action_oob(
            raw_action=raw_action,
            action_names=action_names,
            action_bounds=cfg["safety"]["action_bounds"],
        )

        t_safe0 = time.perf_counter()
        safe_action, flags = safety_filter.apply(raw_action, prev_action, dt_s=target_dt)
        t_safe1 = time.perf_counter()
        prev_action = safe_action.copy()
        simulated_state = safe_action.copy()

        if flags:
            segment_clip_count += 1
        segment_infer_ms_sum += (t_inf1 - t_inf0) * 1000.0

        gt_action = np.asarray(hf[global_idx]["action"], dtype=np.float64).reshape(-1)
        obs_state = np.asarray(obs["observation.state"], dtype=np.float64).reshape(-1)
        if gt_action.shape[0] != len(action_names):
            raise RuntimeError(
                f"Invalid GT action shape at index {global_idx}: expected {len(action_names)}, got {gt_action.shape[0]}"
            )
        if obs_state.shape[0] != len(action_names):
            raise RuntimeError(
                f"Invalid observation.state shape at index {global_idx}: expected {len(action_names)}, got {obs_state.shape[0]}"
            )

        safe_actions_log.append(safe_action.copy())
        raw_actions_log.append(raw_action.copy())
        gt_actions_log.append(gt_action.copy())
        obs_states_log.append(obs_state.copy())

        stat = PseudoStepStat(
            step=local_step,
            global_index=int(global_idx),
            episode_index=int(hf[global_idx]["episode_index"]),
            frame_index=int(hf[global_idx]["frame_index"]),
            segment_id=segment_id,
            step_in_segment=step_in_segment,
            observe_source=observe_source,
            infer_ms=(t_inf1 - t_inf0) * 1000.0,
            safety_ms=(t_safe1 - t_safe0) * 1000.0,
            total_ms=(time.perf_counter() - t0) * 1000.0,
            raw_action_oob_count=raw_oob_count,
            raw_action_oob_ratio=raw_oob_ratio,
            safety_flags="|".join(flags),
        )
        stats_rows.append(stat)

        _append_csv(
            latency_csv,
            [
                stat.step,
                stat.global_index,
                stat.episode_index,
                stat.frame_index,
                stat.segment_id,
                stat.step_in_segment,
                stat.observe_source,
                f"{stat.infer_ms:.4f}",
                f"{stat.safety_ms:.4f}",
                f"{stat.total_ms:.4f}",
                stat.raw_action_oob_count,
                f"{stat.raw_action_oob_ratio:.4f}",
                stat.safety_flags,
            ],
        )
        _append_csv(safe_action_csv, [local_step, *[f"{x:.8f}" for x in safe_action.tolist()]])
        _append_csv(raw_action_csv, [local_step, *[f"{x:.8f}" for x in raw_action.tolist()]])
        _append_csv(gt_action_csv, [local_step, *[f"{x:.8f}" for x in gt_action.tolist()]])
        _append_csv(obs_state_csv, [local_step, *[f"{x:.8f}" for x in obs_state.tolist()]])
        _append_csv(
            compare_csv,
            [
                local_step,
                global_idx,
                segment_id,
                step_in_segment,
                f"{float(np.linalg.norm(raw_action - gt_action)):.8f}",
                f"{float(np.linalg.norm(safe_action - gt_action)):.8f}",
                f"{float(np.linalg.norm(safe_action - raw_action)):.8f}",
            ],
        )

        if local_step < 3 or local_step % 20 == 0:
            print(
                f"[STEP {local_step:04d}] seg={segment_id:03d} in_seg={step_in_segment:02d} "
                f"idx={global_idx} infer={stat.infer_ms:.2f}ms oob={raw_oob_count}/{len(action_names)} "
                f"flags={stat.safety_flags or 'none'} src={observe_source}"
            )

    if segment_id >= 0 and simulated_state is not None and segment_start_state is not None:
        seg_disp = simulated_state[:3].astype(np.float64) - segment_start_state[:3].astype(np.float64)
        segment_summaries.append(
            {
                "segment_id": segment_id,
                "global_step_start": int(segment_start_step),
                "global_step_end": int(run_steps - 1),
                "dataset_index_start": int(anchor_idx),
                "dataset_index_end": int(start_idx + run_steps - 1),
                "steps": int(run_steps - segment_start_step),
                "segment_displacement_xyz": seg_disp.tolist(),
                "segment_displacement_norm_m": float(np.linalg.norm(seg_disp)),
                "segment_clip_count": int(segment_clip_count),
                "segment_mean_infer_ms": float(segment_infer_ms_sum / max(run_steps - segment_start_step, 1)),
            }
        )

    safe_actions_arr = np.stack(safe_actions_log, axis=0)
    gt_actions_arr = np.stack(gt_actions_log, axis=0)
    obs_states_arr = np.stack(obs_states_log, axis=0)

    if initial_state is None:
        raise RuntimeError("Failed to initialize pseudo rollout state.")
    final_state = safe_actions_arr[-1].astype(np.float64)
    final_gt_state = gt_actions_arr[-1].astype(np.float64)
    final_disp = final_state[:3] - initial_state[:3]
    final_gt_disp = final_gt_state[:3] - initial_state[:3]

    displacement_safe = safe_actions_arr[:, :3].astype(np.float64) - initial_state[:3].astype(np.float64)
    displacement_gt = gt_actions_arr[:, :3].astype(np.float64) - initial_state[:3].astype(np.float64)

    summary = {
        "mode": "pseudo_real_no_camera",
        "feedback_mode": args.feedback_mode,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "dataset": {
            "repo_id": cfg["dataset"]["repo_id"],
            "root": cfg["dataset"]["root"],
            "episode_index": int(args.episode_index),
            "start_step": int(args.start_step),
            "run_steps": int(run_steps),
            "global_index_start": int(start_idx),
            "global_index_end": int(start_idx + run_steps - 1),
        },
        "policy": {
            "checkpoint": pretrained_path,
            "device": policy_cfg.device,
            "use_amp": bool(policy_cfg.use_amp),
            "n_obs_steps": int(getattr(policy_cfg, "n_obs_steps", 1)),
            "n_action_steps": int(getattr(policy_cfg, "n_action_steps", 1)),
            "horizon": int(getattr(policy_cfg, "horizon", 0)),
        },
        "segment": {
            "segment_length": int(segment_length),
            "num_segments": int(len(segment_summaries)),
            "keep_prev_action_across_segments": bool(args.keep_prev_action_across_segments),
        },
        "final_state": {
            "initial_state": initial_state.tolist(),
            "pred_final_state": final_state.tolist(),
            "gt_final_state": final_gt_state.tolist(),
            "pred_final_displacement_xyz": final_disp.tolist(),
            "pred_final_displacement_norm_m": float(np.linalg.norm(final_disp)),
            "gt_final_displacement_xyz": final_gt_disp.tolist(),
            "gt_final_displacement_norm_m": float(np.linalg.norm(final_gt_disp)),
            "final_position_error_m": float(np.linalg.norm(final_state[:3] - final_gt_state[:3])),
            "final_action_error_l2": float(np.linalg.norm(final_state - final_gt_state)),
        },
        "runtime": {
            "mean_infer_ms": float(np.mean([s.infer_ms for s in stats_rows])),
            "p95_infer_ms": float(np.percentile([s.infer_ms for s in stats_rows], 95)),
            "mean_total_ms": float(np.mean([s.total_ms for s in stats_rows])),
            "mean_raw_oob_ratio": float(np.mean([s.raw_action_oob_ratio for s in stats_rows])),
            "steps_with_safety_flags": int(sum(1 for s in stats_rows if s.safety_flags)),
        },
        "segments": segment_summaries,
        "outputs": {
            "latency_csv": str(latency_csv),
            "safe_action_csv": str(safe_action_csv),
            "raw_action_csv": str(raw_action_csv),
            "gt_action_csv": str(gt_action_csv),
            "obs_state_csv": str(obs_state_csv),
            "compare_csv": str(compare_csv),
            "plot_png": str(plot_png) if plt is not None else "",
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if plt is not None:
        steps = np.arange(run_steps, dtype=np.int64)
        segment_starts = _segment_boundaries(run_steps, segment_length)
        _plot_rollout(
            out_path=plot_png,
            steps=steps,
            safe_actions=safe_actions_arr,
            gt_actions=gt_actions_arr,
            displacement_safe=displacement_safe,
            displacement_gt=displacement_gt,
            segment_starts_local=segment_starts,
        )

    print(f"[DONE] pseudo latency  : {latency_csv}")
    print(f"[DONE] pseudo safe act : {safe_action_csv}")
    print(f"[DONE] pseudo raw act  : {raw_action_csv}")
    print(f"[DONE] pseudo gt act   : {gt_action_csv}")
    print(f"[DONE] pseudo obs state: {obs_state_csv}")
    print(f"[DONE] pseudo compare  : {compare_csv}")
    print(f"[DONE] pseudo summary  : {summary_json}")
    if plt is not None:
        print(f"[DONE] pseudo plot     : {plot_png}")
    print(
        "[RESULT] pred_final_displacement_xyz = "
        f"{np.array2string(final_disp, precision=6, suppress_small=False)} m, "
        f"norm={np.linalg.norm(final_disp):.6f} m"
    )


if __name__ == "__main__":
    main()
