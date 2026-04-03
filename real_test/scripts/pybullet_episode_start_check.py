#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pybullet as pb
except Exception as exc:  # pragma: no cover
    pb = None
    _PYBULLET_IMPORT_ERROR = exc
else:
    _PYBULLET_IMPORT_ERROR = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class Candidate:
    episode_index: int
    frame_index: int
    dataset_index: int
    state10: np.ndarray


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[:3].astype(np.float64)
    a2 = rot6d[3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2_ortho = a2 - np.dot(a2, b1) * b1
    b2 = a2_ortho / (np.linalg.norm(a2_ortho) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def _matrix_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    tr = float(np.trace(r))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


def _quat_distance_rad(q_a: np.ndarray, q_b: np.ndarray) -> float:
    qa = q_a / (np.linalg.norm(q_a) + 1e-12)
    qb = q_b / (np.linalg.norm(q_b) + 1e-12)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(2.0 * math.acos(dot))


def _manual_to_base_pose(
    pos_manual: np.ndarray,
    rot_manual: np.ndarray,
    manual_origin_base: np.ndarray,
    manual_rotation_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos_base = manual_origin_base + manual_rotation_base @ pos_manual
    rot_base = manual_rotation_base @ rot_manual
    return pos_base, rot_base


def _rewrite_urdf_package_uris(
    urdf_path: Path,
    package_roots: dict[str, Path],
) -> Path:
    text = urdf_path.read_text(encoding="utf-8")

    pattern = re.compile(r"package://([^/]+)/")

    def _repl(match: re.Match[str]) -> str:
        pkg = match.group(1)
        root = package_roots.get(pkg)
        if root is None:
            raise ValueError(
                f"URDF uses package://{pkg}/ but no package root was provided. "
                "Please pass --urdf-package-roots, for example: rm_65_description=/abs/path/to/rm_65_description"
            )
        return str(root.resolve()) + "/"

    rewritten = pattern.sub(_repl, text)
    fd, tmp_path = tempfile.mkstemp(prefix="rm65_pybullet_", suffix=".urdf")
    Path(tmp_path).write_text(rewritten, encoding="utf-8")
    return Path(tmp_path)


def _resolve_package_roots(
    urdf_path: Path,
    urdf_package_roots: str,
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    if urdf_package_roots.strip():
        for item in urdf_package_roots.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    f"Invalid --urdf-package-roots item {item!r}. "
                    "Use format: package_name=/abs/path/to/package_root"
                )
            key, value = item.split("=", 1)
            roots[key.strip()] = Path(value.strip()).expanduser().resolve()

    # Common default for rm_65_description package:
    # .../rm_65_description/urdf/rm_65_description.urdf -> package root .../rm_65_description
    default_pkg_root = urdf_path.parent.parent
    if "rm_65_description" not in roots and (default_pkg_root / "meshes").is_dir():
        roots["rm_65_description"] = default_pkg_root

    return roots


def _collect_candidates(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    steps: list[int],
) -> list[Candidate]:
    hf = dataset.hf_dataset.with_format(None)
    out: list[Candidate] = []
    total_eps = int(dataset.meta.total_episodes)
    for ep in episode_indices:
        if ep < 0 or ep >= total_eps:
            raise ValueError(f"episode index out of range: {ep} / total={total_eps}")
        ep_from = int(dataset.meta.episodes["dataset_from_index"][ep])
        ep_to = int(dataset.meta.episodes["dataset_to_index"][ep])
        for step in steps:
            ds_idx = ep_from + step
            if ds_idx >= ep_to:
                continue
            row = hf[ds_idx]
            state = np.asarray(row["observation.state"], dtype=np.float64).reshape(-1)
            if state.shape[0] != 10:
                raise ValueError(f"observation.state is not 10D at dataset index {ds_idx}: {state.shape}")
            out.append(
                Candidate(
                    episode_index=ep,
                    frame_index=int(row["frame_index"]),
                    dataset_index=ds_idx,
                    state10=state.copy(),
                )
            )
    if not out:
        raise ValueError("No candidate states selected. Check --episodes / --steps.")
    return out


def _list_active_joints(body_id: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    joint_indices: list[int] = []
    joint_lower: list[float] = []
    joint_upper: list[float] = []
    for j in range(pb.getNumJoints(body_id)):
        info = pb.getJointInfo(body_id, j)
        joint_type = int(info[2])
        if joint_type not in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
            continue
        joint_indices.append(j)
        low = float(info[8])
        high = float(info[9])
        if low > high:
            low, high = -1e6, 1e6
        joint_lower.append(low)
        joint_upper.append(high)
    return joint_indices, np.asarray(joint_lower, dtype=np.float64), np.asarray(joint_upper, dtype=np.float64)


def _find_link_index_by_name(body_id: int, link_name: str) -> int:
    link_name = str(link_name).strip()
    for j in range(pb.getNumJoints(body_id)):
        info = pb.getJointInfo(body_id, j)
        child_name = info[12].decode("utf-8")
        if child_name == link_name:
            return j
    raise ValueError(f"Cannot find link_name={link_name!r} in URDF joints")


def _set_joint_state(body_id: int, joint_indices: list[int], q: np.ndarray) -> None:
    for i, jidx in enumerate(joint_indices):
        pb.resetJointState(body_id, jidx, float(q[i]))


def _ik_solve(
    body_id: int,
    ee_link_idx: int,
    joint_indices: list[int],
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
) -> np.ndarray:
    full_q = pb.calculateInverseKinematics(
        body_id,
        ee_link_idx,
        targetPosition=target_pos.tolist(),
        targetOrientation=target_quat_xyzw.tolist(),
        maxNumIterations=256,
        residualThreshold=1e-6,
    )
    full_q = np.asarray(full_q, dtype=np.float64).reshape(-1)
    if full_q.shape[0] < (max(joint_indices) + 1):
        raise RuntimeError(
            "PyBullet IK output length is shorter than expected joint index range: "
            f"{full_q.shape[0]} vs required {max(joint_indices)+1}"
        )
    q = np.asarray([full_q[j] for j in joint_indices], dtype=np.float64)
    return q


def _add_pose_marker(pos: np.ndarray, rot: np.ndarray, scale: float = 0.03) -> None:
    origin = pos.astype(np.float64)
    axes = rot.astype(np.float64)
    colors = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    for i in range(3):
        end = origin + axes[:, i] * scale
        pb.addUserDebugLine(origin.tolist(), end.tolist(), lineColorRGB=colors[i], lineWidth=2.0, lifeTime=0)
    pb.addUserDebugText("target", origin.tolist(), textColorRGB=[1, 1, 0], lifeTime=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyBullet-based episode start pose sanity check (manual->base mapping)")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.generated.json",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        required=True,
        help="Path to RM65 URDF (for example rm_65_description.urdf)",
    )
    parser.add_argument(
        "--urdf-package-roots",
        type=str,
        default="",
        help=(
            "Comma-separated package root map: "
            "package_name=/abs/path/to/root,another_pkg=/abs/path/to/root2 . "
            "If omitted, rm_65_description is auto-inferred from URDF parent."
        ),
    )
    parser.add_argument("--ee-link-name", type=str, default="Link6", help="URDF end-effector link name")
    parser.add_argument(
        "--episodes",
        type=str,
        default="0",
        help="Episode indices, comma separated. Use 'all' for all episodes.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="0",
        help="Frame offsets within each episode, comma separated (e.g. 0,1,2,3).",
    )
    parser.add_argument("--ik-pos-threshold-m", type=float, default=0.01)
    parser.add_argument("--ik-rot-threshold-rad", type=float, default=0.12)
    parser.add_argument(
        "--soft-margin-ratio",
        type=float,
        default=0.10,
        help="Minimum normalized margin to joint limits. Example 0.1 means 10%% range margin.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open PyBullet GUI and render candidate targets / solved poses",
    )
    parser.add_argument(
        "--show-seconds",
        type=float,
        default=8.0,
        help="GUI mode only: time to keep window open after drawing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/results/start_pose_check",
    )
    args = parser.parse_args()

    if pb is None:
        raise RuntimeError(
            "pybullet is not installed in this Python environment. "
            f"Import error: {_PYBULLET_IMPORT_ERROR}"
        )

    cfg = _load_json(args.config)
    action_frame = str(cfg.get("action_schema", {}).get("coordinate_frame", "")).strip().lower()
    if action_frame != "manual_relative_frame":
        raise ValueError(
            f"Expected action_schema.coordinate_frame=manual_relative_frame, got {action_frame!r}"
        )

    ra_cfg = cfg["robot_adapter"]["config"]
    policy_frame = str(ra_cfg.get("policy_frame", "")).strip().lower()
    if policy_frame != "manual_relative_frame":
        raise ValueError(f"Expected robot_adapter.config.policy_frame=manual_relative_frame, got {policy_frame!r}")

    manual_origin = np.asarray(ra_cfg["manual_origin"], dtype=np.float64).reshape(3)
    manual_rotation = np.asarray(ra_cfg["manual_rotation"], dtype=np.float64).reshape(3, 3)
    if not np.allclose(manual_rotation.T @ manual_rotation, np.eye(3), atol=1e-5):
        raise ValueError("manual_rotation is not orthonormal")

    dataset = LeRobotDataset(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])
    if str(args.episodes).strip().lower() == "all":
        episode_indices = list(range(int(dataset.meta.total_episodes)))
    else:
        episode_indices = _parse_int_list(args.episodes)
    steps = _parse_int_list(args.steps)
    if not steps:
        raise ValueError("--steps cannot be empty")
    candidates = _collect_candidates(dataset, episode_indices, steps)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    urdf_path = Path(args.urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    package_roots = _resolve_package_roots(urdf_path, str(args.urdf_package_roots))
    rewritten_urdf = _rewrite_urdf_package_uris(urdf_path, package_roots)

    cid = pb.connect(pb.GUI if args.gui else pb.DIRECT)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet")
    try:
        pb.resetSimulation()
        pb.setGravity(0.0, 0.0, -9.81)
        pb.setTimeStep(1.0 / 240.0)
        robot_id = pb.loadURDF(str(rewritten_urdf), useFixedBase=True)
        ee_link_idx = _find_link_index_by_name(robot_id, args.ee_link_name)
        joint_indices, lower, upper = _list_active_joints(robot_id)
        if not joint_indices:
            raise RuntimeError("No active joints found in URDF")

        rows: list[dict] = []
        for cand in candidates:
            st = cand.state10
            pos_m = st[:3]
            rot_m = _rot6d_to_matrix(st[3:9])
            pos_b, rot_b = _manual_to_base_pose(pos_m, rot_m, manual_origin, manual_rotation)
            quat_b = _matrix_to_quat_xyzw(rot_b)

            q_sol = _ik_solve(robot_id, ee_link_idx, joint_indices, pos_b, quat_b)
            _set_joint_state(robot_id, joint_indices, q_sol)
            link_state = pb.getLinkState(robot_id, ee_link_idx, computeForwardKinematics=True)
            fk_pos = np.asarray(link_state[4], dtype=np.float64)
            fk_quat = np.asarray(link_state[5], dtype=np.float64)
            pos_err = float(np.linalg.norm(fk_pos - pos_b))
            rot_err = _quat_distance_rad(fk_quat, quat_b)

            margin_low = q_sol - lower
            margin_high = upper - q_sol
            min_margin = float(np.min(np.minimum(margin_low, margin_high)))
            joint_range = np.maximum(upper - lower, 1e-8)
            min_margin_ratio = float(np.min(np.minimum(margin_low, margin_high) / joint_range))
            within_limits = bool(np.all(margin_low >= 0.0) and np.all(margin_high >= 0.0))
            ik_ok = bool(
                within_limits
                and (pos_err <= float(args.ik_pos_threshold_m))
                and (rot_err <= float(args.ik_rot_threshold_rad))
            )
            soft_ok = bool(min_margin_ratio >= float(args.soft_margin_ratio))
            usable = bool(ik_ok and soft_ok)

            rows.append(
                {
                    "episode_index": int(cand.episode_index),
                    "frame_index": int(cand.frame_index),
                    "dataset_index": int(cand.dataset_index),
                    "x_manual": float(pos_m[0]),
                    "y_manual": float(pos_m[1]),
                    "z_manual": float(pos_m[2]),
                    "x_base": float(pos_b[0]),
                    "y_base": float(pos_b[1]),
                    "z_base": float(pos_b[2]),
                    "ik_pos_err_m": pos_err,
                    "ik_rot_err_rad": rot_err,
                    "min_joint_margin_rad": min_margin,
                    "min_joint_margin_ratio": min_margin_ratio,
                    "ik_ok": int(ik_ok),
                    "soft_ok": int(soft_ok),
                    "usable_start": int(usable),
                }
            )

            if args.gui:
                color = [0, 1, 0] if usable else [1, 0, 0]
                pb.addUserDebugLine(
                    [float(pos_b[0]), float(pos_b[1]), float(pos_b[2])],
                    [float(pos_b[0]), float(pos_b[1]), float(pos_b[2] + 0.04)],
                    lineColorRGB=color,
                    lineWidth=3.0,
                    lifeTime=0,
                )
                _add_pose_marker(pos_b, rot_b, scale=0.03)

        csv_path = output_dir / "start_pose_check.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        arr_usable = np.asarray([r["usable_start"] for r in rows], dtype=np.int64)
        arr_ik = np.asarray([r["ik_ok"] for r in rows], dtype=np.int64)
        arr_soft = np.asarray([r["soft_ok"] for r in rows], dtype=np.int64)
        summary = {
            "config_path": str(Path(args.config).expanduser().resolve()),
            "dataset_repo_id": str(cfg["dataset"]["repo_id"]),
            "dataset_root": str(cfg["dataset"]["root"]),
            "urdf_path": str(urdf_path),
            "rewritten_urdf_path": str(rewritten_urdf),
            "ee_link_name": str(args.ee_link_name),
            "num_candidates": int(len(rows)),
            "num_usable_start": int(arr_usable.sum()),
            "usable_ratio": float(arr_usable.mean() if len(arr_usable) else 0.0),
            "ik_ok_ratio": float(arr_ik.mean() if len(arr_ik) else 0.0),
            "soft_margin_ok_ratio": float(arr_soft.mean() if len(arr_soft) else 0.0),
            "ik_pos_threshold_m": float(args.ik_pos_threshold_m),
            "ik_rot_threshold_rad": float(args.ik_rot_threshold_rad),
            "soft_margin_ratio": float(args.soft_margin_ratio),
            "episodes": episode_indices,
            "steps": steps,
            "joint_indices": [int(x) for x in joint_indices],
            "joint_lower_rad": [float(x) for x in lower.tolist()],
            "joint_upper_rad": [float(x) for x in upper.tolist()],
            "manual_origin": [float(x) for x in manual_origin.tolist()],
            "manual_rotation": [[float(x) for x in row] for row in manual_rotation.tolist()],
        }
        summary_path = output_dir / "start_pose_check_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if plt is not None:
            x = np.asarray([r["x_base"] for r in rows], dtype=np.float64)
            y = np.asarray([r["y_base"] for r in rows], dtype=np.float64)
            z = np.asarray([r["z_base"] for r in rows], dtype=np.float64)
            c = np.asarray([r["usable_start"] for r in rows], dtype=np.int64)
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x[c == 1], y[c == 1], z[c == 1], c="green", s=30, label="usable")
            ax.scatter(x[c == 0], y[c == 0], z[c == 0], c="red", s=30, label="not_usable")
            ax.set_xlabel("x_base (m)")
            ax.set_ylabel("y_base (m)")
            ax.set_zlabel("z_base (m)")
            ax.set_title("Episode Start Candidates in Base Frame")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(output_dir / "start_pose_scatter_base.png", dpi=160)
            plt.close(fig)

        print(f"[DONE] candidates={len(rows)}")
        print(f"[DONE] usable={int(arr_usable.sum())}/{len(rows)} ({float(arr_usable.mean() if len(arr_usable) else 0.0):.1%})")
        print(f"[DONE] csv={csv_path}")
        print(f"[DONE] summary={summary_path}")
        if plt is not None:
            print(f"[DONE] plot={output_dir / 'start_pose_scatter_base.png'}")
        else:
            print("[WARN] matplotlib not available, skip scatter plot")

        if args.gui:
            print(f"[INFO] GUI mode: keep window for {float(args.show_seconds):.1f}s")
            t_end = time.time() + float(max(args.show_seconds, 0.0))
            while time.time() < t_end:
                pb.stepSimulation()
                time.sleep(1.0 / 240.0)
    finally:
        try:
            pb.disconnect()
        except Exception:
            pass
        try:
            rewritten_urdf.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
