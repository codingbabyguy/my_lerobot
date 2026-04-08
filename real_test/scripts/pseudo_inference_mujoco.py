#!/usr/bin/env python3

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

try:
    import matplotlib.pyplot as plt
except Exception:
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

from frame_transforms import FrameTransformChain
from safety import ActionSafetyFilter, SafetyConfig, matrix_to_rot6d, rot6d_to_matrix


@dataclass
class IKCandidate:
    q: np.ndarray
    total_cost: float
    pos_err_m: float
    rot_err_rad: float
    limit_margin_rad: float
    sigma_min: float
    branch_jump: bool
    wrist_flip_raw: float
    transition_linf_rad: float
    seed_name: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pose_error_components(target_T: np.ndarray, actual_T: np.ndarray) -> tuple[float, float]:
    dp = np.asarray(actual_T[:3, 3] - target_T[:3, 3], dtype=np.float64)
    pos_err_m = float(np.linalg.norm(dp))
    r_err = R.from_matrix(actual_T[:3, :3].T @ target_T[:3, :3]).as_rotvec()
    rot_err_rad = float(np.linalg.norm(r_err))
    return pos_err_m, rot_err_rad


def _as_target_T_from_manual_state(frame_chain: FrameTransformChain, state10: np.ndarray) -> np.ndarray:
    arr = np.asarray(state10, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 9:
        raise ValueError(f"state must be at least 9D, got {arr.shape}")
    pos_manual = arr[:3]
    rot_manual = rot6d_to_matrix(arr[3:9])
    pos_base, rot_base = frame_chain.manual_flange_to_base_flange(pos_manual, rot_manual)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot_base
    T[:3, 3] = pos_base
    return T


def _as_manual_state_from_base_T(frame_chain: FrameTransformChain, T_base_flange: np.ndarray, gripper: float) -> np.ndarray:
    pos_base = np.asarray(T_base_flange[:3, 3], dtype=np.float64).reshape(3)
    rot_base = np.asarray(T_base_flange[:3, :3], dtype=np.float64).reshape(3, 3)
    pos_manual, rot_manual = frame_chain.base_flange_to_manual_flange(pos_base, rot_base)
    state = np.zeros((10,), dtype=np.float64)
    state[:3] = pos_manual
    state[3:9] = matrix_to_rot6d(rot_manual)
    state[9] = float(np.clip(float(gripper), 0.0, 1.0))
    return state


def _resolve_episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
        raise ValueError(f"episode_index out of range: {episode_index}")
    ep_from = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    ep_to = int(dataset.meta.episodes["dataset_to_index"][episode_index])
    return ep_from, ep_to


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


class PinocchioOnlineIKSelector:
    def __init__(
        self,
        urdf_path: str,
        ee_frame_name: str,
        selection_cfg: dict[str, Any],
        solver_cfg: dict[str, Any],
        *,
        seed: int,
    ) -> None:
        try:
            import pinocchio as pin
        except ModuleNotFoundError as exc:
            raise RuntimeError("Pinocchio is required. Install package `pin`.") from exc
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(Path(urdf_path).expanduser().resolve()))
        self.data = self.model.createData()
        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        if self.ee_frame_id >= len(self.model.frames):
            raise ValueError(f"EE frame not found in URDF: {ee_frame_name}")
        self.selection_cfg = dict(selection_cfg)
        self.solver_cfg = dict(solver_cfg)
        self.nq = int(self.model.nq)
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        self.neutral = np.asarray(pin.neutral(self.model), dtype=np.float64).copy()
        self.rng = np.random.default_rng(int(seed))
        self.frame_cache: dict[str, int] = {}
        self.home_q = self._resolve_home_q()

    def _resolve_home_q(self) -> np.ndarray:
        q_deg = self.selection_cfg.get("home_q_deg")
        if isinstance(q_deg, (list, tuple, np.ndarray)):
            arr = np.asarray(q_deg, dtype=np.float64).reshape(-1)
            if arr.shape[0] >= self.nq:
                return self.clip_q(np.deg2rad(arr[: self.nq]))
        return self.neutral.copy()

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(np.asarray(q, dtype=np.float64), self.lower), self.upper)

    def fk_matrix(self, q: np.ndarray) -> np.ndarray:
        q = self.clip_q(q)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        oMf = self.data.oMf[self.ee_frame_id]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = oMf.rotation
        T[:3, 3] = oMf.translation
        return T

    def frame_translation(self, q: np.ndarray, frame_name: str) -> np.ndarray | None:
        key = str(frame_name).strip()
        if len(key) == 0:
            return None
        fid = self.frame_cache.get(key)
        if fid is None:
            fid = self.model.getFrameId(key)
            if fid >= len(self.model.frames):
                return None
            self.frame_cache[key] = int(fid)
        q = self.clip_q(q)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacement(self.model, self.data, fid)
        return np.asarray(self.data.oMf[fid].translation, dtype=np.float64).copy()

    def jacobian_sigma_min(self, q: np.ndarray) -> float:
        q = self.clip_q(q)
        J = self.pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        s = np.linalg.svd(J, compute_uv=False)
        return float(s[-1])

    def limit_margin_rad(self, q: np.ndarray) -> float:
        q = self.clip_q(q)
        margin = np.minimum(q - self.lower, self.upper - q)
        return float(np.min(margin))

    def _normalize_idx(self, idx_raw: Any) -> int | None:
        try:
            idx = int(idx_raw)
        except Exception:
            return None
        if idx < 0:
            idx += self.nq
        if idx < 0 or idx >= self.nq:
            return None
        return idx

    def _joint_center_cost(self, q: np.ndarray) -> float:
        center = 0.5 * (self.lower + self.upper)
        half_span = 0.5 * np.maximum(self.upper - self.lower, 1e-6)
        normed = (q - center) / half_span
        return float(np.linalg.norm(normed) / np.sqrt(self.nq))

    def _joint_range_prior_cost(self, q: np.ndarray) -> float:
        ranges = self.selection_cfg.get("joint_preferred_ranges_deg")
        if not isinstance(ranges, list):
            return 0.0
        q_deg = np.rad2deg(q)
        acc = 0.0
        used = 0
        for i in range(min(self.nq, len(ranges))):
            item = ranges[i]
            if item is None:
                continue
            if isinstance(item, dict):
                lo = item.get("min_deg", item.get("min"))
                hi = item.get("max_deg", item.get("max"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                lo, hi = item[0], item[1]
            else:
                continue
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except Exception:
                continue
            if hi_f < lo_f:
                lo_f, hi_f = hi_f, lo_f
            span = max(hi_f - lo_f, 1e-6)
            qi = float(q_deg[i])
            if qi < lo_f:
                v = (lo_f - qi) / span
            elif qi > hi_f:
                v = (qi - hi_f) / span
            else:
                v = 0.0
            acc += v * v
            used += 1
        if used <= 0:
            return 0.0
        return float(acc / used)

    def _elbow_sign_cost(self, q: np.ndarray) -> float:
        idx = self._normalize_idx(self.selection_cfg.get("elbow_joint_index", 2))
        if idx is None:
            return 0.0
        pref = 1.0 if float(self.selection_cfg.get("elbow_preferred_sign", 1.0)) >= 0.0 else -1.0
        deadband = max(float(self.selection_cfg.get("elbow_sign_deadband_rad", 0.0)), 0.0)
        signed = pref * float(q[idx])
        if signed >= deadband:
            return 0.0
        viol = deadband - signed
        return float((viol / np.pi) ** 2)

    def _elbow_halfspace_cost(self, q: np.ndarray) -> float:
        frame_name = str(self.selection_cfg.get("elbow_frame_name", "")).strip()
        if len(frame_name) == 0:
            return 0.0
        p_elbow = self.frame_translation(q=q, frame_name=frame_name)
        if p_elbow is None:
            return 0.0
        normal = np.asarray(self.selection_cfg.get("elbow_halfspace_normal_xyz", [0.0, 0.0, 1.0]), dtype=np.float64)
        if normal.shape != (3,):
            return 0.0
        n_norm = float(np.linalg.norm(normal))
        if n_norm < 1e-8:
            return 0.0
        normal = normal / n_norm
        offset = float(self.selection_cfg.get("elbow_halfspace_offset_m", 0.0))
        pref = 1.0 if float(self.selection_cfg.get("elbow_halfspace_preferred_sign", 1.0)) >= 0.0 else -1.0
        signed = pref * (float(np.dot(normal, p_elbow)) + offset)
        if signed >= 0.0:
            return 0.0
        viol_m = -signed
        scale_m = max(float(self.selection_cfg.get("elbow_halfspace_scale_m", 0.10)), 1e-4)
        return float((viol_m / scale_m) ** 2)

    def _wrist_flip_transition_cost(self, q_prev: np.ndarray, q_now: np.ndarray) -> float:
        idx_list = self.selection_cfg.get("wrist_joint_indices", [-2, -1])
        if not isinstance(idx_list, list):
            idx_list = [idx_list]
        step_th = max(float(self.selection_cfg.get("wrist_flip_step_threshold_rad", 0.8)), 1e-6)
        sign_eps = max(float(self.selection_cfg.get("wrist_flip_sign_epsilon_rad", 0.15)), 0.0)
        raw = 0.0
        for idx_raw in idx_list:
            idx = self._normalize_idx(idx_raw)
            if idx is None:
                continue
            a = float(q_prev[idx])
            b = float(q_now[idx])
            dq = abs(b - a)
            if dq > step_th:
                raw += (dq - step_th) / step_th
            if abs(a) > sign_eps and abs(b) > sign_eps and (a * b < 0.0):
                raw += 1.0
        return float(raw)

    def _ik_residual(self, q: np.ndarray, target_T: np.ndarray) -> np.ndarray:
        actual_T = self.fk_matrix(q)
        dp = actual_T[:3, 3] - target_T[:3, 3]
        rot_err = R.from_matrix(actual_T[:3, :3].T @ target_T[:3, :3]).as_rotvec()
        return np.concatenate([dp, rot_err], axis=0)

    def _solve_from_seed(self, target_T: np.ndarray, q_seed: np.ndarray) -> tuple[np.ndarray, float, float]:
        max_nfev = int(self.solver_cfg.get("max_nfev", 100))
        result = least_squares(
            fun=lambda x: self._ik_residual(x, target_T),
            x0=self.clip_q(q_seed),
            bounds=(self.lower, self.upper),
            method="trf",
            max_nfev=max_nfev,
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
        )
        q = self.clip_q(result.x)
        actual_T = self.fk_matrix(q)
        pos_err_m, rot_err_rad = _pose_error_components(target_T, actual_T)
        return q, pos_err_m, rot_err_rad

    def _make_seeds(self, q_prev: np.ndarray) -> list[tuple[str, np.ndarray]]:
        seeds: list[tuple[str, np.ndarray]] = [
            ("prev", q_prev.copy()),
            ("home", self.home_q.copy()),
            ("neutral", self.neutral.copy()),
        ]
        seeds.append(("prev_noise_1", self.clip_q(q_prev + self.rng.normal(scale=0.04, size=self.nq))))
        seeds.append(("prev_noise_2", self.clip_q(q_prev + self.rng.normal(scale=0.08, size=self.nq))))
        seeds.append(("home_noise_1", self.clip_q(self.home_q + self.rng.normal(scale=0.03, size=self.nq))))
        n_random = int(self.solver_cfg.get("n_random_seeds", 8))
        for i in range(max(n_random, 0)):
            seeds.append((f"rand_{i}", self.rng.uniform(self.lower, self.upper)))
        return seeds

    def select(self, target_T: np.ndarray, q_prev: np.ndarray) -> IKCandidate:
        dedup_tol = float(self.solver_cfg.get("dedup_joint_tol_rad", 0.02))
        max_keep = int(self.solver_cfg.get("max_candidates_per_frame", 12))
        pos_tol_m = float(self.solver_cfg.get("pos_tol_m", 0.008))
        rot_tol_rad = float(np.deg2rad(float(self.solver_cfg.get("rot_tol_deg", 8.0))))

        seeds = self._make_seeds(q_prev=q_prev)
        cands: list[IKCandidate] = []
        for seed_name, seed in seeds:
            q, pos_err, rot_err = self._solve_from_seed(target_T=target_T, q_seed=seed)
            is_dup = any(float(np.max(np.abs(q - ref.q))) <= dedup_tol for ref in cands)
            if is_dup:
                continue

            sigma_min = self.jacobian_sigma_min(q)
            limit_margin = self.limit_margin_rad(q)
            dq = q - q_prev
            linf = float(np.max(np.abs(dq)))
            branch_jump = linf > float(self.selection_cfg.get("branch_jump_rad", 0.6))
            wrist_flip_raw = self._wrist_flip_transition_cost(q_prev=q_prev, q_now=q)

            total = 0.0
            total += float(self.selection_cfg.get("w_local_pos", 1.0)) * pos_err
            total += float(self.selection_cfg.get("w_local_rot", 0.4)) * rot_err
            total += float(self.selection_cfg.get("w_local_limit", 0.01)) / (limit_margin + 1e-6)
            total += float(self.selection_cfg.get("w_local_sing", 0.01)) / (sigma_min + 1e-6)
            total += float(self.selection_cfg.get("w_local_center", 0.0)) * self._joint_center_cost(q)
            total += float(self.selection_cfg.get("w_home", 0.0)) * float(np.linalg.norm(q - self.home_q))
            total += float(self.selection_cfg.get("w_shape_joint_range", 0.0)) * self._joint_range_prior_cost(q)
            total += float(self.selection_cfg.get("w_elbow_sign", 0.0)) * self._elbow_sign_cost(q)
            total += float(self.selection_cfg.get("w_elbow_halfspace", 0.0)) * self._elbow_halfspace_cost(q)
            total += float(self.selection_cfg.get("w_transition_l2", self.selection_cfg.get("w_transition_smooth", 0.3))) * float(np.linalg.norm(dq))
            total += float(self.selection_cfg.get("w_transition_linf", 0.0)) * linf
            total += float(self.selection_cfg.get("w_wrist_flip", 0.0)) * wrist_flip_raw
            if branch_jump:
                total += float(self.selection_cfg.get("branch_penalty", 2.0))
            hard_max_step = float(self.selection_cfg.get("hard_max_step_rad", 0.0))
            if hard_max_step > 0.0 and linf > hard_max_step:
                excess = linf - hard_max_step
                total += float(self.selection_cfg.get("hard_step_penalty", 300.0)) * (1.0 + excess / hard_max_step)

            if pos_err > pos_tol_m or rot_err > rot_tol_rad:
                total += float(self.selection_cfg.get("failed_candidate_penalty", 20.0))

            cands.append(
                IKCandidate(
                    q=q,
                    total_cost=float(total),
                    pos_err_m=float(pos_err),
                    rot_err_rad=float(rot_err),
                    limit_margin_rad=float(limit_margin),
                    sigma_min=float(sigma_min),
                    branch_jump=bool(branch_jump),
                    wrist_flip_raw=float(wrist_flip_raw),
                    transition_linf_rad=float(linf),
                    seed_name=seed_name,
                )
            )

        if len(cands) == 0:
            raise RuntimeError("IK selection failed: no candidates generated.")
        cands.sort(key=lambda x: x.total_cost)
        return cands[: max(1, max_keep)][0]


def _rewrite_urdf_for_mujoco(urdf_path: Path) -> Path:
    urdf_path = urdf_path.expanduser().resolve()
    text = urdf_path.read_text(encoding="utf-8")
    package_root = urdf_path.parent.parent
    package_name = package_root.name
    workspace_root = package_root.parent

    tmp_dir = Path(tempfile.mkdtemp(prefix="mujoco_pseudo_"))
    tmp_urdf = tmp_dir / "model.urdf"

    import re

    pat = re.compile(r'filename="([^"]+)"')
    matches = list(pat.finditer(text))
    new_text = text
    offset = 0

    def _resolve_ref(raw_ref: str) -> Path | None:
        if raw_ref.startswith("package://"):
            rest = raw_ref[len("package://") :]
            if "/" not in rest:
                return None
            pkg, rel = rest.split("/", 1)
            cand = (package_root / rel) if pkg == package_name else (workspace_root / pkg / rel)
            return cand if cand.is_file() else None
        p = Path(raw_ref)
        if p.is_absolute():
            return p if p.is_file() else None
        cand = urdf_path.parent / p
        return cand if cand.is_file() else None

    copied: set[str] = set()
    for m in matches:
        raw_ref = m.group(1)
        src = _resolve_ref(raw_ref)
        if src is None:
            continue
        base = src.name
        dst = tmp_dir / base
        if base not in copied:
            shutil.copy2(src, dst)
            copied.add(base)
        s, e = m.span(1)
        s += offset
        e += offset
        new_text = new_text[:s] + base + new_text[e:]
        offset += len(base) - (e - s)

    replacement = package_root.as_posix().rstrip("/") + "/"
    new_text = new_text.replace(f"package://{package_name}/", replacement)
    tmp_urdf.write_text(new_text, encoding="utf-8")
    return tmp_urdf


def _render_mujoco_video(
    urdf_path: Path,
    q_traj: np.ndarray,
    out_mp4: Path,
    *,
    fps: int,
    width: int,
    height: int,
    stride: int,
) -> None:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError("MuJoCo python package missing. Install `mujoco`.") from exc
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise RuntimeError("imageio missing. Install `imageio imageio-ffmpeg`.") from exc

    tmp_urdf = _rewrite_urdf_for_mujoco(urdf_path)
    print(f"[MUJOCO] rewritten urdf: {tmp_urdf}")
    model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
    data = mujoco.MjData(model)

    off_w = int(getattr(model.vis.global_, "offwidth", 640))
    off_h = int(getattr(model.vis.global_, "offheight", 480))
    rw = min(int(width), off_w)
    rh = min(int(height), off_h)
    if rw != int(width) or rh != int(height):
        print(f"[MUJOCO] framebuffer {off_w}x{off_h}, fallback render {rw}x{rh}")

    renderer = mujoco.Renderer(model, width=rw, height=rh)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_mp4), fps=int(fps), codec="libx264", quality=8)
    try:
        nq_use = min(int(model.nq), int(q_traj.shape[1]))
        for i in range(0, int(q_traj.shape[0]), max(int(stride), 1)):
            data.qpos[:nq_use] = q_traj[i, :nq_use]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            frame = renderer.render()
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()


def _plot_trajectory(
    out_png: Path,
    target_states: np.ndarray,
    achieved_states: np.ndarray,
    q_traj_deg: np.ndarray,
    sigma_min: np.ndarray,
    limit_margin_deg: np.ndarray,
) -> None:
    if plt is None:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(int(q_traj_deg.shape[0]), dtype=np.int64)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax = axes[0]
    for i, name in enumerate(["x", "y", "z"]):
        ax.plot(steps, target_states[:, i], label=f"target_{name}", linewidth=1.4)
        ax.plot(steps, achieved_states[:, i], "--", label=f"achieved_{name}", linewidth=1.0)
    ax.set_title("Manual Frame XYZ: Target vs Achieved")
    ax.set_xlabel("step")
    ax.set_ylabel("meter")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=3)

    ax = axes[1]
    for j in range(min(int(q_traj_deg.shape[1]), 7)):
        ax.plot(steps, q_traj_deg[:, j], label=f"q{j+1}", linewidth=1.2)
    ax.set_title("Joint Trajectory (deg)")
    ax.set_xlabel("step")
    ax.set_ylabel("deg")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=4)

    ax = axes[2]
    ax.plot(steps, sigma_min, label="sigma_min", linewidth=1.3)
    ax.plot(steps, limit_margin_deg, label="limit_margin_deg", linewidth=1.3)
    ax.set_title("IK Quality")
    ax.set_xlabel("step")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pseudo inference (no real robot) with best_config-consistent mapping + IK + MuJoCo video."
    )
    parser.add_argument("--config", type=str, required=True, help="deployment_config.generated.json path")
    parser.add_argument("--urdf_path", type=str, required=True, help="RM URDF path used by Pinocchio/MuJoCo")
    parser.add_argument("--ee_frame_name", type=str, default="Link6", help="EE frame name in URDF")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default="sim_state_gt_image",
        choices=["sim_state_gt_image", "gt_observation"],
        help="Use gt image always; state uses simulated state or gt state.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/results/pseudo_mujoco",
    )
    parser.add_argument("--render_fps", type=int, default=30)
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--render_stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_json(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_dir(output_dir)
    out_npz = output_dir / "selected_trajectory.npz"
    out_summary = output_dir / "pseudo_summary.json"
    out_plot = output_dir / "trajectory_plot.png"
    out_mp4 = output_dir / "trajectory_mujoco.mp4"

    action_names = cfg["action_schema"]["names"]
    target_hz = float(cfg["control"]["target_hz"])
    target_dt = 1.0 / max(target_hz, 1e-6)
    if action_names != ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper"]:
        raise ValueError(f"Unexpected action_schema.names order: {action_names}")

    dataset = LeRobotDataset(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])
    ep_from, ep_to = _resolve_episode_bounds(dataset, int(args.episode_index))
    start_idx = ep_from + int(args.start_step)
    if start_idx >= ep_to:
        raise ValueError(f"start_step out of range: episode_len={ep_to - ep_from}, start_step={args.start_step}")
    run_steps = max(1, min(int(args.num_steps), ep_to - start_idx))

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
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    safety_cfg = SafetyConfig(
        action_bounds=cfg["safety"]["action_bounds"],
        workspace_bounds=cfg["safety"]["workspace_bounds"],
        max_xyz_speed_mps=float(cfg["safety"]["max_xyz_speed_mps"]),
        max_rot_delta_rad=float(cfg["safety"]["max_rot_delta_rad"]),
        max_gripper_delta_per_step=float(cfg["safety"]["max_gripper_delta_per_step"]),
        clip_workspace_in_action_space=bool(cfg["safety"].get("enable_policy_workspace_clip", True)),
    )
    safety_filter = ActionSafetyFilter(safety_cfg, action_names)
    safe_device = get_safe_torch_device(policy.config.device)

    robot_cfg = cfg.get("robot_adapter", {}).get("config", {})
    if not isinstance(robot_cfg, dict):
        raise ValueError("robot_adapter.config missing in config json")
    frame_chain = FrameTransformChain.from_robot_config(robot_cfg)
    ik_selection = robot_cfg.get("ik_selection", {})
    ik_solver = robot_cfg.get("ik_solver", {})
    ik_selector = PinocchioOnlineIKSelector(
        urdf_path=args.urdf_path,
        ee_frame_name=str(args.ee_frame_name),
        selection_cfg=ik_selection if isinstance(ik_selection, dict) else {},
        solver_cfg=ik_solver if isinstance(ik_solver, dict) else {},
        seed=int(args.seed),
    )

    startup_cfg = cfg.get("startup_pose", {})
    q_prev = ik_selector.home_q.copy()
    q_source = "home_q"
    if isinstance(startup_cfg, dict):
        npz_path = startup_cfg.get("selected_trajectory_npz")
        if isinstance(npz_path, str) and len(npz_path.strip()) > 0 and Path(npz_path).expanduser().is_file():
            with np.load(str(Path(npz_path).expanduser().resolve()), allow_pickle=False) as data:
                if "q_selected" in data:
                    q_sel = np.asarray(data["q_selected"], dtype=np.float64)
                    if q_sel.ndim == 2 and q_sel.shape[0] > 0 and q_sel.shape[1] >= ik_selector.nq:
                        q_index = int(startup_cfg.get("q_index", 0))
                        if q_index < 0:
                            q_index += int(q_sel.shape[0])
                        q_index = int(np.clip(q_index, 0, int(q_sel.shape[0]) - 1))
                        q_row = q_sel[q_index, : ik_selector.nq].copy()
                        q_unit = str(startup_cfg.get("q_selected_unit", "auto")).strip().lower()
                        if q_unit not in {"auto", "rad", "deg"}:
                            q_unit = "auto"
                        if q_unit == "auto":
                            q_unit = "rad" if float(np.max(np.abs(q_row))) <= 8.0 else "deg"
                        if q_unit == "deg":
                            q_row = np.deg2rad(q_row)
                        q_prev = ik_selector.clip_q(q_row)
                        q_source = f"{npz_path}:q_selected[{q_index}]/{q_unit}"

    hf = dataset.hf_dataset.with_format(None)
    row0 = hf[start_idx]
    obs0 = _row_to_observation(row0)
    sim_state = np.asarray(obs0["observation.state"], dtype=np.float64).reshape(-1)
    if sim_state.shape[0] != 10:
        raise ValueError(f"dataset observation.state must be 10D, got {sim_state.shape}")

    print(
        f"[INIT] episode={args.episode_index} start_step={args.start_step} run_steps={run_steps} "
        f"feedback_mode={args.feedback_mode}"
    )
    print(f"[INIT] checkpoint={pretrained_path}")
    print(f"[INIT] startup_q_source={q_source}")
    print(f"[INIT] output_dir={output_dir}")

    raw_actions: list[np.ndarray] = []
    safe_actions: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    achieved_states: list[np.ndarray] = []
    q_track_rad: list[np.ndarray] = []
    pos_err_track: list[float] = []
    rot_err_track: list[float] = []
    sigma_track: list[float] = []
    margin_track: list[float] = []
    branch_track: list[int] = []
    cost_track: list[float] = []

    prev_action = None
    for local_step in range(run_steps):
        global_idx = start_idx + local_step
        row = hf[global_idx]
        obs_gt = _row_to_observation(row)

        if args.feedback_mode == "gt_observation":
            obs_state = np.asarray(obs_gt["observation.state"], dtype=np.float32).copy()
        else:
            obs_state = sim_state.astype(np.float32).copy()
        obs = {
            "observation.image": np.asarray(obs_gt["observation.image"], dtype=np.uint8).copy(),
            "observation.state": obs_state,
        }

        t0 = time.perf_counter()
        action_tensor = predict_action(
            observation=obs,
            policy=policy,
            device=safe_device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task="pseudo_inference_mujoco",
            robot_type="offline_sim",
        )
        raw_dict = make_robot_action(action_tensor, dataset.features)
        raw = np.array([float(raw_dict[n]) for n in action_names], dtype=np.float64)
        safe, flags = safety_filter.apply(raw, prev_action, dt_s=target_dt)
        prev_action = safe.copy()

        T_target = _as_target_T_from_manual_state(frame_chain, safe)
        cand = ik_selector.select(target_T=T_target, q_prev=q_prev)
        q_prev = cand.q.copy()

        T_achieved = ik_selector.fk_matrix(q_prev)
        achieved_state = _as_manual_state_from_base_T(frame_chain, T_achieved, gripper=float(safe[9]))
        sim_state = achieved_state.copy()

        raw_actions.append(raw.copy())
        safe_actions.append(safe.copy())
        target_states.append(safe.copy())
        achieved_states.append(achieved_state.copy())
        q_track_rad.append(q_prev.copy())
        pos_err_track.append(cand.pos_err_m)
        rot_err_track.append(cand.rot_err_rad)
        sigma_track.append(cand.sigma_min)
        margin_track.append(np.rad2deg(cand.limit_margin_rad))
        branch_track.append(1 if cand.branch_jump else 0)
        cost_track.append(cand.total_cost)

        if local_step < 5 or local_step % 20 == 0:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"[STEP {local_step:04d}] idx={global_idx} "
                f"seed={cand.seed_name} cost={cand.total_cost:.4f} "
                f"pos_err={cand.pos_err_m:.4f} rot_err={cand.rot_err_rad:.4f} "
                f"branch={int(cand.branch_jump)} flags={'|'.join(flags) if flags else 'none'} "
                f"latency={dt_ms:.2f}ms"
            )

    raw_arr = np.stack(raw_actions, axis=0)
    safe_arr = np.stack(safe_actions, axis=0)
    target_arr = np.stack(target_states, axis=0)
    achieved_arr = np.stack(achieved_states, axis=0)
    q_arr_rad = np.stack(q_track_rad, axis=0)
    q_arr_deg = np.rad2deg(q_arr_rad)
    pos_err_arr = np.asarray(pos_err_track, dtype=np.float64)
    rot_err_arr = np.asarray(rot_err_track, dtype=np.float64)
    sigma_arr = np.asarray(sigma_track, dtype=np.float64)
    margin_arr = np.asarray(margin_track, dtype=np.float64)
    branch_arr = np.asarray(branch_track, dtype=np.int32)
    cost_arr = np.asarray(cost_track, dtype=np.float64)

    np.savez(
        str(out_npz),
        q_selected=q_arr_rad,
        q_selected_deg=q_arr_deg,
        raw_action=raw_arr,
        safe_action=safe_arr,
        target_state_manual=target_arr,
        achieved_state_manual=achieved_arr,
        pos_err_m=pos_err_arr,
        rot_err_rad=rot_err_arr,
        sigma_min=sigma_arr,
        limit_margin_deg=margin_arr,
        branch_flags=branch_arr,
        total_cost=cost_arr,
    )

    summary = {
        "config_path": str(Path(args.config).expanduser().resolve()),
        "urdf_path": str(Path(args.urdf_path).expanduser().resolve()),
        "ee_frame_name": str(args.ee_frame_name),
        "episode_index": int(args.episode_index),
        "start_step": int(args.start_step),
        "run_steps": int(run_steps),
        "feedback_mode": str(args.feedback_mode),
        "startup_q_source": q_source,
        "metrics": {
            "pos_err_mean_m": float(np.mean(pos_err_arr)),
            "pos_err_max_m": float(np.max(pos_err_arr)),
            "rot_err_mean_deg": float(np.rad2deg(np.mean(rot_err_arr))),
            "rot_err_max_deg": float(np.rad2deg(np.max(rot_err_arr))),
            "sigma_min_min": float(np.min(sigma_arr)),
            "limit_margin_deg_min": float(np.min(margin_arr)),
            "branch_count": int(np.sum(branch_arr)),
            "cost_mean": float(np.mean(cost_arr)),
        },
        "outputs": {
            "selected_trajectory_npz": str(out_npz),
            "trajectory_plot_png": str(out_plot) if plt is not None else "",
            "trajectory_mujoco_mp4": str(out_mp4),
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if plt is not None:
        _plot_trajectory(
            out_png=out_plot,
            target_states=target_arr,
            achieved_states=achieved_arr,
            q_traj_deg=q_arr_deg,
            sigma_min=sigma_arr,
            limit_margin_deg=margin_arr,
        )
        print(f"[DONE] trajectory plot: {out_plot}")
    else:
        print("[WARN] matplotlib not installed; skip trajectory plot.")

    _render_mujoco_video(
        urdf_path=Path(args.urdf_path),
        q_traj=q_arr_rad,
        out_mp4=out_mp4,
        fps=int(args.render_fps),
        width=int(args.render_width),
        height=int(args.render_height),
        stride=int(args.render_stride),
    )
    print(f"[DONE] mujoco video: {out_mp4}")
    print(f"[DONE] selected trajectory npz: {out_npz}")
    print(f"[DONE] summary json: {out_summary}")


if __name__ == "__main__":
    main()
