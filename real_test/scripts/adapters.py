#!/usr/bin/env python3

from __future__ import annotations

import abc
import math
import importlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from frame_transforms import FrameTransformChain, matrix_to_rpy_xyz
except ModuleNotFoundError:  # pragma: no cover
    from .frame_transforms import FrameTransformChain, matrix_to_rpy_xyz
try:
    from safety import matrix_to_rot6d, rot6d_to_matrix
except ModuleNotFoundError:  # pragma: no cover
    from .safety import matrix_to_rot6d, rot6d_to_matrix


class BaseRobotAdapter(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_observation(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def send_action(self, action: dict[str, float]) -> None:
        raise NotImplementedError

    def wait_until_action_complete(self, timeout_s: float) -> bool:
        """Wait until the previously sent action finishes. Returns True on success."""
        return True

    def emergency_stop(self) -> None:
        """Best-effort immediate stop. Adapter may override if robot SDK supports it."""
        return

    def pause_motion(self) -> None:
        return

    def continue_motion(self) -> None:
        return

    def stop_motion(self) -> None:
        return

    def camera_source(self) -> str:
        return "placeholder"

    @abc.abstractmethod
    def robot_type(self) -> str:
        raise NotImplementedError


class DummyRobotAdapter(BaseRobotAdapter):
    def __init__(self, image_shape: tuple[int, int, int] = (224, 224, 3), state_dim: int = 10):
        self.image_shape = image_shape
        self.state_dim = state_dim
        self.connected = False
        self.last_action = np.zeros(state_dim, dtype=np.float32)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_observation(self) -> dict[str, np.ndarray]:
        if not self.connected:
            raise RuntimeError("DummyRobotAdapter is not connected")

        # A deterministic dry-run signal to test end-to-end inference and timing.
        t = time.time()
        state = self.last_action.copy()
        state[0] = np.sin(t) * 0.05
        state[1] = np.cos(t) * 0.05

        image = np.zeros(self.image_shape, dtype=np.uint8)
        return {
            "observation.image": image,
            "observation.state": state.astype(np.float32),
        }

    def send_action(self, action: dict[str, float]) -> None:
        if not self.connected:
            raise RuntimeError("DummyRobotAdapter is not connected")
        self.last_action = np.array(list(action.values()), dtype=np.float32)

    def wait_until_action_complete(self, timeout_s: float) -> bool:
        # Dry-run: emulate execution latency and always succeed.
        time.sleep(min(max(timeout_s, 0.0), 0.005))
        return True

    def robot_type(self) -> str:
        return "dummy"


def _matrix_to_euler_xyz(r: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to intrinsic XYZ Euler angles in radians."""
    sy = float(np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0]))
    singular = sy < 1e-8

    if not singular:
        rx = math.atan2(r[2, 1], r[2, 2])
        ry = math.atan2(-r[2, 0], sy)
        rz = math.atan2(r[1, 0], r[0, 0])
    else:
        rx = math.atan2(-r[1, 2], r[1, 1])
        ry = math.atan2(-r[2, 0], sy)
        rz = 0.0

    return np.array([rx, ry, rz], dtype=np.float64)


def _euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def _rotation_geodesic_distance(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Smallest angle (rad) between two rotation matrices."""
    rel = r_a.T @ r_b
    cos_theta = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def _safe_dof(robot: object, fallback: int = 7) -> int:
    dof = int(getattr(robot, "arm_dof", 0) or 0)
    return dof if dof > 0 else int(fallback)


def _read_joint_degree(robot: object, dof: int) -> np.ndarray:
    ret, joint = robot.rm_get_joint_degree()
    if int(ret) != 0:
        raise RuntimeError(f"rm_get_joint_degree failed with code {ret}")
    arr = np.asarray(joint, dtype=np.float64).reshape(-1)
    if arr.shape[0] < dof:
        raise RuntimeError(f"rm_get_joint_degree returned insufficient length: {arr.shape[0]} < dof {dof}")
    return arr[:dof].copy()


def _read_joint_limits(robot: object, dof: int) -> tuple[np.ndarray, np.ndarray]:
    ret_min, jmin = robot.rm_get_joint_min_pos()
    ret_max, jmax = robot.rm_get_joint_max_pos()
    if int(ret_min) != 0 or int(ret_max) != 0:
        raise RuntimeError(f"failed to read joint limits: min_ret={ret_min}, max_ret={ret_max}")
    min_arr = np.asarray(jmin, dtype=np.float64).reshape(-1)
    max_arr = np.asarray(jmax, dtype=np.float64).reshape(-1)
    if min_arr.shape[0] < dof or max_arr.shape[0] < dof:
        raise RuntimeError(
            f"joint limit length mismatch: min={min_arr.shape[0]}, max={max_arr.shape[0]}, dof={dof}"
        )
    return min_arr[:dof].copy(), max_arr[:dof].copy()


def _to_len7_joint(q_deg: np.ndarray, dof: int) -> list[float]:
    out = np.zeros(7, dtype=np.float64)
    n = min(int(dof), 7, int(q_deg.shape[0]))
    out[:n] = q_deg[:n]
    return out.tolist()


@dataclass
class OnlineIKCandidate:
    q_deg: np.ndarray
    source: str
    total_cost: float
    pos_err_m: float = 0.0
    rot_err_rad: float = 0.0
    limit_margin_deg: float = 0.0
    singular_metric: float = 1.0
    transition_l2_rad: float = 0.0
    transition_linf_rad: float = 0.0
    wrist_flip_raw: float = 0.0
    branch_jump: bool = False
    hard_step_violation: bool = False
    shape_cost: float = 0.0
    elbow_sign_cost: float = 0.0
    elbow_halfspace_cost: float = 0.0
    joint3_strict_cost: float = 0.0
    home_dist_rad: float = 0.0
    center_cost: float = 0.0
    safety_ok: bool = True
    safety_issues: list[str] = field(default_factory=list)


def _check_joint_safety(
    robot: object,
    q_deg: np.ndarray,
    q_min_soft: np.ndarray,
    q_max_soft: np.ndarray,
    dof: int,
    *,
    enable_self_collision_check: bool,
    enable_singularity_check: bool,
    require_algo_checks: bool,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    margin_low = q_deg - q_min_soft
    margin_high = q_max_soft - q_deg
    if np.any(margin_low < 0.0) or np.any(margin_high < 0.0):
        bad = np.where((margin_low < 0.0) | (margin_high < 0.0))[0]
        for idx in bad.tolist():
            issues.append(
                f"joint_limit_soft_violation:j{idx+1}:q={q_deg[idx]:.3f},"
                f"min_soft={q_min_soft[idx]:.3f},max_soft={q_max_soft[idx]:.3f}"
            )

    if enable_self_collision_check and hasattr(robot, "rm_algo_safety_robot_self_collision_detection"):
        try:
            col = int(robot.rm_algo_safety_robot_self_collision_detection(_to_len7_joint(q_deg, dof)))
            if col == 1:
                issues.append("self_collision_detected")
            elif col not in (0, 1):
                msg = f"self_collision_check_unexpected_return={col}"
                if require_algo_checks:
                    issues.append(msg)
                else:
                    print(f"[WARN] {msg}")
        except Exception as exc:
            msg = f"self_collision_check_failed={exc}"
            if require_algo_checks:
                issues.append(msg)
            else:
                print(f"[WARN] {msg}")

    if enable_singularity_check and dof == 6 and hasattr(robot, "rm_algo_kin_robot_singularity_analyse"):
        try:
            sing_ret, sing_dist = robot.rm_algo_kin_robot_singularity_analyse(q_deg[:6].tolist())
            sing_ret = int(sing_ret)
            if sing_ret != 0:
                issues.append(f"singularity_detected:code={sing_ret},dist={float(sing_dist):.6f}")
        except Exception as exc:
            msg = f"singularity_check_failed={exc}"
            if require_algo_checks:
                issues.append(msg)
            else:
                print(f"[WARN] {msg}")

    return len(issues) == 0, issues


def _as_float_vector(value: Any, length: int, default: list[float] | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"Expected a vector of length {length}")
        value = default
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"Expected vector length {length}, got {arr.shape[0]}")
    return arr


def _as_float_matrix(value: Any, shape: tuple[int, int], default: list[list[float]] | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"Expected matrix shape {shape}")
        value = default
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"Expected matrix shape {shape}, got {arr.shape}")
    return arr


def _get_cv2_api_id(cv2_module: Any, api_name: str | None) -> int:
    api_name = str(api_name or "auto").strip().lower()
    api_map = {
        "auto": getattr(cv2_module, "CAP_ANY", 0),
        "any": getattr(cv2_module, "CAP_ANY", 0),
        "opencv": getattr(cv2_module, "CAP_ANY", 0),
        "v4l2": getattr(cv2_module, "CAP_V4L2", getattr(cv2_module, "CAP_ANY", 0)),
    }
    if api_name not in api_map:
        raise ValueError(f"Unsupported camera_api: {api_name}")
    return int(api_map[api_name])


def _infer_camera_index(camera_device_path: str | None) -> int | None:
    if not camera_device_path:
        return None
    path = Path(str(camera_device_path))
    name = path.name
    if name.startswith("video") and name[5:].isdigit():
        return int(name[5:])
    try:
        resolved = path.resolve()
    except Exception:
        return None
    resolved_name = resolved.name
    if resolved_name.startswith("video") and resolved_name[5:].isdigit():
        return int(resolved_name[5:])
    return None


def _open_camera(
    cv2_module: Any,
    camera_api: int,
    camera_device_path: str | None = None,
    camera_index: int | None = None,
):
    attempts: list[tuple[Any, int, str]] = []

    if camera_device_path is not None:
        attempts.append((str(camera_device_path), camera_api, f"path:{camera_device_path}"))
        if camera_api != getattr(cv2_module, "CAP_ANY", 0):
            attempts.append((str(camera_device_path), getattr(cv2_module, "CAP_ANY", 0), f"path:{camera_device_path}:any"))

        inferred_index = _infer_camera_index(camera_device_path)
        if inferred_index is not None:
            attempts.append((int(inferred_index), camera_api, f"index:{inferred_index}"))
            if camera_api != getattr(cv2_module, "CAP_ANY", 0):
                attempts.append((int(inferred_index), getattr(cv2_module, "CAP_ANY", 0), f"index:{inferred_index}:any"))

    if camera_index is not None:
        attempts.append((int(camera_index), camera_api, f"index:{camera_index}"))
        if camera_api != getattr(cv2_module, "CAP_ANY", 0):
            attempts.append((int(camera_index), getattr(cv2_module, "CAP_ANY", 0), f"index:{camera_index}:any"))

    seen: set[tuple[str, int]] = set()
    for source, api, source_desc in attempts:
        dedupe_key = (str(source), int(api))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cap = cv2_module.VideoCapture(source, api)
        if cap.isOpened():
            return cap, source_desc
        cap.release()

    return None, "placeholder"


def _configure_camera_for_uvc(cap: Any, cv2_module: Any, camera_cfg: dict[str, Any]) -> None:
    capture_resolution = camera_cfg.get("camera_resolution")
    if capture_resolution is not None and len(capture_resolution) >= 2:
        cap.set(cv2_module.CAP_PROP_FRAME_WIDTH, int(capture_resolution[0]))
        cap.set(cv2_module.CAP_PROP_FRAME_HEIGHT, int(capture_resolution[1]))

    camera_buffer_size = camera_cfg.get("camera_buffer_size")
    if camera_buffer_size is not None:
        cap.set(cv2_module.CAP_PROP_BUFFERSIZE, int(camera_buffer_size))

    camera_fps = camera_cfg.get("camera_fps")
    if camera_fps is not None:
        cap.set(cv2_module.CAP_PROP_FPS, float(camera_fps))


def _capture_camera_frame(cap: Any, cv2_module: Any, warmup_grabs: int = 1) -> np.ndarray | None:
    # exUMI uses grab/retrieve for UVC capture cards, which is often more stable than read().
    warmup_grabs = max(int(warmup_grabs), 1)
    frame = None
    for _ in range(warmup_grabs):
        ok = cap.grab()
        if not ok:
            return None
        ok, frame = cap.retrieve(frame)
        if not ok or frame is None:
            return None
    return frame


def _apply_image_transform(
    frame_bgr: np.ndarray,
    cv2_module: Any,
    output_hw: tuple[int, int],
    rotation_deg: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    trim_left: float = 0.0,
    trim_right: float = 0.0,
    trim_top: float = 0.0,
    trim_bottom: float = 0.0,
    crop_ratio: float = 1.0,
) -> np.ndarray:
    frame = frame_bgr

    rotation_deg = int(rotation_deg) % 360
    rotate_map = {
        0: None,
        90: cv2_module.ROTATE_90_CLOCKWISE,
        180: cv2_module.ROTATE_180,
        270: cv2_module.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotation_deg not in rotate_map:
        raise ValueError("camera_rotation_deg must be one of 0/90/180/270")
    rotate_code = rotate_map[rotation_deg]
    if rotate_code is not None:
        frame = cv2_module.rotate(frame, rotate_code)

    if flip_horizontal:
        frame = cv2_module.flip(frame, 1)
    if flip_vertical:
        frame = cv2_module.flip(frame, 0)

    in_h, in_w = frame.shape[:2]
    trim_left = float(trim_left)
    trim_right = float(trim_right)
    trim_top = float(trim_top)
    trim_bottom = float(trim_bottom)
    trim_values = [trim_left, trim_right, trim_top, trim_bottom]
    if any(v < 0.0 or v >= 0.5 for v in trim_values):
        raise ValueError("camera_trim_* must be in [0.0, 0.5)")

    x0 = int(round(in_w * trim_left))
    x1 = in_w - int(round(in_w * trim_right))
    y0 = int(round(in_h * trim_top))
    y1 = in_h - int(round(in_h * trim_bottom))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("camera_trim_* removed too much of the image")
    frame = frame[y0:y1, x0:x1]

    frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)

    out_h, out_w = output_hw
    in_h, in_w = frame.shape[:2]
    if in_h <= 0 or in_w <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    crop_ratio = float(crop_ratio)
    if 0.0 < crop_ratio < 1.0:
        target_ratio = out_w / float(out_h)
        crop_h = max(1, min(int(round(in_h * crop_ratio)), in_h))
        crop_w = max(1, int(round(crop_h * target_ratio)))
        if crop_w > in_w:
            crop_w = in_w
            crop_h = max(1, min(int(round(crop_w / target_ratio)), in_h))

        x0 = max((in_w - crop_w) // 2, 0)
        y0 = max((in_h - crop_h) // 2, 0)
        frame = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
        interp = cv2_module.INTER_LINEAR if (crop_w < out_w or crop_h < out_h) else cv2_module.INTER_AREA
        frame = cv2_module.resize(frame, (out_w, out_h), interpolation=interp)
        return frame.astype(np.uint8, copy=False)

    # Match Dual-exumi's behavior: resize while keeping aspect ratio, then center crop.
    scale = max(out_w / float(in_w), out_h / float(in_h))
    resize_w = max(int(math.ceil(in_w * scale)), out_w)
    resize_h = max(int(math.ceil(in_h * scale)), out_h)
    interp = cv2_module.INTER_LINEAR if scale > 1.0 else cv2_module.INTER_AREA
    frame = cv2_module.resize(frame, (resize_w, resize_h), interpolation=interp)

    x0 = max((resize_w - out_w) // 2, 0)
    y0 = max((resize_h - out_h) // 2, 0)
    frame = frame[y0 : y0 + out_h, x0 : x0 + out_w]
    if frame.shape[0] != out_h or frame.shape[1] != out_w:
        frame = cv2_module.resize(frame, (out_w, out_h), interpolation=cv2_module.INTER_LINEAR)
    return frame.astype(np.uint8, copy=False)


class GermanArmAdapter(BaseRobotAdapter):
    """Realman (睿尔曼) adapter for the real_test pipeline."""

    def __init__(self, robot_cfg: dict):
        self.robot_cfg = robot_cfg
        self.connected = False
        self._robot: Any = None
        self._thread_mode_ctor: Any = None
        self._trajectory_enum: Any = None
        self._rm_module: Any = None
        self._rm_module_loaded = False
        self._image_shape = tuple(robot_cfg.get("image_shape", [224, 224, 3]))
        self._disable_gripper_control = bool(robot_cfg.get("disable_gripper_control", True))
        self._observation_gripper_value = float(
            robot_cfg.get(
                "observation_gripper_value",
                0.0 if self._disable_gripper_control else robot_cfg.get("initial_gripper", 0.0),
            )
        )
        self._last_gripper = float(robot_cfg.get("initial_gripper", self._observation_gripper_value))
        self._last_target_pose: np.ndarray | None = None
        self._camera = None
        self._camera_source = "placeholder"
        self._cv2 = None
        self._frame_chain = FrameTransformChain.from_robot_config(robot_cfg)
        # Backward-compatible aliases used by existing debug scripts.
        self._manual_origin_base = self._frame_chain.t_B_from_M.copy()
        self._manual_rotation_base = self._frame_chain.R_B_from_M.copy()

        pose_repr = str(
            robot_cfg.get("input_pose_represents", robot_cfg.get("sdk_pose_represents", "flange"))
        ).strip().lower()
        solve_frame = str(robot_cfg.get("solve_frame", pose_repr)).strip().lower()
        if pose_repr not in {"flange", "tcp"}:
            raise ValueError("robot_adapter.config.input_pose_represents must be 'flange' or 'tcp'")
        if solve_frame not in {"flange", "tcp"}:
            raise ValueError("robot_adapter.config.solve_frame must be 'flange' or 'tcp'")
        if solve_frame != pose_repr:
            raise ValueError(
                "robot_adapter.config.solve_frame must match input_pose_represents. "
                f"Got input_pose_represents={pose_repr}, solve_frame={solve_frame}"
            )
        self._input_pose_represents = pose_repr
        self._solve_frame = solve_frame
        self._sdk_pose_represents = pose_repr
        self._progress_log_interval = int(robot_cfg.get("progress_log_interval", 50))
        if self._progress_log_interval <= 0:
            self._progress_log_interval = 50
        self._action_counter = 0
        self._workspace_clip_in_adapter = bool(robot_cfg.get("workspace_clip_in_adapter", False))
        self._workspace_bounds_base = robot_cfg.get(
            "workspace_bounds_base",
            robot_cfg.get("workspace_bounds_world", {}),
        )
        self._lock_work_tool_frame = bool(robot_cfg.get("lock_work_tool_frame", True))
        self._frame_lock_require_expected_names = bool(robot_cfg.get("frame_lock_require_expected_names", True))
        self._expected_work_frame_names = [
            self._normalize_frame_name(x)
            for x in robot_cfg.get("expected_work_frame_names", [])
            if str(x).strip()
        ]
        self._expected_tool_frame_names = [
            self._normalize_frame_name(x)
            for x in robot_cfg.get("expected_tool_frame_names", [])
            if str(x).strip()
        ]
        self._connected_work_frame_name: str | None = None
        self._connected_tool_frame_name: str | None = None
        self._last_target_pose_base: np.ndarray | None = None
        runtime_joint_guard = robot_cfg.get("runtime_joint_guard", {})
        if not isinstance(runtime_joint_guard, dict):
            runtime_joint_guard = {}
        self._runtime_joint_guard_enabled = bool(runtime_joint_guard.get("enabled", True))
        self._runtime_joint_guard_warn_only = bool(runtime_joint_guard.get("warn_only", False))
        self._runtime_joint_limit_margin_deg = float(runtime_joint_guard.get("joint_limit_margin_deg", 8.0))
        self._runtime_joint_max_step_deg = float(runtime_joint_guard.get("max_joint_step_deg", 6.0))
        self._runtime_enable_self_collision_check = bool(
            runtime_joint_guard.get("enable_self_collision_check", False)
        )
        self._runtime_enable_singularity_check = bool(runtime_joint_guard.get("enable_singularity_check", True))
        self._runtime_require_algo_checks = bool(runtime_joint_guard.get("require_algo_checks", False))
        self._runtime_guard_fail_on_ik_error = bool(runtime_joint_guard.get("fail_on_ik_error", True))
        ik_selection = robot_cfg.get("ik_selection", {})
        if not isinstance(ik_selection, dict):
            ik_selection = {}
        ik_solver = robot_cfg.get("ik_solver", {})
        if not isinstance(ik_solver, dict):
            ik_solver = {}
        self._ik_selection_cfg = dict(ik_selection)
        self._ik_solver_cfg = dict(ik_solver)
        self._online_ik_enabled = bool(robot_cfg.get("online_ik_enabled", True))
        self._online_ik_allow_pose_fallback = bool(robot_cfg.get("online_ik_allow_pose_fallback", True))
        self._online_ik_warn_only = bool(robot_cfg.get("online_ik_warn_only", False))
        self._online_ik_log_interval = int(robot_cfg.get("online_ik_log_interval", self._progress_log_interval))
        if self._online_ik_log_interval <= 0:
            self._online_ik_log_interval = self._progress_log_interval
        self._online_ik_rng = np.random.default_rng(int(self._ik_solver_cfg.get("random_seed", 42)))
        self._last_selected_joint_deg: np.ndarray | None = None
        self._last_target_joint_deg: np.ndarray | None = None
        self._online_ik_warned_halfspace = False

        if self._workspace_clip_in_adapter and not isinstance(self._workspace_bounds_base, dict):
            raise ValueError("workspace_bounds_base must be a dict when workspace_clip_in_adapter=true")
        if self._runtime_joint_limit_margin_deg < 0.0:
            raise ValueError("runtime_joint_guard.joint_limit_margin_deg must be >= 0")
        if self._runtime_joint_max_step_deg <= 0.0:
            raise ValueError("runtime_joint_guard.max_joint_step_deg must be > 0")

    def _connect_camera(self) -> None:
        camera_stream_url = self.robot_cfg.get("camera_stream_url")
        camera_device_path = self.robot_cfg.get("camera_device_path")
        camera_index = self.robot_cfg.get("camera_index")
        if camera_stream_url:
            # Allow opening network streams (e.g. GoPro RTMP/UDP/Webcam URL) without a capture card.
            camera_device_path = str(camera_stream_url)
        if camera_device_path is None and camera_index is None:
            self._camera_source = "placeholder"
            return

        cv2 = importlib.import_module("cv2")
        camera_api = _get_cv2_api_id(cv2, self.robot_cfg.get("camera_api", "auto"))
        cv2.setNumThreads(1)

        retries = int(self.robot_cfg.get("camera_open_retries", 5))
        retry_dt = float(self.robot_cfg.get("camera_open_retry_dt_s", 0.2))
        last_error = "unknown camera init failure"

        for attempt in range(1, max(retries, 1) + 1):
            cap = None
            try:
                cap, source_desc = _open_camera(
                    cv2_module=cv2,
                    camera_api=camera_api,
                    camera_device_path=camera_device_path,
                    camera_index=camera_index,
                )
                if cap is None:
                    raise RuntimeError("OpenCV could not open any configured camera source")

                _configure_camera_for_uvc(cap, cv2, self.robot_cfg)

                frame = None
                for _ in range(10):
                    frame = _capture_camera_frame(cap, cv2)
                    if frame is not None:
                        break
                    time.sleep(0.02)
                if frame is None:
                    raise RuntimeError("Camera opened but failed to return a frame during warmup")

                self._camera = cap
                self._cv2 = cv2
                self._camera_source = f"camera:{source_desc}"
                return
            except Exception as exc:
                last_error = str(exc)
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                self._camera = None
                self._cv2 = None
                self._camera_source = "placeholder"
                if attempt < max(retries, 1):
                    time.sleep(max(retry_dt, 0.0))

        print(f"[WARN] camera initialization failed after {max(retries, 1)} attempts: {last_error}")

    def _load_rm_sdk(self) -> None:
        if self._rm_module_loaded:
            return

        sdk_src_path = self.robot_cfg.get("sdk_src_path")
        if not sdk_src_path:
            raise ValueError("robot_adapter.config.sdk_src_path is required for GermanArmAdapter")

        sdk_src = Path(sdk_src_path).expanduser().resolve()
        if not sdk_src.exists():
            raise FileNotFoundError(f"RM SDK path does not exist: {sdk_src}")

        sdk_src_str = str(sdk_src)
        if sdk_src_str not in sys.path:
            sys.path.insert(0, sdk_src_str)

        rm_module = importlib.import_module("Robotic_Arm.rm_robot_interface")
        self._rm_module = rm_module
        self._robot_ctor = rm_module.RoboticArm
        self._thread_mode_ctor = rm_module.rm_thread_mode_e
        self._trajectory_enum = rm_module.rm_trajectory_connect_config_e
        self._rm_module_loaded = True

    @staticmethod
    def _normalize_frame_name(name: Any) -> str:
        if name is None:
            return ""
        return str(name).replace("\x00", "").strip().lower()

    @staticmethod
    def _extract_frame_name(frame_obj: Any) -> str:
        if not isinstance(frame_obj, dict):
            return ""
        for key in ("name", "frame_name", "frame"):
            if key in frame_obj and frame_obj[key] is not None:
                return GermanArmAdapter._normalize_frame_name(frame_obj[key])
        return ""

    def _refresh_connected_frames(self) -> None:
        if self._robot is None:
            return

        work_name = ""
        tool_name = ""

        if hasattr(self._robot, "rm_get_current_work_frame"):
            try:
                ret_work, work = self._robot.rm_get_current_work_frame()
                if ret_work == 0:
                    work_name = self._extract_frame_name(work)
                else:
                    print(f"[WARN] rm_get_current_work_frame failed with code {ret_work}")
            except Exception as exc:
                print(f"[WARN] rm_get_current_work_frame error: {exc}")

        if hasattr(self._robot, "rm_get_current_tool_frame"):
            try:
                ret_tool, tool = self._robot.rm_get_current_tool_frame()
                if ret_tool == 0:
                    tool_name = self._extract_frame_name(tool)
                else:
                    print(f"[WARN] rm_get_current_tool_frame failed with code {ret_tool}")
            except Exception as exc:
                print(f"[WARN] rm_get_current_tool_frame error: {exc}")

        self._connected_work_frame_name = work_name or None
        self._connected_tool_frame_name = tool_name or None

    def _validate_frame_lock(self) -> None:
        if not self._lock_work_tool_frame:
            return

        if self._frame_lock_require_expected_names:
            if not self._expected_work_frame_names:
                raise RuntimeError(
                    "lock_work_tool_frame=true but expected_work_frame_names is empty. "
                    "Set robot_adapter.config.expected_work_frame_names before inference."
                )
            if not self._expected_tool_frame_names:
                raise RuntimeError(
                    "lock_work_tool_frame=true but expected_tool_frame_names is empty. "
                    "Set robot_adapter.config.expected_tool_frame_names before inference."
                )
            if self._connected_work_frame_name is None:
                raise RuntimeError("Failed to read current RM work frame name while frame lock is enabled.")
            if self._connected_tool_frame_name is None:
                raise RuntimeError("Failed to read current RM tool frame name while frame lock is enabled.")

        if self._expected_work_frame_names and self._connected_work_frame_name is not None:
            if self._connected_work_frame_name not in self._expected_work_frame_names:
                raise RuntimeError(
                    "Current RM work frame does not match expected list: "
                    f"current={self._connected_work_frame_name}, expected={self._expected_work_frame_names}"
                )
        if self._expected_tool_frame_names and self._connected_tool_frame_name is not None:
            if self._connected_tool_frame_name not in self._expected_tool_frame_names:
                raise RuntimeError(
                    "Current RM tool frame does not match expected list: "
                    f"current={self._connected_tool_frame_name}, expected={self._expected_tool_frame_names}"
                )

    def _read_current_pose_base(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._robot is None:
            raise RuntimeError("Robot handle is not initialized")
        ret, state = self._robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed with code {ret}")
        pose = np.asarray(state.get("pose", [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
        if pose.shape[0] < 6:
            raise RuntimeError(f"Invalid RM pose shape: {pose.shape}")
        pos_base = pose[:3]
        rot_base = _euler_xyz_to_matrix(float(pose[3]), float(pose[4]), float(pose[5]))
        pose6 = np.array(
            [float(pos_base[0]), float(pos_base[1]), float(pos_base[2]), float(pose[3]), float(pose[4]), float(pose[5])],
            dtype=np.float64,
        )
        return pos_base, rot_base, pose6

    def _base_to_manual_pose(self, pos_base: np.ndarray, rot_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._frame_chain.base_flange_to_manual_flange(pos_base, rot_base)

    def _manual_to_base_pose(self, pos_manual: np.ndarray, rot_manual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._frame_chain.manual_flange_to_base_flange(pos_manual, rot_manual)

    def _clip_base_position(self, pos_base: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos_base, dtype=np.float64).copy()
        if not self._workspace_clip_in_adapter:
            return pos
        bounds = self._workspace_bounds_base
        if not isinstance(bounds, dict):
            return pos
        for idx, axis in enumerate(("x", "y", "z")):
            axis_bounds = bounds.get(axis)
            if not isinstance(axis_bounds, (list, tuple)) or len(axis_bounds) != 2:
                continue
            pos[idx] = float(np.clip(pos[idx], float(axis_bounds[0]), float(axis_bounds[1])))
        return pos

    def _selection_float(self, key: str, default: float) -> float:
        value = self._ik_selection_cfg.get(key, default)
        try:
            return float(value)
        except Exception:
            return float(default)

    def _solver_float(self, key: str, default: float) -> float:
        value = self._ik_solver_cfg.get(key, default)
        try:
            return float(value)
        except Exception:
            return float(default)

    def _solver_int(self, key: str, default: int) -> int:
        value = self._ik_solver_cfg.get(key, default)
        try:
            return int(value)
        except Exception:
            return int(default)

    def _normalize_joint_index(self, idx_raw: Any, dof: int) -> int | None:
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            return None
        if idx < 0:
            idx += int(dof)
        if idx < 0 or idx >= int(dof):
            return None
        return int(idx)

    def _resolve_home_q_deg(self, dof: int, q_seed_deg: np.ndarray, q_min_deg: np.ndarray, q_max_deg: np.ndarray) -> np.ndarray:
        home = self._ik_selection_cfg.get("home_q_deg")
        if isinstance(home, (list, tuple)):
            arr = np.asarray(home, dtype=np.float64).reshape(-1)
            if arr.shape[0] >= int(dof):
                return np.clip(arr[:dof], q_min_deg[:dof], q_max_deg[:dof])
        return q_seed_deg[:dof].copy()

    @staticmethod
    def _joint_center_cost(q_deg: np.ndarray, q_min_deg: np.ndarray, q_max_deg: np.ndarray) -> float:
        center = 0.5 * (q_min_deg + q_max_deg)
        half_span = 0.5 * np.maximum(q_max_deg - q_min_deg, 1e-6)
        normed = (np.asarray(q_deg, dtype=np.float64) - center) / half_span
        return float(np.linalg.norm(normed) / np.sqrt(max(1, normed.shape[0])))

    def _joint_range_prior_cost(self, q_deg: np.ndarray, dof: int) -> tuple[float, bool]:
        ranges = self._ik_selection_cfg.get("joint_preferred_ranges_deg", None)
        if not isinstance(ranges, list):
            return 0.0, False
        q_deg = np.asarray(q_deg, dtype=np.float64)
        acc = 0.0
        used = 0
        violated = False
        for i in range(min(int(dof), len(ranges))):
            item = ranges[i]
            if item is None:
                continue
            if isinstance(item, dict):
                lo_deg = item.get("min_deg", item.get("min", None))
                hi_deg = item.get("max_deg", item.get("max", None))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                lo_deg, hi_deg = item[0], item[1]
            else:
                continue
            try:
                lo = float(lo_deg)
                hi = float(hi_deg)
            except (TypeError, ValueError):
                continue
            if hi < lo:
                lo, hi = hi, lo
            span = max(hi - lo, 1e-6)
            qi = float(q_deg[i])
            if qi < lo:
                v = (lo - qi) / span
                violated = True
            elif qi > hi:
                v = (qi - hi) / span
                violated = True
            else:
                v = 0.0
            acc += v * v
            used += 1
        if used <= 0:
            return 0.0, False
        return float(acc / used), violated

    def _elbow_sign_cost(self, q_deg: np.ndarray, dof: int) -> tuple[float, bool]:
        idx = self._normalize_joint_index(self._ik_selection_cfg.get("elbow_joint_index", 2), dof=dof)
        if idx is None:
            return 0.0, False
        pref = 1.0 if self._selection_float("elbow_preferred_sign", 1.0) >= 0.0 else -1.0
        deadband_rad = max(self._selection_float("elbow_sign_deadband_rad", 0.0), 0.0)
        signed_rad = pref * math.radians(float(np.asarray(q_deg, dtype=np.float64)[idx]))
        if signed_rad >= deadband_rad:
            return 0.0, False
        viol = deadband_rad - signed_rad
        return float((viol / np.pi) ** 2), bool(signed_rad < 0.0)

    def _elbow_halfspace_cost(self) -> tuple[float, bool]:
        # Online RM SDK path does not expose elbow-frame translation directly.
        if self._selection_float("w_elbow_halfspace", 0.0) > 0.0 and (not self._online_ik_warned_halfspace):
            print("[WARN] online IK: elbow half-space prior requested but elbow frame FK is unavailable in adapter.")
            self._online_ik_warned_halfspace = True
        return 0.0, False

    def _joint3_strict_cost(self, q_deg: np.ndarray, q_home_deg: np.ndarray, dof: int) -> tuple[float, bool]:
        idx = self._normalize_joint_index(self._ik_selection_cfg.get("joint3_index", 2), dof=dof)
        if idx is None:
            return 0.0, False
        q_deg = np.asarray(q_deg, dtype=np.float64)
        q_home_deg = np.asarray(q_home_deg, dtype=np.float64)
        q3 = float(q_deg[idx])
        h3 = float(q_home_deg[idx])

        # Tight preferred range for joint3
        strict_range = self._ik_selection_cfg.get("joint3_strict_range_deg", None)
        lo_deg = None
        hi_deg = None
        if isinstance(strict_range, dict):
            lo_deg = strict_range.get("min_deg", strict_range.get("min", None))
            hi_deg = strict_range.get("max_deg", strict_range.get("max", None))
        elif isinstance(strict_range, (list, tuple)) and len(strict_range) >= 2:
            lo_deg, hi_deg = strict_range[0], strict_range[1]
        if lo_deg is None or hi_deg is None:
            ranges = self._ik_selection_cfg.get("joint_preferred_ranges_deg", None)
            if isinstance(ranges, list) and idx < len(ranges):
                item = ranges[idx]
                if isinstance(item, dict):
                    lo_deg = item.get("min_deg", item.get("min", None))
                    hi_deg = item.get("max_deg", item.get("max", None))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    lo_deg, hi_deg = item[0], item[1]

        range_cost = 0.0
        range_bad = False
        if lo_deg is not None and hi_deg is not None:
            lo = float(lo_deg)
            hi = float(hi_deg)
            if hi < lo:
                lo, hi = hi, lo
            span = max(hi - lo, 1e-6)
            if q3 < lo:
                range_cost = ((lo - q3) / span) ** 2
                range_bad = True
            elif q3 > hi:
                range_cost = ((q3 - hi) / span) ** 2
                range_bad = True

        # Distance-to-home with deadband
        deadband = max(self._selection_float("joint3_home_deadband_deg", 0.0), 0.0)
        scale = max(self._selection_float("joint3_home_scale_deg", 20.0), 1e-6)
        err = abs(q3 - h3)
        home_cost = 0.0
        home_bad = False
        if err > deadband:
            home_cost = ((err - deadband) / scale) ** 2
            home_bad = True

        return float(range_cost + home_cost), bool(range_bad or home_bad)

    def _wrist_flip_transition_cost(self, q_prev_deg: np.ndarray, q_now_deg: np.ndarray, dof: int) -> tuple[float, bool]:
        idx_list = self._ik_selection_cfg.get("wrist_joint_indices", [-2, -1])
        if not isinstance(idx_list, list):
            idx_list = [idx_list]
        step_th_rad = max(self._selection_float("wrist_flip_step_threshold_rad", 0.8), 1e-6)
        sign_eps_rad = max(self._selection_float("wrist_flip_sign_epsilon_rad", 0.15), 0.0)
        q_prev_deg = np.asarray(q_prev_deg, dtype=np.float64)
        q_now_deg = np.asarray(q_now_deg, dtype=np.float64)
        raw = 0.0
        flagged = False
        for idx_raw in idx_list:
            idx = self._normalize_joint_index(idx_raw, dof=dof)
            if idx is None:
                continue
            a = math.radians(float(q_prev_deg[idx]))
            b = math.radians(float(q_now_deg[idx]))
            dq = abs(b - a)
            if dq > step_th_rad:
                raw += (dq - step_th_rad) / step_th_rad
                flagged = True
            if abs(a) > sign_eps_rad and abs(b) > sign_eps_rad and (a * b < 0.0):
                raw += 1.0
                flagged = True
        return float(raw), flagged

    def _evaluate_fk_pose_error(
        self,
        q_deg: np.ndarray,
        target_pos_base: np.ndarray,
        target_rot_base: np.ndarray,
    ) -> tuple[float, float]:
        if self._robot is None:
            return float("inf"), float("inf")
        if not hasattr(self._robot, "rm_algo_forward_kinematics"):
            return 0.0, 0.0
        try:
            pose_fk = self._robot.rm_algo_forward_kinematics(np.asarray(q_deg, dtype=np.float64).tolist(), 1)
            pose_fk = np.asarray(pose_fk, dtype=np.float64).reshape(-1)
            if pose_fk.shape[0] < 6:
                return float("inf"), float("inf")
            fk_pos = pose_fk[:3]
            fk_rot = _euler_xyz_to_matrix(float(pose_fk[3]), float(pose_fk[4]), float(pose_fk[5]))
            pos_err = float(np.linalg.norm(fk_pos - np.asarray(target_pos_base, dtype=np.float64)))
            rot_err = _rotation_geodesic_distance(np.asarray(target_rot_base, dtype=np.float64), fk_rot)
            return pos_err, rot_err
        except Exception:
            return float("inf"), float("inf")

    def _singularity_metric(self, q_deg: np.ndarray, dof: int) -> float:
        if self._robot is None:
            return 1.0
        if (not self._runtime_enable_singularity_check) or dof != 6:
            return 1.0
        if not hasattr(self._robot, "rm_algo_kin_robot_singularity_analyse"):
            return 1.0
        try:
            sing_ret, sing_dist = self._robot.rm_algo_kin_robot_singularity_analyse(
                np.asarray(q_deg, dtype=np.float64)[:6].tolist()
            )
            if int(sing_ret) != 0:
                return 1e-6
            return float(max(abs(float(sing_dist)), 1e-6))
        except Exception:
            return 1e-6

    def _build_ik_seeds(
        self,
        dof: int,
        q_curr_deg: np.ndarray,
        q_home_deg: np.ndarray,
        q_min_deg: np.ndarray,
        q_max_deg: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        seeds: list[tuple[str, np.ndarray]] = []
        seeds.append(("curr", q_curr_deg.copy()))
        if self._last_selected_joint_deg is not None and self._last_selected_joint_deg.shape[0] >= dof:
            seeds.append(("last", self._last_selected_joint_deg[:dof].copy()))
        seeds.append(("home", q_home_deg.copy()))

        noise_small = max(self._solver_float("seed_noise_small_deg", 2.0), 0.0)
        noise_large = max(self._solver_float("seed_noise_large_deg", 5.0), 0.0)
        seeds.append(
            ("curr_noise_small", np.clip(q_curr_deg + self._online_ik_rng.normal(0.0, noise_small, size=dof), q_min_deg, q_max_deg))
        )
        seeds.append(
            ("curr_noise_large", np.clip(q_curr_deg + self._online_ik_rng.normal(0.0, noise_large, size=dof), q_min_deg, q_max_deg))
        )
        seeds.append(
            ("home_noise_small", np.clip(q_home_deg + self._online_ik_rng.normal(0.0, noise_small, size=dof), q_min_deg, q_max_deg))
        )
        seeds.append(
            ("home_noise_large", np.clip(q_home_deg + self._online_ik_rng.normal(0.0, noise_large, size=dof), q_min_deg, q_max_deg))
        )

        n_random = max(self._solver_int("n_random_seeds", 8), 0)
        for idx in range(n_random):
            q_rand = self._online_ik_rng.uniform(q_min_deg, q_max_deg)
            seeds.append((f"uniform_{idx}", q_rand))
        return seeds

    def _score_online_ik_candidate(
        self,
        q_cand_deg: np.ndarray,
        q_ref_deg: np.ndarray,
        q_home_deg: np.ndarray,
        q_min_deg: np.ndarray,
        q_max_deg: np.ndarray,
        q_min_soft_deg: np.ndarray,
        q_max_soft_deg: np.ndarray,
        dof: int,
        target_pos_base: np.ndarray,
        target_rot_base: np.ndarray,
    ) -> OnlineIKCandidate:
        q_cand_deg = np.asarray(q_cand_deg, dtype=np.float64)[:dof].copy()
        q_ref_deg = np.asarray(q_ref_deg, dtype=np.float64)[:dof].copy()
        q_home_deg = np.asarray(q_home_deg, dtype=np.float64)[:dof].copy()
        q_min_deg = np.asarray(q_min_deg, dtype=np.float64)[:dof].copy()
        q_max_deg = np.asarray(q_max_deg, dtype=np.float64)[:dof].copy()
        q_min_soft_deg = np.asarray(q_min_soft_deg, dtype=np.float64)[:dof].copy()
        q_max_soft_deg = np.asarray(q_max_soft_deg, dtype=np.float64)[:dof].copy()

        pos_err_m, rot_err_rad = self._evaluate_fk_pose_error(q_cand_deg, target_pos_base, target_rot_base)
        margin_deg = float(np.min(np.minimum(q_cand_deg - q_min_deg, q_max_deg - q_cand_deg)))
        margin_rad = math.radians(max(margin_deg, 0.0))
        singular_metric = self._singularity_metric(q_cand_deg, dof=dof)

        dq_rad = np.deg2rad(q_cand_deg - q_ref_deg)
        transition_l2 = float(np.linalg.norm(dq_rad))
        transition_linf = float(np.max(np.abs(dq_rad)))
        branch_jump = transition_linf > self._selection_float("branch_jump_rad", 0.6)
        hard_max_step_rad = max(self._selection_float("hard_max_step_rad", 0.0), 0.0)
        hard_step_violation = bool(hard_max_step_rad > 0.0 and transition_linf > hard_max_step_rad)

        wrist_flip_raw, wrist_flip_flag = self._wrist_flip_transition_cost(q_ref_deg, q_cand_deg, dof=dof)
        shape_cost, _ = self._joint_range_prior_cost(q_cand_deg, dof=dof)
        elbow_sign_cost, _ = self._elbow_sign_cost(q_cand_deg, dof=dof)
        elbow_halfspace_cost, _ = self._elbow_halfspace_cost()
        joint3_strict_cost, _ = self._joint3_strict_cost(q_cand_deg, q_home_deg, dof=dof)
        center_cost = self._joint_center_cost(q_cand_deg, q_min_deg, q_max_deg)
        home_dist_rad = float(np.linalg.norm(np.deg2rad(q_cand_deg - q_home_deg)))

        total = 0.0
        total += self._selection_float("w_local_pos", 1.0) * pos_err_m
        total += self._selection_float("w_local_rot", 0.4) * rot_err_rad
        total += self._selection_float("w_local_limit", 0.01) / (margin_rad + 1e-6)
        total += self._selection_float("w_local_sing", 0.01) / (singular_metric + 1e-6)
        total += self._selection_float("w_local_center", 0.0) * center_cost
        total += self._selection_float("w_shape_joint_range", 0.0) * shape_cost
        total += self._selection_float("w_elbow_sign", 0.0) * elbow_sign_cost
        total += self._selection_float("w_elbow_halfspace", 0.0) * elbow_halfspace_cost
        total += self._selection_float("w_joint3_strict", 0.0) * joint3_strict_cost

        w_home = self._selection_float("w_home", 0.0)
        w_start_home = self._selection_float("w_start_home", 0.0)
        home_quad_gain = self._selection_float("home_quadratic_gain", 1.0)
        home_term = home_dist_rad + home_quad_gain * home_dist_rad * home_dist_rad
        start_home_window = max(int(self._ik_selection_cfg.get("start_home_window", 0)), 0)
        total += w_home * home_term
        if start_home_window > 0 and self._action_counter < start_home_window:
            alpha = float(start_home_window - self._action_counter) / float(start_home_window)
            total += w_start_home * alpha * home_term

        w_transition_l2 = self._selection_float("w_transition_l2", self._selection_float("w_transition_smooth", 0.3))
        w_transition_linf = self._selection_float("w_transition_linf", 0.0)
        total += w_transition_l2 * transition_l2
        total += w_transition_linf * transition_linf

        if branch_jump or wrist_flip_flag:
            total += self._selection_float("branch_penalty", 2.0)
        total += self._selection_float("w_wrist_flip", 0.0) * wrist_flip_raw

        if hard_step_violation:
            excess = transition_linf - hard_max_step_rad
            total += self._selection_float("hard_step_penalty", 200.0) * (1.0 + excess / max(hard_max_step_rad, 1e-6))

        pos_tol_m = max(self._solver_float("pos_tol_m", 0.008), 1e-6)
        rot_tol_rad = max(math.radians(self._solver_float("rot_tol_deg", 8.0)), 1e-6)
        if (pos_err_m > pos_tol_m) or (rot_err_rad > rot_tol_rad):
            total += self._selection_float("failed_candidate_penalty", 20.0)

        safety_ok, safety_issues = _check_joint_safety(
            self._robot,
            q_cand_deg,
            q_min_soft_deg,
            q_max_soft_deg,
            dof,
            enable_self_collision_check=self._runtime_enable_self_collision_check,
            enable_singularity_check=False,
            require_algo_checks=self._runtime_require_algo_checks,
        )
        if not safety_ok:
            total += self._selection_float("failed_candidate_penalty", 20.0) + 2.0 * float(len(safety_issues))

        return OnlineIKCandidate(
            q_deg=q_cand_deg,
            source="",
            total_cost=float(total),
            pos_err_m=float(pos_err_m),
            rot_err_rad=float(rot_err_rad),
            limit_margin_deg=float(margin_deg),
            singular_metric=float(singular_metric),
            transition_l2_rad=float(transition_l2),
            transition_linf_rad=float(transition_linf),
            wrist_flip_raw=float(wrist_flip_raw),
            branch_jump=bool(branch_jump),
            hard_step_violation=bool(hard_step_violation),
            shape_cost=float(shape_cost),
            elbow_sign_cost=float(elbow_sign_cost),
            elbow_halfspace_cost=float(elbow_halfspace_cost),
            joint3_strict_cost=float(joint3_strict_cost),
            home_dist_rad=float(home_dist_rad),
            center_cost=float(center_cost),
            safety_ok=bool(safety_ok),
            safety_issues=list(safety_issues),
        )

    def _select_joint_target_with_online_ik(
        self,
        pose_base_xyzrpy: list[float],
        target_pos_base: np.ndarray,
        target_rot_base: np.ndarray,
    ) -> tuple[np.ndarray, OnlineIKCandidate]:
        if self._robot is None or self._rm_module is None:
            raise RuntimeError("online IK selection requires RM robot + RM module")
        raw_joint = np.asarray(self._robot.rm_get_joint_degree()[1], dtype=np.float64).reshape(-1)
        if raw_joint.size <= 0:
            raise RuntimeError("rm_get_joint_degree returned empty list")
        dof = min(_safe_dof(self._robot, fallback=int(raw_joint.size)), int(raw_joint.size))
        q_curr_deg = _read_joint_degree(self._robot, dof)
        q_ref_deg = q_curr_deg.copy()
        if self._last_selected_joint_deg is not None and self._last_selected_joint_deg.shape[0] >= dof:
            q_ref_deg = self._last_selected_joint_deg[:dof].copy()
        q_min_deg, q_max_deg = _read_joint_limits(self._robot, dof)
        margin = float(max(self._runtime_joint_limit_margin_deg, 0.0))
        q_min_soft_deg = q_min_deg + margin
        q_max_soft_deg = q_max_deg - margin
        if np.any(q_min_soft_deg >= q_max_soft_deg):
            q_min_soft_deg = q_min_deg.copy()
            q_max_soft_deg = q_max_deg.copy()
        q_home_deg = self._resolve_home_q_deg(dof, q_curr_deg, q_min_deg, q_max_deg)

        seeds = self._build_ik_seeds(dof, q_curr_deg, q_home_deg, q_min_deg, q_max_deg)
        dedup_tol_deg = max(math.degrees(self._solver_float("dedup_joint_tol_rad", 0.02)), 1e-6)
        max_candidates = max(self._solver_int("max_candidates_per_frame", 12), 1)

        candidates: list[OnlineIKCandidate] = []
        for source, seed in seeds:
            ik_params = self._rm_module.rm_inverse_kinematics_params_t(
                q_in=_to_len7_joint(seed, dof),
                q_pose=[float(x) for x in pose_base_xyzrpy],
                flag=1,
            )
            ik_ret, q_target_raw = self._robot.rm_algo_inverse_kinematics(ik_params)
            if int(ik_ret) != 0:
                continue
            q_target_raw = np.asarray(q_target_raw, dtype=np.float64).reshape(-1)
            if q_target_raw.size < dof:
                continue
            q_cand = np.clip(q_target_raw[:dof].copy(), q_min_deg, q_max_deg)
            duplicated = any(float(np.max(np.abs(ref.q_deg - q_cand))) <= dedup_tol_deg for ref in candidates)
            if duplicated:
                continue
            cand = self._score_online_ik_candidate(
                q_cand_deg=q_cand,
                q_ref_deg=q_ref_deg,
                q_home_deg=q_home_deg,
                q_min_deg=q_min_deg,
                q_max_deg=q_max_deg,
                q_min_soft_deg=q_min_soft_deg,
                q_max_soft_deg=q_max_soft_deg,
                dof=dof,
                target_pos_base=target_pos_base,
                target_rot_base=target_rot_base,
            )
            cand.source = source
            candidates.append(cand)

        if not candidates:
            raise RuntimeError("online IK selection failed: no feasible candidate from RM IK seeds")

        candidates.sort(key=lambda x: x.total_cost)
        best = candidates[:max_candidates][0]
        self._last_selected_joint_deg = best.q_deg.copy()
        self._last_target_joint_deg = best.q_deg.copy()

        if self._action_counter <= 5 or (self._action_counter % self._online_ik_log_interval == 0):
            print(
                f"[IK {self._action_counter}] cands={len(candidates)} best={best.source} "
                f"cost={best.total_cost:.4f} pos_err={best.pos_err_m:.4f} rot_err={best.rot_err_rad:.4f} "
                f"step={best.transition_linf_rad:.4f}rad margin={best.limit_margin_deg:.2f}deg "
                f"wrist_raw={best.wrist_flip_raw:.3f} j3_strict={best.joint3_strict_cost:.3f} branch={int(best.branch_jump)} "
                f"hard_step={int(best.hard_step_violation)} safety={int(best.safety_ok)}"
            )
            if not best.safety_ok:
                print(f"[IK {self._action_counter}] safety_issues={best.safety_issues}")

        return best.q_deg.copy(), best

    def _send_joint_target_movej(self, q_target_deg: np.ndarray) -> None:
        if self._robot is None:
            raise RuntimeError("Robot handle is not initialized")
        speed = int(np.clip(int(self.robot_cfg.get("movej_speed", self.robot_cfg.get("movep_canfd_speed", 30))), 1, 100))
        blend = int(np.clip(int(self.robot_cfg.get("movej_blend", 0)), 0, 100))
        block = int(self.robot_cfg.get("movej_block", 0))
        connect = 0
        if self._trajectory_enum is not None:
            try:
                connect = int(self._trajectory_enum.RM_TRAJECTORY_DISCONNECT_E)
            except Exception:
                connect = 0
        ret = self._robot.rm_movej(np.asarray(q_target_deg, dtype=np.float64).tolist(), speed, blend, connect, block)
        if int(ret) != 0:
            raise RuntimeError(f"rm_movej failed: code={ret}")

    def _send_pose_fallback(self, pose_base_xyzrpy: list[float]) -> None:
        if self._robot is None:
            raise RuntimeError("Robot handle is not initialized")
        speed = int(self.robot_cfg.get("movep_canfd_speed", 50))
        follow = bool(self.robot_cfg.get("movep_canfd_follow", False))
        trajectory_mode = int(self.robot_cfg.get("movep_canfd_trajectory_mode", 0))
        smooth_param = int(self.robot_cfg.get("movep_canfd_radio", 0))
        ret = self._robot.rm_movep_canfd(
            pose_base_xyzrpy,
            follow,
            trajectory_mode=trajectory_mode,
            radio=smooth_param,
        )
        if ret != 0:
            connect = 0
            if self._trajectory_enum is not None:
                try:
                    connect = int(self._trajectory_enum.RM_TRAJECTORY_DISCONNECT_E)
                except Exception:
                    connect = 0
            ret = self._robot.rm_movej_p(pose_base_xyzrpy, speed, 0, connect, 0)
            if ret != 0:
                raise RuntimeError(f"send_action failed in fallback path: rm_movep_canfd={ret}")

    def _runtime_precheck_target_joint(self, q_target_deg: np.ndarray) -> None:
        if (not self._runtime_joint_guard_enabled) or self._robot is None:
            return
        issues: list[str] = []
        try:
            q_target_deg = np.asarray(q_target_deg, dtype=np.float64).reshape(-1)
            if q_target_deg.size <= 0:
                raise RuntimeError("empty target joint")
            dof = int(q_target_deg.shape[0])
            q_seed = _read_joint_degree(self._robot, dof)
            q_min, q_max = _read_joint_limits(self._robot, dof)
            margin = float(max(self._runtime_joint_limit_margin_deg, 0.0))
            q_min_soft = q_min + margin
            q_max_soft = q_max - margin
            if np.any(q_min_soft >= q_max_soft):
                q_min_soft = q_min.copy()
                q_max_soft = q_max.copy()
            ok, safety_issues = _check_joint_safety(
                self._robot,
                q_target_deg[:dof].copy(),
                q_min_soft,
                q_max_soft,
                dof,
                enable_self_collision_check=self._runtime_enable_self_collision_check,
                enable_singularity_check=self._runtime_enable_singularity_check,
                require_algo_checks=self._runtime_require_algo_checks,
            )
            if not ok:
                issues.extend(safety_issues)
            max_step = float(np.max(np.abs(q_target_deg[:dof] - q_seed[:dof])))
            if max_step > float(self._runtime_joint_max_step_deg):
                issues.append(
                    f"joint_step_too_large:max={max_step:.3f}deg,limit={self._runtime_joint_max_step_deg:.3f}deg"
                )
        except Exception as exc:
            issues.append(f"runtime_joint_guard_exception={exc}")

        if issues:
            msg = "runtime joint guard blocked action: " + "; ".join(issues)
            if self._runtime_joint_guard_warn_only:
                print(f"[WARN] {msg}")
                return
            raise RuntimeError(msg)

    def _runtime_precheck_target_pose(self, pose_base_xyzrpy: list[float]) -> None:
        if (not self._runtime_joint_guard_enabled) or self._robot is None:
            return
        if self._rm_module is None:
            if self._runtime_joint_guard_warn_only:
                print("[WARN] runtime joint guard skipped: RM module unavailable")
                return
            raise RuntimeError("runtime joint guard requires RM SDK module")

        issues: list[str] = []
        try:
            raw_joint = np.asarray(self._robot.rm_get_joint_degree()[1], dtype=np.float64).reshape(-1)
            if raw_joint.size <= 0:
                raise RuntimeError("rm_get_joint_degree returned empty list")
            dof = min(_safe_dof(self._robot, fallback=int(raw_joint.size)), int(raw_joint.size))
            q_seed = _read_joint_degree(self._robot, dof)
            q_min, q_max = _read_joint_limits(self._robot, dof)

            margin = float(max(self._runtime_joint_limit_margin_deg, 0.0))
            q_min_soft = q_min + margin
            q_max_soft = q_max - margin
            if np.any(q_min_soft >= q_max_soft):
                q_min_soft = q_min.copy()
                q_max_soft = q_max.copy()

            ik_params = self._rm_module.rm_inverse_kinematics_params_t(
                q_in=_to_len7_joint(q_seed, dof),
                q_pose=[float(x) for x in pose_base_xyzrpy],
                flag=1,
            )
            ik_ret, q_target_raw = self._robot.rm_algo_inverse_kinematics(ik_params)
            if int(ik_ret) != 0:
                msg = f"runtime_ik_failed:code={ik_ret}"
                if self._runtime_guard_fail_on_ik_error:
                    issues.append(msg)
                else:
                    print(f"[WARN] {msg}")
            else:
                q_target_raw = np.asarray(q_target_raw, dtype=np.float64).reshape(-1)
                if q_target_raw.size < dof:
                    issues.append(f"runtime_ik_output_short:len={q_target_raw.size},dof={dof}")
                else:
                    q_target = q_target_raw[:dof].copy()
                    ok, safety_issues = _check_joint_safety(
                        self._robot,
                        q_target,
                        q_min_soft,
                        q_max_soft,
                        dof,
                        enable_self_collision_check=self._runtime_enable_self_collision_check,
                        enable_singularity_check=self._runtime_enable_singularity_check,
                        require_algo_checks=self._runtime_require_algo_checks,
                    )
                    if not ok:
                        issues.extend(safety_issues)
                    max_step = float(np.max(np.abs(q_target - q_seed)))
                    if max_step > float(self._runtime_joint_max_step_deg):
                        issues.append(
                            f"joint_step_too_large:max={max_step:.3f}deg,limit={self._runtime_joint_max_step_deg:.3f}deg"
                        )
        except Exception as exc:
            issues.append(f"runtime_joint_guard_exception={exc}")

        if issues:
            msg = "runtime joint guard blocked action: " + "; ".join(issues)
            if self._runtime_joint_guard_warn_only:
                print(f"[WARN] {msg}")
                return
            raise RuntimeError(msg)

    def connect(self) -> None:
        self._load_rm_sdk()

        host = self.robot_cfg.get("host", "192.168.1.18")
        port = int(self.robot_cfg.get("port", 8080))
        level = int(self.robot_cfg.get("level", 3))
        thread_mode = int(self.robot_cfg.get("thread_mode", 2))
        run_mode = self.robot_cfg.get("run_mode")

        robot = self._robot_ctor(self._thread_mode_ctor(thread_mode))
        handle = robot.rm_create_robot_arm(host, port, level)
        if getattr(handle, "id", -1) == -1:
            raise RuntimeError(f"Failed to connect to Realman arm at {host}:{port}")

        if run_mode is not None:
            ret = robot.rm_set_arm_run_mode(int(run_mode))
            if ret != 0:
                raise RuntimeError(f"rm_set_arm_run_mode failed with code {ret}")

        timeout_ms = self.robot_cfg.get("timeout_ms")
        if timeout_ms is not None:
            robot.rm_set_timeout(int(timeout_ms))

        self._robot = robot
        self.connected = True
        try:
            self._refresh_connected_frames()
            self._validate_frame_lock()
            _, _, pose6 = self._read_current_pose_base()
            self._last_target_pose = pose6.copy()
            self._last_target_pose_base = pose6.copy()
            try:
                raw_joint = np.asarray(self._robot.rm_get_joint_degree()[1], dtype=np.float64).reshape(-1)
                if raw_joint.size > 0:
                    dof = min(_safe_dof(self._robot, fallback=int(raw_joint.size)), int(raw_joint.size))
                    self._last_selected_joint_deg = raw_joint[:dof].copy()
                    self._last_target_joint_deg = raw_joint[:dof].copy()
            except Exception:
                self._last_selected_joint_deg = None
                self._last_target_joint_deg = None
            t_b_rpy = matrix_to_rpy_xyz(self._frame_chain.R_B_from_M)
            t_f_rpy = matrix_to_rpy_xyz(self._frame_chain.R_T_from_F)
            print(
                "[FRAME] policy_pose=manual_relative_frame(flange), "
                f"input_pose_represents={self._input_pose_represents}, "
                f"solve_frame={self._solve_frame}"
            )
            print(
                "[FRAME] T_B_from_pose_frame: "
                f"xyz={self._frame_chain.t_B_from_M.round(6).tolist()} "
                f"rpy_rad={t_b_rpy.round(6).tolist()}"
            )
            print(
                "[FRAME] T_flange_to_tcp: "
                f"xyz={self._frame_chain.t_T_from_F.round(6).tolist()} "
                f"rpy_rad={t_f_rpy.round(6).tolist()}"
            )
            if self._connected_work_frame_name is not None or self._connected_tool_frame_name is not None:
                print(
                    "[INFO] RM frame lock: "
                    f"work={self._connected_work_frame_name or 'unknown'}, "
                    f"tool={self._connected_tool_frame_name or 'unknown'}"
                )
            if self._runtime_joint_guard_enabled:
                print(
                    "[INFO] runtime_joint_guard: "
                    f"warn_only={int(self._runtime_joint_guard_warn_only)}, "
                    f"margin_deg={self._runtime_joint_limit_margin_deg:.2f}, "
                    f"max_joint_step_deg={self._runtime_joint_max_step_deg:.2f}"
                )
            print(
                "[INFO] online_ik_selection: "
                f"enabled={int(self._online_ik_enabled)}, "
                f"pose_fallback={int(self._online_ik_allow_pose_fallback)}, "
                f"warn_only={int(self._online_ik_warn_only)}"
            )
            self._connect_camera()
        except Exception:
            try:
                self._robot.rm_delete_robot_arm()
            except Exception:
                pass
            self._robot = None
            self.connected = False
            raise

    def disconnect(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self._connected_work_frame_name = None
        self._connected_tool_frame_name = None
        self._last_target_pose = None
        self._last_target_pose_base = None
        self._last_selected_joint_deg = None
        self._last_target_joint_deg = None
        if not self.connected or self._robot is None:
            return
        self._robot.rm_delete_robot_arm()
        self._robot = None
        self.connected = False

    def _capture_image(self) -> np.ndarray:
        if self._camera is None or self._cv2 is None:
            return np.zeros(self._image_shape, dtype=np.uint8)

        frame = _capture_camera_frame(self._camera, self._cv2)
        if frame is None:
            return np.zeros(self._image_shape, dtype=np.uint8)

        img_h, img_w, _ = self._image_shape
        rotation_deg = int(self.robot_cfg.get("camera_rotation_deg", 0))
        flip_horizontal = bool(self.robot_cfg.get("camera_flip_horizontal", False))
        flip_vertical = bool(self.robot_cfg.get("camera_flip_vertical", False))
        trim_left = float(self.robot_cfg.get("camera_trim_left", 0.0))
        trim_right = float(self.robot_cfg.get("camera_trim_right", 0.0))
        trim_top = float(self.robot_cfg.get("camera_trim_top", 0.0))
        trim_bottom = float(self.robot_cfg.get("camera_trim_bottom", 0.0))
        crop_ratio = float(self.robot_cfg.get("camera_crop_ratio", 1.0))
        return _apply_image_transform(
            frame_bgr=frame,
            cv2_module=self._cv2,
            output_hw=(img_h, img_w),
            rotation_deg=rotation_deg,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
            trim_left=trim_left,
            trim_right=trim_right,
            trim_top=trim_top,
            trim_bottom=trim_bottom,
            crop_ratio=crop_ratio,
        )

    def get_observation(self) -> dict[str, np.ndarray]:
        if not self.connected or self._robot is None:
            raise RuntimeError("GermanArmAdapter is not connected")

        pos_base_sdk, rot_base_sdk, _ = self._read_current_pose_base()
        pos_base_flange, rot_base_flange = self._frame_chain.sdk_pose_to_base_flange(
            pos_base_sdk,
            rot_base_sdk,
            sdk_pose_represents=self._sdk_pose_represents,
        )
        pos_manual, rot_manual = self._base_to_manual_pose(pos_base_flange, rot_base_flange)
        rot6d = matrix_to_rot6d(rot_manual)

        obs_state = np.zeros(10, dtype=np.float32)
        obs_state[0:3] = pos_manual.astype(np.float32)
        obs_state[3:9] = rot6d.astype(np.float32)
        obs_state[9] = np.float32(self._observation_gripper_value if self._disable_gripper_control else self._last_gripper)

        image = self._capture_image()
        return {
            "observation.image": image,
            "observation.state": obs_state,
        }

    def send_action(self, action: dict[str, float]) -> None:
        if not self.connected or self._robot is None:
            raise RuntimeError("GermanArmAdapter is not connected")

        self._refresh_connected_frames()
        self._validate_frame_lock()

        try:
            rot6d = np.array([float(action[f"rot6d_{i}"]) for i in range(6)], dtype=np.float64)
            pos_manual = np.array([float(action["x"]), float(action["y"]), float(action["z"])], dtype=np.float64)
        except KeyError as exc:
            raise KeyError(f"Missing action key: {exc}") from exc

        rot_manual = rot6d_to_matrix(rot6d)
        pos_base_flange, rot_base_flange = self._manual_to_base_pose(pos_manual, rot_manual)
        pos_base_flange = self._clip_base_position(pos_base_flange)
        pos_base_cmd, rot_base_cmd = self._frame_chain.base_flange_to_sdk_pose(
            pos_base_flange,
            rot_base_flange,
            sdk_pose_represents=self._sdk_pose_represents,
        )
        euler = _matrix_to_euler_xyz(rot_base_cmd)

        pose = [
            float(pos_base_cmd[0]),
            float(pos_base_cmd[1]),
            float(pos_base_cmd[2]),
            float(euler[0]),
            float(euler[1]),
            float(euler[2]),
        ]
        self._action_counter += 1
        if self._action_counter <= 5 or (self._action_counter % self._progress_log_interval == 0):
            print(
                f"[ACTION {self._action_counter}] "
                f"manual_xyz={pos_manual.round(4).tolist()} "
                f"base_flange_xyz={pos_base_flange.round(4).tolist()} "
                f"cmd_{self._sdk_pose_represents}_xyz={pos_base_cmd.round(4).tolist()}"
            )
        self._last_target_pose = np.array(pose, dtype=np.float64)
        self._last_target_pose_base = self._last_target_pose.copy()
        online_exc: Exception | None = None
        if self._online_ik_enabled:
            try:
                q_target_deg, _ = self._select_joint_target_with_online_ik(
                    pose_base_xyzrpy=pose,
                    target_pos_base=pos_base_cmd,
                    target_rot_base=rot_base_cmd,
                )
                self._runtime_precheck_target_joint(q_target_deg)
                self._send_joint_target_movej(q_target_deg)
            except Exception as exc:
                online_exc = exc

        if (not self._online_ik_enabled) or (online_exc is not None):
            if online_exc is not None:
                use_fallback = bool(self._online_ik_allow_pose_fallback or self._online_ik_warn_only)
                if not use_fallback:
                    raise RuntimeError(f"online IK selection failed: {online_exc}") from online_exc
                print(f"[WARN] online IK failed, fallback to pose path: {online_exc}")
            self._runtime_precheck_target_pose(pose)
            self._send_pose_fallback(pose)

        gripper = float(action.get("gripper", self._last_gripper))
        if self._disable_gripper_control:
            self._last_gripper = self._observation_gripper_value
        elif np.isfinite(gripper):
            self._last_gripper = float(np.clip(gripper, 0.0, 1.0))

    def wait_until_action_complete(self, timeout_s: float) -> bool:
        if not self.connected or self._robot is None:
            return False

        timeout_s = float(max(timeout_s, 0.01))
        poll_dt = float(self.robot_cfg.get("completion_poll_dt_s", 0.02))
        pose_tol = float(self.robot_cfg.get("completion_pose_tol_m", 0.005))
        rot_tol = float(self.robot_cfg.get("completion_rot_tol_rad", 0.08))
        stable_polls = int(self.robot_cfg.get("completion_stable_polls", 3))
        stable_polls = max(stable_polls, 1)
        min_settle_s = float(self.robot_cfg.get("completion_min_settle_s", 0.08))
        t_end = time.perf_counter() + timeout_s
        t_start = time.perf_counter()
        target_pose = self._last_target_pose_base
        within_tol_streak = 0

        while time.perf_counter() < t_end:
            traj_type = None
            traj_valid = False
            try:
                traj = self._robot.rm_get_arm_current_trajectory()
                if isinstance(traj, dict) and "trajectory_type" in traj:
                    traj_type = int(traj["trajectory_type"])
                    traj_valid = True
            except Exception:
                traj_valid = False

            if target_pose is not None:
                try:
                    _, _, cur = self._read_current_pose_base()
                except Exception:
                    cur = None
                if cur is not None:
                    pos_err = float(np.linalg.norm(cur[:3] - target_pose[:3]))
                    cur_rot = _euler_xyz_to_matrix(float(cur[3]), float(cur[4]), float(cur[5]))
                    tar_rot = _euler_xyz_to_matrix(float(target_pose[3]), float(target_pose[4]), float(target_pose[5]))
                    rot_err = _rotation_geodesic_distance(tar_rot, cur_rot)
                else:
                    pos_err = float("inf")
                    rot_err = float("inf")

                if pos_err <= pose_tol and rot_err <= rot_tol:
                    within_tol_streak += 1
                else:
                    within_tol_streak = 0

                # Prefer robust pose-convergence check because some RM SDK builds
                # intermittently fail rm_get_arm_current_trajectory.
                if (
                    within_tol_streak >= stable_polls
                    and (time.perf_counter() - t_start) >= min_settle_s
                ):
                    return True

                if traj_valid and traj_type == 0 and pos_err <= pose_tol and rot_err <= rot_tol:
                    return True
            elif traj_valid and traj_type == 0:
                return True

            time.sleep(poll_dt)

        return False

    def camera_source(self) -> str:
        return self._camera_source

    def emergency_stop(self) -> None:
        if not self.connected or self._robot is None:
            return
        try:
            self._robot.rm_set_arm_emergency_stop(True)
        except Exception:
            # Best effort stop path.
            self._robot.rm_set_arm_stop()

    def pause_motion(self) -> None:
        if self.connected and self._robot is not None:
            self._robot.rm_set_arm_pause()

    def continue_motion(self) -> None:
        if self.connected and self._robot is not None:
            self._robot.rm_set_arm_continue()

    def stop_motion(self) -> None:
        if self.connected and self._robot is not None:
            self._robot.rm_set_arm_stop()

    def robot_type(self) -> str:
        return "realman"


def make_robot_adapter(name: str, robot_cfg: dict, dry_run: bool) -> BaseRobotAdapter:
    if dry_run or name.lower() == "dummy":
        return DummyRobotAdapter()
    if name.lower() == "german_arm":
        return GermanArmAdapter(robot_cfg)
    raise ValueError(f"Unsupported robot adapter: {name}")
