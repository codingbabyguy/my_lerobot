#!/usr/bin/env python3

from __future__ import annotations

import abc
import math
import importlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from safety import matrix_to_rot6d, rot6d_to_matrix


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
        self._rm_module_loaded = False
        self._image_shape = tuple(robot_cfg.get("image_shape", [224, 224, 3]))
        self._last_gripper = float(robot_cfg.get("initial_gripper", 1.0))
        self._last_target_pose: np.ndarray | None = None
        self._camera = None
        self._camera_source = "placeholder"
        self._cv2 = None
        self._manual_origin = _as_float_vector(robot_cfg.get("manual_origin"), 3, default=[0.0, 0.0, 0.0])
        self._manual_rotation = _as_float_matrix(
            robot_cfg.get("manual_rotation"),
            (3, 3),
            default=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        self._xyz_mean = _as_float_vector(robot_cfg.get("xyz_mean"), 3, default=[0.0, 0.0, 0.0])
        self._xyz_std = _as_float_vector(robot_cfg.get("xyz_std"), 3, default=[1.0, 1.0, 1.0])
        self._xyz_std = np.maximum(self._xyz_std, 1e-6)

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

        camera_device_path = self.robot_cfg.get("camera_device_path")
        camera_index = self.robot_cfg.get("camera_index")
        if camera_device_path is not None or camera_index is not None:
            try:
                cv2 = importlib.import_module("cv2")
                camera_api = _get_cv2_api_id(cv2, self.robot_cfg.get("camera_api", "auto"))
                cv2.setNumThreads(1)
                cap, source_desc = _open_camera(
                    cv2_module=cv2,
                    camera_api=camera_api,
                    camera_device_path=camera_device_path,
                    camera_index=camera_index,
                )

                if cap is None:
                    self._camera_source = "placeholder"
                    return

                _configure_camera_for_uvc(cap, cv2, self.robot_cfg)

                if cap.isOpened():
                    self._camera = cap
                    self._cv2 = cv2
                    self._camera_source = f"camera:{source_desc}"
                else:
                    cap.release()
                    self._camera_source = "placeholder"
            except Exception:
                self._camera_source = "placeholder"

    def disconnect(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        if not self.connected or self._robot is None:
            return
        self._robot.rm_delete_robot_arm()
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

    def _world_to_policy_pose(self, pos_world: np.ndarray, rot_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos_rel = self._manual_rotation.T @ (pos_world - self._manual_origin)
        rot_rel = self._manual_rotation.T @ rot_world
        pos_norm = (pos_rel - self._xyz_mean) / self._xyz_std
        return pos_norm, rot_rel

    def _policy_to_world_pose(self, pos_policy: np.ndarray, rot_policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos_rel = pos_policy * self._xyz_std + self._xyz_mean
        pos_world = self._manual_rotation @ pos_rel + self._manual_origin
        rot_world = self._manual_rotation @ rot_policy
        return pos_world, rot_world

    def get_observation(self) -> dict[str, np.ndarray]:
        if not self.connected or self._robot is None:
            raise RuntimeError("GermanArmAdapter is not connected")

        ret, state = self._robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed with code {ret}")

        pose = state.get("pose", [0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
        x, y, z, rx, ry, rz = [float(v) for v in pose]
        rot_m = _euler_xyz_to_matrix(rx, ry, rz)
        pos_policy, rot_policy = self._world_to_policy_pose(np.array([x, y, z], dtype=np.float64), rot_m)
        rot6d = matrix_to_rot6d(rot_policy)

        obs_state = np.zeros(10, dtype=np.float32)
        obs_state[0:3] = pos_policy.astype(np.float32)
        obs_state[3:9] = rot6d.astype(np.float32)
        obs_state[9] = np.float32(self._last_gripper)

        image = self._capture_image()
        return {
            "observation.image": image,
            "observation.state": obs_state,
        }

    def send_action(self, action: dict[str, float]) -> None:
        if not self.connected or self._robot is None:
            raise RuntimeError("GermanArmAdapter is not connected")

        rot6d = np.array([action[f"rot6d_{i}"] for i in range(6)], dtype=np.float64)
        rot_policy = rot6d_to_matrix(rot6d)
        pos_policy = np.array([action["x"], action["y"], action["z"]], dtype=np.float64)
        pos_world, rot_world = self._policy_to_world_pose(pos_policy, rot_policy)
        euler = _matrix_to_euler_xyz(rot_world)

        pose = [
            float(pos_world[0]),
            float(pos_world[1]),
            float(pos_world[2]),
            float(euler[0]),
            float(euler[1]),
            float(euler[2]),
        ]
        self._last_target_pose = np.array(pose, dtype=np.float64)

        speed = int(self.robot_cfg.get("movep_canfd_speed", 50))
        follow = bool(self.robot_cfg.get("movep_canfd_follow", False))
        trajectory_mode = int(self.robot_cfg.get("movep_canfd_trajectory_mode", 0))
        smooth_param = int(self.robot_cfg.get("movep_canfd_radio", 0))

        # The wrapper exposes pose pass-through through CANFD for low-latency control.
        ret = self._robot.rm_movep_canfd(pose, follow, trajectory_mode=trajectory_mode, radio=smooth_param)
        if ret != 0:
            # Fallback to non-blocking movej_p if pass-through temporarily fails.
            connect = int(self._trajectory_enum.RM_TRAJECTORY_DISCONNECT_E)
            ret = self._robot.rm_movej_p(pose, speed, 0, connect, 0)
            if ret != 0:
                raise RuntimeError(f"send_action failed: rm_movep_canfd={ret}")

        gripper = float(np.clip(action["gripper"], 0.0, 1.0))
        g_min = int(self.robot_cfg.get("gripper_min", 1))
        g_max = int(self.robot_cfg.get("gripper_max", 1000))
        g_pos = int(round(g_min + gripper * (g_max - g_min)))
        g_block = bool(self.robot_cfg.get("gripper_block", False))
        g_timeout = int(self.robot_cfg.get("gripper_timeout_s", 0))
        g_ret = self._robot.rm_set_gripper_position(g_pos, g_block, g_timeout)
        if g_ret == 0:
            self._last_gripper = gripper

    def wait_until_action_complete(self, timeout_s: float) -> bool:
        if not self.connected or self._robot is None:
            return False

        timeout_s = float(max(timeout_s, 0.01))
        poll_dt = float(self.robot_cfg.get("completion_poll_dt_s", 0.02))
        pose_tol = float(self.robot_cfg.get("completion_pose_tol_m", 0.005))
        rot_tol = float(self.robot_cfg.get("completion_rot_tol_rad", 0.08))
        t_end = time.perf_counter() + timeout_s

        while time.perf_counter() < t_end:
            traj = self._robot.rm_get_arm_current_trajectory()
            traj_type = int(traj.get("trajectory_type", 1))

            ret, state = self._robot.rm_get_current_arm_state()
            if ret == 0 and self._last_target_pose is not None:
                pose = state.get("pose", [0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
                cur = np.array([float(v) for v in pose], dtype=np.float64)
                pos_err = float(np.linalg.norm(cur[:3] - self._last_target_pose[:3]))
                rot_err = float(np.linalg.norm(cur[3:] - self._last_target_pose[3:]))
                if traj_type == 0 and pos_err <= pose_tol and rot_err <= rot_tol:
                    return True
            elif traj_type == 0:
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
