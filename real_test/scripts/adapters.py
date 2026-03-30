#!/usr/bin/env python3

from __future__ import annotations

import abc
import importlib
import math
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

        camera_index = self.robot_cfg.get("camera_index")
        if camera_index is not None:
            try:
                cv2 = importlib.import_module("cv2")
                cap = cv2.VideoCapture(int(camera_index))
                if cap.isOpened():
                    self._camera = cap
                    self._cv2 = cv2
                    self._camera_source = "camera"
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
        if self._camera is None:
            return np.zeros(self._image_shape, dtype=np.uint8)

        ok, frame = self._camera.read()
        if not ok or frame is None:
            return np.zeros(self._image_shape, dtype=np.uint8)

        img_h, img_w, _ = self._image_shape
        frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        frame = self._cv2.resize(frame, (img_w, img_h), interpolation=self._cv2.INTER_LINEAR)
        return frame.astype(np.uint8)

    def get_observation(self) -> dict[str, np.ndarray]:
        if not self.connected or self._robot is None:
            raise RuntimeError("GermanArmAdapter is not connected")

        ret, state = self._robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed with code {ret}")

        pose = state.get("pose", [0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
        x, y, z, rx, ry, rz = [float(v) for v in pose]
        rot_m = _euler_xyz_to_matrix(rx, ry, rz)
        rot6d = matrix_to_rot6d(rot_m)

        obs_state = np.zeros(10, dtype=np.float32)
        obs_state[0] = np.float32(x)
        obs_state[1] = np.float32(y)
        obs_state[2] = np.float32(z)
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
        rot_m = rot6d_to_matrix(rot6d)
        euler = _matrix_to_euler_xyz(rot_m)

        pose = [
            float(action["x"]),
            float(action["y"]),
            float(action["z"]),
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
