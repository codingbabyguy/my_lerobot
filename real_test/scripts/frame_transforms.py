#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def rpy_xyz_to_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    arr = np.asarray(rpy_rad, dtype=np.float64).reshape(3)
    rx = float(arr[0])
    ry = float(arr[1])
    rz = float(arr[2])
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def matrix_to_rpy_xyz(r: np.ndarray) -> np.ndarray:
    m = np.asarray(r, dtype=np.float64).reshape(3, 3)
    sy = float(np.sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0]))
    singular = sy < 1e-8

    if not singular:
        rx = math.atan2(m[2, 1], m[2, 2])
        ry = math.atan2(-m[2, 0], sy)
        rz = math.atan2(m[1, 0], m[0, 0])
    else:
        rx = math.atan2(-m[1, 2], m[1, 1])
        ry = math.atan2(-m[2, 0], sy)
        rz = 0.0

    return np.array([rx, ry, rz], dtype=np.float64)


def _as_vec3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"{name} must be length-3, got shape={arr.shape}")
    return arr


def _as_rot3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3, got shape={arr.shape}")
    orth = arr.T @ arr
    if not np.allclose(orth, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} must be orthonormal")
    if float(np.linalg.det(arr)) <= 0.0:
        raise ValueError(f"{name} must be right-handed (det > 0)")
    return arr


def _parse_xyz_rpy_transform(entry: dict[str, Any], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    xyz = _as_vec3(entry.get("xyz", [0.0, 0.0, 0.0]), f"{prefix}.xyz")
    if "rotation_matrix" in entry:
        rot = _as_rot3(entry["rotation_matrix"], f"{prefix}.rotation_matrix")
    else:
        rpy = _as_vec3(entry.get("rpy_rad", [0.0, 0.0, 0.0]), f"{prefix}.rpy_rad")
        rot = rpy_xyz_to_matrix(rpy)
    return xyz, rot


@dataclass(frozen=True)
class FrameTransformChain:
    """Centralized frame mapping for policy/manual, robot flange, and robot TCP.

    Frames:
    - M: policy/input pose frame (`manual_relative_frame`)
    - B: robot base frame
    - F: robot flange frame
    - T: policy/tool TCP frame
    """

    t_B_from_M: np.ndarray
    R_B_from_M: np.ndarray
    t_T_from_F: np.ndarray
    R_T_from_F: np.ndarray

    @classmethod
    def from_robot_config(cls, robot_cfg: dict[str, Any]) -> "FrameTransformChain":
        frames_cfg = robot_cfg.get("frames", {})
        if frames_cfg is None:
            frames_cfg = {}
        if not isinstance(frames_cfg, dict):
            raise ValueError("robot_adapter.config.frames must be a dict")

        if "T_B_from_pose_frame" in frames_cfg:
            t_B_from_M, R_B_from_M = _parse_xyz_rpy_transform(
                frames_cfg["T_B_from_pose_frame"],
                "frames.T_B_from_pose_frame",
            )
        else:
            manual_origin = robot_cfg.get("manual_origin")
            manual_rotation = robot_cfg.get("manual_rotation")
            if manual_origin is None or manual_rotation is None:
                raise ValueError(
                    "Either robot_adapter.config.frames.T_B_from_pose_frame or "
                    "manual_origin/manual_rotation must be provided."
                )
            t_B_from_M = _as_vec3(manual_origin, "manual_origin")
            R_B_from_M = _as_rot3(manual_rotation, "manual_rotation")

        if "T_flange_to_tcp" in frames_cfg:
            t_T_from_F, R_T_from_F = _parse_xyz_rpy_transform(
                frames_cfg["T_flange_to_tcp"],
                "frames.T_flange_to_tcp",
            )
        else:
            t_T_from_F = np.zeros(3, dtype=np.float64)
            R_T_from_F = np.eye(3, dtype=np.float64)

        return cls(
            t_B_from_M=t_B_from_M,
            R_B_from_M=R_B_from_M,
            t_T_from_F=t_T_from_F,
            R_T_from_F=R_T_from_F,
        )

    @property
    def t_F_from_T(self) -> np.ndarray:
        return -self.R_F_from_T @ self.t_T_from_F

    @property
    def R_F_from_T(self) -> np.ndarray:
        return self.R_T_from_F.T

    def manual_flange_to_base_flange(
        self, pos_manual_flange: np.ndarray, rot_manual_flange: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_m = _as_vec3(pos_manual_flange, "pos_manual_flange")
        rot_m = _as_rot3(rot_manual_flange, "rot_manual_flange")
        pos_b = self.t_B_from_M + self.R_B_from_M @ pos_m
        rot_b = self.R_B_from_M @ rot_m
        return pos_b, rot_b

    def base_flange_to_manual_flange(
        self, pos_base_flange: np.ndarray, rot_base_flange: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_b = _as_vec3(pos_base_flange, "pos_base_flange")
        rot_b = _as_rot3(rot_base_flange, "rot_base_flange")
        pos_m = self.R_B_from_M.T @ (pos_b - self.t_B_from_M)
        rot_m = self.R_B_from_M.T @ rot_b
        return pos_m, rot_m

    def base_flange_to_base_tcp(
        self, pos_base_flange: np.ndarray, rot_base_flange: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_f = _as_vec3(pos_base_flange, "pos_base_flange")
        rot_f = _as_rot3(rot_base_flange, "rot_base_flange")
        pos_t = pos_f + rot_f @ self.t_T_from_F
        rot_t = rot_f @ self.R_T_from_F
        return pos_t, rot_t

    def base_tcp_to_base_flange(
        self, pos_base_tcp: np.ndarray, rot_base_tcp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_t = _as_vec3(pos_base_tcp, "pos_base_tcp")
        rot_t = _as_rot3(rot_base_tcp, "rot_base_tcp")
        pos_f = pos_t + rot_t @ self.t_F_from_T
        rot_f = rot_t @ self.R_F_from_T
        return pos_f, rot_f

    def sdk_pose_to_base_flange(
        self, pos_base_sdk: np.ndarray, rot_base_sdk: np.ndarray, *, sdk_pose_represents: str
    ) -> tuple[np.ndarray, np.ndarray]:
        rep = str(sdk_pose_represents).strip().lower()
        if rep == "flange":
            return _as_vec3(pos_base_sdk, "pos_base_sdk"), _as_rot3(rot_base_sdk, "rot_base_sdk")
        if rep == "tcp":
            return self.base_tcp_to_base_flange(pos_base_sdk, rot_base_sdk)
        raise ValueError(f"sdk_pose_represents must be 'flange' or 'tcp', got {sdk_pose_represents!r}")

    def base_flange_to_sdk_pose(
        self, pos_base_flange: np.ndarray, rot_base_flange: np.ndarray, *, sdk_pose_represents: str
    ) -> tuple[np.ndarray, np.ndarray]:
        rep = str(sdk_pose_represents).strip().lower()
        if rep == "flange":
            return _as_vec3(pos_base_flange, "pos_base_flange"), _as_rot3(rot_base_flange, "rot_base_flange")
        if rep == "tcp":
            return self.base_flange_to_base_tcp(pos_base_flange, rot_base_flange)
        raise ValueError(f"sdk_pose_represents must be 'flange' or 'tcp', got {sdk_pose_represents!r}")

    def to_config_frames(self) -> dict[str, Any]:
        return {
            "T_B_from_pose_frame": {
                "xyz": [float(x) for x in self.t_B_from_M.tolist()],
                "rpy_rad": [float(x) for x in matrix_to_rpy_xyz(self.R_B_from_M).tolist()],
            },
            "T_flange_to_tcp": {
                "xyz": [float(x) for x in self.t_T_from_F.tolist()],
                "rpy_rad": [float(x) for x in matrix_to_rpy_xyz(self.R_T_from_F).tolist()],
            },
        }

