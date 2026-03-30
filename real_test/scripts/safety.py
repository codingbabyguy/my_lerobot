#!/usr/bin/env python3

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyConfig:
    action_bounds: dict[str, list[float]]
    workspace_bounds: dict[str, list[float]]
    max_xyz_speed_mps: float
    max_rot_delta_rad: float
    max_gripper_delta_per_step: float


class EStop:
    def __init__(self, enabled: bool, trigger_file: str):
        self.enabled = enabled
        self.trigger_file = trigger_file

    def triggered(self) -> bool:
        return self.enabled and os.path.exists(self.trigger_file)


def _clip(v: float, low: float, high: float) -> float:
    return float(np.clip(v, low, high))


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[:3].astype(np.float64)
    a2 = rot6d[3:6].astype(np.float64)

    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2_ortho = a2 - np.dot(a2, b1) * b1
    b2 = a2_ortho / (np.linalg.norm(a2_ortho) + 1e-12)
    b3 = np.cross(b1, b2)

    r = np.stack([b1, b2, b3], axis=1)
    return r


def matrix_to_rot6d(r: np.ndarray) -> np.ndarray:
    return np.concatenate([r[:, 0], r[:, 1]], axis=0).astype(np.float64)


def _matrix_to_rotvec(r: np.ndarray) -> np.ndarray:
    tr = float(np.trace(r))
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(cos_theta)

    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)

    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-9:
        # Near pi. Fallback to stable extraction.
        rx = math.sqrt(max((r[0, 0] + 1.0) / 2.0, 0.0))
        ry = math.sqrt(max((r[1, 1] + 1.0) / 2.0, 0.0))
        rz = math.sqrt(max((r[2, 2] + 1.0) / 2.0, 0.0))
        axis = np.array([rx, ry, rz], dtype=np.float64)
        axis /= np.linalg.norm(axis) + 1e-12
        return axis * theta

    kx = (r[2, 1] - r[1, 2]) / (2.0 * sin_theta)
    ky = (r[0, 2] - r[2, 0]) / (2.0 * sin_theta)
    kz = (r[1, 0] - r[0, 1]) / (2.0 * sin_theta)
    axis = np.array([kx, ky, kz], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    return axis * theta


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)

    k = rotvec / theta
    kx, ky, kz = k
    k_mat = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    ident = np.eye(3, dtype=np.float64)
    return ident + math.sin(theta) * k_mat + (1.0 - math.cos(theta)) * (k_mat @ k_mat)


def relative_rotation_angle(r_prev: np.ndarray, r_curr: np.ndarray) -> float:
    r_rel = r_prev.T @ r_curr
    rv = _matrix_to_rotvec(r_rel)
    return float(np.linalg.norm(rv))


class ActionSafetyFilter:
    def __init__(self, config: SafetyConfig, action_names: list[str]):
        self.cfg = config
        self.action_names = action_names
        self.name_to_idx = {name: i for i, name in enumerate(action_names)}

    def apply(
        self,
        action: np.ndarray,
        prev_action: np.ndarray | None,
        dt_s: float,
    ) -> tuple[np.ndarray, list[str]]:
        out = action.astype(np.float64).copy()
        flags: list[str] = []

        for name, bounds in self.cfg.action_bounds.items():
            if name not in self.name_to_idx:
                continue
            idx = self.name_to_idx[name]
            clipped = _clip(out[idx], bounds[0], bounds[1])
            if clipped != out[idx]:
                flags.append(f"clip:{name}")
            out[idx] = clipped

        for axis in ("x", "y", "z"):
            idx = self.name_to_idx[axis]
            low, high = self.cfg.workspace_bounds[axis]
            clipped = _clip(out[idx], low, high)
            if clipped != out[idx]:
                flags.append(f"workspace:{axis}")
            out[idx] = clipped

        rot_idx = [self.name_to_idx[f"rot6d_{i}"] for i in range(6)]
        r_curr = rot6d_to_matrix(out[rot_idx])

        if prev_action is not None:
            r_prev = rot6d_to_matrix(prev_action[rot_idx])
            angle = relative_rotation_angle(r_prev, r_curr)
            if angle > self.cfg.max_rot_delta_rad:
                r_rel = r_prev.T @ r_curr
                rv = _matrix_to_rotvec(r_rel)
                rv_norm = np.linalg.norm(rv)
                if rv_norm > 1e-12:
                    rv = rv / rv_norm * self.cfg.max_rot_delta_rad
                r_curr = r_prev @ _rotvec_to_matrix(rv)
                flags.append("rot_delta_limit")

        out[rot_idx] = matrix_to_rot6d(r_curr)

        if prev_action is not None and dt_s > 1e-6:
            xyz_idx = [self.name_to_idx["x"], self.name_to_idx["y"], self.name_to_idx["z"]]
            delta = out[xyz_idx] - prev_action[xyz_idx]
            speed = float(np.linalg.norm(delta) / dt_s)
            if speed > self.cfg.max_xyz_speed_mps:
                scale = self.cfg.max_xyz_speed_mps / (speed + 1e-12)
                out[xyz_idx] = prev_action[xyz_idx] + delta * scale
                flags.append("xyz_speed_limit")

        if prev_action is not None:
            g_idx = self.name_to_idx["gripper"]
            g_delta = out[g_idx] - prev_action[g_idx]
            max_d = self.cfg.max_gripper_delta_per_step
            if abs(g_delta) > max_d:
                out[g_idx] = prev_action[g_idx] + np.sign(g_delta) * max_d
                flags.append("gripper_delta_limit")

        return out, flags
