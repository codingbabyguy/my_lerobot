#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from frame_transforms import matrix_to_rpy_xyz, rpy_xyz_to_matrix
except ModuleNotFoundError:  # pragma: no cover
    from .frame_transforms import matrix_to_rpy_xyz, rpy_xyz_to_matrix


def _load_mapping(path: Path) -> dict:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyYAML is required to read --best-config YAML file. Please install `pyyaml`."
                ) from exc
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Expected mapping in {path}, got {type(data)}")
            return data
        data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(data)}")
        return data


def _default_observation_log_csv(action_log_csv: str | None) -> str:
    if not action_log_csv:
        return "real_test/results/realtime_observation.csv"
    action_path = Path(action_log_csv)
    return str(action_path.with_name("realtime_observation.csv"))


def _as_vec3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"{name} must be length-3, got {arr.shape}")
    return arr


def _as_rot3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3, got {arr.shape}")
    return arr


def _transform_to_xyz_rpy(entry: dict[str, Any], name: str) -> dict[str, list[float]]:
    xyz = _as_vec3(entry.get("xyz", [0.0, 0.0, 0.0]), f"{name}.xyz")
    if "rotation_matrix" in entry:
        rot = _as_rot3(entry["rotation_matrix"], f"{name}.rotation_matrix")
        rpy = matrix_to_rpy_xyz(rot)
    else:
        rpy = _as_vec3(entry.get("rpy_rad", [0.0, 0.0, 0.0]), f"{name}.rpy_rad")
    return {
        "xyz": [float(x) for x in xyz.tolist()],
        "rpy_rad": [float(x) for x in rpy.tolist()],
    }


def _normalize_frame_semantic(value: Any, name: str) -> str:
    text = str(value).strip().lower()
    if text not in {"flange", "tcp"}:
        raise ValueError(f"{name} must be 'flange' or 'tcp', got {value!r}")
    return text


def _is_identity_transform(entry: dict[str, Any]) -> bool:
    normalized = _transform_to_xyz_rpy(entry, "transform")
    xyz = _as_vec3(normalized.get("xyz", [0.0, 0.0, 0.0]), "transform.xyz")
    rpy = _as_vec3(normalized.get("rpy_rad", [0.0, 0.0, 0.0]), "transform.rpy_rad")
    return bool(np.allclose(xyz, np.zeros(3), atol=1e-9) and np.allclose(rpy, np.zeros(3), atol=1e-9))


def _ensure_frames_dict(ra_cfg: dict) -> dict:
    frames = ra_cfg.get("frames")
    if not isinstance(frames, dict):
        frames = {}
        ra_cfg["frames"] = frames
    return frames


def _promote_legacy_manual_to_frames(ra_cfg: dict) -> None:
    frames = _ensure_frames_dict(ra_cfg)
    if "T_B_from_pose_frame" in frames:
        return
    manual_origin = ra_cfg.get("manual_origin")
    manual_rotation = ra_cfg.get("manual_rotation")
    if manual_origin is None or manual_rotation is None:
        return
    t = _as_vec3(manual_origin, "robot_adapter.config.manual_origin")
    r = _as_rot3(manual_rotation, "robot_adapter.config.manual_rotation")
    frames["T_B_from_pose_frame"] = {
        "xyz": [float(x) for x in t.tolist()],
        "rpy_rad": [float(x) for x in matrix_to_rpy_xyz(r).tolist()],
    }


def _backfill_legacy_manual_from_frames(ra_cfg: dict) -> None:
    frames = _ensure_frames_dict(ra_cfg)
    t_b = frames.get("T_B_from_pose_frame")
    if not isinstance(t_b, dict):
        return
    xyz = _as_vec3(t_b.get("xyz", [0.0, 0.0, 0.0]), "frames.T_B_from_pose_frame.xyz")
    rpy = _as_vec3(t_b.get("rpy_rad", [0.0, 0.0, 0.0]), "frames.T_B_from_pose_frame.rpy_rad")
    rot = rpy_xyz_to_matrix(rpy)
    ra_cfg["manual_origin"] = [float(x) for x in xyz.tolist()]
    ra_cfg["manual_rotation"] = [[float(x) for x in row] for row in rot.tolist()]


def _merge_best_config_into_robot_adapter(ra_cfg: dict, best_cfg: dict) -> None:
    frames_src = best_cfg.get("frames", {})
    if isinstance(frames_src, dict):
        frames_dst = _ensure_frames_dict(ra_cfg)
        if isinstance(frames_src.get("T_B_from_pose_frame"), dict):
            frames_dst["T_B_from_pose_frame"] = _transform_to_xyz_rpy(
                frames_src["T_B_from_pose_frame"], "best_config.frames.T_B_from_pose_frame"
            )
        if isinstance(frames_src.get("T_flange_to_tcp"), dict):
            frames_dst["T_flange_to_tcp"] = _transform_to_xyz_rpy(
                frames_src["T_flange_to_tcp"], "best_config.frames.T_flange_to_tcp"
            )
        if isinstance(frames_src.get("T_pose_to_tcp"), dict):
            frames_dst["T_pose_to_tcp"] = _transform_to_xyz_rpy(
                frames_src["T_pose_to_tcp"], "best_config.frames.T_pose_to_tcp"
            )

    selection_src = best_cfg.get("selection", {})
    if isinstance(selection_src, dict):
        ik_selection = ra_cfg.setdefault("ik_selection", {})
        if isinstance(ik_selection, dict):
            ik_selection.update(selection_src)

    ik_src = best_cfg.get("ik", {})
    if isinstance(ik_src, dict):
        ik_solver = ra_cfg.setdefault("ik_solver", {})
        if isinstance(ik_solver, dict):
            ik_solver.update(ik_src)

    robot_src = best_cfg.get("robot", {})
    if isinstance(robot_src, dict):
        input_pose_represents = robot_src.get("input_pose_represents")
        solve_frame = robot_src.get("solve_frame")
        if input_pose_represents is not None:
            ra_cfg["input_pose_represents"] = _normalize_frame_semantic(
                input_pose_represents, "best_config.robot.input_pose_represents"
            )
        if solve_frame is not None:
            ra_cfg["solve_frame"] = _normalize_frame_semantic(solve_frame, "best_config.robot.solve_frame")


def _finalize_pose_semantics(ra_cfg: dict) -> None:
    input_pose = _normalize_frame_semantic(
        ra_cfg.get("input_pose_represents", ra_cfg.get("sdk_pose_represents", "flange")),
        "robot_adapter.config.input_pose_represents",
    )
    solve_frame = _normalize_frame_semantic(
        ra_cfg.get("solve_frame", input_pose),
        "robot_adapter.config.solve_frame",
    )
    if solve_frame != input_pose:
        raise ValueError(
            "robot_adapter.config.solve_frame must match input_pose_represents in this pipeline. "
            f"Got input_pose_represents={input_pose}, solve_frame={solve_frame}"
        )
    ra_cfg["input_pose_represents"] = input_pose
    ra_cfg["solve_frame"] = solve_frame
    # Internal adapter key kept for backward compatibility.
    ra_cfg["sdk_pose_represents"] = input_pose


def _validate_pose_to_tcp_constraint(ra_cfg: dict) -> None:
    frames = _ensure_frames_dict(ra_cfg)
    input_pose = str(ra_cfg.get("input_pose_represents", "flange")).strip().lower()
    t_pose_to_tcp = frames.get("T_pose_to_tcp")
    if t_pose_to_tcp is None:
        return
    if not isinstance(t_pose_to_tcp, dict):
        raise ValueError("robot_adapter.config.frames.T_pose_to_tcp must be a dict when provided")
    if input_pose == "flange" and not _is_identity_transform(t_pose_to_tcp):
        raise ValueError(
            "For flange-input pipeline, frames.T_pose_to_tcp must be identity. "
            "Set xyz=[0,0,0], rpy_rad=[0,0,0]."
        )


def merge_config(base_cfg: dict, template_cfg: dict, best_cfg: dict | None = None) -> dict:
    out = copy.deepcopy(base_cfg)

    out.setdefault("dataset", {})
    out["dataset"].update(template_cfg.get("dataset", {}))

    out.setdefault("action_schema", {})
    out["action_schema"].update(template_cfg.get("action_schema", {}))
    out["action_schema"]["coordinate_frame"] = "manual_relative_frame"

    out.setdefault("safety", {})
    tpl_safety = template_cfg.get("safety", {})
    if "action_bounds" in tpl_safety:
        out["safety"]["action_bounds"] = tpl_safety["action_bounds"]
    if "workspace_bounds" in tpl_safety:
        out["safety"]["workspace_bounds"] = tpl_safety["workspace_bounds"]
    out["safety"].setdefault("enable_policy_workspace_clip", True)

    out.setdefault("robot_adapter", {})
    out["robot_adapter"].setdefault("config", {})
    ra_cfg = out["robot_adapter"]["config"]
    tpl_ra_cfg = template_cfg.get("robot_adapter", {}).get("config", {})

    for key in (
        "policy_frame",
        "manual_origin",
        "manual_rotation",
        "image_shape",
        "workspace_bounds_base",
        "workspace_bounds_world",
    ):
        if key in tpl_ra_cfg:
            ra_cfg[key] = tpl_ra_cfg[key]
    ra_cfg.pop("use_sdk_pose_transform", None)

    ra_cfg["workspace_clip_in_adapter"] = bool(ra_cfg.get("workspace_clip_in_adapter", False))
    ra_cfg.setdefault("lock_work_tool_frame", True)
    ra_cfg.setdefault("frame_lock_require_expected_names", True)
    ra_cfg.setdefault("expected_work_frame_names", [])
    ra_cfg.setdefault("expected_tool_frame_names", [])
    ra_cfg.setdefault(
        "runtime_joint_guard",
        {
            "enabled": True,
            "warn_only": False,
            "joint_limit_margin_deg": 8.0,
            "max_joint_step_deg": 6.0,
            "enable_self_collision_check": False,
            "enable_singularity_check": True,
            "require_algo_checks": False,
            "fail_on_ik_error": True,
        },
    )
    ra_cfg.pop("xyz_mean", None)
    ra_cfg.pop("xyz_std", None)
    _promote_legacy_manual_to_frames(ra_cfg)
    if best_cfg is not None:
        _merge_best_config_into_robot_adapter(ra_cfg, best_cfg)
    _finalize_pose_semantics(ra_cfg)
    _validate_pose_to_tcp_constraint(ra_cfg)
    _backfill_legacy_manual_from_frames(ra_cfg)

    out.setdefault("control", {})
    out["control"].setdefault("observation_log_csv", _default_observation_log_csv(out["control"].get("action_log_csv")))

    out.setdefault("startup_pose", {})
    startup = out["startup_pose"]
    startup.setdefault("enabled", True)
    startup.setdefault("mode", "safe_positive")
    startup.setdefault("xyz", [0.07, 0.0, 0.07])
    startup.setdefault("map_startup_to_policy_origin", False)
    startup.setdefault("allow_startup_policy_anchor", False)
    startup.setdefault("keep_current_rotation", True)
    startup.setdefault("joint_speed", 6)
    startup.setdefault("max_joint_step_deg", 0.8)
    startup.setdefault("joint_tol_deg", 0.8)
    startup.setdefault("joint_limit_margin_deg", 10.0)
    startup.setdefault("wait_timeout_s", 4.0)
    startup.setdefault("max_timeout_streak", 2)
    startup.setdefault("require_algo_checks", True)
    startup.setdefault("disable_self_collision_check", False)
    startup.setdefault("disable_singularity_check", False)
    startup.setdefault("joint_poll_dt_s", 0.02)
    startup.setdefault("max_steps", 240)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deployment_config.json by merging conversion template into a base config."
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Path to reports/deployment_manual_frame_template.json from convert_session_to_lerobot_dp.py",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        required=True,
        help="Existing deployment config used as runtime baseline (network, control, estop, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output deployment config path",
    )
    parser.add_argument(
        "--best-config",
        type=str,
        default="",
        help="Optional path to offline optimization best_config.yaml/json",
    )
    args = parser.parse_args()

    template_path = Path(args.template).expanduser().resolve()
    base_config_path = Path(args.base_config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    best_cfg_path = Path(args.best_config).expanduser().resolve() if str(args.best_config).strip() else None
    print(f"[INIT] base-config={base_config_path}")
    print(f"[INIT] template={template_path}")
    if best_cfg_path is not None:
        print(f"[INIT] best-config={best_cfg_path}")

    template_cfg = _load_mapping(template_path)
    base_cfg = _load_mapping(base_config_path)
    best_cfg = _load_mapping(best_cfg_path) if best_cfg_path is not None else None
    print("[MERGE] composing deployment config...")
    merged = merge_config(base_cfg, template_cfg, best_cfg=best_cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    ra_cfg = merged.get("robot_adapter", {}).get("config", {})
    frames = ra_cfg.get("frames", {})
    frame_keys = sorted(frames.keys()) if isinstance(frames, dict) else []
    print(f"[DONE] merged deployment config written to: {output_path}")
    print(f"[DONE] frame keys={frame_keys}")
    if isinstance(ra_cfg.get("ik_selection"), dict):
        home_q = ra_cfg["ik_selection"].get("home_q_deg")
        if home_q is not None:
            print(f"[DONE] ik_selection.home_q_deg={home_q}")


if __name__ == "__main__":
    main()
