#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_observation_log_csv(action_log_csv: str | None) -> str:
    if not action_log_csv:
        return "real_test/results/realtime_observation.csv"
    action_path = Path(action_log_csv)
    return str(action_path.with_name("realtime_observation.csv"))


def merge_config(base_cfg: dict, template_cfg: dict) -> dict:
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

    out.setdefault("robot_adapter", {})
    out["robot_adapter"].setdefault("config", {})
    ra_cfg = out["robot_adapter"]["config"]
    tpl_ra_cfg = template_cfg.get("robot_adapter", {}).get("config", {})

    for key in ("policy_frame", "manual_origin", "manual_rotation", "image_shape", "use_sdk_pose_transform"):
        if key in tpl_ra_cfg:
            ra_cfg[key] = tpl_ra_cfg[key]

    ra_cfg["workspace_clip_in_adapter"] = bool(ra_cfg.get("workspace_clip_in_adapter", False))
    ra_cfg.setdefault("lock_work_tool_frame", True)
    ra_cfg.setdefault("expected_work_frame_names", [])
    ra_cfg.setdefault("expected_tool_frame_names", [])
    ra_cfg.pop("xyz_mean", None)
    ra_cfg.pop("xyz_std", None)

    out.setdefault("control", {})
    out["control"].setdefault("observation_log_csv", _default_observation_log_csv(out["control"].get("action_log_csv")))

    out.setdefault("startup_pose", {})
    startup = out["startup_pose"]
    startup.setdefault("enabled", True)
    startup.setdefault("mode", "safe_positive")
    startup.setdefault("xyz", [0.07, 0.0, 0.07])
    startup.setdefault("map_startup_to_policy_origin", True)
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
    args = parser.parse_args()

    template_path = Path(args.template).expanduser().resolve()
    base_config_path = Path(args.base_config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    template_cfg = _load_json(template_path)
    base_cfg = _load_json(base_config_path)
    merged = merge_config(base_cfg, template_cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[DONE] merged deployment config written to: {output_path}")


if __name__ == "__main__":
    main()
