#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one home_q robot posture in MuJoCo and save image."
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="URDF path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional deployment_config*.json. If set, read robot_adapter.config.ik_selection.home_q_deg.",
    )
    parser.add_argument(
        "--home_q_deg",
        type=str,
        default=None,
        help='Comma-separated degrees, e.g. "--home_q_deg 0,-20,70,0,50,0".',
    )
    parser.add_argument(
        "--home_q_rad",
        type=str,
        default=None,
        help='Comma-separated radians, e.g. "--home_q_rad 0,-0.35,1.22,0,0.87,0".',
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="home_q_pose.png",
        help="Output PNG path.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested render width.")
    parser.add_argument("--height", type=int, default=720, help="Requested render height.")
    parser.add_argument(
        "--distance_scale",
        type=float,
        default=1.8,
        help="Camera distance scale relative to model extent.",
    )
    return parser.parse_args()


def _parse_csv_floats(text: str) -> np.ndarray:
    parts = [x.strip() for x in str(text).split(",") if x.strip()]
    if len(parts) == 0:
        raise ValueError("Empty joint vector.")
    return np.asarray([float(x) for x in parts], dtype=np.float64)


def _load_home_q_from_config(config_path: Path) -> np.ndarray:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    robot_cfg = cfg.get("robot_adapter", {}).get("config", {})
    if not isinstance(robot_cfg, dict):
        raise ValueError("Invalid config: robot_adapter.config missing.")
    ik_sel = robot_cfg.get("ik_selection", {})
    if not isinstance(ik_sel, dict):
        raise ValueError("Invalid config: ik_selection missing.")
    home_q_deg = ik_sel.get("home_q_deg")
    if not isinstance(home_q_deg, (list, tuple)) or len(home_q_deg) == 0:
        raise ValueError("Invalid config: ik_selection.home_q_deg missing.")
    return np.asarray(home_q_deg, dtype=np.float64).reshape(-1)


def _rewrite_urdf_package_paths(urdf_path: Path) -> Path:
    urdf_path = urdf_path.expanduser().resolve()
    text = urdf_path.read_text(encoding="utf-8")

    package_root = urdf_path.parent.parent
    package_name = package_root.name
    workspace_root = package_root.parent

    tmp_dir = Path(tempfile.mkdtemp(prefix="mujoco_homeq_"))
    tmp_urdf = tmp_dir / "model.urdf"

    mesh_ref_pat = re.compile(r'filename="([^"]+)"')
    matches = list(mesh_ref_pat.finditer(text))
    new_text = text
    offset = 0

    def _resolve_source(raw_ref: str) -> Path | None:
        if raw_ref.startswith("package://"):
            rest = raw_ref[len("package://") :]
            if "/" not in rest:
                return None
            pkg, rel = rest.split("/", 1)
            if pkg == package_name:
                cand = package_root / rel
            else:
                cand = workspace_root / pkg / rel
            return cand if cand.is_file() else None
        p = Path(raw_ref)
        if p.is_absolute():
            return p if p.is_file() else None
        cand = urdf_path.parent / p
        return cand if cand.is_file() else None

    copied_names: set[str] = set()
    for m in matches:
        raw_ref = m.group(1)
        src = _resolve_source(raw_ref)
        if src is None:
            continue
        basename = src.name
        dst = tmp_dir / basename
        if basename not in copied_names:
            shutil.copy2(src, dst)
            copied_names.add(basename)
        start, end = m.span(1)
        start += offset
        end += offset
        new_text = new_text[:start] + basename + new_text[end:]
        offset += len(basename) - (end - start)

    replacement = package_root.as_posix().rstrip("/") + "/"
    new_text = new_text.replace(f"package://{package_name}/", replacement)

    def _generic_replace(match: re.Match[str]) -> str:
        pkg = match.group(1)
        cand = workspace_root / pkg
        if cand.is_dir():
            return cand.as_posix().rstrip("/") + "/"
        return match.group(0)

    new_text = re.sub(r"package://([^/]+)/", _generic_replace, new_text)
    tmp_urdf.write_text(new_text, encoding="utf-8")
    return tmp_urdf


def _render_views(mujoco, model, data, width: int, height: int, distance_scale: float) -> np.ndarray:
    renderer = mujoco.Renderer(model, width=width, height=height)
    try:
        center = np.asarray(model.stat.center, dtype=np.float64).reshape(3)
        extent = float(model.stat.extent)
        extent = max(extent, 0.1)

        views = [
            ("front", 90.0, -20.0),
            ("side", 180.0, -15.0),
            ("top", 90.0, -75.0),
        ]
        frames: list[np.ndarray] = []
        for _, az, el in views:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = center
            cam.distance = float(extent * max(distance_scale, 0.2))
            cam.azimuth = float(az)
            cam.elevation = float(el)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()

    return np.concatenate(frames, axis=1)


def main() -> None:
    args = parse_args()
    urdf_path = Path(args.urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    q_deg: np.ndarray | None = None
    q_source = ""
    if args.home_q_deg is not None:
        q_deg = _parse_csv_floats(args.home_q_deg)
        q_source = "cli_home_q_deg"
    elif args.home_q_rad is not None:
        q_rad = _parse_csv_floats(args.home_q_rad)
        q_deg = np.rad2deg(q_rad)
        q_source = "cli_home_q_rad"
    elif args.config is not None:
        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        q_deg = _load_home_q_from_config(cfg_path)
        q_source = f"config:{cfg_path}"
    else:
        raise ValueError("Provide one of --home_q_deg / --home_q_rad / --config.")

    if q_deg is None:
        raise RuntimeError("Failed to resolve home_q.")

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError("MuJoCo python package missing. Install with: pip install mujoco") from exc
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise RuntimeError("imageio missing. Install with: pip install imageio") from exc

    rewritten = _rewrite_urdf_package_paths(urdf_path)
    print(f"[INFO] rewritten URDF for MuJoCo: {rewritten}")
    model = mujoco.MjModel.from_xml_path(str(rewritten))
    data = mujoco.MjData(model)

    nq = int(model.nq)
    if q_deg.shape[0] < nq:
        print(f"[WARN] home_q length {q_deg.shape[0]} < model.nq {nq}, remaining joints set to 0.")
    q_use = np.zeros((nq,), dtype=np.float64)
    n_set = min(int(q_deg.shape[0]), nq)
    q_use[:n_set] = np.deg2rad(q_deg[:n_set])

    data.qpos[:nq] = q_use[:nq]
    mujoco.mj_forward(model, data)

    off_w = int(getattr(model.vis.global_, "offwidth", 640))
    off_h = int(getattr(model.vis.global_, "offheight", 480))
    render_w = min(int(args.width), off_w)
    render_h = min(int(args.height), off_h)
    if render_w != int(args.width) or render_h != int(args.height):
        print(
            f"[WARN] requested {args.width}x{args.height} exceeds offscreen {off_w}x{off_h}, "
            f"fallback to {render_w}x{render_h}"
        )

    panel = _render_views(
        mujoco=mujoco,
        model=model,
        data=data,
        width=render_w,
        height=render_h,
        distance_scale=float(args.distance_scale),
    )
    out_png = Path(args.output_png).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out_png), panel)

    print(f"[DONE] source={q_source}")
    print(f"[DONE] q_deg={q_deg.tolist()}")
    print(f"[DONE] saved png: {out_png}")


if __name__ == "__main__":
    main()

