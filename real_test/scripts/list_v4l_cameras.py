#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


def _iter_v4l(dirname: str) -> list[Path]:
    root = Path("/dev/v4l") / dirname
    if not root.exists():
        return []

    paths: list[Path] = []
    for path in sorted(root.glob("*video*")):
        name = path.name
        if "index" in name:
            suffix = name.split("index")[-1]
            if suffix.isdigit() and int(suffix) != 0:
                continue
        paths.append(path)
    return paths


def main() -> None:
    print("[by-id]")
    by_id = _iter_v4l("by-id")
    if not by_id:
        print("  (none)")
    for path in by_id:
        print(f"  {path} -> {path.resolve()}")

    print("\n[by-path]")
    by_path = _iter_v4l("by-path")
    if not by_path:
        print("  (none)")
    for path in by_path:
        print(f"  {path} -> {path.resolve()}")


if __name__ == "__main__":
    main()
