#!/usr/bin/env python3
"""
Check whether rgb.mkv and depth.h5 frame counts match under a directory.

Usage:
  python -m tools.check_rgb_depth_counts data/captures/2025-12-28
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import h5py
except ImportError:
    h5py = None


def find_camera_dirs(root: Path, depth_name: str, rgb_name: str) -> List[Path]:
    dirs = set()
    for path in root.rglob(depth_name):
        dirs.add(path.parent)
    for path in root.rglob(rgb_name):
        dirs.add(path.parent)
    return sorted(dirs)


def read_depth_frames(path: Path) -> Tuple[Optional[int], Optional[str]]:
    if h5py is None:
        return None, "h5py not installed"
    try:
        with h5py.File(path, "r") as f:
            if "depth" not in f:
                return None, "missing /depth dataset"
            dset = f["depth"]
            if dset.ndim != 3:
                return None, f"unexpected depth shape {dset.shape}"
            return int(dset.shape[0]), None
    except Exception as exc:
        return None, f"failed to read depth.h5: {exc}"


def read_video_frames(path: Path) -> Tuple[Optional[int], Optional[str]]:
    if cv2 is None:
        return None, "cv2 not installed"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, "failed to open rgb.mkv"
    count = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        count += 1
    cap.release()
    return count, None


def check_camera_dir(cam_dir: Path, depth_name: str, rgb_name: str) -> List[str]:
    issues: List[str] = []
    depth_path = cam_dir / depth_name
    rgb_path = cam_dir / rgb_name

    if not depth_path.exists():
        issues.append(f"missing {depth_name}")
        return issues
    if not rgb_path.exists():
        issues.append(f"missing {rgb_name}")
        return issues

    depth_frames, depth_err = read_depth_frames(depth_path)
    if depth_err:
        issues.append(depth_err)
    rgb_frames, rgb_err = read_video_frames(rgb_path)
    if rgb_err:
        issues.append(rgb_err)

    if depth_frames is not None and rgb_frames is not None:
        if depth_frames != rgb_frames:
            issues.append(f"frame mismatch: depth={depth_frames} rgb={rgb_frames}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare rgb.mkv and depth.h5 frame counts")
    parser.add_argument("root", help="Root directory containing capture data")
    parser.add_argument("--depth-name", default="depth.h5", help="Depth file name")
    parser.add_argument("--rgb-name", default="rgb.mkv", help="RGB video file name")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"root not found: {root}")
        return 1
    if h5py is None:
        print("h5py not installed; cannot read depth.h5 files")
        return 1
    if cv2 is None:
        print("cv2 not installed; cannot read rgb.mkv files")
        return 1

    camera_dirs = find_camera_dirs(root, args.depth_name, args.rgb_name)
    if not camera_dirs:
        print(f"no camera dirs found under {root}")
        return 1

    any_fail = False
    for cam_dir in camera_dirs:
        issues = check_camera_dir(cam_dir, args.depth_name, args.rgb_name)
        rel = cam_dir.relative_to(root)
        if issues:
            any_fail = True
            print(f"[FAIL] {rel}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"[OK] {rel}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
