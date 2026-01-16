#!/usr/bin/env python3
"""
Delete capture sessions with too few frames.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

MIN_FRAMES = 300


def sanitize_camera_id(cam_id: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in cam_id)


def find_meta_files(root: Path, max_depth: int) -> List[Path]:
    result: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if "meta.json" in filenames:
            result.append(Path(dirpath) / "meta.json")
    return sorted(result)


def list_meta_files(root: Path, find_meta: bool, max_depth: int) -> List[Path]:
    if find_meta:
        return find_meta_files(root, max_depth)
    result: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        meta = child / "meta.json"
        if meta.exists():
            result.append(meta)
    return result


def count_frames(ts_path: Path) -> int:
    if not ts_path.exists():
        return 0
    try:
        with ts_path.open("r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def session_frame_count(capture_root: Path) -> int:
    meta_path = capture_root / "meta.json"
    if not meta_path.exists():
        return 0
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return 0
    cam_ids = [str(c.get("id")) for c in meta.get("cameras", []) if c.get("id") is not None]
    if not cam_ids:
        return 0
    counts = []
    for cam_id in cam_ids:
        cam_dir = capture_root / sanitize_camera_id(cam_id)
        counts.append(count_frames(cam_dir / "timestamps.csv"))
    return max(counts) if counts else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove sessions with too few frames.")
    parser.add_argument("root", help="Root directory containing capture folders or meta.json")
    parser.add_argument("--find-meta", type=str, default="true",
                        choices=["true", "false"],
                        help="Search meta.json recursively (true) or only root/*/meta.json (false)")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    find_meta = args.find_meta.lower() == "true"
    metas = list_meta_files(root, find_meta, 2)
    if not metas:
        print("[filter] no meta.json found")
        return 0

    deleted = 0
    for meta in metas:
        capture_root = meta.parent.resolve()
        if not capture_root.is_relative_to(root):
            print(f"[filter] skip {capture_root} (outside root)")
            continue
        frames = session_frame_count(capture_root)
        if frames < MIN_FRAMES:
            print(f"[filter] deleting {capture_root} (frames={frames} < {MIN_FRAMES})")
            shutil.rmtree(capture_root)
            deleted += 1
    print(f"[filter] deleted {deleted} session(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
