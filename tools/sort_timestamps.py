#!/usr/bin/env python3
"""
Sort timestamps.csv rows by frame index derived from color/depth path.

Usage:
  python3 tools/sort_timestamps.py /path/to/capture_root
  python3 tools/sort_timestamps.py /path/to/dataset_root --find-meta true
  python3 tools/sort_timestamps.py /path/to/timestamps.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone


FRAME_RE = re.compile(r"(\d+)$")


def sanitize_camera_id(cam_id: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in cam_id)

def is_realsense_id(cam_id: str) -> bool:
    return cam_id.startswith("RealSense")


def parse_frame_index(path_value: str) -> Optional[int]:
    if not path_value:
        return None
    stem = Path(path_value).stem
    match = FRAME_RE.search(stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def sort_timestamps_csv(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        print(f"[sort] missing {path}")
        return False
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not fieldnames or not rows:
        print(f"[sort] skip {path}, no rows")
        return False

    def sort_key(item):
        idx, row = item
        frame_idx = parse_frame_index(row.get("color_path", "")) or parse_frame_index(row.get("depth_path", ""))
        return (frame_idx is None, frame_idx or 0, idx)

    indexed_rows = list(enumerate(rows))
    sorted_rows = [row for _, row in sorted(indexed_rows, key=sort_key)]
    if sorted_rows == rows:
        print(f"[sort] already sorted: {path}")
        return False

    if dry_run:
        print(f"[sort] would update {path} ({len(rows)} rows)")
        return False

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_rows)
    tmp.replace(path)
    print(f"[sort] updated {path} ({len(rows)} rows)")
    return True


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


def timestamps_from_capture_root(capture_root: Path) -> List[Path]:
    meta_path = capture_root / "meta.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    cam_ids = [
        str(c["id"]) for c in meta.get("cameras", []) if "id" in c and is_realsense_id(str(c["id"]))
    ]
    paths: List[Path] = []
    for cid in cam_ids:
        cam_dir = capture_root / sanitize_camera_id(cid)
        ts_path = cam_dir / "timestamps.csv"
        if ts_path.exists():
            paths.append(ts_path)
            continue
        alt_dir = capture_root / str(cid).replace("#", "_")
        alt_path = alt_dir / "timestamps.csv"
        if alt_path.exists():
            paths.append(alt_path)
    return paths


def should_skip_step(capture_root: Path, step: str) -> bool:
    marker_path = capture_root / "postprocess_markers.json"
    if not marker_path.exists():
        return False
    try:
        payload = json.loads(marker_path.read_text())
    except json.JSONDecodeError:
        return False
    steps = payload.get("steps")
    if not isinstance(steps, dict):
        return False
    return step in steps


def update_marker(capture_root: Path, step: str, info: Dict) -> None:
    marker_path = capture_root / "postprocess_markers.json"
    payload = {}
    if marker_path.exists():
        try:
            payload = json.loads(marker_path.read_text())
        except json.JSONDecodeError:
            payload = {}
    steps = payload.get("steps")
    if not isinstance(steps, dict):
        steps = {}
    done_at = datetime.now(timezone.utc).isoformat()
    entry = dict(info)
    entry["done_at"] = done_at
    steps[step] = entry
    payload["steps"] = steps
    payload["updated_at"] = done_at
    marker_path.write_text(json.dumps(payload, indent=2))
    print(f"[sort] updated marker {marker_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort timestamps.csv by frame index.")
    parser.add_argument("root", help="timestamps.csv path, capture root, or dataset root")
    parser.add_argument("--find-meta", type=str, default="true",
                        choices=["true", "false"],
                        help="Search meta.json recursively (true) or only root/*/meta.json (false)")
    parser.add_argument("--dry-run", action="store_true", help="Only report files that would be updated")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    find_meta = args.find_meta.lower() == "true"

    if root.is_file():
        if root.name != "timestamps.csv":
            print(f"[sort] skip {root}, not a timestamps.csv")
            return
        sort_timestamps_csv(root, args.dry_run)
        return

    if (root / "meta.json").exists():
        if should_skip_step(root, "sort_timestamps"):
            print(f"[sort] skip {root}, already sorted")
            return
        ts_paths = timestamps_from_capture_root(root)
        updated = False
        for ts_path in ts_paths:
            updated = sort_timestamps_csv(ts_path, args.dry_run) or updated
        if ts_paths and not args.dry_run:
            update_marker(
                root,
                "sort_timestamps",
                {
                    "files": [p.name for p in ts_paths],
                    "updated": updated,
                },
            )
        return

    metas = list_meta_files(root, find_meta, 2)
    if not metas:
        print("[sort] no meta.json found")
        return
    for meta in metas:
        capture_root = meta.parent
        if should_skip_step(capture_root, "sort_timestamps"):
            print(f"[sort] skip {capture_root}, already sorted")
            continue
        ts_paths = timestamps_from_capture_root(capture_root)
        updated = False
        for ts_path in ts_paths:
            updated = sort_timestamps_csv(ts_path, args.dry_run) or updated
        if ts_paths and not args.dry_run:
            update_marker(
                capture_root,
                "sort_timestamps",
                {
                    "files": [p.name for p in ts_paths],
                    "updated": updated,
                },
            )


if __name__ == "__main__":
    main()
