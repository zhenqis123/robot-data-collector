#!/usr/bin/env python3
"""
Timestamp alignment across sensors using nearest-neighbor to a reference camera.

Usage:
  python -m tools.timestamps /path/to/capture --reference CAMERA_ID

Reads timestamps.csv under each camera directory and writes frames_aligned.csv
at the capture root. The reference camera frames are rows; for each other camera,
the closest frame (by timestamp_ms) is recorded.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import bisect
from datetime import datetime, timezone
import numpy as np

TS_FIELDS = [
    "timestamp_iso",
    "timestamp_ms",
    "device_timestamp_ms",
    "frame_index",
    "color_path",
    "rgb_path",
    "depth_path",
    "color_timestamp_iso",
    "color_timestamp_ms",
    "color_device_timestamp_ms",
    "color_frame_index",
    "depth_timestamp_iso",
    "depth_timestamp_ms",
    "depth_device_timestamp_ms",
    "depth_frame_index",
]
ALIGN_MAX_DELTA_MS = 30


@dataclass
class FrameRow:
    timestamp_iso: str
    timestamp_ms: int
    device_timestamp_ms: int
    frame_index: Optional[int]
    color_path: str
    depth_path: str


def read_timestamps(csv_path: Path) -> List[FrameRow]:
    rows: List[FrameRow] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_color_fields = "color_timestamp_ms" in fieldnames or "color_frame_index" in fieldnames
        for r in reader:
            try:
                def parse_int(value: str) -> Optional[int]:
                    if value == "":
                        return None
                    try:
                        return int(value)
                    except ValueError:
                        return None

                color_path = r.get("color_path", "")
                rgb_path = r.get("rgb_path", "")
                depth_path = r.get("depth_path", "")
                if not color_path and rgb_path:
                    color_path = rgb_path

                if has_color_fields:
                    ts_ms = parse_int(r.get("color_timestamp_ms", ""))
                    if ts_ms is None:
                        continue
                    ts_iso = r.get("color_timestamp_iso", "")
                    device_ts = parse_int(r.get("color_device_timestamp_ms", "")) or 0
                    idx = parse_int(r.get("color_frame_index", ""))
                else:
                    ts_ms = parse_int(r.get("timestamp_ms", ""))
                    if ts_ms is None:
                        continue
                    ts_iso = r.get("timestamp_iso", "")
                    device_ts = parse_int(r.get("device_timestamp_ms", "")) or 0
                    idx = parse_int(r.get("frame_index", ""))

                if idx is None and color_path:
                    stem = Path(color_path).stem
                    if stem.isdigit():
                        idx = int(stem)
                rows.append(
                    FrameRow(
                        timestamp_iso=ts_iso,
                        timestamp_ms=int(ts_ms),
                        device_timestamp_ms=int(device_ts),
                        frame_index=idx,
                        color_path=color_path,
                        depth_path=depth_path,
                    )
                )
            except:
                continue
    return rows


def build_timestamp_index(frames: List[FrameRow]) -> Tuple[List[int], List[FrameRow]]:
    if not frames:
        return [], []
    frames_sorted = sorted(frames, key=lambda fr: fr.timestamp_ms)
    ts_list = [fr.timestamp_ms for fr in frames_sorted]
    return ts_list, frames_sorted


def closest_frame(
    target_ts: int,
    ts_list: List[int],
    frames_sorted: List[FrameRow],
) -> Tuple[Optional[FrameRow], Optional[int]]:
    if not ts_list:
        return None, None
    idx = bisect.bisect_left(ts_list, target_ts)
    candidates = []
    if idx < len(ts_list):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best_idx = min(candidates, key=lambda i: abs(ts_list[i] - target_ts))
    return frames_sorted[best_idx], abs(ts_list[best_idx] - target_ts)


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


def align_capture(capture_root: Path, reference: Optional[str]) -> None:
    meta_path = capture_root / "meta.json"
    if not meta_path.exists():
        print(f"[timestamps] skip {capture_root}, no meta.json")
        return
    if should_skip_step(capture_root, "align_timestamps"):
        print(f"[timestamps] skip {capture_root}, already aligned")
        return
    with meta_path.open("r") as f:
        import json

        meta = json.load(f)
    cam_ids_raw = [
        c["id"]
        for c in meta.get("cameras", [])
        if "id" in c and is_realsense_id(str(c["id"]))
    ]
    if not cam_ids_raw:
        print(f"[timestamps] no cameras in {capture_root}")
        return
    cam_ids = []
    seen = set()
    for cid in cam_ids_raw:
        if cid in seen:
            continue
        seen.add(cid)
        cam_ids.append(cid)

    ref_id = reference if reference else cam_ids[0]
    if ref_id not in cam_ids:
        ref_id = cam_ids[0]

    cam_frames: Dict[str, List[FrameRow]] = {}
    for cid in cam_ids:
        cid_path = Path(sanitize_camera_id(str(cid)))
        ts_path = capture_root / cid_path / "timestamps.csv"
        cam_frames[cid] = read_timestamps(ts_path)
    active_cam_ids = [cid for cid in cam_ids if cam_frames.get(cid)]
    if not active_cam_ids:
        print(f"[timestamps] no cameras with timestamps in {capture_root}")
        return
    if ref_id not in active_cam_ids:
        ref_id = active_cam_ids[0]

    cam_indexes: Dict[str, Tuple[List[int], List[FrameRow]]] = {
        cid: build_timestamp_index(cam_frames[cid]) for cid in active_cam_ids
    }

    ref_frames = cam_frames.get(ref_id, [])
    if not ref_frames:
        print(f"[timestamps] reference {ref_id} has no timestamps in {capture_root}")
        return

    # Prepare output header
    header = [
        "ref_camera",
        "ref_timestamp_iso",
        "ref_timestamp_ms",
        "ref_device_timestamp_ms",
        "ref_frame_index",
        "ref_color",
        "ref_depth",
    ]
    for cid in active_cam_ids:
        if cid == ref_id:
            continue
        header.extend([f"{cid}_frame_index", f"{cid}_color", f"{cid}_depth", f"{cid}_delta_ms"])

    out_path = capture_root / "frames_aligned.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # Collect stats per camera
        stats: Dict[str, List[int]] = {cid: [] for cid in active_cam_ids if cid != ref_id}
        kept_rows = 0
        for fr in ref_frames:
            row_ok = True
            row = [
                ref_id,
                fr.timestamp_iso,
                fr.timestamp_ms,
                fr.device_timestamp_ms,
                fr.frame_index if fr.frame_index is not None else "",
                fr.color_path,
                fr.depth_path,
            ]
            for cid in active_cam_ids:
                if cid == ref_id:
                    continue
                ts_list, frames_sorted = cam_indexes.get(cid, ([], []))
                match, delta = closest_frame(fr.timestamp_ms, ts_list, frames_sorted)
                if match and delta is not None and delta <= ALIGN_MAX_DELTA_MS:
                    row.extend(
                        [
                            match.frame_index if match.frame_index is not None else "",
                            match.color_path,
                            match.depth_path,
                            delta if delta is not None else "",
                        ]
                    )
                    if delta is not None:
                        stats[cid].append(delta)
                else:
                    row_ok = False
                    break
            if row_ok:
                writer.writerow(row)
                kept_rows += 1
    # Print stats
    lines = [f"[timestamps] aligned frames written to {out_path}"]
    lines.append(f"  reference: {ref_id}")
    lines.append(f"  frames aligned (rows): {kept_rows}")
    for cid, deltas in stats.items():
        if deltas:
            arr = np.array(deltas, dtype=np.float32)
            lines.append(f"  {cid}: mean_err_ms={arr.mean():.2f}, std_ms={arr.std():.2f}, count={len(arr)}")
        else:
            lines.append(f"  {cid}: no matches")
    print("\n".join(lines))
    update_marker(
        capture_root,
        "align_timestamps",
        {
            "reference_camera": ref_id,
            "output": out_path.name,
            "rows": kept_rows,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Align frames across cameras by timestamp (nearest to reference).")
    parser.add_argument("root", help="Root directory containing capture folders or meta.json")
    parser.add_argument("--reference", help="Reference camera id (defaults to first in meta.json)", default=None)
    parser.add_argument("--find-meta", type=str, default="true",
                        choices=["true", "false"],
                        help="Search meta.json recursively (true) or only root/*/meta.json (false)")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    find_meta = args.find_meta.lower() == "true"
    metas = list_meta_files(root, find_meta, 2)
    if not metas:
        print("No meta.json found")
        return
    for meta in metas:
        align_capture(meta.parent, args.reference)


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
    print(f"[timestamps] updated marker {marker_path}")


def sanitize_camera_id(cam_id: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in cam_id)

def is_realsense_id(cam_id: str) -> bool:
    return cam_id.startswith("RealSense")


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


if __name__ == "__main__":
    main()
