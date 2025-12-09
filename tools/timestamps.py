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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

TS_FIELDS = ["timestamp_iso", "timestamp_ms", "device_timestamp_ms", "color_path", "depth_path"]


@dataclass
class FrameRow:
    timestamp_iso: str
    timestamp_ms: int
    device_timestamp_ms: int
    color_path: str
    depth_path: str


def read_timestamps(csv_path: Path) -> List[FrameRow]:
    rows: List[FrameRow] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    FrameRow(
                        timestamp_iso=r.get("timestamp_iso", ""),
                        timestamp_ms=int(r.get("timestamp_ms", "0")),
                        device_timestamp_ms=int(r.get("device_timestamp_ms", "0")),
                        color_path=r.get("color_path", ""),
                        depth_path=r.get("depth_path", ""),
                    )
                )
            except ValueError:
                continue
    return rows


def closest_frame(target_ts: int, frames: List[FrameRow]) -> Tuple[Optional[FrameRow], Optional[int]]:
    if not frames:
        return None, None
    best = min(frames, key=lambda fr: abs(fr.timestamp_ms - target_ts))
    return best, abs(best.timestamp_ms - target_ts)


def align_capture(capture_root: Path, reference: Optional[str]) -> None:
    meta_path = capture_root / "meta.json"
    if not meta_path.exists():
        print(f"[timestamps] skip {capture_root}, no meta.json")
        return
    with meta_path.open("r") as f:
        import json

        meta = json.load(f)
    cam_ids = [c["id"] for c in meta.get("cameras", [])]
    if not cam_ids:
        print(f"[timestamps] no cameras in {capture_root}")
        return

    ref_id = reference if reference else cam_ids[0]
    if ref_id not in cam_ids:
        ref_id = cam_ids[0]

    cam_frames: Dict[str, List[FrameRow]] = {}
    for cid in cam_ids:
        cid_path = Path(str(cid).replace("#", "_"))
        ts_path = capture_root / cid_path / "timestamps.csv"
        cam_frames[cid] = read_timestamps(ts_path)

    ref_frames = cam_frames.get(ref_id, [])
    if not ref_frames:
        print(f"[timestamps] reference {ref_id} has no timestamps in {capture_root}")
        return

    # Prepare output header
    header = ["ref_camera", "ref_timestamp_iso", "ref_timestamp_ms", "ref_device_timestamp_ms", "ref_color", "ref_depth"]
    for cid in cam_ids:
        if cid == ref_id:
            continue
        header.extend([f"{cid}_color", f"{cid}_depth", f"{cid}_delta_ms"])

    out_path = capture_root / "frames_aligned.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # Collect stats per camera
        stats: Dict[str, List[int]] = {cid: [] for cid in cam_ids if cid != ref_id}
        for fr in ref_frames:
            row = [
                ref_id,
                fr.timestamp_iso,
                fr.timestamp_ms,
                fr.device_timestamp_ms,
                fr.color_path,
                fr.depth_path,
            ]
            for cid in cam_ids:
                if cid == ref_id:
                    continue
                match, delta = closest_frame(fr.timestamp_ms, cam_frames.get(cid, []))
                if match:
                    row.extend([match.color_path, match.depth_path, delta if delta is not None else ""])
                    if delta is not None:
                        stats[cid].append(delta)
                else:
                    row.extend(["", "", ""])
            writer.writerow(row)
    # Print stats
    lines = [f"[timestamps] aligned frames written to {out_path}"]
    lines.append(f"  reference: {ref_id}")
    lines.append(f"  frames aligned (rows): {len(ref_frames)}")
    for cid, deltas in stats.items():
        if deltas:
            arr = np.array(deltas, dtype=np.float32)
            lines.append(f"  {cid}: mean_err_ms={arr.mean():.2f}, std_ms={arr.std():.2f}, count={len(arr)}")
        else:
            lines.append(f"  {cid}: no matches")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Align frames across cameras by timestamp (nearest to reference).")
    parser.add_argument("capture", help="Capture root containing meta.json")
    parser.add_argument("--reference", help="Reference camera id (defaults to first in meta.json)", default=None)
    args = parser.parse_args()
    align_capture(Path(args.capture).expanduser().resolve(), args.reference)


if __name__ == "__main__":
    main()
