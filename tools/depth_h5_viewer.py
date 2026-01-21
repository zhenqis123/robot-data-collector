#!/usr/bin/env python3
"""
Sequentially visualize depth.h5 using RealSense colorizer.

Depth HDF5 format (see tools/align_depth.py):
- dataset name: /depth
- shape: (num_frames, height, width)
- dtype: uint16 (raw depth units)

Usage:
  python -m tools.depth_h5_viewer /path/to/depth.h5
  python -m tools.depth_h5_viewer /path/to/depth.h5 --fps 15 --start 100
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from tools.depth_h5_viz_video import (
    RealSenseDepthColorizer,
    load_meta_for_depth,
    resolve_depth_scale,
    read_fps_from_timestamps,
)


def resolve_depth_scale_for_file(depth_path: Path, override: float) -> Optional[float]:
    if override > 0:
        return override
    meta = load_meta_for_depth(depth_path)
    cam_dir_name = depth_path.parent.name
    return resolve_depth_scale(meta, cam_dir_name)


def visualize_depth(depth_path: Path, fps: float, depth_scale: float, start: int, step: int,
                    loop: bool) -> int:
    if h5py is None:
        print("h5py not installed; cannot read depth.h5")
        return 1
    if rs is None:
        print("pyrealsense2 not installed; cannot use RealSense colorizer")
        return 1
    if not depth_path.exists():
        print(f"depth file not found: {depth_path}")
        return 1

    if fps <= 0:
        fps = read_fps_from_timestamps(depth_path.parent / "timestamps.csv") or 30.0
    wait_ms = max(1, int(1000.0 / max(1.0, fps)))

    depth_scale_val = resolve_depth_scale_for_file(depth_path, depth_scale) or 0.0

    with h5py.File(depth_path, "r") as f:
        if "depth" not in f:
            print(f"missing /depth dataset in {depth_path}")
            return 1
        dset = f["depth"]
        if dset.ndim != 3:
            print(f"unexpected depth shape {dset.shape} in {depth_path}")
            return 1
        total, height, width = dset.shape
        if total <= 0:
            print(f"no frames in {depth_path}")
            return 1

        start_idx = max(0, min(int(start), total - 1))
        step_val = max(1, int(step))

        colorizer = RealSenseDepthColorizer(int(width), int(height), depth_scale_val)
        window_name = f"Depth Viewer: {depth_path.name}"
        paused = False
        idx = start_idx

        try:
            while True:
                if idx < 0 or idx >= total:
                    if loop:
                        idx = 0
                    else:
                        break

                frame = dset[idx]
                if frame is None:
                    idx += step_val
                    continue
                if frame.dtype != np.uint16:
                    frame = frame.astype(np.uint16)
                color = colorizer.colorize(frame)

                info = f"frame {idx + 1}/{total}  fps {fps:.2f}  scale {depth_scale_val:.6f}"
                cv2.putText(color, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(color, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                cv2.imshow(window_name, color)

                key = cv2.waitKey(0 if paused else wait_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord(" "), ord("p")):
                    paused = not paused
                    continue
                if key in (ord("a"), ord("b")):
                    idx = max(0, idx - step_val)
                    paused = True
                    continue
                if key in (ord("d"), ord("n")):
                    idx = min(total - 1, idx + step_val)
                    paused = True
                    continue
                if key == ord("r"):
                    idx = 0
                    paused = True
                    continue

                if not paused:
                    idx += step_val
        finally:
            colorizer.close()
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequentially visualize depth.h5 using RealSense colorizer")
    parser.add_argument("depth_path", help="Path to depth.h5 (or depth_aligned.h5)")
    parser.add_argument("--fps", type=float, default=0.0, help="Playback fps; 0=auto from timestamps.csv")
    parser.add_argument("--depth-scale", type=float, default=0.0, help="Depth scale override in meters")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (0-based)")
    parser.add_argument("--step", type=int, default=1, help="Frame step for playback/seek")
    parser.add_argument("--loop", action="store_true", help="Loop playback when reaching end")
    args = parser.parse_args()

    depth_path = Path(args.depth_path).expanduser().resolve()
    return visualize_depth(depth_path, args.fps, args.depth_scale, args.start, args.step, args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
