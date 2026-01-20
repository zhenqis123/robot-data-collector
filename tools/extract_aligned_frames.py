#!/usr/bin/env python3
"""
Extract a single frame (by index) from each camera's aligned videos.

Inputs per camera:
  - color_aligned.mp4
  - depth_aligned_viz.mp4

Usage:
  python -m tools.extract_aligned_frames /path/to/session --frame 1000
  python -m tools.extract_aligned_frames /path/to/session --frame 1001 --one-based
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2


def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[extract] failed to open {video_path}")
        return False
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[extract] missing frame {frame_index} in {video_path}")
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        return True
    finally:
        cap.release()


def find_camera_dirs(session: Path) -> List[Path]:
    cams = sorted([p for p in session.iterdir() if p.is_dir() and p.name.startswith("RealSense_")])
    if cams:
        return cams
    return sorted([p for p in session.iterdir() if p.is_dir()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract aligned frames by index from each camera")
    parser.add_argument("session", help="Session directory containing camera subfolders")
    parser.add_argument("--frame", type=int, required=True, help="Frame index to extract")
    parser.add_argument("--one-based", action="store_true", help="Interpret frame index as 1-based")
    parser.add_argument("--output-suffix", default=None,
                        help="Optional suffix for output PNGs (default: f<frame>)")
    args = parser.parse_args()

    session = Path(args.session).expanduser().resolve()
    if not session.exists():
        print(f"[extract] session not found: {session}")
        return 1

    frame_index = int(args.frame)
    if args.one_based:
        frame_index -= 1
    if frame_index < 0:
        print("[extract] frame index must be >= 0 after conversion")
        return 1

    suffix = args.output_suffix or f"f{args.frame}"
    cams = find_camera_dirs(session)
    if not cams:
        print(f"[extract] no camera folders found in {session}")
        return 1

    any_ok = False
    for cam_dir in cams:
        color_video = cam_dir / "color_aligned.mp4"
        depth_video = cam_dir / "depth_aligned_viz.mp4"

        if color_video.exists():
            out = cam_dir / f"color_aligned_{suffix}.png"
            ok = extract_frame(color_video, frame_index, out)
            any_ok = any_ok or ok
        else:
            print(f"[extract] missing {color_video}")

        if depth_video.exists():
            out = cam_dir / f"depth_aligned_{suffix}.png"
            ok = extract_frame(depth_video, frame_index, out)
            any_ok = any_ok or ok
        else:
            print(f"[extract] missing {depth_video}")

    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
