#!/usr/bin/env python3
"""
Video utility:
- Encode color PNG sequences to H.264 MP4 for all cameras in a capture

Usage:
  python -m tools.videos /path/to/capture_root --fps 30 --output-name color.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def encode_color_frames_from_list(frames: List[Path], output: Path, fps: float = 30.0) -> None:
    if not frames:
        raise RuntimeError("No frames provided")
    sample = None
    for f in frames:
        sample = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if sample is not None:
            break
    if sample is None:
        raise RuntimeError("Failed to read any sample frame")
    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # H.264-like; use ffmpeg for stricter control if needed
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
    count = 0
    for f in tqdm(frames, desc=f"encode {output.name}", unit="frame", leave=False):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        writer.write(img)
        count += 1
    writer.release()
    print(f"[videos] wrote {output} from {count} frames")


def derive_cameras(root: Path) -> List[str]:
    aligned_path = root / "frames_aligned.csv"
    if aligned_path.exists():
        with aligned_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            cams = set()
            try:
                first_row = next(reader)
                ref_id = first_row.get("ref_camera", "")
                if ref_id:
                    cams.add(ref_id)
            except StopIteration:
                pass
            for field in reader.fieldnames or []:
                if field.endswith("_color"):
                    cid = field.replace("_color", "")
                    if cid == "ref" and "ref_camera" in (reader.fieldnames or []):
                        continue
                    cams.add(cid)
            if cams:
                return list(cams)
    return []


def sanitize_camera_id(cam_id: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in cam_id)


def build_frame_lists(root: Path) -> Dict[str, List[Path]]:
    """
    Build ordered color frame lists per camera.
    If frames_aligned.csv exists, use its order; otherwise fall back to sorted color/*.png.
    """
    aligned_path = root / "frames_aligned.csv"
    result: Dict[str, List[Path]] = {}
    if aligned_path.exists():
        with aligned_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                ref_cam = rows[0].get("ref_camera", "")
                # initialize lists
                cams = derive_cameras(root)
                for c in cams:
                    result[c] = []
                for row in rows:
                    # reference camera
                    if ref_cam:
                        ref_color = row.get("ref_color", "")
                        if ref_color and ref_cam in result:
                            result[ref_cam].append(root / sanitize_camera_id(ref_cam) / ref_color)
                    for key, val in row.items():
                        if key.endswith("_color") and key not in ("ref_color",):
                            cam_id = key.replace("_color", "")
                            if cam_id not in result:
                                result[cam_id] = []
                            if val:
                                result[cam_id].append(root / sanitize_camera_id(cam_id) / val)
    if result:
        return result
    # fallback: per camera sorted color/*.png
    meta_path = root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cams = [c["id"] for c in meta.get("cameras", [])]
        for cid in cams:
            cam_dir = root / sanitize_camera_id(str(cid))
            color_dir = cam_dir / "color"
            frames = sorted(color_dir.glob("*.png"))
            if frames:
                result[str(cid)] = frames
    return result


def auto_process(root: Path, fps: float, output_name: str) -> None:
    root = root.expanduser().resolve()
    frame_lists = build_frame_lists(root)
    if not frame_lists:
        print(f"[videos] skip {root}, no frames_aligned.csv or frames found")
        return
    for cid, frames in frame_lists.items():
        if not frames:
            continue
        out = root / sanitize_camera_id(str(cid)) / output_name
        try:
            encode_color_frames_from_list(frames, out, fps)
        except Exception as exc:
            print(f"[videos] skip {cid}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Encode color frames to MP4 for all cameras in a capture")
    parser.add_argument("root", help="Capture root containing meta.json (and optionally frames_aligned.csv)")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--output-name", default="color.mp4", help="Output video filename per camera")
    args = parser.parse_args()
    auto_process(Path(args.root), args.fps, args.output_name)


if __name__ == "__main__":
    main()
