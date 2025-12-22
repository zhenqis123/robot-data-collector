#!/usr/bin/env python3
"""
Video utility:
- Encode color PNG sequences to H.264 MP4 for all cameras in a capture

Usage:
  python -m tools.videos /path/to/capture_root --fps 30 --output-name color.mp4
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
    annotate_video_positions(root, fps)


def load_ref_timestamps(root: Path) -> Tuple[List[float], str]:
    aligned_path = root / "frames_aligned.csv"
    if aligned_path.exists():
        with aligned_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            ts_list: List[float] = []
            ref_cam = ""
            for i, row in enumerate(reader):
                if i == 0:
                    ref_cam = row.get("ref_camera", "") or ""
                value = row.get("ref_timestamp_ms", "")
                if value == "":
                    continue
                try:
                    ts_list.append(float(value))
                except ValueError:
                    continue
            return ts_list, ref_cam

    meta_path = root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cams = [str(c["id"]) for c in meta.get("cameras", []) if "id" in c]
        if cams:
            cam_id = cams[0]
            cam_dir = root / sanitize_camera_id(cam_id)
            ts_path = cam_dir / "timestamps.csv"
            if ts_path.exists():
                with ts_path.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    ts_list = []
                    for row in reader:
                        value = row.get("timestamp_ms", "")
                        if value == "":
                            continue
                        try:
                            ts_list.append(float(value))
                        except ValueError:
                            continue
                    return ts_list, cam_id
    return [], ""


def nearest_frame_index(timestamps: List[float], ts_ms: float) -> Optional[int]:
    if not timestamps:
        return None
    idx = bisect.bisect_left(timestamps, ts_ms)
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    before = timestamps[idx - 1]
    after = timestamps[idx]
    return idx - 1 if (ts_ms - before) <= (after - ts_ms) else idx


def update_jsonl_with_video_position(
    path: Path,
    timestamps: List[float],
    fps: float,
    ts_getter: Callable[[Dict], Optional[float]],
) -> None:
    if not path.exists():
        return
    updated = 0
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                rows.append({"_raw": line})
                continue
            ts = ts_getter(obj)
            if ts is not None:
                idx = nearest_frame_index(timestamps, ts)
                if idx is not None:
                    obj["video_frame_index"] = idx
                    obj["video_time_s"] = idx / fps
                    updated += 1
            rows.append(obj)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for obj in rows:
            if "_raw" in obj:
                f.write(obj["_raw"] + "\n")
            else:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"[videos] updated {updated} entries with video positions in {path.name}")


def update_camera_poses_with_video_position(path: Path, fps: float) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text())
    poses = data.get("poses", [])
    updated = 0
    for pose in poses:
        frame_index = pose.get("frame_index")
        if isinstance(frame_index, (int, float)):
            frame_idx = int(frame_index)
            pose["video_frame_index"] = frame_idx
            pose["video_time_s"] = frame_idx / fps
            updated += 1
    data["poses"] = poses
    path.write_text(json.dumps(data, indent=2))
    print(f"[videos] updated {updated} pose entries with video_time_s in {path.name}")


def annotate_video_positions(root: Path, fps: float) -> None:
    timestamps, ref_cam = load_ref_timestamps(root)
    if not timestamps:
        print(f"[videos] skip video position update, no timestamps for {root}")
        return
    events_path = root / "events.jsonl"
    annotations_path = root / "annotations.jsonl"
    poses_path = root / "camera_poses_apriltag.json"

    update_jsonl_with_video_position(
        events_path,
        timestamps,
        fps,
        lambda obj: obj.get("ts_ms") if isinstance(obj.get("ts_ms"), (int, float)) else None,
    )
    update_jsonl_with_video_position(
        annotations_path,
        timestamps,
        fps,
        lambda obj: obj.get("timestamp_ms")
        if isinstance(obj.get("timestamp_ms"), (int, float))
        else None,
    )
    update_camera_poses_with_video_position(poses_path, fps)


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


def main():
    parser = argparse.ArgumentParser(description="Encode color frames to MP4 for all cameras in a capture")
    parser.add_argument("root", help="Root directory containing capture folders or meta.json")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--output-name", default="color.mp4", help="Output video filename per camera")
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
        auto_process(meta.parent, args.fps, args.output_name)


if __name__ == "__main__":
    main()
