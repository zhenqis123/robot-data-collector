#!/usr/bin/env python3
"""
Video utility:
- Encode color PNG sequences or MKV videos to H.264 MP4 for all cameras in a capture

Usage:
  python -m tools.videos /path/to/capture_root --fps 30 --output-name color_aligned.mp4
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov"}


def encode_color_frames_from_list(frames: List[Path], output: Path, fps: float = 30.0) -> int:
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
    return count


def encode_video_file(input_path: Path, output: Path, fps: float = 30.0) -> int:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {input_path}")
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size for {input_path}")
    if fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        count += 1
    cap.release()
    writer.release()
    print(f"[videos] wrote {output} from {count} frames")
    return count


def derive_cameras(root: Path) -> List[str]:
    aligned_path = root / "frames_aligned.csv"
    if aligned_path.exists():
        with aligned_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            cams = set()
            try:
                first_row = next(reader)
                ref_id = first_row.get("ref_camera", "")
                if ref_id and is_realsense_id(ref_id):
                    cams.add(ref_id)
            except StopIteration:
                pass
            for field in reader.fieldnames or []:
                if field.endswith("_color"):
                    cid = field.replace("_color", "")
                    if cid == "ref" and "ref_camera" in (reader.fieldnames or []):
                        continue
                    if is_realsense_id(cid):
                        cams.add(cid)
            if cams:
                return list(cams)
    return []


def read_alignment_rows(root: Path) -> Tuple[List[Dict[str, str]], str]:
    aligned_path = root / "frames_aligned.csv"
    if not aligned_path.exists():
        return [], ""
    with aligned_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return [], ""
    ref_cam = rows[0].get("ref_camera", "") or ""
    return rows, ref_cam


def read_camera_timestamps(csv_path: Path) -> Tuple[List[float], List[int]]:
    timestamps: List[float] = []
    indices: List[int] = []
    if not csv_path.exists():
        return timestamps, indices
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        counter = 0
        for row in reader:
            ts_val = row.get("timestamp_ms", "")
            if ts_val is None or ts_val == "":
                continue
            try:
                ts = float(ts_val)
            except (TypeError, ValueError):
                continue
            frame_index = row.get("frame_index", "")
            idx = None
            if frame_index:
                try:
                    idx = int(frame_index)
                except ValueError:
                    idx = None
            if idx is None:
                color_path = row.get("color_path", "") or row.get("rgb_path", "")
                stem = Path(color_path).stem
                if stem.isdigit():
                    idx = int(stem)
            if idx is None:
                counter += 1
                idx = counter
            timestamps.append(ts)
            indices.append(idx)
    return timestamps, indices


def sanitize_camera_id(cam_id: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in cam_id)

def is_realsense_id(cam_id: str) -> bool:
    return cam_id.startswith("RealSense")


def build_inputs(root: Path) -> Dict[str, Dict[str, Optional[Path]]]:
    """
    Build ordered color inputs per camera.
    If frames_aligned.csv exists, use its order; otherwise fall back to sorted color/*.png.
    """
    aligned_path = root / "frames_aligned.csv"
    result: Dict[str, Dict[str, Optional[Path]]] = {}
    if aligned_path.exists():
        with aligned_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                ref_cam = rows[0].get("ref_camera", "")
                # initialize lists
                cams = derive_cameras(root)
                for c in cams:
                    result[c] = {"video": None, "frames": []}
                for row in rows:
                    # reference camera
                    if ref_cam:
                        ref_color = row.get("ref_color", "")
                        if ref_color and ref_cam in result:
                            ref_path = root / sanitize_camera_id(ref_cam) / ref_color
                            if ref_path.suffix.lower() in VIDEO_EXTS:
                                result[ref_cam]["video"] = ref_path
                            else:
                                result[ref_cam]["frames"].append(ref_path)
                    for key, val in row.items():
                        if key.endswith("_color") and key not in ("ref_color",):
                            cam_id = key.replace("_color", "")
                            if not is_realsense_id(cam_id):
                                continue
                            if cam_id not in result:
                                result[cam_id] = {"video": None, "frames": []}
                            if val:
                                path = root / sanitize_camera_id(cam_id) / val
                                if path.suffix.lower() in VIDEO_EXTS:
                                    result[cam_id]["video"] = path
                                else:
                                    result[cam_id]["frames"].append(path)
    if result:
        return result
    # fallback: per camera sorted color/*.png
    meta_path = root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cams = [c["id"] for c in meta.get("cameras", []) if is_realsense_id(str(c.get("id", "")))]
        for cid in cams:
            cam_dir = root / sanitize_camera_id(str(cid))
            color_dir = cam_dir / "color"
            frames = sorted(color_dir.glob("*.png"))
            if frames:
                result[str(cid)] = {"video": None, "frames": frames}
    return result


def build_aligned_inputs(
    root: Path,
) -> Tuple[Dict[str, Dict[str, Optional[Path]]], List[Dict[str, str]], str]:
    rows, ref_cam = read_alignment_rows(root)
    if not rows:
        return {}, [], ""
    result: Dict[str, Dict[str, Optional[Path]]] = {}
    cams = derive_cameras(root)
    for c in cams:
        result[c] = {"video": None, "frames": []}
    for row in rows:
        if ref_cam:
            if not is_realsense_id(ref_cam):
                continue
            ref_color = row.get("ref_color", "")
            if ref_color:
                ref_path = root / sanitize_camera_id(ref_cam) / ref_color
                payload = result.setdefault(ref_cam, {"video": None, "frames": []})
                if ref_path.suffix.lower() in VIDEO_EXTS:
                    payload["video"] = ref_path
                else:
                    payload["frames"].append(ref_path)
        for key, val in row.items():
            if key.endswith("_color") and key not in ("ref_color",):
                cam_id = key.replace("_color", "")
                if not is_realsense_id(cam_id):
                    continue
                if val:
                    path = root / sanitize_camera_id(cam_id) / val
                    payload = result.setdefault(cam_id, {"video": None, "frames": []})
                    if path.suffix.lower() in VIDEO_EXTS:
                        payload["video"] = path
                    else:
                        payload["frames"].append(path)
    return result, rows, ref_cam


def encode_video_from_alignment(
    video_path: Path,
    output: Path,
    fps: float,
    ref_timestamps: List[float],
    aligned_indices: List[int],
    cam_timestamps: List[float],
    cam_indices: List[int],
) -> int:
    if aligned_indices and len(aligned_indices) != len(ref_timestamps):
        raise RuntimeError("Aligned frame indices size mismatch")
    if not ref_timestamps:
        raise RuntimeError("Missing timestamps for aligned encode")
    if not aligned_indices and (not cam_timestamps or not cam_indices):
        raise RuntimeError("Missing timestamps for aligned encode")
    desired_indices: List[int] = []
    for i, ts in enumerate(ref_timestamps):
        if aligned_indices:
            frame_index = aligned_indices[i]
        else:
            idx = nearest_frame_index(cam_timestamps, ts)
            if idx is None:
                frame_index = 0
            else:
                frame_index = cam_indices[idx]
        desired_indices.append(int(frame_index) if frame_index else 0)
    if not desired_indices:
        raise RuntimeError("No aligned frame indices available")

    non_decreasing = True
    last_seen = 0
    for idx in desired_indices:
        if idx <= 0:
            continue
        if idx < last_seen:
            non_decreasing = False
            break
        last_seen = idx
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size for {video_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
    count = 0
    if any(idx <= 0 for idx in desired_indices):
        cap.release()
        writer.release()
        raise RuntimeError(f"Invalid aligned frame indices for {output.name}")

    if non_decreasing:
        next_idx = 0
        current_frame_index = 0
        last_frame = None
        pbar = tqdm(total=len(desired_indices), desc=f"align {output.name}", unit="frame", leave=False)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while next_idx < len(desired_indices):
            target_index = desired_indices[next_idx]
            while current_frame_index < target_index:
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    writer.release()
                    raise RuntimeError(f"Failed to read frame {target_index} from {video_path}")
                current_frame_index += 1
                last_frame = frame
            if last_frame is None or current_frame_index < target_index:
                cap.release()
                writer.release()
                raise RuntimeError(f"Missing frame {target_index} in {video_path}")
            while next_idx < len(desired_indices) and desired_indices[next_idx] == target_index:
                writer.write(last_frame)
                count += 1
                next_idx += 1
                pbar.update(1)
        pbar.close()
    else:
        for i, _ in enumerate(tqdm(ref_timestamps, desc=f"align {output.name}", unit="frame", leave=False)):
            frame_index = desired_indices[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                writer.release()
                raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
            writer.write(frame)
            count += 1

    cap.release()
    writer.release()
    print(f"[videos] wrote {output} from {count} aligned frames")
    return count


def auto_process(root: Path, fps: float, output_name: str) -> None:
    root = root.expanduser().resolve()
    if should_skip_step(root, "encode_videos"):
        print(f"[videos] skip {root}, already encoded")
        return
    inputs, aligned_rows, ref_cam = build_aligned_inputs(root)
    if aligned_rows:
        ref_timestamps = []
        valid_rows: List[Dict[str, str]] = []
        for row in aligned_rows:
            value = row.get("ref_timestamp_ms", "")
            if value == "":
                continue
            try:
                ref_timestamps.append(float(value))
            except ValueError:
                continue
            valid_rows.append(row)
    else:
        inputs = build_inputs(root)
        ref_timestamps = []
        valid_rows = []
        ref_cam = ""

    if not inputs:
        print(f"[videos] skip {root}, no frames_aligned.csv or frames found")
        return
    encoded: Dict[str, int] = {}
    for cid, payload in inputs.items():
        frames = payload.get("frames") or []
        video_path = payload.get("video")
        out = root / sanitize_camera_id(str(cid)) / output_name
        try:
            if video_path:
                if video_path.resolve() == out.resolve():
                    print(f"[videos] skip {cid}: output matches source {video_path.name}")
                    continue
                if aligned_rows and ref_cam and ref_timestamps:
                    aligned_indices = []
                    key = "ref_frame_index" if cid == ref_cam else f"{cid}_frame_index"
                    for row in valid_rows:
                        value = row.get(key, "")
                        if value == "":
                            aligned_indices.append(0)
                            continue
                        try:
                            aligned_indices.append(int(value))
                        except ValueError:
                            aligned_indices.append(0)
                    if not any(idx > 0 for idx in aligned_indices):
                        aligned_indices = []
                    ts_path = root / sanitize_camera_id(str(cid)) / "timestamps.csv"
                    cam_ts, cam_idx = read_camera_timestamps(ts_path)
                    count = encode_video_from_alignment(
                        video_path, out, fps, ref_timestamps, aligned_indices, cam_ts, cam_idx
                    )
                    encoded[str(cid)] = count
                else:
                    count = encode_video_file(video_path, out, fps)
                    encoded[str(cid)] = count
            elif frames:
                count = encode_color_frames_from_list(frames, out, fps)
                encoded[str(cid)] = count
        except Exception as exc:
            print(f"[videos] skip {cid}: {exc}")
    if encoded:
        update_marker(
            root,
            "encode_videos",
            {
                "fps": fps,
                "output_name": output_name,
                "cameras": list(encoded.keys()),
                "frame_counts": encoded,
            },
        )


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
        cams = [
            str(c["id"])
            for c in meta.get("cameras", [])
            if "id" in c and is_realsense_id(str(c["id"]))
        ]
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
    print(f"[videos] updated marker {marker_path}")


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
    parser.add_argument("--output-name", default="color_aligned.mp4",
                        help="Output video filename per camera (aligned)")
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
