#!/usr/bin/env python3
"""
Visualize a session with multi-camera video mosaic, reprojection overlay, 3D poses, and events.

Usage:
  python tools/visualize_session_poses_full.py /path/to/session \
      --tag-map /path/to/apriltag_map.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


OBJECT_POINTS = np.array(
    [
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
    ],
    dtype=np.float64,
)

VIDEO_EXTS = [".mp4", ".mkv", ".avi", ".mov"]
DEFAULT_ALIGNED_NAME = "color_aligned.mp4"


def list_meta_files(root: Path, find_meta: bool, max_depth: int = 2) -> List[Path]:
    if find_meta:
        result: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth > max_depth:
                dirnames[:] = []
                continue
            if "meta.json" in filenames:
                result.append(Path(dirpath) / "meta.json")
        return sorted(result)
    result = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        meta = child / "meta.json"
        if meta.exists():
            result.append(meta)
    return result


def read_encode_output_name(session: Path) -> str:
    marker_path = session / "postprocess_markers.json"
    if not marker_path.exists():
        return ""
    try:
        payload = json.loads(marker_path.read_text())
    except json.JSONDecodeError:
        return ""
    steps = payload.get("steps")
    if not isinstance(steps, dict):
        return ""
    entry = steps.get("encode_videos")
    if not isinstance(entry, dict):
        return ""
    output_name = entry.get("output_name", "")
    return str(output_name) if output_name else ""


def resolve_aligned_output_name(session: Path, cam_ids: List[str]) -> str:
    output_name = read_encode_output_name(session)
    if output_name:
        return output_name
    fallback = DEFAULT_ALIGNED_NAME
    for cam_id in cam_ids:
        cam_dir = session / sanitize_camera_id(cam_id)
        if not (cam_dir / fallback).exists():
            return ""
    return fallback


def read_tag_map(tag_map_path: Path) -> Tuple[str, float, Dict[int, np.ndarray]]:
    data = json.loads(tag_map_path.read_text())
    family = data.get("family")
    tag_length = float(data.get("tag_length_m", 0.0))
    markers = data.get("markers", {})
    if not family or not tag_length or not markers:
        raise RuntimeError("Invalid tag map: missing family/tag_length_m/markers")
    tag_poses: Dict[int, np.ndarray] = {}
    for key, entry in markers.items():
        try:
            mid = int(key)
            T = np.array(entry.get("T_W_M", []), dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if T.shape != (4, 4):
            continue
        tag_poses[mid] = T
    if not tag_poses:
        raise RuntimeError("No valid markers in tag map")
    return family, tag_length, tag_poses


def load_intrinsics(meta_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    meta = json.loads(meta_path.read_text())
    cameras = meta.get("cameras", [])
    intrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cam in cameras:
        cid = cam.get("id")
        streams = cam.get("streams", {})
        color = streams.get("color", {}).get("intrinsics", {})
        fx, fy, cx, cy = (color.get("fx"), color.get("fy"), color.get("cx"), color.get("cy"))
        if cid is None or None in (fx, fy, cx, cy):
            continue
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        coeffs = color.get("coeffs", [0, 0, 0, 0, 0])
        dist = np.array(coeffs, dtype=np.float64)
        intrinsics[str(cid)] = (K, dist)
    return intrinsics


def sanitize_camera_id(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in value)


def parse_frame_index(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def prefer_mp4_path(path: Path) -> Path:
    if path.suffix.lower() == ".mkv":
        mp4 = path.with_suffix(".mp4")
        if mp4.exists():
            return mp4
    return path


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def read_video_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    if frame_index <= 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def read_frames_aligned(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    cam_ids: List[str] = []
    if not csv_path.exists():
        return rows, cam_ids
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames:
            for field in reader.fieldnames:
                if field.endswith("_color"):
                    cid = field.replace("_color", "")
                    if cid == "ref":
                        continue
                    cam_ids.append(cid)
    return rows, sorted(set(cam_ids))


def load_pose_entries(poses_path: Path) -> Dict[Tuple[int, str], Dict[str, object]]:
    data = json.loads(poses_path.read_text())
    entries: Dict[Tuple[int, str], Dict[str, object]] = {}
    for entry in data.get("poses", []):
        frame_index = entry.get("frame_index")
        cam_id = entry.get("camera_id")
        if not isinstance(frame_index, int) or cam_id is None:
            continue
        if entry.get("status") not in ("ok", "interpolated"):
            continue
        T_list = entry.get("T_W_C")
        if not isinstance(T_list, list):
            continue
        T = np.array(T_list, dtype=np.float64)
        if T.shape != (4, 4):
            continue
        entry_copy = dict(entry)
        entry_copy["T_W_C"] = T
        entries[(frame_index, str(cam_id))] = entry_copy
    return entries


def compute_tag_polylines(tag_length: float, tag_poses: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    polylines: Dict[int, np.ndarray] = {}
    scaled = OBJECT_POINTS * tag_length
    scaled = scaled.copy()
    scaled[:, 1] *= -1.0
    scaled[:, 2] *= -1.0
    for mid, T_W_M in tag_poses.items():
        R = T_W_M[:3, :3]
        t = T_W_M[:3, 3]
        poly = (R @ scaled.T).T + t
        polylines[mid] = poly
    return polylines


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def resolve_color_video(cam_dir: Path) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        cand = cam_dir / f"color{ext}"
        if cand.exists():
            return cand
    for ext in VIDEO_EXTS:
        cand = cam_dir / f"rgb{ext}"
        if cand.exists():
            return cand
    return None


def resolve_color_video_from_rows(session: Path, cam_id: str, rows: List[Dict[str, str]]) -> Optional[Path]:
    for row in rows:
        color_rel = ""
        if row.get("ref_camera") == cam_id:
            color_rel = row.get("ref_color", "")
        else:
            color_rel = row.get(f"{cam_id}_color", "")
        if not color_rel:
            continue
        color_path = prefer_mp4_path((session / sanitize_camera_id(cam_id) / color_rel))
        if color_path.exists() and is_video_path(color_path):
            return color_path
    return None


def overlay_reprojection(
    image: np.ndarray,
    pose_entry: Dict[str, object],
    tag_length: float,
    tag_poses: Dict[int, np.ndarray],
    detector: Detector,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    T_w_c = pose_entry.get("T_W_C")
    if not isinstance(T_w_c, np.ndarray):
        return canvas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    det_ids = {int(det.tag_id) for det in detections}

    for det in detections:
        corners = np.array(det.corners, dtype=np.float64).reshape(-1, 2)
        cv2.polylines(canvas, [corners.astype(np.int32)], True, (0, 255, 0), 2)

    scaled = OBJECT_POINTS * tag_length
    scaled = scaled.copy()
    scaled[:, 1] *= -1.0
    scaled[:, 2] *= -1.0

    T_c_w = invert_transform(T_w_c)
    rvec, _ = cv2.Rodrigues(T_c_w[:3, :3])
    tvec = T_c_w[:3, 3].reshape(3)

    for mid, T_W_M in tag_poses.items():
        if det_ids and mid not in det_ids:
            continue
        R_w_m = T_W_M[:3, :3]
        t_w_m = T_W_M[:3, 3]
        world_corners = (R_w_m @ scaled.T).T + t_w_m
        proj, _ = cv2.projectPoints(world_corners, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        cv2.polylines(canvas, [proj.astype(np.int32)], True, (0, 0, 255), 2)
        for pt in proj:
            p = tuple(pt.astype(int))
            cv2.drawMarker(canvas, p, (0, 0, 255), cv2.MARKER_CROSS, 8, 2)

    stats = pose_entry.get("reproj_error_px")
    if isinstance(stats, dict):
        mean = stats.get("mean")
        median = stats.get("median")
        max_err = stats.get("max")
        if all(v is not None for v in (mean, median, max_err)):
            text = f"reproj mean {mean:.2f}px  median {median:.2f}px  max {max_err:.2f}px"
        else:
            text = "reproj error: unavailable"
        cv2.putText(
            canvas,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_event_label(entry: Dict[str, object]) -> str:
    if "event" in entry:
        return f"event: {entry.get('event')}"
    state = entry.get("state")
    step_id = entry.get("current_step_id")
    if state and step_id:
        return f"ann: {state} ({step_id})"
    if state:
        return f"ann: {state}"
    return "ann: update"


def load_events(events_path: Path) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = entry.get("ts_ms")
            if isinstance(ts, (int, float)):
                items.append((int(ts), build_event_label(entry)))
    items.sort(key=lambda x: x[0])
    return items


def events_for_frame(events: List[Tuple[int, str]], ts_ms: int, window_ms: int, max_lines: int) -> List[str]:
    if not events:
        return []
    out: List[str] = []
    for ts, label in events:
        if ts < ts_ms - window_ms:
            continue
        if ts > ts_ms + window_ms:
            break
        delta = ts - ts_ms
        out.append(f"{label} ({delta:+d}ms)")
        if len(out) >= max_lines:
            break
    return out


def prepare_3d_canvas(
    tag_polylines: Dict[int, np.ndarray],
    cam_ids: List[str],
    all_points: np.ndarray,
    width: int,
    height: int,
) -> Tuple[plt.Figure, FigureCanvasAgg, plt.Axes, Dict[str, List], plt.PathCollection]:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    for poly in tag_polylines.values():
        ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], color="orange", linewidth=1.5)
    cam_scatter = ax.scatter([], [], [], c="blue", s=25)
    axis_lines: Dict[str, List] = {}
    axis_colors = {"x": "red", "y": "green", "z": "blue"}
    for cam_id in cam_ids:
        axis_lines[cam_id] = [
            ax.plot([], [], [], color=axis_colors["x"], linewidth=1.0)[0],
            ax.plot([], [], [], color=axis_colors["y"], linewidth=1.0)[0],
            ax.plot([], [], [], color=axis_colors["z"], linewidth=1.0)[0],
        ]
    if all_points.size:
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        center = (mins + maxs) / 2.0
        span = (maxs - mins).max()
        if span <= 0:
            span = 1.0
        half = span / 2.0
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
    return fig, canvas, ax, axis_lines, cam_scatter


def render_session(
    session: Path,
    meta_path: Path,
    family: str,
    tag_length: float,
    tag_poses: Dict[int, np.ndarray],
    tag_polylines: Dict[int, np.ndarray],
    poses_name: str,
    output_name: str,
    fps: int,
    stride: int,
    cols: int,
    scale: float,
    pose_width: int,
    event_window_ms: int,
    max_event_lines: int,
    threads: int,
) -> bool:
    frames_csv = session / "frames_aligned.csv"
    if not frames_csv.exists():
        print(f"[viz] skip {session}, missing frames_aligned.csv")
        return False
    if should_skip_step(session, "session_visualization_full"):
        print(f"[viz] skip {session}, already processed")
        return False
    if poses_name:
        poses_path = session / poses_name
        if not poses_path.exists():
            print(f"[viz] skip {session}, missing {poses_path.name}")
            return False
        print(f"[viz] using poses: {poses_path.name} (override)")
    else:
        preferred = session / "camera_poses_apriltag_post.json"
        fallback = session / "camera_poses_apriltag.json"
        if preferred.exists():
            poses_path = preferred
            print(f"[viz] using poses: {poses_path.name} (preferred)")
        elif fallback.exists():
            poses_path = fallback
            print(f"[viz] using poses: {poses_path.name} (fallback)")
        else:
            print(f"[viz] skip {session}, missing poses files")
            return False

    intrinsics = load_intrinsics(meta_path)

    rows, csv_cam_ids = read_frames_aligned(frames_csv)
    if not rows:
        print(f"[viz] skip {session}, no rows in {frames_csv}")
        return False
    meta = json.loads(meta_path.read_text())
    cam_ids = [str(c.get("id")) for c in meta.get("cameras", []) if c.get("id") is not None]
    cam_ids = [cid for cid in cam_ids if cid in intrinsics]
    if csv_cam_ids:
        cam_ids = [cid for cid in cam_ids if cid in csv_cam_ids or cid == rows[0].get("ref_camera")]
    if not cam_ids:
        print(f"[viz] skip {session}, no cameras found in meta.json")
        return False

    poses = load_pose_entries(poses_path)
    if not poses:
        print(f"[viz] skip {session}, no valid poses in {poses_path}")
        return False

    aligned_video_name = resolve_aligned_output_name(session, cam_ids)
    video_caps: Dict[str, Optional[cv2.VideoCapture]] = {}
    aligned_read_counts: Dict[str, int] = {}
    for cam_id in cam_ids:
        cam_dir = session / sanitize_camera_id(cam_id)
        if aligned_video_name:
            video = cam_dir / aligned_video_name
            if not video.exists():
                print(f"[viz] missing aligned video {video}")
                return False
        else:
            video = resolve_color_video(cam_dir)
            if video is None:
                video = resolve_color_video_from_rows(session, cam_id, rows)
        if video is not None:
            cap = cv2.VideoCapture(str(video))
            video_caps[cam_id] = cap if cap.isOpened() else None
        else:
            video_caps[cam_id] = None

    detector = Detector(
        families=family,
        nthreads=max(1, threads),
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
    )

    # Determine tile size from first available frame.
    tile_w = 0
    tile_h = 0
    for cam_id in cam_ids:
        cap = video_caps.get(cam_id)
        if cap is not None:
            frame = read_video_frame(cap, 1)
            if frame is not None:
                tile_h, tile_w = frame.shape[:2]
                break
    if tile_w <= 0 or tile_h <= 0:
        tile_w, tile_h = 640, 480
    tile_w = int(tile_w * scale)
    tile_h = int(tile_h * scale)

    col_count = cols if cols > 0 else int(math.ceil(math.sqrt(len(cam_ids))))
    rows_count = int(math.ceil(len(cam_ids) / col_count))
    mosaic_w = col_count * tile_w
    mosaic_h = rows_count * tile_h
    pose_w = pose_width if pose_width > 0 else mosaic_h
    pose_h = mosaic_h

    all_points: List[np.ndarray] = []
    for poly in tag_polylines.values():
        all_points.append(poly)
    for entry in poses.values():
        T = entry.get("T_W_C")
        if isinstance(T, np.ndarray):
            all_points.append(T[:3, 3].reshape(1, 3))
    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3))

    fig, canvas, ax, axis_lines, cam_scatter = prepare_3d_canvas(
        tag_polylines,
        cam_ids,
        points,
        pose_w,
        pose_h,
    )

    events = load_events(session / "events.jsonl")

    output_fps = float(fps) / max(1, stride)
    out_path = session / output_name
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (mosaic_w + pose_w, mosaic_h),
    )
    if not writer.isOpened():
        print(f"[viz] failed to open writer for {out_path}")
        return False

    for frame_idx, row in enumerate(tqdm(rows, desc=f"viz {session.name}", unit="frame")):
        if frame_idx % max(1, stride) != 0:
            continue
        tiles: List[np.ndarray] = []
        ref_camera = row.get("ref_camera", "")
        ts_raw = row.get("ref_timestamp_ms", "")
        try:
            ts_ms = int(float(ts_raw)) if ts_raw else 0
        except ValueError:
            ts_ms = 0

        for cam_id in cam_ids:
            frame_number = None
            color_rel = ""
            if cam_id == ref_camera:
                frame_number = parse_frame_index(row.get("ref_frame_index", ""))
                color_rel = row.get("ref_color", "")
            else:
                frame_number = parse_frame_index(row.get(f"{cam_id}_frame_index", ""))
                color_rel = row.get(f"{cam_id}_color", "")

            img = None
            cap = video_caps.get(cam_id)
            if cap is not None and frame_number is not None:
                if aligned_video_name:
                    expected = frame_idx + 1
                    current = aligned_read_counts.get(cam_id, 0)
                    while current < expected:
                        ok, img = cap.read()
                        if not ok:
                            img = None
                            break
                        current += 1
                        aligned_read_counts[cam_id] = current
                    if current != expected:
                        img = None
                else:
                    img = read_video_frame(cap, frame_number)
            if img is None and color_rel:
                color_path = prefer_mp4_path((session / sanitize_camera_id(cam_id) / color_rel))
                if color_path.exists() and not is_video_path(color_path):
                    img = cv2.imread(str(color_path))
            if img is None:
                img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(
                    img,
                    "missing frame",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if img.shape[1] != tile_w or img.shape[0] != tile_h:
                img = cv2.resize(img, (tile_w, tile_h))

            pose_entry = poses.get((frame_idx, cam_id))
            K, dist = intrinsics.get(cam_id, (None, None))
            if pose_entry and K is not None:
                img = overlay_reprojection(
                    img,
                    pose_entry,
                    tag_length,
                    tag_poses,
                    detector,
                    K,
                    dist,
                )
            status = "missing"
            if pose_entry and isinstance(pose_entry, dict):
                status = str(pose_entry.get("status", "unknown"))
            cv2.putText(
                img,
                cam_id,
                (10, tile_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                cam_id,
                (10, tile_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"status: {status}",
                (10, tile_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"status: {status}",
                (10, tile_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            tiles.append(img)

        # Build mosaic
        rows_imgs = []
        for r in range(rows_count):
            row_tiles = tiles[r * col_count : (r + 1) * col_count]
            while len(row_tiles) < col_count:
                row_tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            rows_imgs.append(np.hstack(row_tiles))
        mosaic = np.vstack(rows_imgs)

        # Update 3D pose view
        cam_map = {cid: poses.get((frame_idx, cid)) for cid in cam_ids}
        positions = []
        for cam_id, entry in cam_map.items():
            if not entry:
                continue
            T = entry.get("T_W_C")
            if not isinstance(T, np.ndarray):
                continue
            t = T[:3, 3]
            R = T[:3, :3]
            positions.append(t)
            axes = np.eye(3) * 0.05
            for i, axis_key in enumerate(["x", "y", "z"]):
                vec = R @ axes[:, i]
                line = axis_lines[cam_id][i]
                line.set_data([t[0], t[0] + vec[0]], [t[1], t[1] + vec[1]])
                line.set_3d_properties([t[2], t[2] + vec[2]])
        if positions:
            pos = np.vstack(positions)
            cam_scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        else:
            cam_scatter._offsets3d = ([], [], [])
        ax.set_title(f"Camera poses: {session.name} (frame {frame_idx})")
        canvas.draw()
        if hasattr(canvas, "buffer_rgba"):
            buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
            pose_img = buf[:, :, :3]
        elif hasattr(canvas, "tostring_rgb"):
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            pose_img = buf.reshape(pose_h, pose_w, 3)
        else:
            buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            argb = buf.reshape(pose_h, pose_w, 4)
            pose_img = argb[:, :, 1:4]
        if pose_img.shape[0] != pose_h or pose_img.shape[1] != pose_w:
            pose_img = cv2.resize(pose_img, (pose_w, pose_h))
        pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)

        combined = np.hstack([mosaic, pose_img])
        if combined.shape[0] != mosaic_h or combined.shape[1] != (mosaic_w + pose_w):
            combined = cv2.resize(combined, (mosaic_w + pose_w, mosaic_h))

        # Overlay events/annotations
        event_lines = events_for_frame(events, ts_ms, event_window_ms, max_event_lines)
        y = 25
        for line in event_lines:
            cv2.putText(
                combined,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                combined,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y += 24

        writer.write(combined)

    writer.release()
    for cap in video_caps.values():
        if cap is not None:
            cap.release()
    plt.close(fig)
    print(f"[viz] wrote {out_path}")
    update_marker(
        session,
        "session_visualization_full",
        {
            "output": out_path.name,
            "poses": poses_path.name,
            "frames": len(rows),
            "fps": output_fps,
            "stride": stride,
        },
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Ultimate session visualization.")
    parser.add_argument("root", help="Session directory or capture root")
    parser.add_argument("--tag-map", required=True, type=Path, help="Path to apriltag_map.json")
    parser.add_argument("--find-meta", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--poses-name", default="", help="Pose JSON name in session (override auto)")
    parser.add_argument("--output-name", default="calib_vis.mp4", help="Output video name")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--cols", type=int, default=0, help="Grid columns (0 = auto)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale each camera tile")
    parser.add_argument("--pose-width", type=int, default=0, help="3D view width (0 = mosaic height)")
    parser.add_argument("--event-window-ms", type=int, default=400, help="Event window around frame time")
    parser.add_argument("--max-event-lines", type=int, default=6, help="Max event lines to render")
    parser.add_argument("--threads", type=int, default=1, help="AprilTag detector threads")
    args = parser.parse_args()

    tag_map_path = args.tag_map.expanduser().resolve()
    family, tag_length, tag_poses = read_tag_map(tag_map_path)
    tag_polylines = compute_tag_polylines(tag_length, tag_poses)
    root = Path(args.root).expanduser().resolve()
    if (root / "meta.json").exists():
        metas = [root / "meta.json"]
    else:
        find_meta = args.find_meta.lower() == "true"
        metas = list_meta_files(root, find_meta, 2)
    if not metas:
        print("No meta.json found")
        return 1

    wrote_any = False
    any_processed = False
    for meta in metas:
        session = meta.parent
        wrote = render_session(
            session,
            meta,
            family,
            tag_length,
            tag_poses,
            tag_polylines,
            args.poses_name,
            args.output_name,
            args.fps,
            args.stride,
            args.cols,
            args.scale,
            args.pose_width,
            args.event_window_ms,
            args.max_event_lines,
            args.threads,
        )
        any_processed = True
        wrote_any = wrote_any or wrote
    if any_processed and not wrote_any:
        return 0
    return 0 if wrote_any else 1


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
    print(f"[viz] updated marker {marker_path}")


if __name__ == "__main__":
    raise SystemExit(main())
