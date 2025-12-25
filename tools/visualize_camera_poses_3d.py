#!/usr/bin/env python3
"""
Render AprilTag map + camera poses in a 3D matplotlib view and save as a video.

Usage:
  python tools/visualize_camera_poses_3d.py /path/to/captures \
      --tag-map /path/to/apriltag_map.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


OBJECT_POINTS = np.array(
    [
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ],
    dtype=np.float64,
)


def read_tag_map(tag_map_path: Path) -> Tuple[float, Dict[int, np.ndarray]]:
    data = json.loads(tag_map_path.read_text())
    tag_length = float(data.get("tag_length_m", 0.0))
    markers = data.get("markers", {})
    if not tag_length or not markers:
        raise RuntimeError("Invalid tag map: missing tag_length_m/markers")
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
    return tag_length, tag_poses


def compute_tag_polylines(tag_length: float, tag_poses: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    polylines: Dict[int, np.ndarray] = {}
    scaled = OBJECT_POINTS * tag_length
    for mid, T_W_M in tag_poses.items():
        R = T_W_M[:3, :3]
        t = T_W_M[:3, 3]
        poly = (R @ scaled.T).T + t
        polylines[mid] = poly
    return polylines


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


def load_pose_frames(poses_path: Path) -> Tuple[List[int], Dict[int, Dict[str, np.ndarray]]]:
    data = json.loads(poses_path.read_text())
    frames: Dict[int, Dict[str, np.ndarray]] = {}
    for entry in data.get("poses", []):
        if entry.get("status") != "ok":
            continue
        frame_index = entry.get("frame_index")
        if not isinstance(frame_index, int):
            continue
        cam_id = str(entry.get("camera_id", ""))
        T_list = entry.get("T_W_C")
        if not cam_id or not T_list:
            continue
        T = np.array(T_list, dtype=np.float64)
        if T.shape != (4, 4):
            continue
        frames.setdefault(frame_index, {})[cam_id] = T
    frame_ids = sorted(frames.keys())
    return frame_ids, frames


def set_equal_axes(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max()
    if span <= 0:
        span = 1.0
    half = span / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def render_capture(
    capture_root: Path,
    poses_path: Path,
    tag_polylines: Dict[int, np.ndarray],
    output_name: str,
    fps: int,
    stride: int,
    axis_len: float,
) -> bool:
    if should_skip_step(capture_root, "visualize_camera_poses_3d"):
        print(f"[pose3d] skip {capture_root}, already visualized")
        return False
    frame_ids, frames = load_pose_frames(poses_path)
    if not frame_ids:
        print(f"[pose3d] no valid poses in {poses_path}")
        return False
    frame_ids = frame_ids[:: max(1, stride)]

    cam_ids = sorted({cid for cam_map in frames.values() for cid in cam_map.keys()})
    if not cam_ids:
        print(f"[pose3d] no camera ids in {poses_path}")
        return False

    all_points: List[np.ndarray] = []
    for poly in tag_polylines.values():
        all_points.append(poly)
    for cam_map in frames.values():
        for T in cam_map.values():
            all_points.append(T[:3, 3].reshape(1, 3))
    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Camera poses: {capture_root.name}")
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

    set_equal_axes(ax, points)

    def update(frame_idx: int):
        cam_map = frames.get(frame_idx, {})
        positions = []
        for cam_id, T in cam_map.items():
            t = T[:3, 3]
            R = T[:3, :3]
            positions.append(t)
            axes = np.eye(3) * axis_len
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
        ax.set_title(f"Camera poses: {capture_root.name} (frame {frame_idx})")
        return [cam_scatter]

    anim = animation.FuncAnimation(fig, update, frames=frame_ids, interval=1000 / max(1, fps))
    out_path = capture_root / output_name
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"[pose3d] wrote {out_path}")
    update_marker(
        capture_root,
        "visualize_camera_poses_3d",
        {
            "output": out_path.name,
            "frames": len(frame_ids),
            "stride": stride,
            "fps": fps,
        },
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Render 3D camera poses + AprilTag map to video.")
    parser.add_argument("root", help="Root directory containing capture folders or meta.json")
    parser.add_argument("--tag-map", required=True, type=Path, help="Path to apriltag_map.json")
    parser.add_argument("--find-meta", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--poses-name", default="camera_poses_apriltag.json", help="Pose JSON name per capture")
    parser.add_argument("--output-name", default="camera_poses_3d.mp4", help="Output video name per capture")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--axis-length", type=float, default=0.05, help="Camera axis length (meters)")
    args = parser.parse_args()

    tag_map_path = args.tag_map.expanduser().resolve()
    if not tag_map_path.exists():
        print(f"[pose3d] tag map not found: {tag_map_path}")
        return 2
    try:
        tag_length, tag_poses = read_tag_map(tag_map_path)
    except RuntimeError as exc:
        print(f"[pose3d] failed to read tag map: {exc}")
        return 2
    tag_polylines = compute_tag_polylines(tag_length, tag_poses)

    root = Path(args.root).expanduser().resolve()
    find_meta = args.find_meta.lower() == "true"
    metas = list_meta_files(root, find_meta, 2)
    if not metas:
        print("No meta.json found")
        return 1

    wrote_any = False
    for meta in metas:
        capture_root = meta.parent
        poses_path = capture_root / args.poses_name
        if not poses_path.exists():
            print(f"[pose3d] skip {capture_root}, missing {poses_path.name}")
            continue
        wrote = render_capture(
            capture_root,
            poses_path,
            tag_polylines,
            args.output_name,
            args.fps,
            args.stride,
            args.axis_length,
        )
        wrote_any = wrote_any or wrote
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
    print(f"[pose3d] updated marker {marker_path}")


if __name__ == "__main__":
    raise SystemExit(main())
