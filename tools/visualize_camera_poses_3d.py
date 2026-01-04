#!/usr/bin/env python3
"""
Render AprilTag map + camera poses in a 3D matplotlib view and save as a video.

Usage:
  python tools/visualize_camera_poses_3d.py /path/to/captures \
      --tag-map /path/to/apriltag_map.json
"""
from __future__ import annotations

import argparse
import csv
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
import cv2
from pupil_apriltags import Detector


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
        intrinsics[cid] = (K, dist)
    return intrinsics


def sanitize_camera_id(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def resolve_color_path(capture_root: Path, camera_id: str, color_rel: str) -> Path:
    p = Path(color_rel)
    if p.is_absolute():
        return p
    cam_dir = sanitize_camera_id(camera_id)
    return capture_root / cam_dir / p


def parse_frame_index(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in {".mkv", ".mp4", ".avi", ".mov"}


def prefer_mp4_path(path: Path) -> Path:
    if path.suffix.lower() == ".mkv":
        mp4 = path.with_suffix(".mp4")
        if mp4.exists():
            return mp4
    return path


def read_video_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    if frame_index <= 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


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


def load_pose_frames(poses_path: Path) -> Tuple[List[int], Dict[int, Dict[str, Dict[str, object]]]]:
    data = json.loads(poses_path.read_text())
    frames: Dict[int, Dict[str, Dict[str, object]]] = {}
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
        frames.setdefault(frame_index, {})[cam_id] = {
            "T_W_C": T,
            "reproj_error_px": entry.get("reproj_error_px"),
            "num_tags": entry.get("num_tags"),
            "used_tag_ids": entry.get("used_tag_ids"),
        }
    frame_ids = sorted(frames.keys())
    return frame_ids, frames


def load_frames_csv(frames_csv: Path) -> Dict[int, Dict[str, object]]:
    frames: Dict[int, Dict[str, object]] = {}
    if not frames_csv.exists():
        return frames
    with frames_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            frames[idx] = {
                "ref_camera": row.get("ref_camera", ""),
                "ref_color": row.get("ref_color", ""),
                "ref_frame_index": parse_frame_index(row.get("ref_frame_index", "")),
            }
    return frames


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def overlay_reprojection(
    image: np.ndarray,
    T_W_C: np.ndarray,
    tag_length: float,
    tag_poses: Dict[int, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    detector: Detector,
    reproj_stats: Optional[Dict[str, float]],
) -> np.ndarray:
    canvas = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    det_ids = {int(det.tag_id) for det in detections}
    scaled = OBJECT_POINTS * tag_length
    scaled = scaled.copy()
    scaled[:, 1] *= -1.0
    scaled[:, 2] *= -1.0
    T_C_W = invert_transform(T_W_C)
    rvec, _ = cv2.Rodrigues(T_C_W[:3, :3])
    tvec = T_C_W[:3, 3].reshape(3)

    for det in detections:
        corners = np.array(det.corners, dtype=np.float64).reshape(-1, 2)
        cv2.polylines(canvas, [corners.astype(np.int32)], True, (0, 255, 0), 2)

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

    if reproj_stats:
        mean = reproj_stats.get("mean")
        median = reproj_stats.get("median")
        max_err = reproj_stats.get("max")
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
    meta_path: Path,
    poses_path: Path,
    tag_polylines: Dict[int, np.ndarray],
    tag_length: float,
    tag_poses: Dict[int, np.ndarray],
    family: str,
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
    frames_csv = capture_root / "frames_aligned.csv"
    csv_frames = load_frames_csv(frames_csv)
    if not csv_frames:
        print(f"[pose3d] missing frames_aligned.csv in {capture_root}")
        return False
    frame_ids = [fid for fid in frame_ids if fid in csv_frames]
    frame_ids = frame_ids[:: max(1, stride)]

    cam_ids = sorted({cid for cam_map in frames.values() for cid in cam_map.keys()})
    if not cam_ids:
        print(f"[pose3d] no camera ids in {poses_path}")
        return False

    intrinsics = load_intrinsics(meta_path)
    if not intrinsics:
        print(f"[pose3d] no intrinsics found in {meta_path}")
        return False

    all_points: List[np.ndarray] = []
    for poly in tag_polylines.values():
        all_points.append(poly)
    for cam_map in frames.values():
        for entry in cam_map.values():
            T = entry.get("T_W_C")
            if isinstance(T, np.ndarray):
                all_points.append(T[:3, 3].reshape(1, 3))
    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3))

    fig = plt.figure(figsize=(12, 6))
    ax_img = fig.add_subplot(1, 2, 1)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax_img.axis("off")
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

    detector = Detector(
        families=family,
        nthreads=max(1, cv2.getNumberOfCPUs()),
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
    )
    img_artist = ax_img.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
    video_caps: Dict[Path, cv2.VideoCapture] = {}

    def update(frame_idx: int):
        cam_map = frames.get(frame_idx, {})
        positions = []
        for cam_id, T in cam_map.items():
            if not isinstance(T, dict):
                continue
            T_w_c = T.get("T_W_C")
            if not isinstance(T_w_c, np.ndarray):
                continue
            t = T_w_c[:3, 3]
            R = T_w_c[:3, :3]
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

        csv_entry = csv_frames.get(frame_idx, {})
        ref_cam = str(csv_entry.get("ref_camera", ""))
        color_rel = str(csv_entry.get("ref_color", ""))
        frame_number = csv_entry.get("ref_frame_index")
        img = None
        if ref_cam and color_rel:
            color_path = prefer_mp4_path(resolve_color_path(capture_root, ref_cam, color_rel))
            if color_path.exists():
                if is_video_path(color_path):
                    if frame_number is not None:
                        cap = video_caps.get(color_path)
                        if cap is None or not cap.isOpened():
                            cap = cv2.VideoCapture(str(color_path))
                            video_caps[color_path] = cap
                        img = read_video_frame(cap, frame_number) if cap is not None else None
                else:
                    img = cv2.imread(str(color_path))

        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                img,
                "missing image",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            img_artist.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return [cam_scatter, img_artist]

        K, dist = intrinsics.get(ref_cam, (None, None))
        pose_entry = cam_map.get(ref_cam)
        if K is None or dist is None or not isinstance(pose_entry, dict):
            img_artist.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return [cam_scatter, img_artist]

        T_w_c = pose_entry.get("T_W_C")
        if not isinstance(T_w_c, np.ndarray):
            img_artist.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return [cam_scatter, img_artist]

        reproj_stats = pose_entry.get("reproj_error_px")
        canvas = overlay_reprojection(
            img,
            T_w_c,
            tag_length,
            tag_poses,
            K,
            dist,
            detector,
            reproj_stats if isinstance(reproj_stats, dict) else None,
        )
        img_artist.set_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        return [cam_scatter, img_artist]

    anim = animation.FuncAnimation(fig, update, frames=frame_ids, interval=1000 / max(1, fps))
    out_path = capture_root / output_name
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    for cap in video_caps.values():
        cap.release()
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
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--axis-length", type=float, default=0.1, help="Camera axis length (meters)")
    args = parser.parse_args()

    tag_map_path = args.tag_map.expanduser().resolve()
    if not tag_map_path.exists():
        print(f"[pose3d] tag map not found: {tag_map_path}")
        return 2
    try:
        family, tag_length, tag_poses = read_tag_map(tag_map_path)
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
            meta,
            poses_path,
            tag_polylines,
            tag_length,
            tag_poses,
            family,
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
