#!/usr/bin/env python3
"""
Visualize AprilTag pose results by overlaying detections and reprojections.

Usage:
  python tools/visualize_apriltag_pose_results.py \
      --poses /path/to/camera_poses_apriltag.json \
      --tag-map /path/to/apriltag_map.json \
      --count 10 \
      --output-dir /path/to/output
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector

OBJECT_POINTS = np.array(
    [
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
    ],
    dtype=np.float64,
)


def rotation_matrix_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def rt_to_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def sanitize_camera_id(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def read_tag_map(tag_map_path: Path) -> Tuple[str, float, Dict[int, np.ndarray]]:
    with tag_map_path.open("r") as f:
        data = json.load(f)
    family = data.get("family")
    tag_length = float(data.get("tag_length_m", 0.0))
    markers = data.get("markers", {})
    if not family or not tag_length or not markers:
        raise RuntimeError("Invalid tag map: missing family/tag_length/markers")
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
    with meta_path.open("r") as f:
        meta = json.load(f)
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


def resolve_color_path(capture_root: Path, camera_id: str, color_rel: str) -> Path:
    p = Path(color_rel)
    if p.is_absolute():
        return p
    cam_dir = sanitize_camera_id(camera_id)
    return capture_root / cam_dir / p


def compute_world_corners(tag_poses: Dict[int, np.ndarray], tag_length: float) -> Dict[int, np.ndarray]:
    corners: Dict[int, np.ndarray] = {}
    scaled_obj = OBJECT_POINTS * tag_length
    for mid, T_W_M in tag_poses.items():
        R = T_W_M[:3, :3]
        t = T_W_M[:3, 3]
        world_corners = (R @ scaled_obj.T).T + t
        corners[mid] = world_corners
    return corners


def draw_polyline(image: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], thickness: int) -> None:
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts_i], isClosed=True, color=color, thickness=thickness)


def draw_points(image: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], radius: int) -> None:
    for p in pts:
        cv2.circle(image, (int(round(p[0])), int(round(p[1]))), radius, color, -1)


def draw_text_lines(
    image: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
    line_gap: int = 4,
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += int(14 * font_scale + line_gap)


def project_world_points(
    world_points: np.ndarray,
    T_W_C: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    T_C_W = invert_transform(T_W_C)
    R_c_w = T_C_W[:3, :3]
    t_c_w = T_C_W[:3, 3]
    rvec = rotation_matrix_to_rvec(R_c_w)
    proj, _ = cv2.projectPoints(world_points, rvec, t_c_w, K, dist)
    return proj.reshape(-1, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize AprilTag pose results with detections and reprojections.")
    parser.add_argument("--poses", required=True, type=Path, help="Path to camera_poses_apriltag.json")
    parser.add_argument("--tag-map", required=True, type=Path, help="Path to apriltag_map.json")
    parser.add_argument("--count", type=int, default=10, help="Number of random frames to visualize")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save visualizations")
    parser.add_argument("--threads", type=int, default=0, help="Detector threads (0 = hardware concurrency)")
    parser.add_argument("--decimate", type=float, default=1.0, help="quad_decimate for speed/accuracy tradeoff")
    parser.add_argument("--sigma", type=float, default=0.0, help="quad_sigma (Gaussian blur) for robustness")
    args = parser.parse_args()

    poses_path = args.poses.expanduser().resolve()
    tag_map_path = args.tag_map.expanduser().resolve()
    if not poses_path.exists():
        print(f"[viz] poses not found: {poses_path}")
        return 2
    if not tag_map_path.exists():
        print(f"[viz] tag map not found: {tag_map_path}")
        return 2

    with poses_path.open("r") as f:
        poses_data = json.load(f)
    capture_root = Path(poses_data.get("capture_root", poses_path.parent)).expanduser().resolve()
    meta_path = capture_root / "meta.json"
    if not meta_path.exists():
        print(f"[viz] meta.json not found: {meta_path}")
        return 2

    try:
        family, tag_length, tag_poses = read_tag_map(tag_map_path)
    except RuntimeError as exc:
        print(f"[viz] failed to read tag map: {exc}")
        return 2

    intrinsics = load_intrinsics(meta_path)
    world_corners = compute_world_corners(tag_poses, tag_length)
    tag_ids = set(tag_poses.keys())

    if args.output_dir is None:
        output_dir = poses_path.parent / "pose_viz"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        entry
        for entry in poses_data.get("poses", [])
        if entry.get("status") == "ok" and entry.get("color_path")
    ]
    if not candidates:
        print("[viz] no valid poses to visualize")
        return 1

    if args.seed is not None:
        random.seed(args.seed)
    sample_count = min(args.count, len(candidates))
    samples = random.sample(candidates, sample_count)

    threads = args.threads if args.threads > 0 else max(1, cv2.getNumberOfCPUs())
    detector = Detector(
        families=family,
        nthreads=threads,
        quad_decimate=args.decimate,
        quad_sigma=args.sigma,
        refine_edges=True,
    )

    saved = 0
    for idx, entry in enumerate(samples, start=1):
        cam_id = entry.get("camera_id")
        color_rel = entry.get("color_path")
        T_W_C_list = entry.get("T_W_C")
        if cam_id is None or color_rel is None or T_W_C_list is None:
            continue
        if cam_id not in intrinsics:
            print(f"[viz] missing intrinsics for camera {cam_id}")
            continue
        color_path = resolve_color_path(capture_root, cam_id, color_rel)
        if not color_path.exists():
            print(f"[viz] image not found: {color_path}")
            continue
        image = cv2.imread(str(color_path))
        if image is None:
            print(f"[viz] failed to read image: {color_path}")
            continue

        K, dist = intrinsics[cam_id]
        T_W_C = np.array(T_W_C_list, dtype=np.float64)
        if T_W_C.shape != (4, 4):
            print(f"[viz] invalid T_W_C for {cam_id} at {color_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)
        detections = [det for det in detections if int(det.tag_id) in tag_ids]

        overlay = image.copy()
        reproj_stats = entry.get("reproj_error_px", {}) or {}
        err_lines = [
            f"reproj mean: {reproj_stats.get('mean')}",
            f"reproj median: {reproj_stats.get('median')}",
            f"reproj max: {reproj_stats.get('max')}",
            f"reproj count: {reproj_stats.get('count')}",
        ]
        draw_text_lines(overlay, err_lines, (12, 24), (255, 255, 255), font_scale=0.5, thickness=1)
        for det in detections:
            corners = np.array(det.corners, dtype=np.float64)
            draw_polyline(overlay, corners, (0, 200, 0), 2)
            center = np.mean(corners, axis=0)
            cv2.putText(
                overlay,
                f"id:{int(det.tag_id)}",
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )

        for mid, world_pts in world_corners.items():
            proj = project_world_points(world_pts, T_W_C, K, dist)
            draw_polyline(overlay, proj, (0, 0, 255), 1)
            draw_points(overlay, proj, (0, 0, 255), 2)

        label = f"{cam_id}_frame{entry.get('frame_index', idx)}"
        out_path = output_dir / f"{sanitize_camera_id(label)}.png"
        cv2.imwrite(str(out_path), overlay)
        saved += 1
        print(f"[viz] saved {out_path}")

    print(f"[viz] done, saved {saved} images to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
