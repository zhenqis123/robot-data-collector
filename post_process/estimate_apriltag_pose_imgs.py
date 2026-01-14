#!/usr/bin/env python3
"""
Estimate AprilTag marker poses from a RealSense capture directory using pupil-apriltag.

Usage:
  python tools/estimate_apriltag_pose.py \
      --images RealSense_151322070562/color \
      --meta RealSense_151322070562/meta.json \
      --tag-length 0.04 \
      --family tagStandard41h12 \
      --output apriltag_map.json \
      --figure apriltag_map_3d.png

流程概要：
  - 读取 meta.json 的彩色相机内参与畸变。
  - pupil-apriltag 检测指定家族的标记（默认 tagStandard41h12），仅保留允许的 ID。
  - 仅使用同帧检测到 ≥2 个标记的帧，构建同帧相对边并以 tag0 为世界原点求解姿态。
  - 默认假设标记共面：传播时将姿态投影到 z=0、只保留 yaw。
  - 导出 JSON（T_W_M、rvec/tvec、四元数、观测计数、重投影误差中位值）并绘制 3D 图（弹窗 + 可选保存）。
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pupil_apriltags import Detector

REF_TAG_ID = 0
OBJECT_POINTS = np.array(
    [
        [-0.5, 0.5, 0],
        [0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [-0.5, -0.5, 0],
    ],
    dtype=np.float64,
)


def load_intrinsics(meta_path: Path, image_path: Path = None) -> Tuple[np.ndarray, np.ndarray]:
    with meta_path.open("r") as f:
        meta = json.load(f)
    cams = meta.get("cameras", [])
    if not cams:
        raise RuntimeError("No cameras section in meta.json")
    
    selected_cam = None
    if image_path:
        # Simple heuristic: check if serial or ID is in the path string
        path_str = str(image_path)
        for cam in cams:
            serial = cam.get("serial")
            cid = cam.get("id")
            if serial and serial in path_str:
                selected_cam = cam
                break
            if cid and cid in path_str:
                selected_cam = cam
                break
    
    if selected_cam is None:
        if len(cams) > 1 and image_path:
            print(f"[Warn] Could not match image path '{image_path}' to a specific camera in meta.json. Using the first camera.")
        selected_cam = cams[0]

    streams = selected_cam.get("streams", {})
    color = streams.get("color", {}).get("intrinsics", {})
    fx = color.get("fx")
    fy = color.get("fy")
    cx = color.get("cx")
    cy = color.get("cy")
    if None in (fx, fy, cx, cy):
        raise RuntimeError("Missing intrinsics fx/fy/cx/cy in meta.json")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    coeffs = color.get("coeffs", [0, 0, 0, 0, 0])
    dist = np.array(coeffs, dtype=np.float64)
    return K, dist


def average_rotation_matrices(rmats: List[np.ndarray]) -> np.ndarray:
    if not rmats:
        return np.eye(3)
    M = np.zeros((3, 3))
    for R in rmats:
        M += R
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def rotation_matrix_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    trace = np.trace(R)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


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


def compose_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def average_transform(transforms: List[np.ndarray]) -> np.ndarray:
    if not transforms:
        return np.eye(4)
    rmats = [T[:3, :3] for T in transforms]
    tvecs = [T[:3, 3] for T in transforms]
    R_mean = average_rotation_matrices(rmats)
    t_mean = np.mean(np.stack(tvecs, axis=0), axis=0)
    return rt_to_transform(R_mean, t_mean)


def project_pose_to_plane(T: np.ndarray) -> np.ndarray:
    t = T[:3, 3].copy()
    t[2] = 0.0
    R = T[:3, :3]
    yaw = math.atan2(R[1, 0], R[0, 0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rt_to_transform(R_yaw, t)


def parse_allowed_ids(expr: str) -> List[int]:
    ids: List[int] = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            s, e = part.split("-", 1)
            ids.extend(list(range(int(s), int(e) + 1)))
        else:
            ids.append(int(part))
    return ids


def process_images(
    image_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
    tag_length: float,
    family: str,
    allowed_ids: List[int],
    threads: int,
    decimate: float,
    sigma: float,
):
    # Determine source: directory of PNGs or single video file
    frame_list: List[Path] = []
    video_path = None
    is_video = False
    
    if image_path.is_file() and image_path.suffix.lower() in {".mkv", ".mp4", ".avi", ".mov"}:
        is_video = True
        video_path = image_path
    elif image_path.is_dir():
        # Try finding PNGs first
        pngs = sorted(list(image_path.glob("*.png")))
        if pngs:
            frame_list = pngs
        else:
            # Try finding video
            videos = sorted([p for p in image_path.glob("*") if p.suffix.lower() in {".mkv", ".mp4", ".avi", ".mov"}])
            if videos:
                is_video = True
                video_path = videos[0]
                print(f"Using video for tag estimation: {video_path}")
    
    if not frame_list and not is_video:
        raise RuntimeError(f"No PNG images or suitable video found in {image_path}")

    detector = Detector(
        families=family,
        nthreads=threads,
        quad_decimate=decimate,
        quad_sigma=sigma,
        refine_edges=True,
    )

    frames = []

    def detect_in_image(img, frame_name):
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections_raw = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
            tag_size=tag_length,
        )
        filtered = []
        scaled_obj = OBJECT_POINTS * tag_length
        for det in detections_raw:
            if allowed_ids and det.tag_id not in allowed_ids:
                continue
            if det.pose_R is None or det.pose_t is None:
                continue
            R = det.pose_R
            t = det.pose_t.reshape(3)
            # 固定坐标系修正：x 平面内，y 平面内，z 指向相机
            R_FIX = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1],
            ], dtype=np.float64)
            R = R @ R_FIX   # ★关键：右乘，表示“重新定义 tag 坐标系”
            T_c_m = rt_to_transform(R, t)
            proj, _ = cv2.projectPoints(scaled_obj, rotation_matrix_to_rvec(R), t, K, dist)
            err = np.linalg.norm(proj.reshape(-1, 2) - np.array(det.corners), axis=1)
            filtered.append(
                {
                    "id": int(det.tag_id),
                    "T_C_M": T_c_m,
                    "median_error_px": float(np.median(err)),
                    "frame": frame_name,
                    "corners": det.corners,
                }
            )
        if len(filtered) < 2:
            return None
        return {"frame": frame_name, "detections": filtered}

    if is_video:
        cap = cv2.VideoCapture(str(video_path))
        stride = 5 # Process every 5th frame
        frame_idx = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                res = detect_in_image(img, f"frame_{frame_idx:06d}")
                if res:
                    frames.append(res)
            frame_idx += 1
        cap.release()
    else:
        for img_path in frame_list:
            img = cv2.imread(str(img_path))
            res = detect_in_image(img, img_path.name)
            if res:
                frames.append(res)
                
    return frames


def build_pose_graph(frames: List[Dict]) -> Dict[int, Dict[int, List[np.ndarray]]]:
    edges: Dict[int, Dict[int, List[np.ndarray]]] = {}
    for frame in frames:
        detections = frame["detections"]
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                a = detections[i]
                b = detections[j]
                T_mi_mj = compose_transform(invert_transform(a["T_C_M"]), b["T_C_M"])
                T_mj_mi = invert_transform(T_mi_mj)
                edges.setdefault(a["id"], {}).setdefault(b["id"], []).append(T_mi_mj)
                edges.setdefault(b["id"], {}).setdefault(a["id"], []).append(T_mj_mi)
    return edges


def solve_poses(edges: Dict[int, Dict[int, List[np.ndarray]]], planar: bool) -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {REF_TAG_ID: np.eye(4)}
    queue = [REF_TAG_ID]
    visited = set(queue)
    while queue:
        current = queue.pop(0)
        for neighbor, transforms in edges.get(current, {}).items():
            if neighbor in visited:
                continue
            T_curr_neigh = average_transform(transforms)
            candidate = compose_transform(poses[current], T_curr_neigh)
            poses[neighbor] = project_pose_to_plane(candidate) if planar else candidate
            visited.add(neighbor)
            queue.append(neighbor)
    return poses


def collect_stats(frames: List[Dict]) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    for frame in frames:
        for det in frame["detections"]:
            mid = det["id"]
            stats.setdefault(mid, {"count": 0, "errors": []})
            stats[mid]["count"] += 1
            stats[mid]["errors"].append(det["median_error_px"])
    for mid, s in stats.items():
        s["median_reproj_error_px"] = float(np.median(s["errors"])) if s["errors"] else None
    return stats


def plot_map(poses: Dict[int, np.ndarray], tag_length: float, output_path: Path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    axis_len = tag_length * 0.5
    colors = ["r", "g", "b"]

    for mid, T in poses.items():
        origin = T[:3, 3]
        R = T[:3, :3]
        axes = [R[:, i] * axis_len for i in range(3)]
        for i, vec in enumerate(axes):
            end = origin + vec
            ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], color=colors[i])
        ax.scatter(origin[0], origin[1], origin[2], color="k", s=20)
        ax.text(origin[0], origin[1], origin[2], f"{mid}", fontsize=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("AprilTag Map (Tag0 as World Origin)")
    if poses:
        points = np.array([T[:3, 3] for T in poses.values()])
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        max_range = (maxs - mins).max() or 0.1
        mid = (maxs + mins) / 2.0
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved 3D map visualization to {output_path}")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Estimate AprilTag map with tag0 as world origin (pupil-apriltag)")
    parser.add_argument("--images", required=True, type=Path, help="Directory with color PNGs")
    parser.add_argument("--meta", required=True, type=Path, help="Path to meta.json with intrinsics")
    parser.add_argument("--tag-length", required=True, type=float, help="Tag side length in meters")
    parser.add_argument("--family", type=str, default="tagStandard41h12", help="AprilTag family (e.g., tag36h11)")
    parser.add_argument("--allowed-ids", type=str, default="0-9", help="Comma/range list, e.g., 0-9,12,15")
    parser.add_argument("--output", type=Path, default=Path("apriltag_map.json"), help="Output JSON path")
    parser.add_argument("--figure", type=Path, default=Path("apriltag_map_3d.png"), help="Output 3D figure path")
    parser.add_argument("--threads", type=int, default=0, help="Detector threads (0 = hardware concurrency)")
    parser.add_argument("--decimate", type=float, default=1.0, help="quad_decimate for speed/accuracy tradeoff")
    parser.add_argument("--sigma", type=float, default=0.0, help="quad_sigma (Gaussian blur) for robustness")
    parser.add_argument("--no-planar", action="store_true", help="Disable planar projection (keep full 6DoF)")
    args = parser.parse_args()

    K, dist = load_intrinsics(args.meta, args.images)
    allowed_ids = parse_allowed_ids(args.allowed_ids)
    frames = process_images(
        args.images,
        K,
        dist,
        args.tag_length,
        args.family,
        allowed_ids,
        args.threads if args.threads > 0 else max(1, cv2.getNumberOfCPUs()),
        args.decimate,
        args.sigma,
    )
    edges = build_pose_graph(frames)
    poses = solve_poses(edges, planar=not args.no_planar)
    stats = collect_stats(frames)

    markers_out = {}
    for mid, T in poses.items():
        R = T[:3, :3]
        t = T[:3, 3]
        markers_out[str(mid)] = {
            "T_W_M": T.tolist(),
            "rvec": rotation_matrix_to_rvec(R).tolist(),
            "tvec_m": t.tolist(),
            "quaternion_wxyz": rotation_matrix_to_quaternion(R).tolist(),
            "observations": stats.get(mid, {}).get("count", 0),
            "median_reproj_error_px": stats.get(mid, {}).get("median_reproj_error_px"),
        }

    out = {
        "detector": "pupil-apriltag",
        "family": args.family,
        "ref_id": REF_TAG_ID,
        "tag_length_m": args.tag_length,
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "markers": markers_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved poses for {len(markers_out)} tags to {args.output}")
    update_marker(
        args.meta.parent,
        "apriltag_map",
        {
            "tag_map": str(args.output),
            "figure": str(args.figure),
            "images": str(args.images),
            "family": args.family,
            "tag_length_m": args.tag_length,
        },
    )

    plot_map(poses, args.tag_length, args.figure)


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
    print(f"[apriltag] updated marker {marker_path}")


if __name__ == "__main__":
    main()
