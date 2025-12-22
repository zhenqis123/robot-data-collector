#!/usr/bin/env python3
"""
Estimate AprilTag poses from a single image using pupil-apriltag.

Usage:
  python tools/estimate_apriltag_pose_single.py \
      --image image.png \
      --meta meta.json \
      --tag-length 0.04 \
      --family tagStandard41h12 \
      --output apriltag_map.json

Notes:
  - Uses tag0 as world origin by default (same as estimate_apriltag_pose.py).
  - Output JSON format matches estimate_apriltag_pose.py.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

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


def load_intrinsics(meta_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with meta_path.open("r") as f:
        meta = json.load(f)
    cams = meta.get("cameras", [])
    if not cams:
        raise RuntimeError("No cameras section in meta.json")
    streams = cams[0].get("streams", {})
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


def match_projected_corners(det_corners: np.ndarray, proj_corners: np.ndarray) -> Tuple[np.ndarray, float]:
    det = np.array(det_corners, dtype=np.float64).reshape(-1, 2)
    proj = np.array(proj_corners, dtype=np.float64).reshape(-1, 2)
    best_err = float("inf")
    best_proj = proj
    for flip in (False, True):
        cand = proj[::-1] if flip else proj
        for shift in range(4):
            shifted = np.roll(cand, shift, axis=0)
            err = np.linalg.norm(shifted - det, axis=1)
            med = float(np.median(err))
            if med < best_err:
                best_err = med
                best_proj = shifted
    return best_proj, best_err


def detect_tags(
    image_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
    tag_length: float,
    family: str,
    allowed_ids: List[int],
    threads: int,
    decimate: float,
    sigma: float,
) -> List[Dict]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = Detector(
        families=family,
        nthreads=threads,
        quad_decimate=decimate,
        quad_sigma=sigma,
        refine_edges=True,
    )
    detections_raw = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
        tag_size=tag_length,
    )
    scaled_obj = OBJECT_POINTS * tag_length
    detections = []
    for det in detections_raw:
        if allowed_ids and det.tag_id not in allowed_ids:
            continue
        if det.pose_R is None or det.pose_t is None:
            continue
        R = det.pose_R
        t = det.pose_t.reshape(3)
        # Coordinate fix: keep x/y in plane, z toward camera.
        R_FIX = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )
        R = R @ R_FIX
        T_c_m = rt_to_transform(R, t)
        proj, _ = cv2.projectPoints(
            scaled_obj, rotation_matrix_to_rvec(R), t, K, dist
        )
        corners = np.array(det.corners, dtype=np.float64).reshape(-1, 2)
        proj = proj.reshape(-1, 2)
        proj_matched, median_err = match_projected_corners(corners, proj)
        detections.append(
            {
                "id": int(det.tag_id),
                "T_C_M": T_c_m,
                "median_error_px": float(median_err),
                "corners": corners,
                "proj": proj_matched,
            }
        )
    return detections, img


def draw_overlay(image: np.ndarray, detections: List[Dict], output_path: Path):
    canvas = image.copy()
    for det in detections:
        corners = np.array(det["corners"], dtype=np.float32).reshape(-1, 2)
        proj = np.array(det["proj"], dtype=np.float32).reshape(-1, 2)
        # Projected corners: red (draw first), detected corners: green (draw last).
        cv2.polylines(canvas, [proj.astype(np.int32)], True, (0, 0, 255), 2)
        for pt in proj:
            p = tuple(pt.astype(int))
            cv2.drawMarker(canvas, p, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
        cv2.polylines(canvas, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        for pt in corners:
            p = tuple(pt.astype(int))
            cv2.circle(canvas, p, 6, (0, 0, 0), 2)
            cv2.circle(canvas, p, 4, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            f"id {det['id']}",
            tuple(corners[0].astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"Saved overlay visualization to {output_path}")


def plot_map(
    poses: Dict[int, np.ndarray],
    tag_length: float,
    output_path: Path,
    show: bool,
    camera_pose: np.ndarray | None,
):
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

    if camera_pose is not None:
        cam_origin = camera_pose[:3, 3]
        R = camera_pose[:3, :3]
        cam_axis_len = tag_length * 0.75
        axes = [R[:, i] * cam_axis_len for i in range(3)]
        for i, vec in enumerate(axes):
            end = cam_origin + vec
            ax.plot([cam_origin[0], end[0]], [cam_origin[1], end[1]], [cam_origin[2], end[2]], color=colors[i])
        ax.scatter(cam_origin[0], cam_origin[1], cam_origin[2], color="c", s=40, marker="^")
        ax.text(cam_origin[0], cam_origin[1], cam_origin[2], "Camera", fontsize=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("AprilTag Map (Tag as World Origin)")
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
    if show:
        plt.show()
    plt.close(fig)


def camera_pose_world(detections: List[Dict], ref_id: int) -> np.ndarray:
    ref_det = None
    for det in detections:
        if det["id"] == ref_id:
            ref_det = det
            break
    if ref_det is None:
        raise RuntimeError(f"Reference tag id {ref_id} not detected")
    T_c_ref = ref_det["T_C_M"]
    return invert_transform(T_c_ref)


def distance_between_markers(poses: Dict[int, np.ndarray], id_a: int, id_b: int) -> float:
    if id_a not in poses or id_b not in poses:
        raise KeyError(f"Missing marker ids in poses: {id_a}, {id_b}")
    a = poses[id_a][:3, 3]
    b = poses[id_b][:3, 3]
    return float(np.linalg.norm(a - b))


def solve_poses_single(
    detections: List[Dict],
    ref_id: int,
    planar: bool,
) -> Dict[int, np.ndarray]:
    ref_det = None
    for det in detections:
        if det["id"] == ref_id:
            ref_det = det
            break
    if ref_det is None:
        raise RuntimeError(f"Reference tag id {ref_id} not detected")
    T_c_ref = ref_det["T_C_M"]
    T_ref_c = invert_transform(T_c_ref)
    poses: Dict[int, np.ndarray] = {}
    for det in detections:
        T_c_m = det["T_C_M"]
        T_ref_m = T_ref_c @ T_c_m
        poses[det["id"]] = project_pose_to_plane(T_ref_m) if planar else T_ref_m
    return poses


def main():
    parser = argparse.ArgumentParser(
        description="Estimate AprilTag poses from a single image (pupil-apriltag)"
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--meta", required=True, type=Path, help="Path to meta.json with intrinsics")
    parser.add_argument("--tag-length", required=True, type=float, help="Tag side length in meters")
    parser.add_argument("--family", type=str, default="tagStandard41h12", help="AprilTag family")
    parser.add_argument("--allowed-ids", type=str, default="0-9", help="Comma/range list, e.g., 0-9,12,15")
    parser.add_argument("--ref-id", type=int, default=REF_TAG_ID, help="Reference tag id (world origin)")
    parser.add_argument("--output", type=Path, default=Path("apriltag_map.json"), help="Output JSON path")
    parser.add_argument("--vis-out", type=Path, default=Path("apriltag_overlay.png"),
                        help="Output overlay image path")
    parser.add_argument("--figure", type=Path, default=Path("apriltag_map_3d.png"),
                        help="Output 3D figure path")
    parser.add_argument("--no-show-3d", action="store_true",
                        help="Do not open interactive 3D plot window")
    parser.add_argument("--camera-dist-ids", type=str, default="0-3",
                        help="Marker ids for camera distance (comma/range list)")
    parser.add_argument("--threads", type=int, default=0, help="Detector threads (0 = hardware concurrency)")
    parser.add_argument("--decimate", type=float, default=1.0, help="quad_decimate for speed/accuracy tradeoff")
    parser.add_argument("--sigma", type=float, default=0.0, help="quad_sigma (Gaussian blur) for robustness")
    parser.add_argument("--no-planar", action="store_true", help="Disable planar projection (keep full 6DoF)")
    args = parser.parse_args()

    K, dist = load_intrinsics(args.meta)
    allowed_ids = parse_allowed_ids(args.allowed_ids)
    detections, image = detect_tags(
        args.image,
        K,
        dist,
        args.tag_length,
        args.family,
        allowed_ids,
        args.threads if args.threads > 0 else max(1, cv2.getNumberOfCPUs()),
        args.decimate,
        args.sigma,
    )
    if not detections:
        raise RuntimeError("No valid AprilTag detections in image")
    draw_overlay(image, detections, args.vis_out)
    poses = solve_poses_single(detections, args.ref_id, planar=not args.no_planar)
    cam_pose = camera_pose_world(detections, args.ref_id)

    stats = {det["id"]: {"count": 1, "median_reproj_error_px": det["median_error_px"]} for det in detections}
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
        "ref_id": args.ref_id,
        "tag_length_m": args.tag_length,
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "markers": markers_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved poses for {len(markers_out)} tags to {args.output}")
    cam_pos = cam_pose[:3, 3]
    print(f"Camera position (world): [{cam_pos[0]:.6f}, {cam_pos[1]:.6f}, {cam_pos[2]:.6f}] m")
    camera_dist_ids = parse_allowed_ids(args.camera_dist_ids)
    if len(camera_dist_ids) > 4:
        camera_dist_ids = camera_dist_ids[:4]
    for mid in camera_dist_ids:
        if mid in poses:
            dist = float(np.linalg.norm(poses[mid][:3, 3] - cam_pos))
            print(f"Camera -> marker {mid}: {dist:.6f} m")
        else:
            print(f"Camera -> marker {mid}: unavailable (missing tag id)")
    try:
        dist_34 = distance_between_markers(poses, 3, 4)
        print(f"Distance between marker 3 and 4: {dist_34:.6f} m")
    except KeyError:
        print("Distance between marker 3 and 4: unavailable (missing tag id)")
    plot_map(poses, args.tag_length, args.figure, show=not args.no_show_3d, camera_pose=cam_pose)


if __name__ == "__main__":
    main()
