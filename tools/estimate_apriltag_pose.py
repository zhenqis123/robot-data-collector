#!/usr/bin/env python3
"""
Estimate AprilTag marker poses from a RealSense .bag recording using pupil-apriltag.

Usage:
  python tools/estimate_apriltag_pose.py \
      --bag 20260117_152613.bag \
      --tag-length 0.04 \
      --family tagStandard41h12 \
      --output apriltag_map.json \
      --figure apriltag_map_3d.png

Notes:
  - Uses tag2 as world origin by default (same as estimate_apriltag_pose_single.py when ref-id=2).
  - Only frames containing all required tags and passing clarity thresholds are used.
  - Bundle adjustment optimizes tag + per-frame camera poses (tags constrained to XY+yaw unless --no-planar).
  - Saves detection/reprojection overlays for all used frames.
  - Output JSON format matches estimate_apriltag_pose_single.py.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pupil_apriltags import Detector
from tqdm import tqdm

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - runtime dependency
    rs = None
try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - runtime dependency
    least_squares = None

REF_TAG_ID = 2
BASE_OBJECT_POINTS = np.array(
    [
        [-0.5, 0.5, 0],
        [0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [-0.5, -0.5, 0],
    ],
    dtype=np.float64,
)


@dataclass
class Detection:
    tag_id: int
    T_C_M: np.ndarray
    median_error_px: float
    decision_margin: float
    corners: np.ndarray


@dataclass
class FrameData:
    index: int
    image: np.ndarray
    detections: Dict[int, Detection]


def require_scipy() -> None:
    if least_squares is None:
        raise RuntimeError("scipy is required for bundle adjustment (install scipy).")


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


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return rt_to_transform(R, tvec.reshape(3))


def pose_from_xyyaw(x: float, y: float, yaw: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rt_to_transform(R, np.array([x, y, 0.0], dtype=np.float64))


def yaw_from_rotation(R: np.ndarray) -> float:
    return float(math.atan2(R[1, 0], R[0, 0]))


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def average_rotation_matrices(rmats: List[np.ndarray]) -> np.ndarray:
    if not rmats:
        return np.eye(3)
    M = np.zeros((3, 3))
    for R in rmats:
        M += R
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def average_transform(transforms: List[np.ndarray]) -> np.ndarray:
    if not transforms:
        return np.eye(4)
    rmats = [T[:3, :3] for T in transforms]
    tvecs = [T[:3, 3] for T in transforms]
    R_mean = average_rotation_matrices(rmats)
    t_mean = np.mean(np.stack(tvecs, axis=0), axis=0)
    return rt_to_transform(R_mean, t_mean)


def rotation_delta_deg(R_ref: np.ndarray, R: np.ndarray) -> float:
    R_delta = R_ref.T @ R
    trace = float(np.trace(R_delta))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(math.degrees(math.acos(cos_theta)))


def stats_from_samples(samples: List[float]) -> Dict[str, float | None]:
    if not samples:
        return {"count": 0, "mean": None, "median": None, "std": None}
    arr = np.array(samples, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
    }


def project_corners(
    T_c_m: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    R = T_c_m[:3, :3]
    t = T_c_m[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(object_points, rvec, t, K, dist)
    return proj.reshape(-1, 2)


def compute_median_reproj_error(
    T_c_m: np.ndarray,
    corners: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    proj = project_corners(T_c_m, object_points, K, dist)
    err = np.linalg.norm(corners - proj, axis=1)
    return float(np.median(err))


def best_shift_for_corners(
    det_corners: np.ndarray,
    proj_corners: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    best_err = float("inf")
    best_shift = 0
    best_corners = det_corners
    for shift in range(4):
        shifted = np.roll(det_corners, shift, axis=0)
        err = np.linalg.norm(shifted - proj_corners, axis=1)
        med = float(np.median(err))
        if med < best_err:
            best_err = med
            best_shift = shift
            best_corners = shifted
    return best_shift, best_err, best_corners


def select_object_points_base(
    det_corners: np.ndarray,
    T_c_m: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    tag_length: float,
) -> Tuple[np.ndarray, Dict[str, int | float]]:
    base_obj = BASE_OBJECT_POINTS * tag_length
    best_mode = "normal"
    best_err = float("inf")
    best_shift = 0
    for mode, obj in (("normal", base_obj), ("reversed", base_obj[::-1])):
        proj = project_corners(T_c_m, obj, K, dist)
        shift, err, _ = best_shift_for_corners(det_corners, proj)
        if err < best_err:
            best_err = err
            best_mode = mode
            best_shift = shift
    chosen = base_obj if best_mode == "normal" else base_obj[::-1]
    info = {"order_mode": best_mode, "ref_shift": best_shift, "median_error_px": best_err}
    return chosen, info


def compute_stats(
    frames: List[FrameData],
    tag_poses: Dict[int, np.ndarray],
    cam_poses: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    object_points: np.ndarray,
    required_ids: List[int],
) -> Tuple[Dict[int, Dict[str, float | None]], Dict[int, Dict[str, Dict[str, float | None]]]]:
    reproj_errors: Dict[int, List[float]] = {mid: [] for mid in required_ids}
    consistency_samples: Dict[int, Dict[str, List[float]]] = {
        mid: {"translation_m": [], "rotation_deg": []} for mid in required_ids
    }
    for frame_idx, frame in enumerate(frames):
        T_w_c = cam_poses[frame_idx]
        T_c_w = invert_transform(T_w_c)
        for mid, det in frame.detections.items():
            if mid not in reproj_errors:
                continue
            T_w_m = tag_poses[mid]
            T_c_m = T_c_w @ T_w_m
            reproj_errors[mid].append(
                compute_median_reproj_error(T_c_m, det.corners, object_points, K, dist)
            )

            T_w_m_obs = T_w_c @ det.T_C_M
            consistency_samples[mid]["translation_m"].append(
                float(np.linalg.norm(T_w_m_obs[:3, 3] - T_w_m[:3, 3]))
            )
            consistency_samples[mid]["rotation_deg"].append(
                rotation_delta_deg(T_w_m[:3, :3], T_w_m_obs[:3, :3])
            )

    stats = {mid: stats_from_samples(errors) for mid, errors in reproj_errors.items()}
    consistency_stats: Dict[int, Dict[str, Dict[str, float | None]]] = {}
    for mid, samples in consistency_samples.items():
        consistency_stats[mid] = {
            "translation_m": stats_from_samples(samples["translation_m"]),
            "rotation_deg": stats_from_samples(samples["rotation_deg"]),
        }
    return stats, consistency_stats


def print_stats(
    title: str,
    stats: Dict[int, Dict[str, float | None]],
    consistency: Dict[int, Dict[str, Dict[str, float | None]]],
    required_ids: List[int],
) -> None:
    print(title)
    for mid in required_ids:
        err = stats.get(mid, {})
        cons = consistency.get(mid, {})
        print(
            "Tag {}: reproj_px(median={:.3f}, mean={:.3f}, std={:.3f}), "
            "trans_m(median={:.6f}, mean={:.6f}, std={:.6f}), "
            "rot_deg(median={:.3f}, mean={:.3f}, std={:.3f})".format(
                mid,
                (err.get("median") or 0.0),
                (err.get("mean") or 0.0),
                (err.get("std") or 0.0),
                (cons.get("translation_m", {}).get("median") or 0.0),
                (cons.get("translation_m", {}).get("mean") or 0.0),
                (cons.get("translation_m", {}).get("std") or 0.0),
                (cons.get("rotation_deg", {}).get("median") or 0.0),
                (cons.get("rotation_deg", {}).get("mean") or 0.0),
                (cons.get("rotation_deg", {}).get("std") or 0.0),
            )
        )


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
    return sorted(set(ids))


def detections_from_image(
    img: np.ndarray,
    detector: Detector,
    K: np.ndarray,
    dist: np.ndarray,
    tag_length: float,
    allowed_ids: List[int],
) -> List[Detection]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections_raw = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
        tag_size=tag_length,
    )
    detections: List[Detection] = []
    for det in detections_raw:
        if allowed_ids and det.tag_id not in allowed_ids:
            continue
        if det.pose_R is None or det.pose_t is None:
            continue
        R = det.pose_R
        t = det.pose_t.reshape(3)
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
        corners = np.array(det.corners, dtype=np.float64).reshape(-1, 2)
        decision_margin = (
            float(det.decision_margin) if hasattr(det, "decision_margin") else float("inf")
        )
        detections.append(
            Detection(
                tag_id=int(det.tag_id),
                T_C_M=T_c_m,
                median_error_px=float("inf"),
                decision_margin=decision_margin,
                corners=corners,
            )
        )
    return detections


def solve_poses_single(
    detections: List[Detection],
    ref_id: int,
    planar: bool,
) -> Dict[int, np.ndarray]:
    ref_det = None
    for det in detections:
        if det.tag_id == ref_id:
            ref_det = det
            break
    if ref_det is None:
        raise RuntimeError(f"Reference tag id {ref_id} not detected")
    T_c_ref = ref_det.T_C_M
    T_ref_c = invert_transform(T_c_ref)
    poses: Dict[int, np.ndarray] = {}
    for det in detections:
        T_c_m = det.T_C_M
        T_ref_m = T_ref_c @ T_c_m
        poses[det.tag_id] = project_pose_to_plane(T_ref_m) if planar else T_ref_m
    return poses


def plot_map(
    poses: Dict[int, np.ndarray],
    tag_length: float,
    output_path: Path,
    show: bool,
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


def intrinsics_from_profile(profile: rs.video_stream_profile) -> Tuple[np.ndarray, np.ndarray]:
    intr = profile.get_intrinsics()
    K = np.array(
        [
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist = np.array(list(intr.coeffs), dtype=np.float64)
    return K, dist


def bundle_adjust(
    frames: List[FrameData],
    required_ids: List[int],
    ref_id: int,
    init_tag_poses: Dict[int, np.ndarray],
    init_cam_poses: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    object_points: np.ndarray,
    max_nfev: int,
    verbose: int,
    planar_tags: bool,
) -> Tuple[Dict[int, np.ndarray], List[np.ndarray], Dict[str, float]]:
    require_scipy()
    tag_ids = [mid for mid in required_ids if mid != ref_id]
    def pack_params() -> np.ndarray:
        params = []
        for mid in tag_ids:
            T = init_tag_poses[mid]
            if planar_tags:
                yaw = yaw_from_rotation(T[:3, :3])
                params.extend([float(T[0, 3]), float(T[1, 3]), float(yaw)])
            else:
                rvec, _ = cv2.Rodrigues(T[:3, :3])
                params.extend(rvec.flatten())
                params.extend(T[:3, 3].flatten())
        for T in init_cam_poses:
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            params.extend(rvec.flatten())
            params.extend(T[:3, 3].flatten())
        return np.array(params, dtype=np.float64)

    def unpack_params(params: np.ndarray) -> Tuple[Dict[int, np.ndarray], List[np.ndarray]]:
        tag_poses: Dict[int, np.ndarray] = {ref_id: np.eye(4)}
        offset = 0
        for mid in tag_ids:
            if planar_tags:
                x, y, yaw = params[offset:offset + 3]
                offset += 3
                tag_poses[mid] = pose_from_xyyaw(float(x), float(y), float(yaw))
            else:
                rvec = params[offset:offset + 3]
                tvec = params[offset + 3:offset + 6]
                offset += 6
                tag_poses[mid] = rvec_tvec_to_transform(rvec, tvec)
        cam_poses: List[np.ndarray] = []
        for _ in frames:
            rvec = params[offset:offset + 3]
            tvec = params[offset + 3:offset + 6]
            offset += 6
            cam_poses.append(rvec_tvec_to_transform(rvec, tvec))
        return tag_poses, cam_poses

    def residuals(params: np.ndarray) -> np.ndarray:
        tag_poses, cam_poses = unpack_params(params)
        res = []
        for frame_idx, frame in enumerate(frames):
            T_w_c = cam_poses[frame_idx]
            T_c_w = invert_transform(T_w_c)
            for det in frame.detections.values():
                T_w_m = tag_poses.get(det.tag_id)
                if T_w_m is None:
                    continue
                T_c_m = T_c_w @ T_w_m
                rvec, _ = cv2.Rodrigues(T_c_m[:3, :3])
                tvec = T_c_m[:3, 3]
                proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
                proj = proj.reshape(-1, 2)
                res.extend((proj - det.corners).reshape(-1))
        return np.array(res, dtype=np.float64)

    x0 = pack_params()
    result = least_squares(
        residuals,
        x0,
        loss="huber",
        f_scale=1.0,
        max_nfev=max_nfev,
        verbose=verbose,
    )
    tag_poses, cam_poses = unpack_params(result.x)
    metrics = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "message": str(result.message),
    }
    return tag_poses, cam_poses, metrics


def draw_detection_overlay(image: np.ndarray, detections: Dict[int, Detection]) -> np.ndarray:
    canvas = image.copy()
    for det in detections.values():
        corners = det.corners.astype(np.float32).reshape(-1, 2)
        cv2.polylines(canvas, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        for pt in corners:
            p = tuple(pt.astype(int))
            cv2.circle(canvas, p, 6, (0, 0, 0), 2)
            cv2.circle(canvas, p, 4, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            f"id {det.tag_id}",
            tuple(corners[0].astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def draw_reprojection_overlay(
    image: np.ndarray,
    detections: Dict[int, Detection],
    tag_poses: Dict[int, np.ndarray],
    T_w_c: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    T_c_w = invert_transform(T_w_c)
    for det in detections.values():
        T_w_m = tag_poses.get(det.tag_id)
        if T_w_m is None:
            continue
        T_c_m = T_c_w @ T_w_m
        rvec, _ = cv2.Rodrigues(T_c_m[:3, :3])
        tvec = T_c_m[:3, 3]
        proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        corners = det.corners.astype(np.float32).reshape(-1, 2)
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
            f"id {det.tag_id}",
            tuple(corners[0].astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_visualizations(
    frames: List[FrameData],
    tag_poses: Dict[int, np.ndarray],
    cam_poses: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    object_points: np.ndarray,
    output_dir: Path,
) -> None:
    det_dir = output_dir / "detected"
    rep_dir = output_dir / "reprojected"
    det_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(tqdm(frames, desc="Saving visualizations", unit="frame")):
        det_img = draw_detection_overlay(frame.image, frame.detections)
        rep_img = draw_reprojection_overlay(
            frame.image,
            frame.detections,
            tag_poses,
            cam_poses[idx],
            K,
            dist,
            object_points,
        )
        stem = f"frame_{idx:06d}_idx{frame.index}"
        cv2.imwrite(str(det_dir / f"{stem}.png"), det_img)
        cv2.imwrite(str(rep_dir / f"{stem}.png"), rep_img)


def run_on_bag(
    bag_path: Path,
    tag_length: float,
    family: str,
    allowed_ids: List[int],
    required_ids: List[int],
    ref_id: int,
    min_decision_margin: float,
    max_reproj_error: float,
    max_frames: int,
    max_valid_frames: int,
    planar: bool,
    threads: int,
    decimate: float,
    sigma: float,
):
    if rs is None:
        raise RuntimeError("pyrealsense2 is required to read .bag files")

    detector = Detector(
        families=family,
        nthreads=threads,
        quad_decimate=decimate,
        quad_sigma=sigma,
        refine_edges=True,
    )

    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    K, dist = intrinsics_from_profile(color_profile)

    total_frames = 0
    frames_with_required = 0
    frames_pass_clarity = 0
    used_frames = 0
    frame_data: List[FrameData] = []
    per_tag_transforms: Dict[int, List[np.ndarray]] = {mid: [] for mid in required_ids}
    id_counts: Dict[int, int] = {}
    object_points = None
    order_info = None
    corner_shifts: Dict[int, int] = {}

    pbar = tqdm(total=max_frames if max_frames > 0 else None, desc="Reading frames", unit="frame")
    try:
        while True:
            if max_frames > 0 and total_frames >= max_frames:
                break
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                break
            if playback.current_status() == rs.playback_status.stopped:
                break
            total_frames += 1
            pbar.update(1)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
            detections = detections_from_image(
                img,
                detector,
                K,
                dist,
                tag_length,
                allowed_ids,
            )
            if not detections:
                continue

            found: Dict[int, Detection] = {}
            for det in detections:
                id_counts[det.tag_id] = id_counts.get(det.tag_id, 0) + 1
                existing = found.get(det.tag_id)
                if existing is None or det.median_error_px < existing.median_error_px:
                    found[det.tag_id] = det

            if not all(mid in found for mid in required_ids):
                continue

            if object_points is None and ref_id in found:
                object_points, order_info = select_object_points_base(
                    found[ref_id].corners,
                    found[ref_id].T_C_M,
                    K,
                    dist,
                    tag_length,
                )
                ref_proj = project_corners(found[ref_id].T_C_M, object_points, K, dist)
                ref_shift, _, _ = best_shift_for_corners(found[ref_id].corners, ref_proj)
                corner_shifts[ref_id] = ref_shift
                print(
                    "[apriltag] fixed object point order using tag {} (mode={}, ref_shift={}, median_err_px={:.3f})".format(
                        ref_id,
                        order_info.get("order_mode") if order_info else "normal",
                        order_info.get("ref_shift") if order_info else 0,
                        order_info.get("median_error_px") if order_info else 0.0,
                    )
                )

            if object_points is None:
                continue

            for det in found.values():
                proj = project_corners(det.T_C_M, object_points, K, dist)
                shift = corner_shifts.get(det.tag_id)
                if shift is None:
                    shift, _, _ = best_shift_for_corners(det.corners, proj)
                    corner_shifts[det.tag_id] = shift
                det.corners = np.roll(det.corners, shift, axis=0)
                det.median_error_px = float(np.median(np.linalg.norm(det.corners - proj, axis=1)))

            frames_with_required += 1
            clarity_ok = True
            for mid in required_ids:
                det = found[mid]
                if min_decision_margin > 0 and det.decision_margin < min_decision_margin:
                    clarity_ok = False
                    break
                if max_reproj_error > 0 and det.median_error_px > max_reproj_error:
                    clarity_ok = False
                    break
            if not clarity_ok:
                continue

            frames_pass_clarity += 1
            frame_data.append(FrameData(index=total_frames, image=img.copy(), detections=found))
            poses = solve_poses_single(list(found.values()), ref_id, planar=planar)
            for mid in required_ids:
                per_tag_transforms[mid].append(poses[mid])
            used_frames += 1
            if max_valid_frames > 0 and used_frames >= max_valid_frames:
                break
    finally:
        pbar.close()
        pipeline.stop()

    if used_frames == 0:
        raise RuntimeError(
            "No frames contained all required tags with sufficient clarity. "
            f"frames_with_required={frames_with_required}, "
            f"frames_pass_clarity={frames_pass_clarity}, "
            f"seen_ids={sorted(id_counts.keys())}"
        )

    if object_points is None:
        object_points = BASE_OBJECT_POINTS * tag_length
        order_info = {"order_mode": "normal", "ref_shift": 0, "median_error_px": None}

    return (
        frame_data,
        per_tag_transforms,
        K,
        dist,
        object_points,
        order_info,
        corner_shifts,
        total_frames,
        frames_with_required,
        frames_pass_clarity,
        used_frames,
        id_counts,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Estimate AprilTag map from a RealSense .bag using pupil-apriltag"
    )
    parser.add_argument("--bag", required=True, type=Path, help="Input RealSense .bag file")
    parser.add_argument("--tag-length", required=True, type=float, help="Tag side length in meters")
    parser.add_argument("--family", type=str, default="tagStandard41h12", help="AprilTag family")
    parser.add_argument("--allowed-ids", type=str, default="0-4", help="Comma/range list, e.g., 0-4,7")
    parser.add_argument(
        "--required-ids",
        type=str,
        default="",
        help="Comma/range list of required IDs (default: same as allowed-ids)",
    )
    parser.add_argument("--ref-id", type=int, default=REF_TAG_ID, help="Reference tag id (world origin)")
    parser.add_argument("--output", type=Path, default=Path("apriltag_map.json"), help="Output JSON path")
    parser.add_argument("--figure", type=Path, default=Path("apriltag_map_3d.png"), help="Output 3D figure path")
    parser.add_argument("--no-show-3d", action="store_true", help="Do not open interactive 3D plot window")
    parser.add_argument("--threads", type=int, default=0, help="Detector threads (0 = hardware concurrency)")
    parser.add_argument("--decimate", type=float, default=1.0, help="quad_decimate for speed/accuracy tradeoff")
    parser.add_argument("--sigma", type=float, default=0.0, help="quad_sigma (Gaussian blur) for robustness")
    parser.add_argument("--min-decision-margin", type=float, default=10.0, help="Min decision_margin to accept (0=disable)")
    parser.add_argument("--max-reproj-error", type=float, default=1.0, help="Max reprojection error (px, 0=disable)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to scan (0 = all)")
    parser.add_argument("--max-valid-frames", type=int, default=0, help="Max valid frames to use (0 = all)")
    parser.add_argument("--ba-max-nfev", type=int, default=50, help="Max BA iterations")
    parser.add_argument("--ba-verbose", type=int, default=2, help="BA verbose level (0-2)")
    parser.add_argument("--vis-dir", type=Path, default=Path("apriltag_vis"), help="Output dir for visualizations")
    parser.add_argument("--no-vis", action="store_true", help="Disable saving visualization images")
    parser.add_argument("--no-planar", action="store_true", help="Disable planar projection (keep full 6DoF)")
    args = parser.parse_args()

    if rs is None:
        raise RuntimeError("pyrealsense2 is required to read .bag files")

    allowed_ids = parse_allowed_ids(args.allowed_ids)
    required_ids = parse_allowed_ids(args.required_ids) if args.required_ids else allowed_ids
    if len(required_ids) != 5:
        print(f"[warn] required ids count is {len(required_ids)} (expected 5)")
    if args.ref_id not in required_ids:
        raise RuntimeError(f"ref-id {args.ref_id} must be included in required-ids")

    (
        frame_data,
        per_tag_transforms,
        K,
        dist,
        object_points,
        order_info,
        corner_shifts,
        total_frames,
        frames_with_required,
        frames_pass_clarity,
        used_frames,
        id_counts,
    ) = run_on_bag(
        args.bag,
        args.tag_length,
        args.family,
        allowed_ids,
        required_ids,
        args.ref_id,
        args.min_decision_margin,
        args.max_reproj_error,
        args.max_frames,
        args.max_valid_frames,
        planar=not args.no_planar,
        threads=args.threads if args.threads > 0 else max(1, cv2.getNumberOfCPUs()),
        decimate=args.decimate,
        sigma=args.sigma,
    )

    init_tag_poses = {mid: average_transform(per_tag_transforms[mid]) for mid in required_ids}
    if not args.no_planar:
        init_tag_poses = {mid: project_pose_to_plane(T) for mid, T in init_tag_poses.items()}
    init_tag_poses[args.ref_id] = np.eye(4)
    init_cam_poses = [
        invert_transform(frame.detections[args.ref_id].T_C_M)
        for frame in frame_data
    ]
    init_stats, init_consistency = compute_stats(
        frame_data,
        init_tag_poses,
        init_cam_poses,
        K,
        dist,
        object_points,
        required_ids,
    )
    print_stats("Initial stats (pre-BA):", init_stats, init_consistency, required_ids)
    poses, cam_poses, ba_metrics = bundle_adjust(
        frame_data,
        required_ids,
        args.ref_id,
        init_tag_poses,
        init_cam_poses,
        K,
        dist,
        object_points,
        args.ba_max_nfev,
        args.ba_verbose,
        planar_tags=not args.no_planar,
    )

    stats, consistency_stats = compute_stats(
        frame_data,
        poses,
        cam_poses,
        K,
        dist,
        object_points,
        required_ids,
    )

    if not args.no_vis:
        save_visualizations(
            frame_data,
            poses,
            cam_poses,
            K,
            dist,
            object_points,
            args.vis_dir,
        )

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
            "reproj_error_px": stats.get(mid, {}),
            "pose_consistency": consistency_stats.get(mid, {}),
        }

    out = {
        "detector": "pupil-apriltag",
        "family": args.family,
        "ref_id": args.ref_id,
        "tag_length_m": args.tag_length,
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "object_points_order": order_info,
        "corner_shifts": {str(mid): int(shift) for mid, shift in sorted(corner_shifts.items())},
        "frames_total": total_frames,
        "frames_with_required": frames_with_required,
        "frames_pass_clarity": frames_pass_clarity,
        "frames_used": used_frames,
        "seen_ids": {str(k): v for k, v in sorted(id_counts.items())},
        "initial_stats": {
            str(mid): {
                "reproj_error_px": init_stats.get(mid, {}),
                "pose_consistency": init_consistency.get(mid, {}),
            }
            for mid in required_ids
        },
        "optimizer": {
            "name": "scipy_least_squares_huber",
            "max_nfev": args.ba_max_nfev,
            "verbose": args.ba_verbose,
            "planar_tags": not args.no_planar,
            "metrics": ba_metrics,
        },
        "visualizations": {
            "enabled": not args.no_vis,
            "dir": str(args.vis_dir) if not args.no_vis else None,
        },
        "markers": markers_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved poses for {len(markers_out)} tags to {args.output}")
    print(
        "Frames: total={}, with_required={}, pass_clarity={}, used={}".format(
            total_frames, frames_with_required, frames_pass_clarity, used_frames
        )
    )
    print(f"Seen tag IDs: {sorted(id_counts.keys())}")
    print_stats("Final stats (post-BA):", stats, consistency_stats, required_ids)

    plot_map(poses, args.tag_length, args.figure, show=not args.no_show_3d)


if __name__ == "__main__":
    main()
