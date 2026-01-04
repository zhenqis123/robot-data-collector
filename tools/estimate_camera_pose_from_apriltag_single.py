#!/usr/bin/env python3
"""
Estimate camera pose from a single image using an AprilTag map, with visualization.

Usage:
  python tools/estimate_camera_pose_from_apriltag_single.py \
      --image /path/to/image.png \
      --meta /path/to/meta.json \
      --tag-map /path/to/apriltag_map.json \
      --camera-id 000000000000
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass
class PoseResult:
    status: str
    error: Optional[str] = None
    used_tag_ids: Optional[List[int]] = None
    num_tags: int = 0
    T_W_C: Optional[np.ndarray] = None
    rvec_w_c: Optional[np.ndarray] = None
    tvec_w_c: Optional[np.ndarray] = None
    quat_wxyz: Optional[np.ndarray] = None
    reproj_stats: Optional[Dict[str, float]] = None


def rotation_matrix_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
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


def build_correspondences(
    detections: List,
    tag_length: float,
    tag_poses: Dict[int, np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    used_ids: List[int] = []
    scaled_obj = OBJECT_POINTS * tag_length
    scaled_obj = scaled_obj.copy()
    scaled_obj[:, 1] *= -1.0
    scaled_obj[:, 2] *= -1.0
    for det in detections:
        mid = int(det.tag_id)
        T_W_M = tag_poses.get(mid)
        if T_W_M is None:
            continue
        R = T_W_M[:3, :3]
        t = T_W_M[:3, 3]
        world_corners = (R @ scaled_obj.T).T + t
        obj_points.append(world_corners)
        img_points.append(np.array(det.corners, dtype=np.float64))
        used_ids.append(mid)
    if not obj_points:
        return None, None, []
    obj = np.concatenate(obj_points, axis=0)
    img = np.concatenate(img_points, axis=0)
    return obj, img, sorted(set(used_ids))


def solve_camera_pose(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    detector: Detector,
    tag_length: float,
    tag_poses: Dict[int, np.ndarray],
    pnp_reproj: float,
    pnp_iterations: int,
    pnp_confidence: float,
    allow_ippe_fallback: bool,
    use_ransac: bool,
) -> PoseResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    obj_points, img_points, used_ids = build_correspondences(
        detections,
        tag_length,
        tag_poses,
    )
    if obj_points is None or img_points is None:
        return PoseResult(status="error", error="no_tags", used_tag_ids=used_ids, num_tags=len(used_ids))
    if obj_points.shape[0] < 4:
        return PoseResult(status="error", error="insufficient_points", used_tag_ids=used_ids, num_tags=len(used_ids))
    success = False
    rvec = None
    tvec = None
    inliers = None
    if use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points,
            img_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=pnp_reproj,
            iterationsCount=pnp_iterations,
            confidence=pnp_confidence,
        )
    if not success:
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inliers = None
    if not success and allow_ippe_fallback and len(used_ids) == 1:
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        inliers = None
    if not success or rvec is None or tvec is None:
        return PoseResult(status="error", error="pnp_failed", used_tag_ids=used_ids, num_tags=len(used_ids))
    proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_points, axis=1)
    if inliers is not None and len(inliers) > 0:
        idx = inliers.flatten()
        err = err[idx]
    stats = {
        "mean": float(np.mean(err)) if err.size else None,
        "median": float(np.median(err)) if err.size else None,
        "max": float(np.max(err)) if err.size else None,
        "count": int(err.size),
    }
    R_c_w, _ = cv2.Rodrigues(rvec)
    T_c_w = rt_to_transform(R_c_w, tvec.reshape(3))
    T_w_c = invert_transform(T_c_w)
    R_w_c = T_w_c[:3, :3]
    t_w_c = T_w_c[:3, 3]
    return PoseResult(
        status="ok",
        used_tag_ids=used_ids,
        num_tags=len(used_ids),
        T_W_C=T_w_c,
        rvec_w_c=rotation_matrix_to_rvec(R_w_c),
        tvec_w_c=t_w_c,
        quat_wxyz=rotation_matrix_to_quaternion(R_w_c),
        reproj_stats=stats,
    )


def draw_overlay(
    image: np.ndarray,
    detections: List,
    tag_poses: Dict[int, np.ndarray],
    tag_length: float,
    K: np.ndarray,
    dist: np.ndarray,
    T_W_C: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    canvas = image.copy()
    scaled_obj = OBJECT_POINTS * tag_length
    scaled_obj = scaled_obj.copy()
    scaled_obj[:, 1] *= -1.0
    scaled_obj[:, 2] *= -1.0
    T_C_W = invert_transform(T_W_C)
    R_c_w = T_C_W[:3, :3]
    t_c_w = T_C_W[:3, 3]
    for det in detections:
        mid = int(det.tag_id)
        T_W_M = tag_poses.get(mid)
        if T_W_M is None:
            continue
        R_w_m = T_W_M[:3, :3]
        t_w_m = T_W_M[:3, 3]
        world_corners = (R_w_m @ scaled_obj.T).T + t_w_m
        obj = world_corners.reshape(-1, 3).astype(np.float64)
        rvec, _ = cv2.Rodrigues(R_c_w)
        proj, _ = cv2.projectPoints(obj, rvec, t_c_w.reshape(3), K, dist)
        proj = proj.reshape(-1, 2)
        det_corners = np.array(det.corners, dtype=np.float64).reshape(-1, 2)
        cv2.polylines(canvas, [proj.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.polylines(canvas, [det_corners.astype(np.int32)], True, (0, 255, 0), 2)
        for pt in det_corners:
            p = tuple(pt.astype(int))
            cv2.circle(canvas, p, 5, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            f"id {mid}",
            tuple(det_corners[0].astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"[pose] wrote overlay: {output_path}")
    if show:
        cv2.imshow("apriltag_pose_overlay", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate camera pose from a single image using AprilTag map."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path (PNG/JPG)")
    parser.add_argument("--meta", required=True, type=Path, help="Path to meta.json")
    parser.add_argument("--tag-map", required=True, type=Path, help="Path to apriltag_map.json")
    parser.add_argument("--camera-id", type=str, default="", help="Camera id in meta.json (default: first)")
    parser.add_argument("--output", type=Path, default=Path("camera_pose_single.json"))
    parser.add_argument("--vis-out", type=Path, default=Path("camera_pose_overlay.png"))
    parser.add_argument("--show", action="store_true", help="Show overlay window")
    parser.add_argument("--threads", type=int, default=0, help="Detector threads (0 = hardware concurrency)")
    parser.add_argument("--decimate", type=float, default=1.0, help="quad_decimate for speed/accuracy tradeoff")
    parser.add_argument("--sigma", type=float, default=0.0, help="quad_sigma (Gaussian blur) for robustness")
    parser.add_argument("--pnp-reproj", type=float, default=5.0, help="PnP RANSAC reprojection error (px)")
    parser.add_argument("--pnp-iterations", type=int, default=100, help="PnP RANSAC iterations")
    parser.add_argument("--pnp-confidence", type=float, default=0.99, help="PnP RANSAC confidence")
    parser.add_argument(
        "--fallback-ippesquare",
        action="store_true",
        help="Fallback to SOLVEPNP_IPPE_SQUARE when only one tag is visible",
    )
    parser.add_argument("--no-ransac", action="store_true", help="Disable PnP RANSAC stage")
    parser.add_argument("--debug", action="store_true", help="Print detection/debug info")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"[pose] failed to read image: {args.image}")
        return 2

    try:
        family, tag_length, tag_poses = read_tag_map(args.tag_map)
    except RuntimeError as exc:
        print(f"[pose] failed to read tag map: {exc}")
        return 2

    intrinsics = load_intrinsics(args.meta)
    if not intrinsics:
        print("[pose] no valid intrinsics found in meta.json")
        return 2
    if args.camera_id:
        if args.camera_id not in intrinsics:
            print(f"[pose] camera id not found in meta.json: {args.camera_id}")
            return 2
        camera_id = args.camera_id
    else:
        camera_id = sorted(intrinsics.keys())[0]
    K, dist = intrinsics[camera_id]

    threads = args.threads if args.threads > 0 else max(1, cv2.getNumberOfCPUs())
    detector = Detector(
        families=family,
        nthreads=threads,
        quad_decimate=args.decimate,
        quad_sigma=args.sigma,
        refine_edges=True,
    )
    result = solve_camera_pose(
        image,
        K,
        dist,
        detector,
        tag_length,
        tag_poses,
        args.pnp_reproj,
        args.pnp_iterations,
        args.pnp_confidence,
        args.fallback_ippesquare,
        not args.no_ransac,
    )

    output = {
        "image": str(args.image),
        "meta": str(args.meta),
        "tag_map": str(args.tag_map),
        "camera_id": camera_id,
        "status": result.status,
        "error": result.error,
        "num_tags": result.num_tags,
        "used_tag_ids": result.used_tag_ids,
        "T_W_C": result.T_W_C.tolist() if result.T_W_C is not None else None,
        "rvec_w_c": result.rvec_w_c.tolist() if result.rvec_w_c is not None else None,
        "tvec_w_c_m": result.tvec_w_c.tolist() if result.tvec_w_c is not None else None,
        "quaternion_wxyz": result.quat_wxyz.tolist() if result.quat_wxyz is not None else None,
        "reproj_error_px": result.reproj_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[pose] wrote pose json: {args.output}")

    if result.status != "ok" or result.T_W_C is None:
        print(f"[pose] estimation failed: {result.error}")
        if args.debug:
            print(f"[pose] debug camera_id={camera_id}")
            print(f"[pose] debug tag_map_family={family} tag_length={tag_length}")
            print(f"[pose] debug used_tag_ids={result.used_tag_ids} num_tags={result.num_tags}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)
            det_ids = sorted({int(det.tag_id) for det in detections})
            print(f"[pose] debug detected_tag_ids={det_ids}")
        return 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    draw_overlay(
        image,
        detections,
        tag_poses,
        tag_length,
        K,
        dist,
        result.T_W_C,
        args.vis_out,
        args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
