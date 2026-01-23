#!/usr/bin/env python3
"""
Estimate moving board (tracking board) poses in the world frame using per-frame
camera poses and tag detections on the board.

Assumptions / inputs:
  - A per-capture JSON containing per-frame camera poses (produced by
    `estimate_camera_poses_from_apriltag.py`), e.g. `camera_poses_apriltag.json`.
  - A `board_map.json` describing markers attached to the moving board. Format:
    {
      "family": "tag36h11",
      "tag_length_m": 0.05,
      "markers": {
        "10": { "T_B_M": [[...4x4...]] },
        "11": { "T_B_M": [[...]] }
      }
    }
    where T_B_M is the 4x4 transform from board frame B to marker frame M.
  - The capture directory contains `frames_aligned.csv` and the image/video
    assets referenced there (same layout as used by camera pose estimator).

High level algorithm:
  - For each capture:
      - Load camera poses JSON (per frame, per camera) produced earlier.
      - Load camera intrinsics from `meta.json`.
      - For each frame & camera:
          - Read image (from aligned video or image path in frames_aligned.csv).
          - Detect AprilTags in the image.
          - For detected tags that are in the `board_map`, build 3D object
            points in board frame using T_B_M and canonical tag corner layout.
          - Use solvePnP (RANSAC optional) to estimate board pose in camera
            frame: T_C_B. Then compute T_W_B = T_W_C * T_C_B using known
            camera pose T_W_C for that frame.
          - Record result. If no tags or insufficient points, mark as
            "visible": false and include an error code. If previously visible
            but now not, mark a disappearance event for that frame.

Outputs:
  - Per-capture JSON (default name `board_poses.json`) with entries per
    frame/camera containing: status, used_tag_ids, num_tags, T_W_B (4x4),
    rvec/tvec, quaternion, reprojection stats, visible flag, disappeared flag.

Notes / choices:
  - The script is intentionally general: board marker poses are relative to
    the board frame (so the board can move). It does not assume markers are
    fixed in world.
  - We reuse the same detection (pupil_apriltags) and OpenCV PnP.
  - Missing/occluded frames are recorded; a simple visibility state per-camera
    is tracked to flag disappearance events.

Usage example:
  python tools/estimate_board_poses_from_camera.py /path/to/capture \
      --board-map /path/to/board_map.json \
      --camera-poses camera_poses_apriltag.json

"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov"}
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
class BoardPoseResult:
    status: str
    error: Optional[str] = None
    used_tag_ids: Optional[List[int]] = None
    num_tags: int = 0
    T_W_B: Optional[np.ndarray] = None
    rvec_b_c: Optional[np.ndarray] = None
    tvec_b_c: Optional[np.ndarray] = None
    rvec_w_b: Optional[np.ndarray] = None
    tvec_w_b: Optional[np.ndarray] = None
    quat_wxyz: Optional[np.ndarray] = None
    reproj_stats: Optional[Dict[str, float]] = None


def sanitize_camera_id(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


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


def parse_frame_index(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


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


def build_board_correspondences(
    detections: List,
    tag_length: float,
    tag_poses_B: Dict[int, np.ndarray],
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
        T_B_M = tag_poses_B.get(mid)
        if T_B_M is None:
            continue
        R = T_B_M[:3, :3]
        t = T_B_M[:3, 3]
        board_corners = (R @ scaled_obj.T).T + t
        obj_points.append(board_corners)
        img_points.append(np.array(det.corners, dtype=np.float64))
        used_ids.append(mid)
    if not obj_points:
        return None, None, []
    obj = np.concatenate(obj_points, axis=0)
    img = np.concatenate(img_points, axis=0)
    return obj, img, sorted(set(used_ids))


def solve_board_pose(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    detector: Detector,
    tag_length: float,
    tag_poses_B: Dict[int, np.ndarray],
    pnp_reproj: float,
    pnp_iterations: int,
    pnp_confidence: float,
    use_ransac: bool,
    pnp_flag: int,
) -> BoardPoseResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    obj_points, img_points, used_ids = build_board_correspondences(detections, tag_length, tag_poses_B)
    if obj_points is None or img_points is None:
        return BoardPoseResult(status="error", error="no_tags", used_tag_ids=used_ids, num_tags=len(used_ids))
    if obj_points.shape[0] < 4:
        return BoardPoseResult(status="error", error="insufficient_points", used_tag_ids=used_ids, num_tags=len(used_ids))
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
            flags=pnp_flag,
            reprojectionError=pnp_reproj,
            iterationsCount=pnp_iterations,
            confidence=pnp_confidence,
        )
    if not success:
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist, flags=pnp_flag)
        inliers = None
    if not success or rvec is None or tvec is None:
        return BoardPoseResult(status="error", error="pnp_failed", used_tag_ids=used_ids, num_tags=len(used_ids))
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
    R_b_c, _ = cv2.Rodrigues(rvec)
    T_b_c = rt_to_transform(R_b_c, tvec.reshape(3))
    # We return board pose in camera frame as rvec/tvec (b in c), caller can transform
    # to world using camera pose.
    R_c_b = invert_transform(T_b_c)[:3, :3]
    T_c_b = invert_transform(T_b_c)
    R_w_b = None
    t_w_b = None
    # R_w_b and t_w_b are not computed here because T_W_B requires T_W_C from caller
    return BoardPoseResult(
        status="ok",
        used_tag_ids=used_ids,
        num_tags=len(used_ids),
        T_W_B=None,
        rvec_b_c=rvec.flatten(),
        tvec_b_c=tvec.flatten(),
        rvec_w_b=None,
        tvec_w_b=None,
        quat_wxyz=None,
        reproj_stats=stats,
    )


def count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", newline="") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 1)


def resolve_color_path(capture_root: Path, camera_id: str, color_rel: str) -> Path:
    p = Path(color_rel)
    if p.is_absolute():
        return p
    cam_dir = sanitize_camera_id(camera_id)
    return capture_root / cam_dir / p


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
    print(f"[board] updated marker {marker_path}")


def process_capture(
    capture_root: Path,
    meta_path: Path,
    camera_poses_path: Path,
    board_map_path: Path,
    detector: Detector,
    output_name: str,
    pnp_reproj: float,
    pnp_iterations: int,
    pnp_confidence: float,
    use_ransac: bool,
    pnp_flag: int,
) -> bool:
    with meta_path.open("r") as f:
        meta = json.load(f)
    intrinsics = load_intrinsics(meta_path)
    # load camera poses file
    cam_poses = json.loads(camera_poses_path.read_text())
    # load board map
    board_map = json.loads(board_map_path.read_text())
    family = board_map.get("family")
    tag_length = float(board_map.get("tag_length_m", 0.0))
    markers = board_map.get("markers", {})
    tag_poses_B: Dict[int, np.ndarray] = {}
    for key, entry in markers.items():
        try:
            mid = int(key)
            T = np.array(entry.get("T_B_M", []), dtype=np.float64)
        except Exception:
            continue
        if T.shape != (4, 4):
            continue
        tag_poses_B[mid] = T
    if not tag_poses_B:
        print(f"[board] no markers in board map: {board_map_path}")
        return False

    frames_csv = capture_root / "frames_aligned.csv"
    if not frames_csv.exists():
        print(f"[board] missing frames_aligned.csv in {capture_root}")
        return False

    row_count = count_csv_rows(frames_csv)
    output = {
        "board_map": str(board_map_path),
        "capture_root": str(capture_root),
        "frames_aligned": str(frames_csv),
        "poses": [],
    }

    video_caps: Dict[str, cv2.VideoCapture] = {}
    video_paths: Dict[str, Path] = {}
    aligned_video_name = None
    # if camera poses file contains video-based color_path names, we try aligned
    # But we will simply follow frames_aligned.csv entries

    last_visible: Dict[str, bool] = {}

    with frames_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for frame_index, row in enumerate(reader):
            ref_id = row.get("ref_camera")
            for cam_entry in cam_poses.get("poses", []):
                # cam_poses produced earlier might be keyed differently; try to
                # find matching entry by frame_index and camera id.
                pass
            # Instead, we iterate cameras in meta
            for cam in meta.get("cameras", []):
                cid = cam.get("id")
                if cid not in intrinsics:
                    output["poses"].append({
                        "frame_index": frame_index,
                        "camera_id": cid,
                        "status": "error",
                        "error": "missing_intrinsics",
                    })
                    continue
                if cid == ref_id:
                    color_rel = row.get("ref_color", "")
                    frame_idx = parse_frame_index(row.get("ref_frame_index", ""))
                else:
                    color_rel = row.get(f"{cid}_color", "")
                    frame_idx = parse_frame_index(row.get(f"{cid}_frame_index", ""))

                if not color_rel:
                    output["poses"].append({
                        "frame_index": frame_index,
                        "camera_id": cid,
                        "status": "error",
                        "error": "missing_color_path",
                    })
                    continue
                color_path = prefer_mp4_path(resolve_color_path(capture_root, cid, color_rel))
                if not color_path.exists():
                    output["poses"].append({
                        "frame_index": frame_index,
                        "camera_id": cid,
                        "status": "error",
                        "error": "missing_image",
                        "color_path_resolved": str(color_path),
                    })
                    continue
                if is_video_path(color_path):
                    if frame_idx is None:
                        output["poses"].append({
                            "frame_index": frame_index,
                            "camera_id": cid,
                            "status": "error",
                            "error": "missing_frame_index",
                            "color_path_resolved": str(color_path),
                        })
                        continue
                    cap = video_caps.get(cid)
                    if cap is None or video_paths.get(cid) != color_path:
                        if cap is not None:
                            cap.release()
                        cap = cv2.VideoCapture(str(color_path))
                        if not cap.isOpened():
                            output["poses"].append({
                                "frame_index": frame_index,
                                "camera_id": cid,
                                "status": "error",
                                "error": "video_open_failed",
                                "color_path_resolved": str(color_path),
                            })
                            continue
                        video_caps[cid] = cap
                        video_paths[cid] = color_path
                    image = read_video_frame(cap, frame_idx)
                else:
                    image = cv2.imread(str(color_path))
                if image is None:
                    output["poses"].append({
                        "frame_index": frame_index,
                        "camera_id": cid,
                        "status": "error",
                        "error": "read_failed",
                        "color_path_resolved": str(color_path),
                    })
                    continue

                # need camera world pose for this frame
                # Trying to find T_W_C in camera_poses file: it commonly stores
                # per-frame entries with keys frame_index and camera_id. We'll
                # perform a lookup fallback: if camera_poses is keyed by
                # camera->frames, attempt both.
                T_W_C = None
                cam_poses_entries = {}
                # flexible lookup
                if isinstance(cam_poses, dict):
                    # search in top-level poses list
                    for p in cam_poses.get("poses", []):
                        if p.get("frame_index") == frame_index and str(p.get("camera_id")) == str(cid):
                            T_W_C = np.array(p.get("T_W_C")) if p.get("T_W_C") is not None else None
                            break
                if T_W_C is None:
                    # try to find camera block
                    for cblock in cam_poses.get("cameras", []) if isinstance(cam_poses.get("cameras"), list) else []:
                        if str(cblock.get("camera_id")) == str(cid):
                            frames = cblock.get("frames", [])
                            for fr in frames:
                                if fr.get("frame_index") == frame_index:
                                    T_W_C = np.array(fr.get("T_W_C")) if fr.get("T_W_C") is not None else None
                                    break
                            if T_W_C is not None:
                                break

                if T_W_C is None:
                    output["poses"].append({
                        "frame_index": frame_index,
                        "camera_id": cid,
                        "status": "error",
                        "error": "missing_camera_pose",
                    })
                    continue

                K, dist = intrinsics[cid]
                result = solve_board_pose(
                    image,
                    K,
                    dist,
                    detector,
                    tag_length,
                    tag_poses_B,
                    pnp_reproj,
                    pnp_iterations,
                    pnp_confidence,
                    use_ransac,
                    pnp_flag,
                )
                entry: Dict = {
                    "frame_index": frame_index,
                    "camera_id": cid,
                }
                previously_visible = last_visible.get(str(cid), False)
                if result.status != "ok":
                    entry.update({
                        "status": "error",
                        "error": result.error,
                        "num_tags": result.num_tags,
                        "used_tag_ids": result.used_tag_ids,
                        "visible": False,
                    })
                    # mark disappearance if it was visible previously
                    if previously_visible:
                        entry["disappeared"] = True
                    last_visible[str(cid)] = False
                    output["poses"].append(entry)
                    continue

                # convert rvec_b_c,tvec_b_c into T_C_B then to T_W_B
                rvec = result.rvec_b_c
                tvec = result.tvec_b_c
                R_b_c, _ = cv2.Rodrigues(rvec)
                T_b_c = rt_to_transform(R_b_c, tvec.reshape(3))
                T_c_b = invert_transform(T_b_c)
                # T_W_C is 4x4 world <- camera
                T_W_C = np.array(T_W_C, dtype=np.float64)
                T_W_B = T_W_C @ T_c_b
                R_w_b = T_W_B[:3, :3]
                t_w_b = T_W_B[:3, 3]
                quat = rotation_matrix_to_quaternion(R_w_b)
                entry.update({
                    "status": "ok",
                    "visible": True,
                    "T_W_B": T_W_B.tolist(),
                    "rvec_w_b": rotation_matrix_to_rvec(R_w_b).tolist(),
                    "tvec_w_b_m": t_w_b.tolist(),
                    "quaternion_wxyz": quat.tolist(),
                    "num_tags": result.num_tags,
                    "used_tag_ids": result.used_tag_ids,
                    "reproj_error_px": result.reproj_stats,
                })
                # if previously not visible and now visible, mark appeared
                if not previously_visible:
                    entry["appeared"] = True
                last_visible[str(cid)] = True
                output["poses"].append(entry)

    out_path = capture_root / output_name
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    for cap in video_caps.values():
        cap.release()
    update_marker(capture_root, "board_poses", {"output": out_path.name})
    print(f"[board] wrote {out_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate moving board poses using camera poses and board map")
    parser.add_argument("root", help="Root directory containing capture folders or meta.json")
    parser.add_argument("--board-map", required=True, type=Path, help="Path to board_map.json (marker poses in board frame)")
    parser.add_argument("--camera-poses", required=True, type=Path, help="Path to camera poses JSON (per-capture or single file)")
    parser.add_argument("--output-name", default="board_poses.json", help="Output JSON name per capture")
    parser.add_argument("--decimate", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--pnp-reproj", type=float, default=2.0)
    parser.add_argument("--pnp-iterations", type=int, default=1000)
    parser.add_argument("--pnp-confidence", type=float, default=0.99)
    parser.add_argument("--pnp-method", default="iterative", choices=["iterative", "ippe", "ippe_square"])
    parser.add_argument("--no-ransac", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    board_map_path = args.board_map.expanduser().resolve()
    camera_poses_path = args.camera_poses.expanduser().resolve()
    if not board_map_path.exists():
        print(f"board map not found: {board_map_path}")
        return 2
    if not camera_poses_path.exists():
        print(f"camera poses not found: {camera_poses_path}")
        return 2

    metas = []
    if (root / "meta.json").exists():
        metas = [root / "meta.json"]
    else:
        # try root/*/meta.json
        for child in sorted(root.iterdir()):
            meta = child / "meta.json"
            if meta.exists():
                metas.append(meta)
    if not metas:
        print("No meta.json found")
        return 1

    pnp_flags = {
        "iterative": cv2.SOLVEPNP_ITERATIVE,
        "ippe": cv2.SOLVEPNP_IPPE,
        "ippe_square": cv2.SOLVEPNP_IPPE_SQUARE,
    }
    pnp_flag = pnp_flags[args.pnp_method]

    threads = max(1, cv2.getNumberOfCPUs())
    detector = Detector(families=None, nthreads=threads, quad_decimate=args.decimate, quad_sigma=args.sigma, refine_edges=True)

    any_written = False
    for meta in metas:
        capture_root = meta.parent
        wrote = process_capture(
            capture_root,
            meta,
            camera_poses_path,
            board_map_path,
            detector,
            args.output_name,
            args.pnp_reproj,
            args.pnp_iterations,
            args.pnp_confidence,
            not args.no_ransac,
            pnp_flag,
        )
        any_written = any_written or wrote

    detector = None
    gc.collect()
    return 0 if any_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
