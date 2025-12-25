#!/usr/bin/env python3
"""
Estimate per-frame camera poses in the AprilTag world frame using aligned frames.

Requirements:
  - frames_aligned.csv is already generated under each capture directory.
  - Tag map JSON path is provided via --tag-map.
  - color paths can be PNG sequences or MKV videos (uses frame_index columns when present).

Usage:
  python tools/estimate_camera_poses_from_apriltag.py /path/to/captures \
      --tag-map /path/to/apriltag_map.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
except ImportError as exc:
    raise SystemExit("rich is required: pip install rich") from exc

OBJECT_POINTS = np.array(
    [
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
    ],
    dtype=np.float64,
)

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov"}


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


def build_correspondences(
    detections: List,
    tag_length: float,
    tag_poses: Dict[int, np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    used_ids: List[int] = []
    scaled_obj = OBJECT_POINTS * tag_length
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
) -> PoseResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    obj_points, img_points, used_ids = build_correspondences(detections, tag_length, tag_poses)
    if obj_points is None or img_points is None:
        return PoseResult(status="error", error="no_tags", used_tag_ids=used_ids, num_tags=len(used_ids))
    if obj_points.shape[0] < 4:
        return PoseResult(status="error", error="insufficient_points", used_tag_ids=used_ids, num_tags=len(used_ids))
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


def parse_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def process_capture(
    capture_root: Path,
    meta_path: Path,
    tag_map_path: Path,
    detector: Detector,
    tag_length: float,
    tag_poses: Dict[int, np.ndarray],
    output_name: str,
    pnp_reproj: float,
    pnp_iterations: int,
    pnp_confidence: float,
    progress: Progress,
    console: Console,
    capture_index: int,
    capture_total: int,
) -> bool:
    frames_csv = capture_root / "frames_aligned.csv"
    if not frames_csv.exists():
        console.print(f"[pose] missing frames_aligned.csv in {capture_root}")
        return False
    if should_skip_step(capture_root, "camera_poses_apriltag"):
        console.print(f"[pose] skip {capture_root}, already processed")
        return False

    with meta_path.open("r") as f:
        meta = json.load(f)
    cam_ids = [c.get("id") for c in meta.get("cameras", []) if c.get("id") is not None]
    if not cam_ids:
        console.print(f"[pose] no cameras in {capture_root}")
        return False
    intrinsics = load_intrinsics(meta_path)

    row_count = count_csv_rows(frames_csv)
    total_steps = row_count * max(1, len(cam_ids))
    task_label = f"[{capture_index}/{capture_total}] {capture_root.name}"
    task_id = progress.add_task(task_label, total=total_steps)

    output: Dict[str, object] = {
        "tag_map": str(tag_map_path),
        "tag_length_m": tag_length,
        "capture_root": str(capture_root),
        "frames_aligned": str(frames_csv),
        "poses": [],
    }

    ok_count = 0
    err_count = 0
    video_caps: Dict[str, cv2.VideoCapture] = {}
    video_paths: Dict[str, Path] = {}

    with frames_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for frame_index, row in enumerate(reader):
            ref_id = row.get("ref_camera", cam_ids[0])
            ref_ts_ms = parse_float(row.get("ref_timestamp_ms", ""))
            ref_ts_iso = row.get("ref_timestamp_iso", "")
            for cid in cam_ids:
                if cid not in intrinsics:
                    output["poses"].append(
                        {
                            "frame_index": frame_index,
                            "ref_camera": ref_id,
                            "ref_timestamp_ms": ref_ts_ms,
                            "ref_timestamp_iso": ref_ts_iso,
                            "camera_id": cid,
                            "status": "error",
                            "error": "missing_intrinsics",
                        }
                    )
                    err_count += 1
                    progress.advance(task_id)
                    continue
                if cid == ref_id:
                    color_rel = row.get("ref_color", "")
                    frame_idx = parse_frame_index(row.get("ref_frame_index", ""))
                    delta_ms = 0.0
                else:
                    color_rel = row.get(f"{cid}_color", "")
                    frame_idx = parse_frame_index(row.get(f"{cid}_frame_index", ""))
                    delta_ms = parse_float(row.get(f"{cid}_delta_ms", ""))
                entry = {
                    "frame_index": frame_index,
                    "ref_camera": ref_id,
                    "ref_timestamp_ms": ref_ts_ms,
                    "ref_timestamp_iso": ref_ts_iso,
                    "camera_id": cid,
                    "delta_ms": delta_ms,
                    "color_path": color_rel,
                }
                if not color_rel:
                    entry.update({"status": "error", "error": "missing_color_path"})
                    output["poses"].append(entry)
                    err_count += 1
                    progress.advance(task_id)
                    continue
                color_path = prefer_mp4_path(resolve_color_path(capture_root, cid, color_rel))
                if not color_path.exists():
                    entry.update(
                        {
                            "status": "error",
                            "error": "missing_image",
                            "color_path_resolved": str(color_path),
                        }
                    )
                    output["poses"].append(entry)
                    err_count += 1
                    progress.advance(task_id)
                    continue
                if is_video_path(color_path):
                    if frame_idx is None:
                        entry.update(
                            {
                                "status": "error",
                                "error": "missing_frame_index",
                                "color_path_resolved": str(color_path),
                            }
                        )
                        output["poses"].append(entry)
                        err_count += 1
                        progress.advance(task_id)
                        continue
                    cap = video_caps.get(cid)
                    if cap is None or video_paths.get(cid) != color_path:
                        if cap is not None:
                            cap.release()
                        cap = cv2.VideoCapture(str(color_path))
                        if not cap.isOpened():
                            entry.update(
                                {
                                    "status": "error",
                                    "error": "video_open_failed",
                                    "color_path_resolved": str(color_path),
                                }
                            )
                            output["poses"].append(entry)
                            err_count += 1
                            progress.advance(task_id)
                            continue
                        video_caps[cid] = cap
                        video_paths[cid] = color_path
                    image = read_video_frame(cap, frame_idx)
                else:
                    if frame_idx is None and color_path.suffix.lower() == ".png":
                        frame_idx = parse_frame_index(color_path.stem)
                    image = cv2.imread(str(color_path))
                if image is None:
                    entry.update(
                        {
                            "status": "error",
                            "error": "read_failed",
                            "color_path_resolved": str(color_path),
                        }
                    )
                    output["poses"].append(entry)
                    err_count += 1
                    progress.advance(task_id)
                    continue
                K, dist = intrinsics[cid]
                result = solve_camera_pose(
                    image,
                    K,
                    dist,
                    detector,
                    tag_length,
                    tag_poses,
                    pnp_reproj,
                    pnp_iterations,
                    pnp_confidence,
                )
                if result.status != "ok":
                    entry.update(
                        {
                            "status": "error",
                            "error": result.error,
                            "num_tags": result.num_tags,
                            "used_tag_ids": result.used_tag_ids,
                        }
                    )
                    output["poses"].append(entry)
                    err_count += 1
                    progress.advance(task_id)
                    continue
                entry.update(
                    {
                        "status": "ok",
                        "T_W_C": result.T_W_C.tolist() if result.T_W_C is not None else None,
                        "rvec_w_c": result.rvec_w_c.tolist() if result.rvec_w_c is not None else None,
                        "tvec_w_c_m": result.tvec_w_c.tolist() if result.tvec_w_c is not None else None,
                        "quaternion_wxyz": result.quat_wxyz.tolist() if result.quat_wxyz is not None else None,
                        "num_tags": result.num_tags,
                        "used_tag_ids": result.used_tag_ids,
                        "reproj_error_px": result.reproj_stats,
                    }
                )
                output["poses"].append(entry)
                ok_count += 1
                progress.advance(task_id)

    out_path = capture_root / output_name
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    for cap in video_caps.values():
        cap.release()
    progress.remove_task(task_id)
    console.print(
        f"[pose] wrote {out_path} (ok={ok_count}, error={err_count}, frames={row_count})"
    )
    update_marker(
        capture_root,
        "camera_poses_apriltag",
        {
            "output": out_path.name,
            "ok_frames": ok_count,
            "error_frames": err_count,
            "frames": row_count,
            "tag_map": str(tag_map_path),
        },
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate camera poses using aligned frames and AprilTag map.")
    parser.add_argument("root", help="Root directory containing capture folders or meta.json")
    parser.add_argument("--tag-map", required=True, type=Path, help="Path to apriltag_map.json")
    parser.add_argument(
        "--find-meta",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Search meta.json recursively (true) or only root/*/meta.json (false)",
    )
    parser.add_argument("--output-name", default="camera_poses_apriltag.json", help="Output JSON name per capture")
    parser.add_argument("--threads", type=int, default=0, help="Detector threads (0 = hardware concurrency)")
    parser.add_argument("--decimate", type=float, default=1.0, help="quad_decimate for speed/accuracy tradeoff")
    parser.add_argument("--sigma", type=float, default=0.0, help="quad_sigma (Gaussian blur) for robustness")
    parser.add_argument("--pnp-reproj", type=float, default=5.0, help="PnP RANSAC reprojection error (px)")
    parser.add_argument("--pnp-iterations", type=int, default=100, help="PnP RANSAC iterations")
    parser.add_argument("--pnp-confidence", type=float, default=0.99, help="PnP RANSAC confidence")
    args = parser.parse_args()

    tag_map_path = args.tag_map.expanduser().resolve()
    if not tag_map_path.exists():
        print(f"[pose] tag map not found: {tag_map_path}")
        return 2
    try:
        family, tag_length, tag_poses = read_tag_map(tag_map_path)
    except RuntimeError as exc:
        print(f"[pose] failed to read tag map: {exc}")
        return 2

    root = Path(args.root).expanduser().resolve()
    find_meta = args.find_meta.lower() == "true"
    metas = list_meta_files(root, find_meta, 2)
    if not metas:
        print("No meta.json found")
        return 1

    threads = args.threads if args.threads > 0 else max(1, cv2.getNumberOfCPUs())
    detector = Detector(
        families=family,
        nthreads=threads,
        quad_decimate=args.decimate,
        quad_sigma=args.sigma,
        refine_edges=True,
    )

    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    any_written = False
    with progress:
        for idx, meta in enumerate(metas, start=1):
            capture_root = meta.parent
            wrote = process_capture(
                capture_root,
                meta,
                tag_map_path,
                detector,
                tag_length,
                tag_poses,
                args.output_name,
                args.pnp_reproj,
                args.pnp_iterations,
                args.pnp_confidence,
                progress,
                console,
                idx,
                len(metas),
            )
            any_written = any_written or wrote
    if not any_written:
        return 1
    return 0


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
    print(f"[pose] updated marker {marker_path}")


if __name__ == "__main__":
    raise SystemExit(main())
