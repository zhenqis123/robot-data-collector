#!/usr/bin/env python3
"""
Post-process camera pose estimates: interpolate bad frames and smooth sequences.

Usage:
  python tools/postprocess_camera_poses.py /path/to/captures \
      --poses-name camera_poses_apriltag.json \
      --output-name camera_poses_apriltag_post.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


@dataclass
class PoseSample:
    frame_index: int
    entry_index: int
    camera_id: str
    T_w_c: Optional[np.ndarray]
    status: str
    reproj_error: Optional[float]


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


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm == 0:
        return q
    return q / norm


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return normalize_quaternion(out)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def average_quaternions(quats: List[np.ndarray]) -> np.ndarray:
    if not quats:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ref = quats[0]
    acc = np.zeros(4, dtype=np.float64)
    for q in quats:
        if np.dot(ref, q) < 0:
            acc -= q
        else:
            acc += q
    return normalize_quaternion(acc)


def savgol_coeffs(window: int, polyorder: int) -> np.ndarray:
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be odd and >= 3")
    if polyorder < 0 or polyorder >= window:
        raise ValueError("polyorder must be in [0, window-1]")
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, polyorder + 1, increasing=True)
    # Solve A^T w = e0 (0th derivative at 0).
    ATA_inv = np.linalg.pinv(A.T @ A)
    coeffs = (ATA_inv @ A.T)[0]
    return coeffs


def rt_to_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def parse_reproj_error(entry: Dict, metric: str) -> Optional[float]:
    stats = entry.get("reproj_error_px")
    if not isinstance(stats, dict):
        return None
    value = stats.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_pose_samples(data: Dict, metric: str) -> Tuple[List[PoseSample], List[Dict]]:
    poses = data.get("poses", [])
    samples: List[PoseSample] = []
    for idx, entry in enumerate(poses):
        frame_index = entry.get("frame_index")
        cam_id = entry.get("camera_id")
        if not isinstance(frame_index, int) or cam_id is None:
            continue
        T_list = entry.get("T_W_C")
        T = None
        if isinstance(T_list, list):
            arr = np.array(T_list, dtype=np.float64)
            if arr.shape == (4, 4):
                T = arr
        status = str(entry.get("status", ""))
        reproj = parse_reproj_error(entry, metric)
        samples.append(
            PoseSample(
                frame_index=frame_index,
                entry_index=idx,
                camera_id=str(cam_id),
                T_w_c=T,
                status=status,
                reproj_error=reproj,
            )
        )
    return samples, poses


def rotation_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    dot = float(np.dot(q0, q1))
    dot = max(-1.0, min(1.0, abs(dot)))
    return math.degrees(2.0 * math.acos(dot))


def hampel_outliers(
    samples: List[PoseSample],
    window: int,
    k_trans: float,
    k_rot: float,
) -> Dict[Tuple[str, int], List[str]]:
    reasons: Dict[Tuple[str, int], List[str]] = {}
    if window < 3 or window % 2 == 0:
        return reasons
    half = window // 2

    by_cam: Dict[str, List[PoseSample]] = {}
    for s in samples:
        if s.T_w_c is None or s.status != "ok":
            continue
        by_cam.setdefault(s.camera_id, []).append(s)

    for cam_id, cam_samples in by_cam.items():
        cam_samples.sort(key=lambda s: s.frame_index)
        positions = [s.T_w_c[:3, 3] for s in cam_samples]
        quats = [rotation_matrix_to_quaternion(s.T_w_c[:3, :3]) for s in cam_samples]
        for i in range(len(cam_samples)):
            start = max(0, i - half)
            end = min(len(cam_samples), i + half + 1)
            if end - start < 3:
                continue
            pos_win = positions[start:end]
            med_pos = np.median(np.stack(pos_win, axis=0), axis=0)
            dev = np.linalg.norm(np.stack(pos_win, axis=0) - med_pos, axis=1)
            med_dev = float(np.median(dev))
            mad = float(np.median(np.abs(dev - med_dev)))
            if mad > 0:
                threshold = k_trans * 1.4826 * mad
                if dev[i - start] > threshold:
                    reasons.setdefault((cam_id, cam_samples[i].frame_index), []).append("hampel_trans")

            quat_win = quats[start:end]
            ref = average_quaternions(quat_win)
            ang = [rotation_angle_deg(ref, q) for q in quat_win]
            med_ang = float(np.median(ang))
            mad_ang = float(np.median(np.abs(np.array(ang) - med_ang)))
            if mad_ang > 0:
                threshold = k_rot * 1.4826 * mad_ang
                if ang[i - start] > threshold:
                    reasons.setdefault((cam_id, cam_samples[i].frame_index), []).append("hampel_rot")
    return reasons


def mark_bad_frames(
    samples: List[PoseSample],
    reproj_threshold: Optional[float],
    require_reproj: bool,
    max_trans_m: float,
    max_rot_deg: float,
) -> Dict[Tuple[str, int], List[str]]:
    reasons: Dict[Tuple[str, int], List[str]] = {}

    def add_reason(cam_id: str, frame_index: int, reason: str) -> None:
        key = (cam_id, frame_index)
        reasons.setdefault(key, []).append(reason)

    by_cam: Dict[str, List[PoseSample]] = {}
    for s in samples:
        by_cam.setdefault(s.camera_id, []).append(s)

    for cam_id, cam_samples in by_cam.items():
        cam_samples.sort(key=lambda s: s.frame_index)
        prev_valid: Optional[PoseSample] = None
        for s in cam_samples:
            if s.status != "ok" or s.T_w_c is None:
                add_reason(cam_id, s.frame_index, "status")
                continue
            if reproj_threshold is not None:
                if s.reproj_error is None and require_reproj:
                    add_reason(cam_id, s.frame_index, "missing_reproj")
                    continue
                if s.reproj_error is not None and s.reproj_error > reproj_threshold:
                    add_reason(cam_id, s.frame_index, "reproj")
                    continue
            if prev_valid is not None and prev_valid.T_w_c is not None:
                t0 = prev_valid.T_w_c[:3, 3]
                t1 = s.T_w_c[:3, 3]
                trans = float(np.linalg.norm(t1 - t0))
                R0 = prev_valid.T_w_c[:3, :3]
                R1 = s.T_w_c[:3, :3]
                dR = R0.T @ R1
                angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
                if trans > max_trans_m:
                    add_reason(cam_id, s.frame_index, "jump_trans")
                    continue
                if angle > max_rot_deg:
                    add_reason(cam_id, s.frame_index, "jump_rot")
                    continue
            prev_valid = s
    return reasons


def interpolate_pose(
    left: PoseSample,
    right: PoseSample,
    frame_index: int,
) -> np.ndarray:
    if left.T_w_c is None or right.T_w_c is None:
        raise ValueError("Missing pose for interpolation")
    t0 = left.T_w_c[:3, 3]
    t1 = right.T_w_c[:3, 3]
    if right.frame_index == left.frame_index:
        alpha = 0.0
    else:
        alpha = (frame_index - left.frame_index) / (right.frame_index - left.frame_index)
    t = (1.0 - alpha) * t0 + alpha * t1
    q0 = rotation_matrix_to_quaternion(left.T_w_c[:3, :3])
    q1 = rotation_matrix_to_quaternion(right.T_w_c[:3, :3])
    q = slerp(q0, q1, alpha)
    R = quaternion_to_rotation_matrix(q)
    return rt_to_transform(R, t)


def savgol_smooth_sequence(
    frames: List[int],
    poses: Dict[int, np.ndarray],
    window: int,
    polyorder: int,
) -> Dict[int, np.ndarray]:
    if window <= 1 or len(frames) < window:
        return poses
    coeffs = savgol_coeffs(window, polyorder)
    half = window // 2
    smoothed: Dict[int, np.ndarray] = {}

    seq_t = []
    seq_q = []
    for f in frames:
        T = poses.get(f)
        if T is None:
            seq_t.append(None)
            seq_q.append(None)
            continue
        seq_t.append(T[:3, 3])
        seq_q.append(rotation_matrix_to_quaternion(T[:3, :3]))

    # Enforce quaternion continuity.
    last_q = None
    for i, q in enumerate(seq_q):
        if q is None:
            continue
        if last_q is not None and np.dot(last_q, q) < 0:
            seq_q[i] = -q
        last_q = seq_q[i]

    for idx, frame_index in enumerate(frames):
        if seq_t[idx] is None or seq_q[idx] is None:
            continue
        t_acc = np.zeros(3, dtype=np.float64)
        q_acc = np.zeros(4, dtype=np.float64)
        for k in range(window):
            offset = k - half
            j = idx + offset
            j = max(0, min(len(frames) - 1, j))
            if seq_t[j] is None or seq_q[j] is None:
                continue
            w = coeffs[k]
            t_acc += w * seq_t[j]
            q_acc += w * seq_q[j]
        q_acc = normalize_quaternion(q_acc)
        R = quaternion_to_rotation_matrix(q_acc)
        smoothed[frame_index] = rt_to_transform(R, t_acc)
    return smoothed


def postprocess_capture(
    poses_path: Path,
    output_path: Path,
    reproj_threshold: Optional[float],
    reproj_metric: str,
    require_reproj: bool,
    max_gap: int,
    max_trans_m: float,
    max_rot_deg: float,
    hampel_window: int,
    hampel_k: float,
    hampel_rot_k: float,
    disable_hampel: bool,
    smooth_window: int,
    smooth_poly: int,
    disable_smooth: bool,
) -> bool:
    data = json.loads(poses_path.read_text())
    samples, poses = load_pose_samples(data, reproj_metric)
    if not samples:
        print(f"[post] no valid poses in {poses_path}")
        return False

    reasons = mark_bad_frames(samples, reproj_threshold, require_reproj, max_trans_m, max_rot_deg)
    if not disable_hampel:
        hampel = hampel_outliers(samples, hampel_window, hampel_k, hampel_rot_k)
        for key, vals in hampel.items():
            reasons.setdefault(key, []).extend(vals)
    by_cam: Dict[str, List[PoseSample]] = {}
    for s in samples:
        by_cam.setdefault(s.camera_id, []).append(s)

    interpolated = 0
    smoothed = 0

    for cam_id, cam_samples in by_cam.items():
        cam_samples.sort(key=lambda s: s.frame_index)
        valid_frames = [s for s in cam_samples if (cam_id, s.frame_index) not in reasons and s.T_w_c is not None]
        valid_idx = {s.frame_index: s for s in valid_frames}
        for s in cam_samples:
            if (cam_id, s.frame_index) in reasons:
                left = None
                right = None
                for other in reversed(valid_frames):
                    if other.frame_index < s.frame_index:
                        left = other
                        break
                for other in valid_frames:
                    if other.frame_index > s.frame_index:
                        right = other
                        break
                if left and right and (s.frame_index - left.frame_index) <= max_gap and (right.frame_index - s.frame_index) <= max_gap:
                    T = interpolate_pose(left, right, s.frame_index)
                    poses[s.entry_index]["T_W_C"] = T.tolist()
                    poses[s.entry_index]["status"] = "interpolated"
                    poses[s.entry_index]["rvec_w_c"] = cv2.Rodrigues(T[:3, :3])[0].reshape(3).tolist()
                    poses[s.entry_index]["tvec_w_c_m"] = T[:3, 3].tolist()
                    poses[s.entry_index]["quaternion_wxyz"] = rotation_matrix_to_quaternion(T[:3, :3]).tolist()
                    pp = poses[s.entry_index].setdefault("postprocess", {})
                    pp["interpolated"] = True
                    pp["reasons"] = reasons.get((cam_id, s.frame_index), [])
                    interpolated += 1
                    valid_idx[s.frame_index] = PoseSample(
                        frame_index=s.frame_index,
                        entry_index=s.entry_index,
                        camera_id=cam_id,
                        T_w_c=T,
                        status="interpolated",
                        reproj_error=None,
                    )

        if disable_smooth or smooth_window <= 1:
            continue
        frames = sorted([s.frame_index for s in cam_samples if s.frame_index in valid_idx])
        pose_map = {fid: valid_idx[fid].T_w_c for fid in frames if valid_idx[fid].T_w_c is not None}
        smoothed_map = savgol_smooth_sequence(frames, pose_map, smooth_window, smooth_poly)
        for fid, T in smoothed_map.items():
            sample = valid_idx.get(fid)
            if sample is None:
                continue
            entry = poses[sample.entry_index]
            entry["T_W_C"] = T.tolist()
            entry["rvec_w_c"] = cv2.Rodrigues(T[:3, :3])[0].reshape(3).tolist()
            entry["tvec_w_c_m"] = T[:3, 3].tolist()
            entry["quaternion_wxyz"] = rotation_matrix_to_quaternion(T[:3, :3]).tolist()
            pp = entry.setdefault("postprocess", {})
            pp["smoothed"] = True
            smoothed += 1

    data["poses"] = poses
    data["postprocess"] = {
        "poses_in": poses_path.name,
        "reproj_metric": reproj_metric,
        "reproj_threshold": reproj_threshold,
        "require_reproj": require_reproj,
        "max_gap": max_gap,
        "max_trans_m": max_trans_m,
        "max_rot_deg": max_rot_deg,
        "hampel_window": hampel_window,
        "hampel_k": hampel_k,
        "hampel_rot_k": hampel_rot_k,
        "disable_hampel": disable_hampel,
        "smooth_window": smooth_window,
        "smooth_poly": smooth_poly,
        "interpolated": interpolated,
        "smoothed": smoothed,
    }
    output_path.write_text(json.dumps(data, indent=2))
    print(f"[post] wrote {output_path} (interpolated={interpolated}, smoothed={smoothed})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process camera poses (interpolate + smooth).")
    parser.add_argument("root", help="Root directory containing capture folders or meta.json")
    parser.add_argument("--find-meta", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--poses-name", default="camera_poses_apriltag.json", help="Input poses name per capture")
    parser.add_argument("--output-name", default="camera_poses_apriltag_post.json", help="Output poses name per capture")
    parser.add_argument("--reproj-metric", default="mean", choices=["mean", "median", "max"])
    parser.add_argument("--reproj-threshold", type=float, default=3, help="Mark frames above this reproj error (px)")
    parser.add_argument("--require-reproj", action="store_true", help="Treat missing reproj stats as bad")
    parser.add_argument("--max-gap", type=int, default=120, help="Max frame gap for interpolation")
    parser.add_argument("--max-trans-m", type=float, default=5, help="Max translation jump (m)")
    parser.add_argument("--max-rot-deg", type=float, default=90.0, help="Max rotation jump (deg)")
    parser.add_argument("--smooth-window", type=int, default=11, help="Smoothing window size (odd)")
    parser.add_argument("--smooth-poly", type=int, default=3, help="Savgol poly order (< window)")
    parser.add_argument("--hampel-window", type=int, default=15, help="Hampel window size (odd)")
    parser.add_argument("--hampel-k", type=float, default=4.0, help="Hampel threshold for translation")
    parser.add_argument("--hampel-rot-k", type=float, default=4.0, help="Hampel threshold for rotation")
    parser.add_argument("--no-hampel", action="store_true", help="Disable Hampel outlier detection")
    parser.add_argument("--no-smooth", action="store_true", help="Disable smoothing stage")
    args = parser.parse_args()

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
            print(f"[post] skip {capture_root}, missing {poses_path.name}")
            continue
        output_path = capture_root / args.output_name
        wrote = postprocess_capture(
            poses_path,
            output_path,
            args.reproj_threshold,
            args.reproj_metric,
            args.require_reproj,
            args.max_gap,
            args.max_trans_m,
            args.max_rot_deg,
            args.hampel_window,
            args.hampel_k,
            args.hampel_rot_k,
            args.no_hampel,
            args.smooth_window,
            args.smooth_poly,
            args.no_smooth,
        )
        wrote_any = wrote_any or wrote
    return 0 if wrote_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
