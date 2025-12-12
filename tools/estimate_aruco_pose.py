#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

REQ_IDS = [0, 1, 2, 3, 4]

def load_intrinsics(meta_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    meta = json.loads(meta_path.read_text())
    color = meta["cameras"][0]["streams"]["color"]["intrinsics"]
    fx, fy, cx, cy = color["fx"], color["fy"], color["cx"], color["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(color.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)
    return K, dist

def rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def project_to_plane_se2(T: np.ndarray) -> np.ndarray:
    """强制 z=0，且只保留 yaw（丢 roll/pitch）。返回 (x,y,theta)。"""
    x, y = float(T[0, 3]), float(T[1, 3])
    R = T[:3, :3]
    theta = math.atan2(R[1, 0], R[0, 0])
    return np.array([x, y, theta], dtype=np.float64)

def se2_to_T(p: np.ndarray) -> np.ndarray:
    x, y, th = float(p[0]), float(p[1]), float(p[2])
    c, s = math.cos(th), math.sin(th)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]], dtype=np.float64)
    T[:3, 3] = np.array([x, y, 0], dtype=np.float64)
    return T

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def marker_obj_points(marker_len: float) -> np.ndarray:
    s = marker_len / 2.0
    return np.array([[-s,  s, 0],
                     [ s,  s, 0],
                     [ s, -s, 0],
                     [-s, -s, 0]], dtype=np.float32)

def estimate_pose_from_corners(corners_1x4x2: np.ndarray,
                               marker_len: float,
                               K: np.ndarray,
                               dist: np.ndarray):
    objp = marker_obj_points(marker_len)
    imgp = corners_1x4x2.reshape(4, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return ok, rvec.reshape(3), tvec.reshape(3)

def reproj_error_px(corners_1x4x2: np.ndarray,
                    rvec: np.ndarray,
                    tvec: np.ndarray,
                    marker_len: float,
                    K: np.ndarray,
                    dist: np.ndarray) -> float:
    objp = marker_obj_points(marker_len)
    proj, _ = cv2.projectPoints(objp, rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist)
    err = np.linalg.norm(proj.reshape(4, 2) - corners_1x4x2.reshape(4, 2), axis=1)
    return float(np.median(err))

def corner_span_px(corners_1x4x2: np.ndarray) -> float:
    pts = corners_1x4x2.reshape(4, 2)
    # 近似用 bbox 对角线作为“尺寸”
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return float(np.linalg.norm(mx - mn))

def detect_frame(img_bgr, detector) -> Tuple[List[np.ndarray], np.ndarray]:
    corners, ids, _ = detector.detectMarkers(img_bgr)
    if ids is None:
        return [], None
    return corners, ids.flatten().astype(int)

def try_import_scipy():
    try:
        from scipy.optimize import least_squares  # noqa: F401
        return True
    except Exception:
        return False

def optimize_se2(meas_by_i: Dict[int, List[np.ndarray]], use_scipy: bool):
    """
    meas_by_i: 对每个 i(1..4)，存多帧观测到的 z_1i = (x,y,theta)  (marker1<-markeri)
    目标：优化出每个 marker i 在 marker1 世界系下的 pose p_i=(x,y,theta)
    marker1 固定为 (0,0,0)。对每个观测，残差为 p_i - z_1i（角度wrap）。
    """
    var_ids = [1, 2, 3, 4]
    # 初值：用观测的中位数（鲁棒）
    x0 = []
    for mid in var_ids:
        Z = np.stack(meas_by_i[mid], axis=0)
        x0.extend([np.median(Z[:, 0]), np.median(Z[:, 1]), np.median(Z[:, 2])])
    x0 = np.array(x0, dtype=np.float64)

    def unpack(x, mid):
        k = var_ids.index(mid) * 3
        return np.array([x[k], x[k + 1], x[k + 2]], dtype=np.float64)

    def residual(x):
        r = []
        for mid in var_ids:
            p = unpack(x, mid)
            for z in meas_by_i[mid]:
                r.append(p[0] - z[0])
                r.append(p[1] - z[1])
                r.append(wrap_pi(p[2] - z[2]))
        return np.array(r, dtype=np.float64)

    if use_scipy:
        from scipy.optimize import least_squares
        res = least_squares(residual, x0, loss="huber", f_scale=1.0, max_nfev=50, verbose=0)
        x_opt = res.x
    else:
        # 没有 scipy：退化为鲁棒统计（中位数）= 最简稳健解
        x_opt = x0

    poses = {0: np.array([0.0, 0.0, 0.0], dtype=np.float64)}
    for mid in var_ids:
        poses[mid] = unpack(x_opt, mid)
    return poses

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path, help="PNG folder")
    ap.add_argument("--meta", required=True, type=Path, help="meta.json with intrinsics")
    ap.add_argument("--marker-length", required=True, type=float, help="marker side length (m)")
    ap.add_argument("--max-frames", type=int, default=200, help="max good frames to keep")
    ap.add_argument("--max-median-reproj", type=float, default=1.5, help="px threshold (median reproj error)")
    ap.add_argument("--min-span-px", type=float, default=60.0, help="min marker bbox diag (px)")
    ap.add_argument("--out", type=Path, default=Path("aruco_map_1to5.json"))
    args = ap.parse_args()

    K, dist = load_intrinsics(args.meta)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    files = sorted(args.images.glob("*.png"))
    if not files:
        raise RuntimeError(f"No PNG in {args.images}")

    candidates = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        corners, ids = detect_frame(img, detector)
        if ids is None:
            continue

        id_set = set(int(x) for x in ids.tolist())
        if not all(i in id_set for i in REQ_IDS):
            continue

        # 组装 id->corners
        id_to_corner = {int(mid): corners[k] for k, mid in enumerate(ids)}
        # 只取 1..5
        id_to_corner = {mid: id_to_corner[mid] for mid in REQ_IDS}

        # 估计每个 marker 的 T_C_M 和质量
        T_C = {}
        errs = []
        spans = []
        ok_all = True
        for mid in REQ_IDS:
            ok, rvec, tvec = estimate_pose_from_corners(id_to_corner[mid], args.marker_length, K, dist)
            if not ok:
                ok_all = False
                break
            T_C[mid] = rt_to_T(rvec, tvec)
            errs.append(reproj_error_px(id_to_corner[mid], rvec, tvec, args.marker_length, K, dist))
            spans.append(corner_span_px(id_to_corner[mid]))

        if not ok_all:
            continue

        med_err = float(np.median(errs))
        min_span = float(np.min(spans))

        # 质量筛选：误差要小，尺寸要大
        if med_err > args.max_median_reproj:
            continue
        if min_span < args.min_span_px:
            continue

        # 打分（越小越好）：误差优先，其次鼓励更大的 span
        score = med_err + 30.0 / (min_span + 1e-6)
        candidates.append((score, p.name, T_C))

    if not candidates:
        raise RuntimeError("No high-quality frames containing IDs 1..5. Relax thresholds or improve capture.")

    candidates.sort(key=lambda x: x[0])
    selected = candidates[: args.max_frames]
    print(f"[OK] selected {len(selected)} frames (best scores).")

    # 对每帧，形成相对观测 z_1i = T_1<-i = inv(T_C_1) @ T_C_i，然后投影到平面SE2
    meas_by_i = {1: [], 2: [], 3: [], 4: []}
    used_frames = []
    for score, fname, T_C in selected:
        T_0_C = inv_T(T_C[0])
        for mid in [1, 2, 3, 4]:
            T_0_i = T_0_C @ T_C[mid]          # T_{0<-i}
            z = project_to_plane_se2(T_0_i)
            meas_by_i[mid].append(z)
        used_frames.append({"frame": fname, "score": score})


    use_scipy = try_import_scipy()
    poses_se2 = optimize_se2(meas_by_i, use_scipy=use_scipy)

    # 输出
    out = {
        "dictionary": "DICT_4X4_50",
        "required_ids": REQ_IDS,
        "world_ref_id": 1,
        "marker_length_m": args.marker_length,
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "selection": {
            "max_frames": args.max_frames,
            "max_median_reproj_px": args.max_median_reproj,
            "min_span_px": args.min_span_px,
            "selected_frames": used_frames,
            "optimizer": "scipy_least_squares_huber" if use_scipy else "robust_median_only",
        },
        "markers": {}
    }

    for mid in REQ_IDS:
        p = poses_se2[mid] if mid != 0 else np.array([0.0, 0.0, 0.0])
        T = se2_to_T(p)
        out["markers"][str(mid)] = {
            "x_m": float(p[0]),
            "y_m": float(p[1]),
            "yaw_rad": float(p[2]),
            "T_W_M": T.tolist()
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote map to {args.out}")
    if not use_scipy:
        print("[WARN] scipy not found; used robust median (install scipy to run huber least_squares).")

if __name__ == "__main__":
    main()
