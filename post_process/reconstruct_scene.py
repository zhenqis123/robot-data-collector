#!/usr/bin/env python3
"""
Reconstruct scene from multi-view RGB-D and camera poses.
Fuses point clouds from multiple cameras into a single world-space point cloud,
then optionally crops it to a target camera's view frustum.

Usage:
  python tools/reconstruct_scene.py /path/to/captures \
      --poses session_poses.json \
      --target-camera "RealSense_123" \
      --output-dir reconstructions \
      --stride 10

Outputs:
  .ply files containing colored point clouds.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None


# --- Helper Functions (Shared/Adapted) ---

def load_intrinsics(meta_path: Path) -> Dict[str, Dict]:
    """Load intrinsics (K, dist, width, height) from meta.json."""
    meta = json.loads(meta_path.read_text())
    cameras = meta.get("cameras", [])
    intrinsics: Dict[str, Dict] = {}
    for cam in cameras:
        cid = cam.get("id")
        streams = cam.get("streams", {})
        # Prefer alignment intrinsics if available (often same as color)
        # But here we adhere to what align_depth.py produces (color frame)
        color = streams.get("color", {}).get("intrinsics", {})
        fx, fy, cx, cy = (color.get("fx"), color.get("fy"), color.get("cx"), color.get("cy"))
        width, height = (color.get("width"), color.get("height"))
        
        if cid is None or None in (fx, fy, cx, cy, width, height):
            continue
            
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        coeffs = color.get("coeffs", [0, 0, 0, 0, 0])
        dist = np.array(coeffs, dtype=np.float64)
        
        intrinsics[sanitize_camera_id(str(cid))] = {
            "K": K,
            "dist": dist,
            "width": width,
            "height": height
        }
    return intrinsics


def sanitize_camera_id(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in value)


def load_pose_entries(poses_path: Path) -> Dict[Tuple[int, str], np.ndarray]:
    """Load T_W_C from JSON pose file. Returns dict[(frame_idx, cam_id)] -> 4x4 mat."""
    data = json.loads(poses_path.read_text())
    entries: Dict[Tuple[int, str], np.ndarray] = {}
    for entry in data.get("poses", []):
        frame_index = entry.get("frame_index")
        cam_id = entry.get("camera_id")
        status = entry.get("status")
        
        # Accept 'ok' and 'interpolated' poses
        if status not in ("ok", "interpolated"):
            continue
            
        T_list = entry.get("T_W_C")
        if frame_index is None or not cam_id or not T_list:
            continue
            
        T = np.array(T_list, dtype=np.float64)
        if T.shape == (4, 4):
            entries[(frame_index, sanitize_camera_id(str(cam_id)))] = T
            
    return entries


def read_frames_aligned(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read frames_aligned.csv to get synchronized frame filenames."""
    rows: List[Dict[str, str]] = []
    cam_ids: List[str] = []
    if not csv_path.exists():
        return rows, cam_ids
        
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows, cam_ids
            
        rows = list(reader)
        for field in reader.fieldnames:
            if field.endswith("_color"):
                cid = field.replace("_color", "")
                if cid == "ref":
                    continue
                cam_ids.append(cid)
                
    return rows, sorted(set(cam_ids))


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# --- Point Cloud Functions ---

def depth_to_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Back-project depth map to 3D points in camera frame.
    Returns (N, 3) array.
    """
    h, w = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Create meshgrid of pixel coordinates
    # Note: adding 0.5 to center pixel (optional but common)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Filter valid depth
    valid = (depth > 0)
    z = depth[valid]
    u = u[valid]
    v = v[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.column_stack((x, y, z))


def get_colors_for_points(color_img: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Get RGB colors corresponding to valid depth pixels."""
    if color_img.shape[:2] != depth.shape:
        # Resize color to match depth if needed (simple approach)
        color_img = cv2.resize(color_img, (depth.shape[1], depth.shape[0]))
        
    valid = (depth > 0)
    # OpenCV loads BGR, convert to RGB
    rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    return rgb[valid] / 255.0  # Normalize to 0-1


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply rigid transform T (4x4) to points (N, 3)."""
    if points.size == 0:
        return points
    
    # Rotation
    R = T[:3, :3]
    t = T[:3, 3]
    
    # P_new = R * P_old + t
    # (N,3) = (N,3) @ R.T + t
    return points @ R.T + t


def crop_to_frustum(points: np.ndarray, colors: np.ndarray, 
                   T_w_c: np.ndarray, K: np.ndarray, 
                   width: int, height: int, z_min: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop world-space points to valid view frustum of a camera.
    T_w_c: Camera pose in world (Camera-to-World).
    """
    # Transform World -> Camera
    T_c_w = invert_transform(T_w_c)
    points_cam = transform_points(points, T_c_w)
    
    # Filter Z > z_min
    mask_z = points_cam[:, 2] > z_min
    
    # Project to image plane
    # uv = K * (X,Y,Z)
    # u = fx * X/Z + cx
    # v = fy * Y/Z + cy
    
    valid_indices = np.where(mask_z)[0]
    if len(valid_indices) == 0:
        return np.empty((0,3)), np.empty((0,3))
        
    p_z = points_cam[mask_z]
    
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    X = p_z[:, 0]
    Y = p_z[:, 1]
    Z = p_z[:, 2]
    
    u = (fx * X / Z) + cx
    v = (fy * Y / Z) + cy
    
    mask_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    final_mask = np.zeros(len(points), dtype=bool)
    # Map valid_uv back to original indices
    final_indices = valid_indices[mask_uv]
    
    return points[final_indices], colors[final_indices]


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    """Save colored point cloud to PLY ASCII format."""
    if len(points) == 0:
        return
        
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Interleave data
        colors_u8 = (colors * 255).astype(np.uint8)
        
        for i in range(len(points)):
            p = points[i]
            c = colors_u8[i]
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


# --- Main Processing ---

def process_frame(
    row: Dict[str, str],
    cam_ids: List[str],
    capture_root: Path,
    intrinsics: Dict[str, Dict],
    poses: Dict[Tuple[int, str], np.ndarray],
    target_cam: str,
    output_dir: Path,
    frame_idx: int
):
    """Process a single timestep."""
    
    all_points = []
    all_colors = []
    
    for cid in cam_ids:
        c_safe = sanitize_camera_id(cid)
        
        # 1. Get Pose
        # Poses are keyed by frame index and camera ID
        # Note: frames_aligned.csv might have gaps, but 'frame_index' usually monotonic.
        # We need the alignment between CSV row index and pose frame_index.
        # By convention in this pipeline, frame_index in poses usually corresponds 
        # to the sequence frame number.
        # Let's assume input row "frame" field exists or we imply it.
        # Actually estimate_camera_poses uses image index.
        
        T_w_c = poses.get((frame_idx, c_safe))
        if T_w_c is None:
            # Skip this camera for this frame if no pose
            continue
            
        intr = intrinsics.get(c_safe)
        if not intr:
            continue
            
        # 2. Load Images
        # Color
        color_rel = row.get(f"{cid}_color")
        if not color_rel: continue
        
        # Assume aligned depth is in depth_aligned/ folder with same stem as color
        # Or look for it next to color
        color_path = capture_root / c_safe / color_rel
        if not color_path.exists():
            continue
            
        color_img = cv2.imread(str(color_path))
        if color_img is None: continue
        
        # Depth
        # Construct depth path: replace 'color' with 'depth_aligned' and keep filename
        # Or use row[f"{cid}_depth"] if existed (it doesn't in standard csv)
        # Check standard convention: color/...png -> depth_aligned/...png
        depth_path = color_path.parent.parent / "depth_aligned" / color_path.name
        
        depth_img = None
        if depth_path.exists():
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        
        # TODO: Handle .h5 if implemented in pipeline
        if depth_img is None:
            continue
            
        # Depth is usually mm (uint16)
        depth_m = depth_img.astype(np.float32) * 0.001
        
        # 3. Access Intrinsics
        K = intr["K"]
        
        # 4. Backproject
        points = depth_to_points(depth_m, K)
        colors = get_colors_for_points(color_img, depth_m)
        
        # 5. Transform to World
        points_w = transform_points(points, T_w_c)
        
        all_points.append(points_w)
        all_colors.append(colors)
        
    if not all_points:
        return

    # Fuse
    full_cloud_pts = np.concatenate(all_points, axis=0)
    full_cloud_col = np.concatenate(all_colors, axis=0)
    
    # 6. Frustum Culling / Crop to Target
    # We need the pose of the target camera at this frame
    target_safe = sanitize_camera_id(target_cam)
    T_target = poses.get((frame_idx, target_safe))
    
    cloud_to_save_pts = full_cloud_pts
    cloud_to_save_col = full_cloud_col
    
    suffix = "full"
    
    if T_target is not None and target_safe in intrinsics:
        intr_t = intrinsics[target_safe]
        pts_crop, col_crop = crop_to_frustum(
            full_cloud_pts, full_cloud_col,
            T_target, intr_t["K"], intr_t["width"], intr_t["height"]
        )
        if len(pts_crop) > 0:
            cloud_to_save_pts = pts_crop
            cloud_to_save_col = col_crop
            suffix = f"crop_{target_safe}"
    
    # 7. Save
    out_name = output_dir / f"frame_{frame_idx:06d}_{suffix}.ply"
    save_ply(out_name, cloud_to_save_pts, cloud_to_save_col)
    

def main():
    parser = argparse.ArgumentParser(description="Multi-view reconstruction pipeline")
    parser.add_argument("capture_root", type=Path, help="Root capture directory")
    parser.add_argument("--poses", type=Path, required=True, help="Path to session_poses.json")
    parser.add_argument("--target-camera", type=str, required=True, help="Camera ID to use for viewpoint pruning (e.g. RealSense#...)")
    parser.add_argument("--output-dir", type=Path, default=Path("reconstructions"), help="Output directory for PLY files")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--count", type=int, default=1000000, help="Max frames to process")

    args = parser.parse_args()
    
    root = args.capture_root.resolve()
    if not root.exists():
        print(f"Error: {root} does not exist")
        sys.exit(1)
        
    # 1. Load Meta/Intrinsics (look in first camera folder found or assume root structure)
    # The standard structure is root/CameraID/meta.json
    # Or root/meta.json?
    # tools/align_depth.py looks recursively. Let's look for any meta.json
    metas = list(root.rglob("meta.json"))
    if not metas:
        print("Error: No meta.json found in capture root")
        sys.exit(1)
        
    # Use the first meta found to load ALL camera intrinsics 
    # (assuming all plugged cams are in one meta, which is typical for this collector)
    intrinsics = load_intrinsics(metas[0])
    
    # 2. Load Poses
    print(f"Loading poses from {args.poses}...")
    poses = load_pose_entries(args.poses)
    if not poses:
        print("Error: No valid poses found")
        sys.exit(1)

    # 3. Load aligned frames CSV
    csv_path = root / "frames_aligned.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run align_timestamps.py first.")
        sys.exit(1)
        
    rows, cam_ids = read_frames_aligned(csv_path)
    print(f"Found {len(rows)} aligned frames, cameras: {cam_ids}")
    
    # 4. Process Loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for i, row in tqdm(enumerate(rows), total=len(rows), desc="Reconstructing"):
        if i < args.start:
            continue
        if (i - args.start) % args.stride != 0:
            continue
        if count >= args.count:
            break
            
        # We assume frame_idx matches i (csv row index)
        # Verify this assumption against poses keys if possible?
        # poses are keyed by stored frame_index. 
        # Usually align_timestamps produces 0-based index.
        process_frame(row, cam_ids, root, intrinsics, poses, args.target_camera, args.output_dir, i)
        count += 1
        
    print(f"Done. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
