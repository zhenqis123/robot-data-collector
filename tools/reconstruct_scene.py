#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import open3d as o3d
import cv2
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

try:
    import h5py
except ImportError:
    h5py = None

# --- Original Helper Functions (Unchanged) ---

def filter_by_color(colors: np.ndarray, color_min: list, color_max: list):
    """
    colors: (N, 3) 范围 0.0-1.0
    color_min/max: [R, G, B] 阈值
    """
    # 转换为 0-255 整数进行处理更高效
    c_u8 = (colors * 255).astype(np.uint8)
    
    # 如果是简单的RGB过滤
    mask = (c_u8[:, 0] >= color_min[0]) & (c_u8[:, 0] <= color_max[0]) & \
           (c_u8[:, 1] >= color_min[1]) & (c_u8[:, 1] <= color_max[1]) & \
           (c_u8[:, 2] >= color_min[2]) & (c_u8[:, 2] <= color_max[2])
    return mask

def sanitize_camera_id(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in value)

def load_intrinsics(meta_path: Path) -> Dict[str, Dict]:
    if not meta_path.exists(): return {}
    meta = json.loads(meta_path.read_text())
    intrinsics: Dict[str, Dict] = {}
    for cam in meta.get("cameras", []):
        cid = cam.get("id")
        color = cam.get("streams", {}).get("color", {}).get("intrinsics", {})
        fx, fy, cx, cy = (color.get("fx"), color.get("fy"), color.get("cx"), color.get("cy"))
        width, height = (color.get("width"), color.get("height"))
        if cid is None or None in (fx, fy, cx, cy, width, height): continue
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        intrinsics[sanitize_camera_id(str(cid))] = {"K": K, "width": width, "height": height}
    return intrinsics

def load_pose_entries(poses_path: Path) -> Dict[Tuple[int, str], np.ndarray]:
    data = json.loads(poses_path.read_text())
    entries = {}
    for entry in data.get("cam_poses", []):
        status = entry.get("status")
        if status not in ("ok", "interpolated", "static_fixed"): continue
        T = np.array(entry.get("T_W_C"), dtype=np.float64)
        if T.shape == (4, 4):
            entries[(entry.get("frame_index"), sanitize_camera_id(str(entry.get("camera_id"))))] = T
    return entries

# --- PyTorch Accelerated Functions ---

def check_consistency_torch(
    points_w_torch: torch.Tensor, 
    check_cameras: List[Dict], 
    threshold: float,
    device: torch.device
) -> torch.Tensor:
    """GPU accelerated version of check_consistency."""
    N = points_w_torch.shape[0]
    final_mask = torch.ones(N, dtype=torch.bool, device=device)
    
    for cam in check_cameras:
        # T_w_c: World -> Camera
        T_w_c = torch.from_numpy(cam['T_w_c']).to(device).float()
        K = torch.from_numpy(cam['K']).to(device).float()
        d_map = torch.from_numpy(cam['depth']).to(device).float()
        w, h = cam['width'], cam['height']
        
        # World to Camera space
        # R_inv = R.T, t_inv = -R.T @ t
        R = T_w_c[:3, :3]
        t = T_w_c[:3, 3:4]
        R_inv = R.t()
        t_inv = -R_inv @ t
        
        # P_c = R_inv @ P_w + t_inv
        points_c = (R_inv @ points_w_torch.t() + t_inv).t()
        
        z_proj = points_c[:, 2]
        mask_z = z_proj > 0.001
        
        # Project to UV
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = (fx * points_c[:, 0] / torch.clamp(z_proj, min=0.001)) + cx
        v = (fy * points_c[:, 1] / torch.clamp(z_proj, min=0.001)) + cy
        
        mask_uv = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        valid_proj = mask_z & mask_uv
        final_mask &= valid_proj
        
        if not torch.any(final_mask): break
            
        # Sample Depth Map (Nearest Neighbor consistent with original np.round)
        u_int = torch.round(u).long().clamp(0, w - 1)
        v_int = torch.round(v).long().clamp(0, h - 1)
        z_measured = d_map[v_int, u_int]
        
        consistent = (torch.abs(z_proj - z_measured) < threshold) & (z_measured > 0)
        final_mask &= consistent
        
    return final_mask

# --- Parallel Processing Worker ---

def process_frame_range(frame_indices, args, intrinsics, cam_poses, rows):
    """Worker function to process a subset of frames."""
    # Each process gets its own GPU device context and resource cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resource_cache = {}
    results = []

    # Local helper for caching inside worker
    def get_cached_handles_local(safe_id, color_rel, src_idx):
        entry = resource_cache.setdefault(safe_id, {})
        cam_folder = Path(args.capture_root) / safe_id
        color_img, depth_img = None, None

        # Color logic
        color_path = cam_folder / color_rel
        if color_path.suffix.lower() in {".mkv", ".mp4", ".avi"}:
            cap_entry = entry.get('color_cap')
            if cap_entry is None or cap_entry['path'] != str(color_path):
                cap = cv2.VideoCapture(str(color_path))
                entry['color_cap'] = {'cap': cap, 'path': str(color_path), 'last_idx': -1}
            cap = entry['color_cap']['cap']
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_idx))
            ret, color_img = cap.read()
        else:
            if color_path.exists(): color_img = cv2.imread(str(color_path))

        # Depth logic
        depth_png = cam_folder / "depth_aligned" / f"{src_idx:06d}.png"
        if depth_png.exists():
            depth_img = cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED)
        elif h5py:
            h5_path = cam_folder / "depth_aligned.h5"
            if h5_path.exists():
                if 'h5' not in entry: entry['h5'] = h5py.File(str(h5_path), 'r')
                f = entry['h5']
                if 'depth' in f and src_idx < f['depth'].shape[0]: depth_img = f['depth'][src_idx]
        return color_img, depth_img

    for frame_idx in frame_indices:
        target_row = rows[frame_idx]
        # Identical camera selection logic
        cam_ids = []
        for k, v in target_row.items():
            if k.endswith("_color") and "RealSense" in k:
                cid = k.replace("_color", "")
                if cid != "ref": cam_ids.append(cid)
        ref_cam = target_row.get("ref_camera")
        if ref_cam and "RealSense" in ref_cam: cam_ids.append(ref_cam)
        cam_ids = sorted(list(set(cam_ids)))

        if len(cam_ids) < args.min_views: continue

        cam_data = []
        for cid in cam_ids:
            safe_id = sanitize_camera_id(cid)
            if (frame_idx, safe_id) not in cam_poses: continue
            
            prefix = "ref" if (target_row.get("ref_camera") == cid) else cid
            color_rel = target_row.get(f"{prefix}_color")
            src_idx = int(target_row.get(f"{prefix}_frame_index", frame_idx))
            
            color_img, depth_img = get_cached_handles_local(safe_id, color_rel, src_idx)
            if color_img is None or depth_img is None: continue

            cam_data.append({
                "id": safe_id,
                "color": cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) / 255.0,
                "depth": depth_img.astype(np.float32) * 0.001,
                "T_w_c": cam_poses[(frame_idx, safe_id)],
                "K": intrinsics[safe_id]["K"],
                "width": intrinsics[safe_id]["width"],
                "height": intrinsics[safe_id]["height"]
            })

        if not cam_data: continue

        # Processing Intersection
        frame_pts, frame_clrs = [], []
        bbox_min = np.array(args.bbox_min) if args.bbox_min else None
        bbox_max = np.array(args.bbox_max) if args.bbox_max else None

        for i, src_cam in enumerate(cam_data):
            # Back-project (NumPy for initial grid, then to Tensor)
            h, w = src_cam['depth'].shape
            u_idx = np.arange(0, w, args.spatial_subsample)
            v_idx = np.arange(0, h, args.spatial_subsample)
            u_full, v_full = np.meshgrid(u_idx, v_idx)
            sampled_depth = src_cam['depth'][v_full, u_full]
            valid = sampled_depth > 0
            
            z = sampled_depth[valid]
            u, v = u_full[valid], v_full[valid]
            fx, fy, cx, cy = src_cam['K'][0,0], src_cam['K'][1,1], src_cam['K'][0,2], src_cam['K'][1,2]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pts_c = np.column_stack((x, y, z))
            
            # To World
            R, t = src_cam['T_w_c'][:3, :3], src_cam['T_w_c'][:3, 3]
            pts_w = pts_c @ R.T + t
            
            # BBox filter
            if bbox_min is not None:
                in_bbox = np.all((pts_w >= bbox_min) & (pts_w <= bbox_max), axis=1)
                pts_w = pts_w[in_bbox]; u = u[in_bbox]; v = v[in_bbox]
            
            if len(pts_w) == 0: continue
            if args.max_source_points and len(pts_w) > args.max_source_points:
                idxs = np.random.choice(len(pts_w), args.max_source_points, replace=False)
                pts_w = pts_w[idxs]; u = u[idxs]; v = v[idxs]

            if args.color_min and args.color_max:
                color_mask = filter_by_color(colors_flat, args.color_min, args.color_max)
                pts_w = pts_w[color_mask]
                pts_c_flat = pts_c_flat[color_mask]
                colors_flat = colors_flat[color_mask]
                if len(pts_w) == 0: continue

            # GPU Consistency Check
            pts_w_torch = torch.from_numpy(pts_w).to(device).float()
            others = [c for k, c in enumerate(cam_data) if k != i]
            confirm_counts = torch.zeros(len(pts_w), dtype=torch.int, device=device)
            
            for other in others:
                mask = check_consistency_torch(pts_w_torch, [other], args.consistency_thresh, device)
                confirm_counts[mask] += 1
            
            final_mask = (confirm_counts + 1) >= args.min_views
            
            if torch.any(final_mask):
                valid_pts = pts_w[final_mask.cpu().numpy()]
                # Match color resizing logic
                if src_cam['color'].shape[:2] != src_cam['depth'].shape:
                    src_cam['color'] = cv2.resize(src_cam['color'], (w, h))
                valid_clrs = src_cam['color'][v[final_mask.cpu().numpy()].astype(int), 
                                              u[final_mask.cpu().numpy()].astype(int)]
                frame_pts.append(valid_pts)
                frame_clrs.append(valid_clrs)

        if frame_pts:
            full_pts = np.concatenate(frame_pts, axis=0)
            full_clrs = np.concatenate(frame_clrs, axis=0)

            # Default to final arrays (may be replaced by filtered versions below)
            final_pts = full_pts
            final_clrs = full_clrs

            # Configurable outlier removal + clustering
            if args.remove_outliers and len(full_pts) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(full_pts)
                pcd.colors = o3d.utility.Vector3dVector(full_clrs)

                # Optional voxel downsample to speed up subsequent filters
                if args.voxel_size and args.voxel_size > 0.0:
                    try:
                        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
                    except Exception:
                        pass

                # Outlier removal: radius-based or statistical
                try:
                    if args.radius_outlier:
                        cl, ind = pcd.remove_radius_outlier(nb_points=args.radius_nb, radius=args.radius_radius)
                    else:
                        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=args.sor_nb_neighbors, std_ratio=args.sor_std_ratio)
                    pcd = cl
                except Exception:
                    # fallback: keep original pcd on failure
                    pass

                # DBSCAN clustering to remove small clusters / noise
                if len(pcd.points) > 0:
                    labels = np.array(pcd.cluster_dbscan(eps=args.cluster_eps, min_points=args.cluster_min_points, print_progress=False))
                    if labels.size > 0:
                        kept_idx = []
                        unique_labels = np.unique(labels)

                        if args.keep_largest_cluster:
                            valid_labels = unique_labels[unique_labels >= 0]
                            if valid_labels.size > 0:
                                counts = [(lbl, int((labels == lbl).sum())) for lbl in valid_labels]
                                largest_lbl = max(counts, key=lambda x: x[1])[0]
                                kept_idx = np.where(labels == largest_lbl)[0].tolist()
                        else:
                            for lbl in unique_labels:
                                if lbl < 0:
                                    continue
                                cnt = int((labels == lbl).sum())
                                if cnt >= args.min_cluster_size:
                                    kept_idx.extend(np.where(labels == lbl)[0].tolist())

                        if len(kept_idx) > 0:
                            pcd = pcd.select_by_index(kept_idx)

                final_pts = np.asarray(pcd.points)
                final_clrs = np.asarray(pcd.colors)

            out_path = Path(args.output_dir) / f"frame_{frame_idx:06d}_intersect.ply"
            save_ply_fast(out_path, final_pts, final_clrs)
            
    # Cleanup handles
    for entry in resource_cache.values():
        if 'color_cap' in entry: entry['color_cap']['cap'].release()
        if 'h5' in entry: entry['h5'].close()

def save_ply_fast(path: Path, points: np.ndarray, colors: np.ndarray):
    """Faster PLY saving using binary or direct write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        colors_u8 = (colors * 255).astype(np.uint8)
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--cam_poses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reconstruction_intersection"))
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--consistency-thresh", type=float, default=0.03)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--spatial-subsample", type=int, default=1)
    parser.add_argument("--max-source-points", type=int, default=200000)
    parser.add_argument("--working-space", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--color-min", type=int, nargs=3, help="RGB min threshold e.g. 0 0 0")
    parser.add_argument("--color-max", type=int, nargs=3, help="RGB max threshold e.g. 255 255 255")
    # Outlier / cluster filtering options (off by default)
    parser.add_argument("--remove-outliers", action="store_true", help="Enable SOR/Radius + DBSCAN filtering (default: False)")
    parser.add_argument("--sor-nb-neighbors", type=int, default=20, help="nb_neighbors for statistical outlier removal")
    parser.add_argument("--sor-std-ratio", type=float, default=2.0, help="std_ratio for statistical outlier removal")
    parser.add_argument("--radius-outlier", action="store_true", help="Use radius outlier removal instead of statistical")
    parser.add_argument("--radius-nb", type=int, default=16, help="min number of neighbours for radius outlier removal")
    parser.add_argument("--radius-radius", type=float, default=0.05, help="radius for radius outlier removal (meters)")
    parser.add_argument("--cluster-eps", type=float, default=0.02, help="eps for DBSCAN clustering (meters)")
    parser.add_argument("--cluster-min-points", type=int, default=50, help="min_points for DBSCAN clustering")
    parser.add_argument("--min-cluster-size", type=int, default=200, help="minimum cluster size to keep after DBSCAN (points)")
    parser.add_argument("--keep-largest-cluster", action="store_true", help="If set, keep only the largest DBSCAN cluster")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional voxel downsample before filtering (meters). 0 disables.")
    parser.add_argument("--filter-config", type=Path, default=Path("post_process/resources/point_cloud_filter.json"),
                        help="Path to JSON file containing point-cloud filter parameters. If missing, defaults will be written to it.")
    
    args = parser.parse_args()

    # Load filter configuration from JSON (if present) and merge with CLI args.
    # CLI flags take precedence over values in the config file.
    filter_config_path: Path = args.filter_config

    # Build a config dict from current argparse defaults (to write if file missing)
    default_config = {
        "color_min": args.color_min,
        "color_max": args.color_max,
        "remove_outliers": args.remove_outliers,
        "sor_nb_neighbors": args.sor_nb_neighbors,
        "sor_std_ratio": args.sor_std_ratio,
        "radius_outlier": args.radius_outlier,
        "radius_nb": args.radius_nb,
        "radius_radius": args.radius_radius,
        "cluster_eps": args.cluster_eps,
        "cluster_min_points": args.cluster_min_points,
        "min_cluster_size": args.min_cluster_size,
        "keep_largest_cluster": args.keep_largest_cluster,
        "voxel_size": args.voxel_size
    }

    # Ensure parent dir exists when creating defaults
    try:
        if not filter_config_path.exists() or filter_config_path.stat().st_size == 0:
            filter_config_path.parent.mkdir(parents=True, exist_ok=True)
            # Write defaults (JSON serializable)
            with filter_config_path.open("w") as cf:
                json.dump(default_config, cf, indent=2)
        else:
            # Try load existing config, but be tolerant to parse errors
            try:
                cfg = json.loads(filter_config_path.read_text())
                # For each config key, if user did NOT provide the corresponding CLI flag, override arg
                provided_flags = set()
                argv = sys.argv[1:]
                # collect flags like --voxel-size
                for i, a in enumerate(argv):
                    if a.startswith("--"):
                        provided_flags.add(a)
                for k, v in cfg.items():
                    arg_name = "--" + k.replace("_", "-")
                    if arg_name in provided_flags:
                        continue
                    # set attribute if it exists on args
                    if hasattr(args, k):
                        setattr(args, k, v)
            except Exception:
                # If file exists but is invalid, overwrite with defaults for clarity
                with filter_config_path.open("w") as cf:
                    json.dump(default_config, cf, indent=2)
    except Exception:
        # Don't fail hard on config IO issues; just proceed with CLI args
        pass

    intrinsics = load_intrinsics(list(args.capture_root.rglob("meta.json"))[0])
    cam_poses = load_pose_entries(args.cam_poses)
    
    rows = []
    with open(args.capture_root / "frames_aligned.csv", "r") as f:
        rows = list(csv.DictReader(f))

    # Frame Range Logic
    if args.frame_index is not None:
        frames = [args.frame_index]
    else:
        start = args.start_frame
        end = args.end_frame if args.end_frame is not None else len(rows) - 1
        frames = list(range(start, end + 1, args.stride))

    # BBox Pre-processing
    args.bbox_min, args.bbox_max = None, None
    if args.working_space and args.working_space.exists():
        ws = json.loads(args.working_space.read_text())
        args.bbox_min = ws['bbox']['min']
        args.bbox_max = ws['bbox']['max']

    # Parallel Execution
    print(f"Processing {len(frames)} frames with {args.workers} workers...")
    
    
    # Split frames into chunks for workers
    chunks = np.array_split(frames, args.workers)
    chunks = [c.tolist() for c in chunks if len(c) > 0]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_frame_range, chunk, args, intrinsics, cam_poses, rows) for chunk in chunks]
        for _ in tqdm(futures, desc="Overall Progress"):
            _.result()

if __name__ == "__main__":
    # Crucial for CUDA + Multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()