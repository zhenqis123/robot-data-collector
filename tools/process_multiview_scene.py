#!/usr/bin/env python3
"""
Multi-view RGB-D Scene Reconstruction Pipeline
Handles:
1. Depth & Timestamp Alignment
2. Raw Camera Pose Estimation (Apriltag)
3. Pose Refinement (Enforce Static Constraints for fixed cameras)
4. Scene Reconstruction (Fused & Cropped to Target View)

Usage:
    python tools/process_multiview_scene.py /path/to/capture_dir \
        --tag-map /path/to/apriltag_map.json \
        --fixed-cameras "CamFixed1,CamFixed2" \
        --moving-cameras "CamMoving" \
        --target-view "CamMoving" \
        --output-dir reconstructions
"""

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("Pipeline")

def run_command(cmd: List[str], check: bool = True, allow_failure_if_output_exists: Optional[Path] = None):
    """
    Run a subprocess command and log it.
    
    Args:
        cmd: The command to run.
        check: If True, check the return code. 
               (Modified behavior: if allow_failure_if_output_exists is set, we check that file instead of return code)
        allow_failure_if_output_exists: If provided, and the command fails, check if this file exists and is non-empty.
                                        If so, proceed with a warning instead of exiting.
    """
    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")
    
    try:
        # If we have a fallback check, disable internal check so we can catch it manually
        start_time = os.times()
        proc = subprocess.run(cmd, check=check if not allow_failure_if_output_exists else False)
        
        if proc.returncode != 0:
            if allow_failure_if_output_exists:
                if allow_failure_if_output_exists.exists() and allow_failure_if_output_exists.stat().st_size > 100:
                    logger.warning(f"Command returned {proc.returncode} but output file {allow_failure_if_output_exists} seems valid. Proceeding.")
                else:
                    logger.error(f"Command failed with {proc.returncode} and output file is missing/empty: {allow_failure_if_output_exists}")
                    sys.exit(1)
            elif check:
                 # This branch shouldn't be reached if check=True was passed to subprocess.run, 
                 # but for completeness if we passed check=False manually:
                 logger.error(f"Command failed with {proc.returncode}")
                 sys.exit(1)

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {cmd_str}")
        sys.exit(1)

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

def should_skip_step(step_name: str, args) -> bool:
    """Check if a step should be skipped based on flags."""
    # This could be enhanced to check file existence, but for now simple logic
    return False

# ...existing code...

    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

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

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else q

def average_quaternions(quats: List[np.ndarray]) -> np.ndarray:
    """
    Calculate the average quaternion using the eigenvalue method (or simple iterative mean).
    Here using a simpler averaging suitable for close rotations.
    """
    if not quats:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    # Simple averaging ensuring same hemisphere
    ref = quats[0]
    acc = np.zeros(4, dtype=np.float64)
    for q in quats:
        if np.dot(ref, q) < 0:
            acc -= q
        else:
            acc += q
    return normalize_quaternion(acc)

def refine_poses(
    input_json: Path,
    output_json: Path,
    fixed_cameras: List[str],
    moving_cameras: List[str]
):
    """
    Refines camera poses:
    - Fixed Cameras: Compute one static pose (median translation, average rotation) from all valid frames.
    - Moving Cameras: Keep per-frame poses (optionally could smooth).
    """
    logger.info(f"Refining poses. Fixed: {fixed_cameras}, Moving: {moving_cameras}")
    
    with open(input_json, 'r') as f:
        data = json.load(f)
        
    poses = data.get("poses", [])
    
    # Organize by camera
    cam_poses: Dict[str, List[dict]] = {}
    for entry in poses:
        cam_id = entry.get("camera_id")
        if not cam_id: 
            continue
        # Sanitize ID matching
        # Users might pass "RealSense_1" but json has "RealSense_1_Color" or similar
        # We'll do a loose containment check or exact match
        clean_id = cam_id
        if clean_id not in cam_poses:
            cam_poses[clean_id] = []
        cam_poses[clean_id].append(entry)
        
    refined_poses = []
    
    for cam_id, entries in cam_poses.items():
        is_fixed = any(f in cam_id for f in fixed_cameras)
        is_moving = any(m in cam_id for m in moving_cameras)
        
        # Determine strategy if camera matches both or neither (default to moving/per-frame if unsure)
        if is_fixed:
            logger.info(f"Processing FIXED camera: {cam_id}")
            # Collect valid transforms
            valid_ts = []
            valid_qs = []
            
            for entry in entries:
                if entry.get("status") != "ok":
                    continue
                T = np.array(entry["T_W_C"])
                R = T[:3, :3]
                t = T[:3, 3]
                valid_ts.append(t)
                valid_qs.append(rotation_matrix_to_quaternion(R))
            
            if not valid_ts:
                logger.warning(f"No valid frames for fixed camera {cam_id}! All frames will be invalid.")
                for entry in entries:
                    refined_poses.append(entry) # Keep original (likely failed)
                continue
                
            # Compute Statistics
            valid_ts = np.array(valid_ts)
            median_t = np.median(valid_ts, axis=0) # Robust to outliers
            avg_q = average_quaternions(valid_qs)
            avg_R = quaternion_to_rotation_matrix(avg_q)
            
            # Construct static transform
            T_static = np.eye(4)
            T_static[:3, :3] = avg_R
            T_static[:3, 3] = median_t
            
            T_static_list = T_static.tolist()
            
            # Overwrite all entries for this camera
            logger.info(f"  -> Calculated static pose: t={median_t}")
            for entry in entries:
                # Even if original frame failed, for a fixed camera we can now 'recover' it 
                # if we assume the camera never moved. 
                # But to be safe, we only overwrite if 'ok' or mark 'interpolated'
                # Here we force set it to 'ok' or 'static_fixed'
                entry["T_W_C"] = T_static_list
                entry["status"] = "ok" # Mark as usable
                refined_poses.append(entry)
                
        else:
            # Moving or unspecified -> Keep original
            # (Optional: Add smoothing here if needed, but postprocess_camera_poses.py does that)
            if is_moving:
                logger.info(f"Processing MOVING camera: {cam_id} (Checking pass-through)")
            for entry in entries:
                refined_poses.append(entry)

    # Sort by frame index for tidiness
    refined_poses.sort(key=lambda x: x.get("frame_index", 0))
    
    # Save
    out_data = {"poses": refined_poses}
    with open(output_json, 'w') as f:
        json.dump(out_data, f, indent=2)
    logger.info(f"Saved refined poses to {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Multi-view RGB-D Scene Reconstruction Pipeline")
    parser.add_argument("capture_root", type=Path, help="Root directory of the capture session")
    parser.add_argument("--tag-map", type=Path, required=True, help="Path to AprilTag map JSON")
    parser.add_argument("--fixed-cameras", type=str, default="", help="Comma-separated list of keywords for fixed camera IDs")
    parser.add_argument("--moving-cameras", type=str, default="", help="Comma-separated list of keywords for moving camera IDs")
    parser.add_argument("--target-view", type=str, required=True, help="Keyword for the camera view to crop reconstruction to")
    parser.add_argument("--output-dir", type=Path, default=Path("reconstructions"), help="Output directory for PLY files")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter to use")
    
    args = parser.parse_args()
    
    if not args.capture_root.exists():
        logger.error(f"Capture root does not exist: {args.capture_root}")
        sys.exit(1)

    if not args.tag_map.exists():
         logger.error(f"Tag map does not exist: {args.tag_map}")
         sys.exit(1)
        
    fixed_list = [x.strip() for x in args.fixed_cameras.split(',') if x.strip()]
    moving_list = [x.strip() for x in args.moving_cameras.split(',') if x.strip()]
    
    # Path to existing tools
    tools_dir = Path(__file__).parent
    post_process_dir = tools_dir.parent / "post_process" # Assuming structure
    
    tool_align_depth = tools_dir / "align_depth.py"
    tool_align_timestamps = tools_dir / "align_timestamps.py"
    tool_estimate_pose = tools_dir / "estimate_camera_poses_from_apriltag.py"
    # Note: reconstruct_scene might be in post_process based on workspace info
    tool_reconstruct = post_process_dir / "reconstruct_scene.py"
    
    if not tool_reconstruct.exists():
        # Fallback check
        tool_reconstruct = tools_dir / "reconstruct_scene.py"

    # -- Auto-detect moving/fixed cameras if not specified --
    # Strategy: 
    # 1. Read 'meta.json' or folder structure to list all camera (Serial numbers)
    # 2. Assume cameras NOT in 'fixed-cameras' are moving, or vice versa
    # 3. For now, we require at least the target view or fixed list to be partially known
    #    User prompt suggests: "1 moving, 2 fixed". 
    #    So if user supplies only moving, others are fixed.
    
    # Let's inspect directories to find all 'RealSense_*'
    all_camera_dirs = [d.name for d in args.capture_root.iterdir() if d.is_dir() and "RealSense" in d.name]
    
    if not args.fixed_cameras and not args.moving_cameras:
         # Rough heuristic: If target view is set, that's moving, others fixed?
         # Or if no args provided, we can't guess.
         # But the user script flow implies manual input usually.
         # We will proceed with whatever lists we have.
         pass
    
    if args.moving_cameras and not args.fixed_cameras:
        # Infer fixed cameras
        fixed_list = [c for c in all_camera_dirs if not any(m in c for m in moving_list)]
        logger.info(f"Auto-inferred Fixed Cameras: {fixed_list}")
    
    if args.fixed_cameras and not args.moving_cameras:
        # Infer moving cameras
        moving_list = [c for c in all_camera_dirs if not any(f in c for f in fixed_list)]
        logger.info(f"Auto-inferred Moving Cameras: {moving_list}")

    logger.info("=== Step 1: Aligning Depth ===")
    run_command([args.python_bin, str(tool_align_depth), str(args.capture_root), "--delete-original-depth", "false"])
    
    logger.info("=== Step 2: Aligning Timestamps ===")

    run_command([args.python_bin, str(tools_dir / "sort_timestamps.py"), str(args.capture_root), "--find-meta", "true"])
    run_command([args.python_bin, str(tool_align_timestamps), str(args.capture_root), "--find-meta", "true"])
    
    logger.info("=== Step 3: Estimating Raw Poses ===")
    raw_poses_path = args.capture_root / "camera_poses_raw.json"
    
    # Check if we have HDF5 depth, if so, we might need to be careful with Apriltag tool which expects images
    # The estimate_camera_poses_from_apriltag.py tool uses 'frames_aligned.csv' usually.
    # It reads images from 'color_path' in frames_aligned.csv.
    
    run_command([
        args.python_bin, str(tool_estimate_pose), str(args.capture_root),
        "--tag-map", str(args.tag_map),
        "--output-name", raw_poses_path.name,
        "--pnp-method", "ippe",
        "--no-ransac" # Use standard PnP, outlier removal handled in static refinement
    ], allow_failure_if_output_exists=raw_poses_path)
    
    logger.info("=== Step 4: Refining Poses (Static/Moving Split) ===")
    refined_poses_path = args.capture_root / "session_poses_refined.json"
    refine_poses(raw_poses_path, refined_poses_path, fixed_list, moving_list)
    
    logger.info(f"=== Step 5: Reconstructing Scene (Target: {args.target_view}) ===")
    final_output_dir = args.capture_root / args.output_dir
    
    reconstruct_cmd = [
        args.python_bin, str(tool_reconstruct), str(args.capture_root),
        "--poses", str(refined_poses_path),
        "--target-camera", args.target_view,
        "--output-dir", str(final_output_dir),
        "--stride", "5" # Adjust stride for speed vs density
    ]
    
    # If user provided fixed cameras, we can optionally pass them to reconstruct 
    # to optimize or strictly crop (though reconstruct_scene currently just uses target-camera for crop)
    
    run_command(reconstruct_cmd)
    
    logger.info(f"Pipeline complete. Outputs in {final_output_dir}")


if __name__ == "__main__":
    main()
