#!/usr/bin/env python3
"""
Repair corrupt frames in a session by freezing the previous valid frame.
STRATEGY: REPAIR (Fill-in) instead of REMOVE.
1. Detects corrupt frames in Video files.
2. Replaces corrupt video frames with the last valid frame.
3. Replaces corrupt depth frames (if aligned) with the last valid frame.
4. Keeps timestamps and other sensor data UNCHANGED (Timeline is preserved).

Usage:
    python tools/repair_corrupt_frames.py /path/to/session
    (Optional: --no-backup to delete originals immediately)
"""

import os
import sys
import argparse
import subprocess
import re
import shutil
import multiprocessing
import numpy as np
import cv2
from pathlib import Path
from typing import List, Set, Optional
from tqdm import tqdm

try:
    import h5py
except ImportError:
    print("Error: h5py not installed. Please pip install h5py")
    sys.exit(1)

# Configuration
EXT_VIDEO = {'.mkv', '.mp4', '.avi'}

def detect_corrupt_frames(file_path: Path) -> Set[int]:
    """
    Scans video using ffmpeg to find specific decoding errors.
    Returns a set of corrupt frame indices.
    """
    command = ["ffmpeg", "-v", "info", "-i", str(file_path), "-f", "null", "-"]
    
    error_frames = set()
    current_frame = 0
    # Regex to capture current frame processing
    frame_pattern = re.compile(r"frame=\s*(\d+)")
    
    try:
        process = subprocess.Popen(
            command, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace'
        )
        
        buffer = ""
        while True:
            chunk = process.stderr.read(4096)
            if not chunk and process.poll() is not None:
                break
            if chunk:
                buffer += chunk
                while '\n' in buffer or '\r' in buffer:
                    if '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                    else:
                        line, buffer = buffer.split('\r', 1)
                    line = line.strip()
                    if not line: continue
                    
                    # Track current frame index
                    match_frame = frame_pattern.search(line)
                    if match_frame:
                        current_frame = int(match_frame.group(1))
                    
                    # Detect Error keywords
                    lower_line = line.lower()
                    is_error = "error" in lower_line or \
                               ("[h264" in lower_line and "frame=" not in lower_line)
                    
                    # Exclude metadata info
                    if is_error:
                        if any(x in line for x in ["Input #", "Output #", "Metadata:", "Duration:"]):
                            continue
                        error_frames.add(current_frame)
                        
    except Exception as e:
        print(f"Warning: ffmpeg scan failed for {file_path.name}: {e}")
        return set()

    return error_frames

def process_camera_worker(args):
    """
    Worker to handle a single camera folder.
    1. Scans video for corruption.
    2. Reads original -> Writes Repaired (Video & Depth).
    3. Handles Backups.
    """
    cam_dir, backup_dir, backup_enabled = args
    
    # Identify Video File
    video_file = None
    for f in cam_dir.iterdir():
        if f.suffix in EXT_VIDEO and "viz" not in f.name:
            video_file = f
            break
            
    if not video_file:
        return # No video found
    
    # Identify Depth File
    depth_file = cam_dir / "depth.h5"
    has_depth = depth_file.exists()
    
    # 1. Detect Corruption
    # We scan the original file first
    bad_indices = detect_corrupt_frames(video_file)
    
    if not bad_indices:
        return # Nothing to do for this camera
        
    print(f"  [Repairing] {cam_dir.name}: Found {len(bad_indices)} corrupt frames.")

    # Prepare Backup Directory
    cam_backup = backup_dir / cam_dir.name
    if backup_enabled:
        cam_backup.mkdir(parents=True, exist_ok=True)

    # --- REPAIR VIDEO ---
    # Setup paths
    if backup_enabled:
        src_video = cam_backup / video_file.name
        if not src_video.exists():
            shutil.copy2(video_file, src_video)
    else:
        # Rename original to .orig
        src_video = video_file.with_suffix(video_file.suffix + ".orig")
        if not src_video.exists():
            video_file.rename(src_video)

    tmp_video = video_file.with_suffix(".mp4.tmp")
    
    cap = cv2.VideoCapture(str(src_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))
    
    last_valid_frame = np.zeros((height, width, 3), dtype=np.uint8) # Default black
    
    # Processing Loop
    for i in range(total_frames):
        ret, frame = cap.read()
        
        # Logic: 
        # If frame index is known bad -> Use last_valid
        # If read failed (ret=False) -> Use last_valid
        # Else -> Use new frame AND update last_valid
        
        is_bad = (i in bad_indices) or (not ret)
        
        if is_bad:
            out_vid.write(last_valid_frame)
        else:
            last_valid_frame = frame
            out_vid.write(frame)
            
    cap.release()
    out_vid.release()
    
    # Replace Video
    if tmp_video.exists():
        tmp_video.replace(video_file) # Overwrite original location
        if not backup_enabled and src_video.exists():
            src_video.unlink() # Delete .orig

    # --- REPAIR DEPTH (Optional) ---
    if has_depth:
        if backup_enabled:
            src_depth = cam_backup / depth_file.name
            if not src_depth.exists():
                shutil.copy2(depth_file, src_depth)
        else:
            src_depth = depth_file.with_suffix(depth_file.suffix + ".orig")
            if not src_depth.exists():
                depth_file.rename(src_depth)
                
        tmp_depth = depth_file.with_suffix(".h5.tmp")
        
        try:
            with h5py.File(src_depth, "r") as f_in, h5py.File(tmp_depth, "w") as f_out:
                if "depth" in f_in:
                    d_in = f_in["depth"]
                    # Create dataset
                    d_out_ds = f_out.create_dataset(
                        "depth",
                        shape=d_in.shape,
                        dtype=d_in.dtype,
                        compression="gzip"
                    )
                    
                    total_depth = d_in.shape[0]
                    # We assume depth is aligned with video frame-by-frame
                    # Initialize last valid buffer
                    if total_depth > 0:
                        last_valid_depth = d_in[0]
                    else:
                        last_valid_depth = None
                        
                    for i in range(total_depth):
                        # Determine if this index is bad (using video indices)
                        # or if we are out of bounds of video bad_indices
                        is_bad = i in bad_indices
                        
                        if is_bad and last_valid_depth is not None:
                            d_out_ds[i] = last_valid_depth
                        else:
                            # Read current
                            current = d_in[i]
                            d_out_ds[i] = current
                            last_valid_depth = current
                            
            # Replace Depth
            if tmp_depth.exists():
                tmp_depth.replace(depth_file)
                if not backup_enabled and src_depth.exists():
                    src_depth.unlink()
                    
        except Exception as e:
            print(f"Error processing depth for {cam_dir.name}: {e}")
            if tmp_depth.exists(): tmp_depth.unlink()
            # Restore original if something failed
            if not backup_enabled and src_depth.exists() and not depth_file.exists():
                src_depth.rename(depth_file)

    print(f"  [Done] {cam_dir.name} repaired.")

def main():
    parser = argparse.ArgumentParser(description="Repair corrupt frames (In-Place, Keep Timeline)")
    parser.add_argument("session_dir", type=Path, help="Path to the session directory")
    parser.add_argument("--backup", action="store_true", help="Keep backups in _corrupt_backup folder")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2), 
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    session_dir = args.session_dir.resolve()
    if not session_dir.exists():
        print(f"Error: Session directory not found: {session_dir}")
        sys.exit(1)
        
    print(f"=== Frame Repair Started: {session_dir.name} ===")
    print("Mode: REPLACE bad frames with PREVIOUS valid frame (Timestamp invariant)")
    
    # 1. Setup Backup Directory
    backup_dir = session_dir / "_corrupt_backup"
    if args.backup:
        backup_dir.mkdir(exist_ok=True)
        print(f"Backups enabled: {backup_dir}")
    else:
        print("Backups DISABLED. Original files will be modified/deleted.")

    # 2. Find Camera Directories
    camera_dirs = []
    for root, dirs, files in os.walk(session_dir):
        if "_corrupt_backup" in root: continue
        # Check if folder has video
        has_vid = any(f.endswith(tuple(EXT_VIDEO)) and "viz" not in f for f in files)
        if has_vid:
            camera_dirs.append(Path(root))
            
    if not camera_dirs:
        print("No camera directories found.")
        return

    print(f"Found {len(camera_dirs)} cameras. Processing...")
    
    # 3. Process in Parallel
    pool_args = [(d, backup_dir, args.backup) for d in camera_dirs]
    
    with multiprocessing.Pool(processes=args.workers) as pool:
        # Use tqdm to show progress bar
        list(tqdm(pool.imap_unordered(process_camera_worker, pool_args), total=len(camera_dirs)))

    # 4. Cleanup
    if not args.backup and backup_dir.exists():
        try:
            shutil.rmtree(backup_dir)
        except:
            pass
            
    print("\nAll tasks completed.")

if __name__ == "__main__":
    # Windows/Mac compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main()