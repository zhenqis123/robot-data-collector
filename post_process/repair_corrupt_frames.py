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

from ast import arg
import os
import sys
import argparse
import subprocess
import re
import shutil
import multiprocessing
import numpy as np
import cv2
import time
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

def detect_corrupt_frames(file_path: Path, total_frames: Optional[int] = None, log_prefix: str = "") -> Set[int]:
    """
    Scans video using ffmpeg to find specific decoding errors.
    Returns a set of corrupt frame indices.
    """
    command = ["ffmpeg", "-v", "info", "-i", str(file_path), "-f", "null", "-"]
    
    error_frames = set()
    current_frame = 0
    # Regex to capture current frame processing
    frame_pattern = re.compile(r"frame=\s*(\d+)")
    
    start_time = time.time()
    last_log_time = start_time

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
                        # Log progress
                        now = time.time()
                        if now - last_log_time > 2.0: # Log every 2 seconds
                            if total_frames and total_frames > 0:
                                percent = (current_frame / total_frames) * 100
                                print(f"{log_prefix}Scanning: {percent:.1f}% (Frame {current_frame}/{total_frames})")
                            else:
                                print(f"{log_prefix}Scanning: Frame {current_frame}")
                            last_log_time = now

                    
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


class FFmpegWriter:
    """
    Writes video frames to an ffmpeg subprocess for faster encoding.
    """
    def __init__(self, filename, width, height, fps, crf=23, preset="ultrafast", output_format=None):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        
        # Build command
        # Input: raw video from pipe (bgr24 from opencv)
        # Output: h264 mp4
        self.cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",  # Input from stdin
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p", # Standard for compatibility
        ]
        
        if output_format:
            self.cmd.extend(["-f", output_format])
            
        self.cmd.append(str(filename))

    def start(self):
        # Hide output unless error
        self.process = subprocess.Popen(
            self.cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )

    def write(self, frame):
        if self.process is None:
            self.start()
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print(f"Error: FFmpeg pipe broken for {self.filename}")
            # Try to read stderr
            _, stderr = self.process.communicate()
            if stderr:
                print(f"FFmpeg stderr: {stderr.decode('utf-8', errors='replace')}")

    def release(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            self.process = None

def process_camera_worker(args):
    """
    Worker to handle a single camera folder.
    1. Scans video for corruption.
    2. Reads original -> Writes Repaired (Video & Depth).
    3. Handles Backups.
    """
    cam_dir, backup_dir, backup_enabled, fast_mode, process_depth = args
    cam_name = cam_dir.name
    log_prefix = f"[{cam_name}] "
    
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
    
    # Get total frames using OpenCV
    total_frames = 0
    try:
        tmp_cap = cv2.VideoCapture(str(video_file))
        if tmp_cap.isOpened():
            total_frames = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tmp_cap.release()
    except:
        pass

    # 1. Detect Corruption
    bad_indices = set()
    if not fast_mode:
        # Full scan
        bad_indices = detect_corrupt_frames(video_file, total_frames, log_prefix)
    else:
        print(f"{log_prefix}Fast mode: Skipping ffmpeg scan. Will repair only unreadable frames.")

    if not fast_mode and not bad_indices:
        return # Nothing to do

    if bad_indices:
        print(f"{log_prefix}Found {len(bad_indices)} corrupt frames via scan.")

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
    
    # Check if src_video exists now
    if not src_video.exists():
        print(f"{log_prefix}Error: Source video not found: {src_video}")
        return

    # Use original extension for tmp file to imply format, but still specify explicitly for safety with .tmp
    input_ext = video_file.suffix.lower()
    tmp_video = video_file.with_suffix(input_ext + ".tmp")
    
    # Determine format for ffmpeg
    ffmpeg_fmt = "mp4" # Default
    if input_ext == ".mkv":
        ffmpeg_fmt = "matroska"
    elif input_ext == ".avi":
        ffmpeg_fmt = "avi"
    elif input_ext == ".mp4":
        ffmpeg_fmt = "mp4"
    
    cap = cv2.VideoCapture(str(src_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames == 0:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use FFmpegWriter instead of cv2.VideoWriter for speed (ultrafast preset)
    out_vid = FFmpegWriter(tmp_video, width, height, fps, preset="ultrafast", output_format=ffmpeg_fmt)
    out_vid.start()
    
    last_valid_frame = np.zeros((height, width, 3), dtype=np.uint8) # Default black
    
    # Processing Loop
    last_log_time = time.time()
    repaired_count = 0
    
    for i in range(total_frames):
        ret, frame = cap.read()
        
        # Logic: 
        # If frame index is known bad -> Use last_valid
        # If read failed (ret=False) -> Use last_valid
        # Else -> Use new frame AND update last_valid
        
        is_bad = (i in bad_indices) or (not ret)
        
        if is_bad:
            out_vid.write(last_valid_frame)
            repaired_count += 1
            if fast_mode and i not in bad_indices:
                bad_indices.add(i) # Add to set for depth repair later
        else:
            last_valid_frame = frame
            out_vid.write(frame)
        
        # Progress Log
        if i % 100 == 0:
            now = time.time()
            if now - last_log_time > 5.0:
                percent = (i / total_frames) * 100
                print(f"{log_prefix}Repairing: {percent:.1f}% ({i}/{total_frames}). Repaired: {repaired_count}")
                last_log_time = now
            
    cap.release()
    out_vid.release()
    
    # If fast_mode was ON and we found NO errors, we might have just re-encoded for nothing.
    # But checking 'repaired_count' handles this. 
    # However, we already overwrote the file. 
    # If repaired_count == 0, the output is identical (but re-encoded) to input.
    # Ideally, we should check this before committing, but we've already streamed it.
    
    if repaired_count == 0 and fast_mode:
        print(f"{log_prefix}No corrupt frames found during read. Keeping re-encoded version.")
        # Optimization: We could restore original if we trust it, but re-encoding ensures standard format.
        # Let's keep it consistent.
    
    # Replace Video
    if tmp_video.exists():
        tmp_video.replace(video_file) # Overwrite original location
        if not backup_enabled and src_video.exists():
            src_video.unlink() # Delete .orig

    # --- REPAIR DEPTH (Optional) ---
    if has_depth and bad_indices and process_depth:
        print(f"{log_prefix}Repairing depth (Found {len(bad_indices)} bad frames)...")
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
            print(f"{log_prefix}Error processing depth: {e}")
            if tmp_depth.exists(): tmp_depth.unlink()
            # Restore original if something failed
            if not backup_enabled and src_depth.exists() and not depth_file.exists():
                src_depth.rename(depth_file)

    print(f"{log_prefix}Done. Repaired {repaired_count} frames.")

def main():
    parser = argparse.ArgumentParser(description="Repair corrupt frames (In-Place, Keep Timeline)")
    parser.add_argument("session_dir", type=Path, help="Path to the session directory")
    parser.add_argument("--backup", action="store_true", help="Keep backups in _corrupt_backup folder")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2), 
                        help="Number of parallel workers")
    parser.add_argument("--fast", action="store_true", 
                        help="Fast mode: Skip ffmpeg scan, only repair frames that fail to read in OpenCV. Faster but less thorough.")
    parser.add_argument("--depth", action="store_true", 
                        help="Process depth as video frames (Default: False)")
    
    args = parser.parse_args()
    
    session_dir = args.session_dir.resolve()
    if not session_dir.exists():
        print(f"Error: Session directory not found: {session_dir}")
        sys.exit(1)
        
    print(f"=== Frame Repair Started: {session_dir.name} ===")
    print("Mode: REPLACE bad frames with PREVIOUS valid frame (Timestamp invariant)")
    if args.fast:
        print("Option: FAST mode enabled (Skipping ffmpeg scan)")
    
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
    pool_args = [(d, backup_dir, args.backup, args.fast, args.depth) for d in camera_dirs]
    
    # If only 1 worker or 1 camera, we can use simple loop for cleaner output
    if args.workers == 1 or len(camera_dirs) == 1:
        for p_arg in pool_args:
            process_camera_worker(p_arg)
    else:
        with multiprocessing.Pool(processes=args.workers) as pool:
            # Use tqdm to show total progress (files processed)
            list(tqdm(pool.imap_unordered(process_camera_worker, pool_args), total=len(camera_dirs), unit="cam"))

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
