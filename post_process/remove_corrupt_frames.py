#!/usr/bin/env python3
"""
Remove corrupt frames from a session (Video, Depth, Manus, Tactile).
IN-PLACE / FAST VERSION
1. Detects corrupt frames in MKV/MP4 files using ffmpeg (Parallel).
2. Identifies corresponding timestamps.
3. Modifies files IN-PLACE (with backup) to remove corrupt frames.
   - Moves original files to '_corrupt_backup/' folder.
   - Writes filtered content to original location.
4. Uses multiprocessing for camera processing.

Usage:
    python tools/remove_corrupt_frames.py /path/to/session
    (Optional: --no-backup to delete originals immediately, saves space but risky)
"""

import os
import sys
import argparse
import subprocess
import re
import json
import shutil
import csv
import multiprocessing
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm

try:
    import h5py
except ImportError:
    print("Error: h5py not installed.")
    sys.exit(1)

# Configuration
EXT_VIDEO = {'.mkv', '.mp4', '.avi'}
EXT_H5 = {'.h5', '.hdf5'}
EXT_CSV = {'.csv'}

def get_video_metadata(file_path: Path) -> int:
    """Retrieves total frame count using ffprobe."""
    try:
        command = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", "-select_streams", "v:0",
            str(file_path)
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        data = json.loads(result.stdout)
        if not data.get("streams"): return 0
        stream = data["streams"][0]
        frames = stream.get("nb_frames")
        if not frames:
            duration = float(data["format"].get("duration", 0))
            avg_frame_rate = stream.get("avg_frame_rate", "30/1")
            num, den = map(float, avg_frame_rate.split('/'))
            fps = num / den if den > 0 else 30
            frames = int(duration * fps)
        return int(frames)
    except Exception as e:
        return 0

def detect_corrupt_frames(file_path: Path) -> List[int]:
    """
    Runs ffmpeg to scan for decoding errors.
    """
    # print(f"Scanning: {file_path.name}")
    command = ["ffmpeg", "-v", "info", "-i", str(file_path), "-f", "null", "-"]
    
    error_frames = set()
    current_frame = 0
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
                    
                    match_frame = frame_pattern.search(line)
                    if match_frame:
                        current_frame = int(match_frame.group(1))
                    
                    lower_line = line.lower()
                    if "error" in lower_line or ("[h264" in lower_line and "frame=" not in lower_line):
                         if any(x in line for x in ["Input #", "Output #", "Metadata:", "Duration:"]):
                             continue
                         error_frames.add(current_frame)
                         
    except Exception:
        return []

    return sorted(list(error_frames))

def read_timestamps(csv_path: Path) -> Dict[int, int]:
    mapping = {}
    if not csv_path.exists():
        return mapping
    try:
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    idx_str = row.get("frame_index") or row.get("color_frame_index") or row.get("index")
                    ts_str = row.get("timestamp_ms") or row.get("color_timestamp_ms") or row.get("timestamp")
                    if idx_str and ts_str:
                        mapping[int(idx_str)] = int(float(ts_str))
                except ValueError:
                    continue
    except Exception:
        pass
    return mapping

def scan_worker(video_path: Path) -> Set[int]:
    """Worker function for parallel scanning."""
    bad_indices = detect_corrupt_frames(video_path)
    if not bad_indices:
        return set()
    
    ts_csv = video_path.parent / "timestamps.csv"
    idx_to_ts = read_timestamps(ts_csv)
    
    bad_ts = set()
    for idx in bad_indices:
        ts = idx_to_ts.get(idx)
        if ts is not None:
            bad_ts.add(ts)
    
    if bad_ts:
        print(f"  [Found Corruption] {video_path.name}: {len(bad_indices)} bad frames")
    return bad_ts

def filter_csv_file(src: Path, dst: Path, bad_timestamps: Set[int], threshold_ms: int = 15):
    """Filter CSV and write to dst."""
    if not src.exists(): return

    def clean_lines(f):
        for line in f:
            yield line.replace('\0', '')

    try:
        with src.open("r", newline="", encoding="utf-8", errors="replace") as f_in, \
             dst.open("w", newline="", encoding="utf-8") as f_out:
            
            reader = csv.DictReader(clean_lines(f_in))
            fieldnames = reader.fieldnames
            
            if not fieldnames: return

            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            
            ts_col = None
            for cand in ["timestamp_ms", "color_timestamp_ms", "timestamp", "ts", "Time", "time"]:
                if cand in fieldnames:
                    ts_col = cand
                    break
            
            if not ts_col:
                # Copy all
                for row in reader: writer.writerow(row)
                return

            for row in reader:
                try:
                    ts_val = row.get(ts_col)
                    if not ts_val:
                        writer.writerow(row)
                        continue
                    ts = int(float(ts_val))
                    
                    is_bad = False
                    if ts in bad_timestamps:
                        is_bad = True
                    else:
                        for bad_ts in bad_timestamps:
                            if abs(ts - bad_ts) <= threshold_ms:
                                is_bad = True
                                break
                    if not is_bad:
                        writer.writerow(row)
                except:
                    writer.writerow(row)
    except Exception as e:
        print(f"Error filtering {src}: {e}")

def process_camera(args):
    """
    Worker for processing a single camera folder.
    Args: (cam_dir, bad_timestamps, backup_dir, threshold_ms, backup_enabled)
    """
    cam_dir, bad_timestamps, backup_dir, threshold_ms, backup_enabled = args
    
    video_file = None
    for f in cam_dir.iterdir():
        if f.suffix in EXT_VIDEO and "viz" not in f.name:
            video_file = f
            break
            
    if not video_file: return
    
    ts_file = cam_dir / "timestamps.csv"
    depth_file = cam_dir / "depth.h5"
    
    # Backup structure
    cam_backup = backup_dir / cam_dir.name
    if backup_enabled:
        cam_backup.mkdir(parents=True, exist_ok=True)
    
    # 1. Filter Video
    # If backup enabled, copy to backup then read from backup.
    # If backup disabled, rename original to .orig, read from .orig, write to original, then delete .orig
    
    source_video = video_file
    
    if backup_enabled:
        v_backup = cam_backup / video_file.name
        if not v_backup.exists():
            shutil.copy2(video_file, v_backup)
        source_video = v_backup
    else:
        # No backup mode: use a temporary rename for source
        # But wait, we can't rename if we are iterating? 
        # Actually, safely: rename video_file -> video_file.orig
        v_orig = video_file.with_suffix(video_file.suffix + ".orig")
        if not v_orig.exists():
            os.rename(video_file, v_orig)
        source_video = v_orig
    
    # Use fast decoding/encoding settings
    cap = cv2.VideoCapture(str(source_video)) # Read from source
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Temp output
    v_tmp = cam_dir / (video_file.name + ".tmp.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(v_tmp), fourcc, fps, (width, height))
    
    # 2. Filter Depth
    d_in = None
    d_out = None
    d_tmp = None
    d_dset_out = None
    source_depth = None
    
    if depth_file.exists():
        if backup_enabled:
            d_backup = cam_backup / depth_file.name
            if not d_backup.exists():
                shutil.copy2(depth_file, d_backup)
            source_depth = d_backup
        else:
            d_orig = depth_file.with_suffix(depth_file.suffix + ".orig")
            if not d_orig.exists():
                os.rename(depth_file, d_orig)
            source_depth = d_orig

        # Try to open source depth
        try:
             d_in = h5py.File(source_depth, "r")
             if "depth" in d_in:
                 d_dset = d_in["depth"]
                 d_tmp = cam_dir / (depth_file.name + ".tmp.h5")
                 d_out = h5py.File(d_tmp, "w")
                 d_dset_out = d_out.create_dataset(
                     "depth",
                     shape=(0, d_dset.shape[1], d_dset.shape[2]),
                     maxshape=(None, d_dset.shape[1], d_dset.shape[2]),
                     dtype=d_dset.dtype,
                     compression="gzip" 
                 )
        except Exception as e:
             # If depth broken, cleanup
             if d_in: d_in.close(); d_in = None
             if d_out: d_out.close(); d_out = None
    
    # 3. Filter Timestamps
    source_ts = None
    if backup_enabled:
        t_backup = cam_backup / ts_file.name
        if not t_backup.exists():
            shutil.copy2(ts_file, t_backup)
        source_ts = t_backup
    else:
        # For small CSV, just read into memory from original, overwrite original later
        source_ts = ts_file
    
    idx_to_ts = read_timestamps(source_ts)
    
    # Read all timestamps rows
    ts_rows = []
    try:
        with source_ts.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            ts_rows = list(reader)
    except:
        pass
        
    new_ts_rows = []
    kept_count = 0
    
    # Loop
    for i in range(total):
        ret, frame = cap.read()
        if not ret: break
        
        ts = idx_to_ts.get(i, -1)
        is_bad = False
        if ts != -1:
            if ts in bad_timestamps:
                is_bad = True
            else:
                for b in bad_timestamps:
                    if abs(ts - b) <= threshold_ms:
                        is_bad = True
                        break
        
        if not is_bad:
            out_video.write(frame)
            
            if d_dset_out is not None and d_in:
                # Check bounds
                if i < len(d_in["depth"]):
                     d_dset_out.resize(kept_count + 1, axis=0)
                     d_dset_out[kept_count] = d_in["depth"][i]
            
            if i < len(ts_rows):
                row = ts_rows[i].copy()
                # Update indices
                for k in ["frame_index", "color_frame_index", "depth_frame_index"]:
                    if k in row: row[k] = kept_count
                new_ts_rows.append(row)
                
            kept_count += 1
            
    cap.release()
    out_video.release()
    if d_in: d_in.close()
    if d_out: d_out.close()
    
    # Replace Files
    # Video
    if v_tmp.exists():
        os.replace(v_tmp, video_file)
        if not backup_enabled and source_video.exists():
             os.remove(source_video) # Delete .orig
    
    # Depth
    if d_tmp and d_tmp.exists():
        os.replace(d_tmp, depth_file)
        if not backup_enabled and source_depth and source_depth.exists():
             os.remove(source_depth) # Delete .orig
        
    # Timestamps
    if new_ts_rows:
        with ts_file.open("w", newline="", encoding="utf-8") as f:
            if ts_rows:
                writer = csv.DictWriter(f, fieldnames=ts_rows[0].keys())
                writer.writeheader()
                writer.writerows(new_ts_rows)
    
    print(f"  [Processed] {cam_dir.name}: Kept {kept_count}/{total} frames")

def main():
    parser = argparse.ArgumentParser(description="Remove corrupt frames IN-PLACE")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--backup", action="store_true", help="Keep backups in _corrupt_backup (Default: False, deletes originals)")
    args = parser.parse_args()
    
    session_dir = args.session_dir.resolve()
    if not session_dir.exists():
        print("Session not found")
        sys.exit(1)
        
    print(f"=== Optimized Corruption Removal: {session_dir.name} ===")
    
    # Clean up any leftover tmp files from previous runs
    print("Cleaning up leftover .tmp files...")
    for tmp in session_dir.rglob("*.tmp*"):
        try:
            tmp.unlink()
        except:
            pass

    # 1. Scan Videos (Parallel)
    videos = [p for p in session_dir.rglob("*") if p.suffix in EXT_VIDEO and "viz" not in p.name and "_corrupt_backup" not in str(p)]
    
    print(f"Scanning {len(videos)} videos for corruption...")
    bad_timestamps = set()
    
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap_unordered(scan_worker, videos), total=len(videos)))
        for r in results:
            bad_timestamps.update(r)
            
    if not bad_timestamps:
        print("No corruption detected. Exiting.")
        return

    print(f"Found {len(bad_timestamps)} unique corrupt timestamps.")
    
    # 2. Setup Backup
    backup_dir = session_dir / "_corrupt_backup"
    if not args.backup:
        print("Running in NO-BACKUP mode. Originals will be deleted.")
    else:
        backup_dir.mkdir(exist_ok=True)
        print(f"Backups will be stored in: {backup_dir}")

    # 3. Process Cameras (Parallel)
    # Identify camera directories
    camera_dirs = []
    for root, dirs, files in os.walk(session_dir):
        if "_corrupt_backup" in root: continue
        if "timestamps.csv" in files:
            # Check for video
            has_vid = any(f.endswith(tuple(EXT_VIDEO)) and "viz" not in f for f in files)
            if has_vid:
                camera_dirs.append(Path(root))
    
    print(f"Processing {len(camera_dirs)} cameras...")
    
    pool_args = [(d, bad_timestamps, backup_dir, 15, args.backup) for d in camera_dirs]
    with multiprocessing.Pool() as pool:
         # Use fewer processes for writing to avoid disk thrashing?
         # 4-8 workers usually fine.
         list(tqdm(pool.imap_unordered(process_camera, pool_args), total=len(camera_dirs)))

    # 4. Process Global CSVs (Main thread)
    print("Filtering global CSVs...")
    for f in session_dir.iterdir():
        if f.suffix == ".csv" and f.name != "frames_aligned.csv":
             # Check if it looks like data
             # Usually: manus_data.csv, tactile_data.csv, etc.
             # We want to be inclusive but exclude meta.
             if "manus" in f.name.lower() or "tactile" in f.name.lower() or "glove" in f.name.lower() or "data" in f.name.lower():
                 print(f"  Filtering {f.name}")
                 # Backup
                 if args.backup:
                     shutil.copy2(f, backup_dir / f.name)
                 
                 # Filter to tmp
                 tmp_csv = f.with_suffix(".csv.tmp")
                 # We use a larger threshold for global data as frequencies might differ (e.g. 1000Hz vs 30Hz)
                 # However, "align with cleaned video" implies removing the time chunk corresponding to the bad video frame.
                 # Video frame duration ~33ms. We remove [t-15, t+15].
                 filter_csv_file(f, tmp_csv, bad_timestamps, threshold_ms=16) 
                 os.replace(tmp_csv, f)
    
    # Also recursively check for tactile_left/tactile_data.csv etc if structure is nested
    for root, dirs, files in os.walk(session_dir):
        if "_corrupt_backup" in root: continue
        for f in files:
            if f.endswith(".csv") and f != "frames_aligned.csv" and f != "timestamps.csv":
                 if "manus" in f.lower() or "tactile" in f.lower() or "glove" in f.lower():
                     f_path = Path(root) / f
                     # Avoid re-processing if already handled in root loop
                     if f_path.parent == session_dir: continue 
                     
                     print(f"  Filtering {f_path.relative_to(session_dir)}")
                     
                     # Backup
                     if args.backup:
                         rel_backup = backup_dir / f_path.parent.relative_to(session_dir)
                         rel_backup.mkdir(parents=True, exist_ok=True)
                         shutil.copy2(f_path, rel_backup / f)
                         
                     tmp_csv = f_path.with_suffix(".csv.tmp")
                     filter_csv_file(f_path, tmp_csv, bad_timestamps, threshold_ms=16)
                     os.replace(tmp_csv, f_path)

    # 5. Cleanup
    aligned_csv = session_dir / "frames_aligned.csv"
    if aligned_csv.exists():
        aligned_csv.unlink()
        
    # Remove backup dir if no backup requested (cleanup any empty dirs created by logic)
    if not args.backup and backup_dir.exists():
        shutil.rmtree(backup_dir)

    print("\nDone. Corrupt frames removed.")
    if args.backup:
        print(f"Original files backed up to: {backup_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # Safer for opencv/cuda interaction
    main()
