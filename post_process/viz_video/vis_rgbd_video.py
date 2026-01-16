import cv2
import h5py
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import time

def play_aligned_video(capture_root, cam_id, fps=30):
    capture_root = Path(capture_root)
    cam_name = f"RealSense_{cam_id}"
    cam_dir = capture_root / cam_name
    csv_path = capture_root / "frames_aligned.csv"
    
    if not cam_dir.exists():
        print(f"Camera directory not found: {cam_dir}")
        return

    # 1. Load Alignment Table
    print(f"Loading alignment data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 2. Find the correct column for this camera
    # Check for various naming conventions
    col_candidates = [
        f"{cam_name}#frame_index",
        f"{cam_name}_frame_index",
        f"RealSense#{cam_id}_frame_index",
        f"RealSense_{cam_id}_frame_index"
    ]
    
    frame_col = None
    for col in df.columns:
        if col in col_candidates:
            frame_col = col
            break
        # Fuzzy match if exact match fails
        if cam_id in col and "frame_index" in col:
            frame_col = col
            break
            
    # Check if this is the reference camera (sometimes column name is just ref_frame_index)
    # usually we might not know if it is ref without metadata, but this is a fallback
    if frame_col is None and "ref_frame_index" in df.columns:
        # Check metadata or assume based on user input (risky, but printing info helps)
        print("Warning: Specific column not found, checking if 'ref_frame_index' applies...")
        # For safety, let's just warn for now.
        
    # Check column names
    ref_ts_col = "ref_timestamp_ms"
    delta_col = f"{cam_name}_delta_ms" 
    # Try alternate delta col name if needed
    if delta_col not in df.columns:
        delta_col = f"RealSense#{cam_id}_delta_ms"

    if frame_col is None:
        print(f"Could not find frame index column for camera {cam_id}")
        print(f"Available columns: {list(df.columns)}")
        return
        
    print(f"Using column '{frame_col}' for frame indices.")

    # 3. Open Resources
    rgb_path = cam_dir / "rgb.mkv"
    depth_path = cam_dir / "depth.h5"
    
    cap = None
    depth_file = None
    depth_dataset = None
    
    if rgb_path.exists():
        cap = cv2.VideoCapture(str(rgb_path))
    else:
        print(f"RGB file not found: {rgb_path}")
        
    if depth_path.exists():
        depth_file = h5py.File(depth_path, 'r')
        if "depth" in depth_file:
            depth_dataset = depth_file["depth"]
    else:
        print(f"Depth h5 file not found: {depth_path}")

    if cap is None and depth_dataset is None:
        print("No data found to play.")
        return

    cv2.namedWindow(f"RGB-D Player {cam_id}", cv2.WINDOW_NORMAL)
    
    print("Starting playback. Press 'q' to quit, 'SPACE' to pause.")
    
    # Pre-calculate or just loop through row indices
    num_rows = len(df)
    
    paused = False
    
    current_log_frame = 0 # Logical aligned frame index
    
    # Cache for tracking video seek
    last_phy_idx = -1
    
    while current_log_frame < num_rows:
        if not paused:
            # Get physical frame index
            row_data = df.iloc[current_log_frame]
            phy_idx_raw = row_data[frame_col]
            
            # Get timestamps
            ref_ts = row_data.get(ref_ts_col, float('nan'))
            delta_ts = row_data.get(delta_col, float('nan'))
            
            # Prepare images
            rgb_img = None
            depth_vis = None
            
            valid_frame = not pd.isna(phy_idx_raw)
            
            if valid_frame:
                phy_idx = int(phy_idx_raw)
                
                # Fetch RGB
                if cap is not None:
                    # Optimize seeking: if next frame is sequential, just read
                    if phy_idx == last_phy_idx + 1:
                        ret, frame = cap.read()
                        if ret:
                            rgb_img = frame
                            last_phy_idx = phy_idx
                        else:
                            # Read failed, try seeking
                            cap.set(cv2.CAP_PROP_POS_FRAMES, phy_idx)
                            ret, frame = cap.read()
                            if ret: 
                                rgb_img = frame
                                last_phy_idx = phy_idx
                    else:
                        # Non-sequential (frame drop or jump), must seek
                        cap.set(cv2.CAP_PROP_POS_FRAMES, phy_idx)
                        ret, frame = cap.read()
                        if ret:
                            rgb_img = frame
                            last_phy_idx = phy_idx
                
                # Fetch Depth
                if depth_dataset is not None:
                    try:
                        # Assuming depth dataset is indexed by physical frame index
                        # OR is it aligned to something else? 
                        # If align_depth.py was used, it aligns depth to RGB frames.
                        # So depth index == RGB frame index (physical).
                        if phy_idx < len(depth_dataset):
                            d_frame = depth_dataset[phy_idx]
                            
                            # Normalize for visualization (0-2000mm mapped to 0-255)
                            # Using fixed range for better visual consistency
                            depth_clipped = np.clip(d_frame, 0, 2000)
                            depth_norm = (depth_clipped / 2000.0 * 255).astype(np.uint8)
                            depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                    except Exception as e:
                        print(f"Error reading depth frame {phy_idx}: {e}")

            # Construct display image
            h, w = 480, 640 # Default fallback
            if rgb_img is not None:
                h, w = rgb_img.shape[:2]
            elif depth_vis is not None:
                h, w = depth_vis.shape[:2]
                
            if rgb_img is None:
                rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(rgb_img, "No RGB", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            if depth_vis is None:
                depth_vis = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(depth_vis, "No Depth", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Mark dropped frames
            if not valid_frame:
                cv2.putText(rgb_img, "DROPPED/INVALID", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Concatenate
            combined = np.hstack((rgb_img, depth_vis))
            
            # Add Info Overlay
            info_text = f"Log Frame: {current_log_frame} | Phy Frame: {phy_idx_raw if valid_frame else 'N/A'}"
            cv2.putText(combined, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            ts_text = f"Ref TS: {ref_ts:.0f} ms | Delta: {delta_ts:.1f} ms"
            cv2.putText(combined, ts_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow(f"RGB-D Player {cam_id}", combined)
            
            current_log_frame += 1
        
        # Handle Input
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            
    # Cleanup
    if cap:
        cap.release()
    if depth_file:
        depth_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play synchronized RGB and Depth video.")
    parser.add_argument("--root", type=str, default="my_project/resources/logs/captures/2026-01-15/sess_20260115_135327", help="Session root directory")
    parser.add_argument("--cam", type=str, default="151322070562", help="Camera Serial Number or ID")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    
    args = parser.parse_args()
    
    play_aligned_video(args.root, args.cam, args.fps)
