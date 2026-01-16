import h5py
import cv2
import numpy as np
import argparse
import sys
import os

def visualize_depth_h5(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Opening HDF5 file: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            if 'depth' not in f:
                print("Error: 'depth' dataset not found in HDF5 file.")
                print(f"Available keys: {list(f.keys())}")
                return

            depth_dataset = f['depth']
            num_frames = depth_dataset.shape[0]
            height = depth_dataset.shape[1]
            width = depth_dataset.shape[2]
            
            print(f"Dataset info: Shape={depth_dataset.shape}, Dtype={depth_dataset.dtype}")
            print("Control: Press 'q' to quit, 'SPACE' to pause/resume.")

            # Create a window
            window_name = "Depth Visualization"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Use chunks for efficient reading if possible, but reading frame by frame is okay
            for i in range(num_frames):
                # Read frame
                depth_frame = depth_dataset[i]
                
                # Normalize for visualization
                # Depth is usually uint16 in millimeters. 
                # To visualize, we can clip and normalize or apply a colormap.
                
                # Simple normalization (0 to 2 meters visualized as full range, for example)
                # Or just Min/Max normalization per frame for contrast
                
                # Option A: Dynamic scaling per frame (good for visibility)
                min_val = np.min(depth_frame[depth_frame > 0]) if np.any(depth_frame > 0) else 0
                max_val = np.max(depth_frame)
                
                if max_val > min_val:
                    # Scale to 0-255
                    depth_vis = ((depth_frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    depth_vis = np.zeros_like(depth_frame, dtype=np.uint8)

                # Option B: Fixed range (e.g. 0-1000mm) just for checking specific range
                # depth_vis = cv2.convertScaleAbs(depth_frame, alpha=0.03) 

                # Apply colormap
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # Add frame index text
                cv2.putText(depth_colormap, f"Frame: {i}/{num_frames}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow(window_name, depth_colormap)
                
                # Keyboard control
                key = cv2.waitKey(30) # ~30fps
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0) # Pause until key press

            cv2.destroyAllWindows()
            print("Visualization finished.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default path based on your workspace context
        file_path = "/home/zc/HandMotion/temp_new/temp/robot-data-collector/my_project/resources/logs/captures/2026-01-15/sess_20260115_210650/RealSense_213222079798/depth.h5"
        
    visualize_depth_h5(file_path)
