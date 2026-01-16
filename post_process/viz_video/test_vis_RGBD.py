import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_first_frame(capture_root, cam_id):
    cam_dir = capture_root / f"RealSense_{cam_id}"
    
    # Load Color
    # Try mkv first
    color_path = cam_dir / "rgb.mkv"
    color_img = None
    if color_path.exists():
        cap = cv2.VideoCapture(str(color_path))
        ret, frame = cap.read()
        if ret:
            color_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
    
    # Fallback to PNGs if needed (though structure suggests mkv)
    if color_img is None:
        png_path = cam_dir / "color/000000.png" 
        if png_path.exists():
             color_img = cv2.cvtColor(cv2.imread(str(png_path)), cv2.COLOR_BGR2RGB)

    # Load Aligned Depth from H5
    depth_h5 = cam_dir / "depth_aligned.h5"
    depth_img = None
    if depth_h5.exists():
        with h5py.File(depth_h5, 'r') as f:
            if "depth" in f:
                depth_img = f["depth"][0] # Read first frame

    # Plot
    if color_img is not None and depth_img is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(color_img)
        ax1.set_title(f"RGB Camera {cam_id}")
        
        # Depth is usually mm, clip for visibility
        ax2.imshow(depth_img, cmap='plasma', vmin=0, vmax=2000) 
        ax2.set_title(f"Aligned Depth Camera {cam_id}")
        
        plt.show()
    else:
        print(f"Failed to load data for {cam_id}")

root = Path("my_project/resources/logs/captures/2026-01-15/sess_20260115_135327")
cameras = ["151322070562", "152122075515", "922612070441"]

print(f"Checking {root}...")
for cam in cameras:
    visualize_first_frame(root, cam)
