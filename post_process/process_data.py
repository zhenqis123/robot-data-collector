import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path

# Add project root to path to import modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "post_process"))

# Mock tactask.glove_hand if missing
try:
    from tactask.glove_hand import HandType
except ImportError:
    # Create mock module
    from types import ModuleType
    tactask = ModuleType("tactask")
    sys.modules["tactask"] = tactask
    
    glove_hand = ModuleType("glove_hand")
    tactask.glove_hand = glove_hand
    sys.modules["tactask.glove_hand"] = glove_hand
    
    class HandType:
        RIGHT = 0
        LEFT = 1
    
    glove_hand.HandType = HandType
    print("Mocked tactask.glove_hand.HandType")

# Import VDGlove processing
from post_process.mocap.VDGloves.VDHand import VD_to_mano_keypoints

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_camera_serial(config, camera_id):
    for cam in config.get('cameras', []):
        if cam.get('type') == 'RealSense' and cam.get('id') == camera_id:
            return cam.get('serial')
    raise ValueError(f"No RealSense camera found with ID {camera_id} in config.")

def load_calibration(serial):
    # Try different naming conventions
    # Convention 1: .npz from VIVE_Station_Calibration
    path_npz = PROJECT_ROOT / "post_process/handeye" / f"cam2station_calib_{serial}.npz"
    if path_npz.exists():
        print(f"Loading calibration from {path_npz}")
        data = np.load(path_npz)
        if 'RT_cam2station' in data:
            return data['RT_cam2station']
    
    # Convention 2: .npy from test_calibration or others
    path_npy_test = PROJECT_ROOT / "post_process/handeye" / f"cam2station_test_{serial}.npy"
    if path_npy_test.exists():
        print(f"Loading calibration from {path_npy_test}")
        return np.load(path_npy_test)
        
    path_npy_base = PROJECT_ROOT / "post_process/handeye" / f"cam2base_{serial}.npy"
    if path_npy_base.exists():
        print(f"Loading calibration from {path_npy_base}")
        return np.load(path_npy_base)
        
    raise FileNotFoundError(f"Calibration file for camera {serial} not found in post_process/handeye/")

def process_vive_data(input_path, output_path, T_cam_station):
    if not input_path.exists():
        print(f"Vive data not found at {input_path}")
        return

    print(f"Processing Vive data: {input_path}")
    df = pd.read_csv(input_path)
    
    # Columns format: timestamp, python_timestamp, t0_valid, t0_px, t0_py, t0_pz, t0_qw, t0_qx, t0_qy, t0_qz, ...
    
    # We want to transform the pose (p, q) from Station frame to Camera frame.
    # P_cam = T_cam_station * P_station
    # T_cam_station is the inverse of what is usually stored as "RT_cam2station" (Station from Cam)
    
    # However, let's verify what we have in T_cam_station passed here.
    # The caller should pass the matrix that transforms Station -> Camera.
    
    # Identify tracker columns
    # We assume standard naming t{i}_...
    
    # Find max tracker index
    cols = df.columns
    trackers = set()
    for c in cols:
        if c.startswith('t') and '_' in c:
            parts = c.split('_')
            if parts[0][1:].isdigit():
                trackers.add(int(parts[0][1:]))
    
    sorted_trackers = sorted(list(trackers))
    
    for t_idx in sorted_trackers:
        prefix = f"t{t_idx}"
        valid_col = f"{prefix}_valid"
        
        # Transform positions
        px_col, py_col, pz_col = f"{prefix}_px", f"{prefix}_py", f"{prefix}_pz"
        qw_col, qx_col, qy_col, qz_col = f"{prefix}_qw", f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz"
        
        if valid_col not in df.columns:
            continue
            
        # Only process valid rows
        valid_mask = df[valid_col] == 1
        
        if not valid_mask.any():
            continue
            
        # Extract positions (N, 3)
        positions = df.loc[valid_mask, [px_col, py_col, pz_col]].values
        
        # Apply transformation: P_new = R * P + T
        # T_cam_station = [[R, T], [0, 1]]
        R_mat = T_cam_station[:3, :3]
        T_vec = T_cam_station[:3, 3]
        
        # (R @ P.T).T + T = P @ R.T + T
        positions_cam = positions @ R_mat.T + T_vec
        
        # Update DataFrame
        df.loc[valid_mask, [px_col, py_col, pz_col]] = positions_cam
        
        # Extract Quaternions (N, 4) -> (w, x, y, z)
        quats = df.loc[valid_mask, [qx_col, qy_col, qz_col, qw_col]].values # Scipy uses (x, y, z, w)
        
        # Apply rotation to orientation
        # R_new = R_transform * R_old
        r_old = R.from_quat(quats)
        r_trans = R.from_matrix(R_mat)
        r_new = r_trans * r_old
        
        quats_new = r_new.as_quat() # x, y, z, w
        
        df.loc[valid_mask, qx_col] = quats_new[:, 0]
        df.loc[valid_mask, qy_col] = quats_new[:, 1]
        df.loc[valid_mask, qz_col] = quats_new[:, 2]
        df.loc[valid_mask, qw_col] = quats_new[:, 3]
        
    df.to_csv(output_path, index=False)
    print(f"Saved processed Vive data to {output_path}")

def process_glove_data(input_path, output_path):
    if not input_path.exists():
        print(f"Glove data not found at {input_path}")
        return

    print(f"Processing Glove data: {input_path}")
    df = pd.read_csv(input_path)
    
    # Process Left Hand
    if 'left_detected' in df.columns:
        process_hand_keypoints(df, 'l', HandType.LEFT)
        
    # Process Right Hand
    if 'right_detected' in df.columns:
        process_hand_keypoints(df, 'r', HandType.RIGHT)
        
    df.to_csv(output_path, index=False)
    print(f"Saved processed Glove data to {output_path}")

def process_hand_keypoints(df, prefix, hand_type):
    # Columns: {prefix}_x0..20, {prefix}_y0..20, {prefix}_z0..20
    # Wrist: {prefix}_wrist_qw...
    
    # 1. Collect Keypoints
    # Shape: (N, 21, 3)
    kp_cols = []
    for i in range(21):
        kp_cols.extend([f"{prefix}_x{i}", f"{prefix}_y{i}", f"{prefix}_z{i}"])
    
    # Check if columns exist
    if not all(c in df.columns for c in kp_cols):
        return

    keypoints_flat = df[kp_cols].values
    N = len(df)
    keypoints = keypoints_flat.reshape(N, 21, 3)
    
    # 2. Collect Wrist Rotation
    # Quat: w, x, y, z
    w_col = f"{prefix}_wrist_qw"
    x_col = f"{prefix}_wrist_qx"
    y_col = f"{prefix}_wrist_qy"
    z_col = f"{prefix}_wrist_qz"
    
    if not all(c in df.columns for c in [w_col, x_col, y_col, z_col]):
        return
        
    quats = df[[x_col, y_col, z_col, w_col]].values # Scipy (x, y, z, w)
    
    # Scipy Rotation expects (x, y, z, w)
    rotations = R.from_quat(quats).as_matrix() # (N, 3, 3)
    
    # 3. Call VD_to_mano_keypoints
    # VD_to_mano_keypoints(hand_keypoints, hand_wrist_rot, hand_type)
    # It supports batch processing if implemented correctly (N, 21, 3) and (N, 3, 3)
    
    # Since VD_to_mano_keypoints implementation modifies input in place or returns new?
    # Checking implementation:
    # hand_keypoints = hand_keypoints - ... (creates new)
    # Returns hand_keypoints.
    
    mano_kps = VD_to_mano_keypoints(keypoints.copy(), rotations, hand_type) # (N, 21, 3)
    
    # 4. Save back to DataFrame as new columns
    # We'll use 'mano_{prefix}_x{i}'
    
    mano_flat = mano_kps.reshape(N, -1)
    
    new_cols = []
    for i in range(21):
        new_cols.extend([f"mano_{prefix}_x{i}", f"mano_{prefix}_y{i}", f"mano_{prefix}_z{i}"])
        
    # Assign columns
    df[new_cols] = mano_flat

def main():
    parser = argparse.ArgumentParser(description="Post-process Vive and Glove data.")
    parser.add_argument("--camera_id", type=int, default=0, help="ID of the camera to use as base frame")
    parser.add_argument("--input_dir", type=str, default="my_project/resources/logs/captures", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="my_project/resources/logs/post_process", help="Output directory")
    
    args = parser.parse_args()
    
    input_path = PROJECT_ROOT / args.input_dir
    output_path = PROJECT_ROOT / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Calibration
    config_path = PROJECT_ROOT / "my_project/resources/config.json"
    config = load_config(config_path)
    serial = get_camera_serial(config, args.camera_id)
    
    print(f"Target Camera: ID={args.camera_id}, Serial={serial}")
    
    try:
        RT_station_cam = load_calibration(serial)
        # Based on analysis: The saved matrix is T_station_cam (Camera pose in Station frame)
        # We need T_cam_station = inv(T_station_cam)
        T_cam_station = np.linalg.inv(RT_station_cam)
        print("Calibration matrix loaded and inverted.")
    except Exception as e:
        print(f"Warning: Calibration failed ({e}). Vive data will NOT be transformed.")
        T_cam_station = None

    # 2. Process Vive Data
    if T_cam_station is not None:
        process_vive_data(input_path / "vive_data.csv", output_path / "vive_data.csv", T_cam_station)
        
    # 3. Process Glove Data
    process_glove_data(input_path / "glove_data.csv", output_path / "glove_data.csv")

if __name__ == "__main__":
    main()
