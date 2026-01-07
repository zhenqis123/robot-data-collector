import os
import json
import pandas as pd
import argparse
from pathlib import Path

class SessionAligner:
    """
    Handles time synchronization and alignment of multimodal data (Video, VIVE, Gloves)
    based on host system timestamps.
    """
    def __init__(self, session_path):
        self.session_path = Path(session_path)
        if not self.session_path.exists():
            raise FileNotFoundError(f"Session path does not exist: {session_path}")
        
        self.meta = self._load_meta()
        self.aligned_df = None

    def _load_meta(self):
        meta_path = self.session_path / "meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        return {}

    def get_camera_folder(self, camera_id=None):
        """
        Locate the RealSense camera folder.
        If camera_id is provided, tries to match it.
        Otherwise defaults to the first 'RealSense' folder found.
        """
        # Strategy 1: Look at file system for RealSense folders
        subdirs = [d for d in self.session_path.iterdir() if d.is_dir()]
        rs_dirs = sorted([d for d in subdirs if "RealSense" in d.name])
        
        if not rs_dirs:
            raise FileNotFoundError("No RealSense camera folders found in session.")
            
        chosen_dir = None
        
        # Strategy 2: If we have metadata, try to find the directory matching the camera ID
        if camera_id is not None and self.meta:
            cameras = self.meta.get('cameras', [])
            target_serial = None
            for cam in cameras:
                # Check for integer ID match or string ID match
                if cam.get('id') == camera_id or cam.get('id') == str(camera_id):
                    # Found the camera config, now find its serial to match folder
                    # format usually "RealSense#<Serial>" or just "<Serial>"
                    if 'serial' in cam:
                        target_serial = cam['serial']
                    elif 'id' in cam and 'RealSense#' in cam['id']:
                         target_serial = cam['id'].split('#')[1]
                    break
            
            if target_serial:
                for d in rs_dirs:
                    if target_serial in d.name:
                        chosen_dir = d
                        break

        # Fallback: Just take the first one or valid index
        if chosen_dir is None:
            if camera_id is not None and isinstance(camera_id, int) and camera_id < len(rs_dirs):
                chosen_dir = rs_dirs[camera_id]
            else:
                chosen_dir = rs_dirs[0]

        print(f"Selected master camera folder: {chosen_dir.name}")
        return chosen_dir

    def align_session(self, tolerance_ms=50):
        """
        Main method to perform alignment.
        """
        # 1. Load Master Clock (Video Timestamps)
        cam_folder = self.get_camera_folder()
        video_timestamps_path = cam_folder / "timestamps.csv"
        
        if not video_timestamps_path.exists():
            raise FileNotFoundError(f"Video timestamps not found: {video_timestamps_path}")
            
        print(f"Loading master clock from: {video_timestamps_path}")
        # Columns: frame_index,timestamp_iso,timestamp_ms,device_timestamp_ms,rgb_path,depth_path
        master_df = pd.read_csv(video_timestamps_path)
        
        # Ensure timestamp_ms is integer and sorted
        if 'timestamp_ms' not in master_df.columns:
            raise KeyError("timestamps.csv missing 'timestamp_ms' column")

        master_df['timestamp_ms'] = master_df['timestamp_ms'].astype('int64')
        master_df = master_df.sort_values('timestamp_ms')
        
        # Prefix columns to avoid collisions
        master_df = master_df.add_prefix('video_')
        
        print(f"Base frames: {len(master_df)}")

        # 2. Align VIVE
        master_df = self._align_device(
            master_df, 
            "vive_data.csv", 
            prefix="vive_", 
            time_col="timestamp", 
            tolerance_ms=tolerance_ms
        )

        # 3. Align VDGlove
        master_df = self._align_device(
            master_df,
            "glove_data.csv",
            prefix="glove_",
            time_col="timestamp",
            tolerance_ms=tolerance_ms
        )

        # 4. Align TacGlove (TODO)
        master_df = self._handle_tac_glove(master_df)
        
        self.aligned_df = master_df
        return master_df

    def _align_device(self, master_df, filename, prefix, time_col='timestamp', tolerance_ms=50):
        file_path = self.session_path / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found, skipping alignment for {prefix}")
            return master_df

        print(f"Aligning {filename}...")
        try:
            device_df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return master_df
        
        # Verify timestamp column
        if time_col not in device_df.columns:
            print(f"Error: Column '{time_col}' not found in {filename}. Columns: {device_df.columns}")
            # Try to infer if it's the first column
            if len(device_df.columns) > 0:
                col0 = device_df.columns[0]
                print(f"Falling back to first column '{col0}' as timestamp")
                time_col = col0
            else:
                return master_df
            
        device_df[time_col] = device_df[time_col].astype('int64')
        device_df = device_df.sort_values(time_col)
        
        # Prepare for merge
        prefixed_df = device_df.add_prefix(prefix)
        right_on_key = f"{prefix}{time_col}"
        
        # merge_asof performs a left-join finding the nearest match
        merged_df = pd.merge_asof(
            master_df,
            prefixed_df,
            left_on='video_timestamp_ms',
            right_on=right_on_key,
            direction='nearest',
            tolerance=tolerance_ms
        )
        
        # Calculate alignment metrics
        matched_mask = merged_df[right_on_key].notna()
        match_count = matched_mask.sum()
        
        # Calculate time diff for matched frames (Sync Error)
        if match_count > 0:
            diff = merged_df.loc[matched_mask, 'video_timestamp_ms'] - merged_df.loc[matched_mask, right_on_key]
            merged_df[f'{prefix}sync_error_ms'] = diff
            mean_diff = diff.abs().mean()
            print(f"  -> Matched {match_count} / {len(master_df)} frames. Mean sync error: {mean_diff:.2f} ms")
        else:
            print(f"  -> No matches found within {tolerance_ms} ms tolerance.")

        return merged_df

    def _handle_tac_glove(self, master_df):
        """
        TODO: Implement TacGlove alignment.
        TacGlove data is likely stored in 'tactile_left' and 'tactile_right' folders.
        Each contains 'timestamps.csv' and binary/csv data.
        
        Expected structure:
        - tactile_left/
            - timestamps.csv (contains timestamp_ms mapping to data index)
            - tactile_data.csv (or .bin)
        """
        # Check for presence
        left_path = self.session_path / "tactile_left"
        right_path = self.session_path / "tactile_right"
        
        if left_path.exists() or right_path.exists():
            print("[INFO] TacGlove data detected. Alignment TODO")
            # Placeholder for logic:
            # 1. Load tactile_left/timestamps.csv
            # 2. merge_asof with master_df
            # 3. Load actual tactile data using the index from timestamps if needed
        
        return master_df

    def save_aligned_data(self, output_filename=None):
        if self.aligned_df is None:
            print("No aligned data to save.")
            return
        
        # 默认使用 Parquet 格式，扩展名决定存储格式
        if output_filename is None:
            output_filename = "aligned_session.parquet"
        
        out_path = self.session_path / output_filename
        
        if out_path.suffix == '.parquet':
            try:
                # 使用 zstd 压缩，兼顾速度和压缩率
                self.aligned_df.to_parquet(out_path, index=False, compression='zstd')
                print(f"Saved aligned data to {out_path} (Format: Parquet, Compression: ZSTD)")
            except ImportError:
                print("Error: 'pyarrow' or 'fastparquet' library not found. Please install one (e.g., `pip install pyarrow`).")
                print("Falling back to CSV...")
                self.aligned_df.to_csv(out_path.with_suffix('.csv'), index=False)
        else:
            self.aligned_df.to_csv(out_path, index=False)
            print(f"Saved aligned data to {out_path} (Format: CSV)")

    @staticmethod
    def load_aligned_data(file_path):
        """
        Efficiently load aligned data from Parquet or CSV.
        Usage: df = SessionAligner.load_aligned_data("path/to/aligned_session.parquet")
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        else:
            return pd.read_csv(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align session data (VIVE, VDGlove) to Video timestamps.")
    # parser.add_argument("session_path", help="Path to the session folder")
    parser.add_argument("--tol", type=int, default=50, help="Alignment tolerance in ms")
    parser.add_argument("--cam_id", type=int, default=0, help="Specific camera ID (optional)")
    parser.add_argument("--format", type=str, default="parquet", choices=["csv", "parquet"], help="Output format")
    
    args = parser.parse_args()
    
    args.session_path = "/home/zc/HandMotion/temp_new/robot-data-collector/my_project/resources/logs/captures/2026-01-07/sess_20260107_165517"
    aligner = SessionAligner(args.session_path)
    try:
        # Use a wrapper to mimic method signature if needed, or update call
        aligner.get_camera_folder(camera_id=args.cam_id)
        aligner.align_session(tolerance_ms=args.tol)
        
        output_name = f"aligned_session.{args.format}"
        aligner.save_aligned_data(output_name)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Alignment failed: {e}")