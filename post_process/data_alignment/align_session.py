import os
import json
import pandas as pd
import argparse
from pathlib import Path

class SessionAligner:
    def __init__(self, config_path):
        self.config = self._load_json(config_path)
        
    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def get_main_camera_serial(self, camera_id):
        """根据 ID 在 config.json 中查找对应的序列号"""
        for cam in self.config.get('cameras', []):
            if cam.get('type') == 'RealSense' and cam.get('id') == camera_id:
                return cam.get('serial')
        raise ValueError(f"No RealSense camera found with ID {camera_id} in config.")

    def align_session(self, session_path, main_cam_id=0, tolerance_ms=50):
        """
        对齐单个采集会话的数据
        :param session_path: 会话文件夹路径 (e.g., .../sess_20251211_...)
        :param main_cam_id: config.json 中定义的主摄像头 ID
        :param tolerance_ms: 时间对齐的最大容差(毫秒)
        """
        session_path = Path(session_path)
        
        # 1. 确定主摄像头文件夹
        main_serial = self.get_main_camera_serial(main_cam_id)
        main_cam_folder = session_path / f"RealSense_{main_serial}"
        
        if not main_cam_folder.exists():
            raise FileNotFoundError(f"Main camera folder not found: {main_cam_folder}")

        # 2. 读取主摄像头时间戳 (Master Clock)
        # 假设 RealSense 的 timestamps.csv 中 timestamp_ms 是第 2 列
        main_df = pd.read_csv(main_cam_folder / "timestamps.csv")
        
        # 确保时间戳是整数并排序 (merge_asof 要求已排序)
        main_df['timestamp_ms'] = main_df['timestamp_ms'].astype(int)
        main_df = main_df.sort_values('timestamp_ms')
        
        # 重命名列以区分来源
        main_df = main_df.add_prefix(f'cam{main_cam_id}_')
        # 恢复用于对齐的 key
        align_key = f'cam{main_cam_id}_timestamp_ms'
        
        print(f"Main Camera (ID {main_cam_id}) frames: {len(main_df)}")

        # 3. 读取并对齐其他数据源
        
        # --- 对齐 Vive 数据 ---
        vive_csv = session_path / "vive_data.csv"
        if vive_csv.exists():
            print("Aligning Vive data...")
            vive_df = pd.read_csv(vive_csv)
            # vive_data.csv 第一列通常是 timestamp (ms)
            vive_df['timestamp'] = vive_df['timestamp'].astype(int)
            vive_df = vive_df.sort_values('timestamp')
            
            # 使用 merge_asof 进行最近邻匹配
            # direction='nearest': 找最近的
            # tolerance: 如果时间差超过 tolerance_ms，则视为无匹配 (NaN)
            main_df = pd.merge_asof(
                main_df, 
                vive_df.add_prefix('vive_'), 
                left_on=align_key, 
                right_on='vive_timestamp', 
                direction='nearest',
                tolerance=tolerance_ms
            )

        # --- 对齐手套数据 (Glove) ---
        glove_csv = session_path / "glove_data.csv"
        if glove_csv.exists():
            print("Aligning Glove data...")
            glove_df = pd.read_csv(glove_csv)
            glove_df['timestamp'] = glove_df['timestamp'].astype(int)
            glove_df = glove_df.sort_values('timestamp')
            
            main_df = pd.merge_asof(
                main_df, 
                glove_df.add_prefix('glove_'), 
                left_on=align_key, 
                right_on='glove_timestamp', 
                direction='nearest',
                tolerance=tolerance_ms
            )

        # --- 对齐其他 RealSense 摄像头 (如果有) ---
        # 遍历目录下所有 RealSense_ 开头的文件夹，排除主摄像头
        for entry in session_path.iterdir():
            if entry.is_dir() and entry.name.startswith("RealSense_") and entry.name != main_cam_folder.name:
                other_serial = entry.name.replace("RealSense_", "")
                print(f"Aligning secondary camera {other_serial}...")
                
                other_df = pd.read_csv(entry / "timestamps.csv")
                other_df['timestamp_ms'] = other_df['timestamp_ms'].astype(int)
                other_df = other_df.sort_values('timestamp_ms')
                
                # 对齐
                main_df = pd.merge_asof(
                    main_df,
                    other_df.add_prefix(f'rs_{other_serial}_'),
                    left_on=align_key,
                    right_on=f'rs_{other_serial}_timestamp_ms',
                    direction='nearest',
                    tolerance=tolerance_ms
                )

        return main_df

if __name__ == "__main__":
    # 使用示例
    # 假设脚本位于 post_process/data_alignment/ 下，我们需要向上回溯找到项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parents[2] # 根据实际层级调整
    CONFIG_PATH = PROJECT_ROOT / "my_project/resources/config.json"
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Align session data to main camera timestamps.")
    parser.add_argument("session_path", type=str, help="Path to the session directory")
    parser.add_argument("--cam_id", type=int, default=0, help="Main camera ID from config (default: 0)")
    
    args = parser.parse_args()
    
    aligner = SessionAligner(CONFIG_PATH)
    
    try:
        aligned_df = aligner.align_session(args.session_path, main_cam_id=args.cam_id)
        
        output_file = Path(args.session_path) / "aligned_data.csv"
        aligned_df.to_csv(output_file, index=False)
        print(f"\nSuccess! Aligned data saved to:\n{output_file}")
        print(f"Total aligned frames: {len(aligned_df)}")
        
    except Exception as e:
        print(f"Error: {e}")