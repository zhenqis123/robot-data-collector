import os
import sys
import pandas as pd
import numpy as np
import torch
import pyvista as pv
from trimesh import Trimesh
from scipy.spatial.transform import Rotation as R
import time

# 添加路径以确保可以导入 teleop 和 manotorch 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from PathConfig import PROJECT_ROOT
try:
    from .VDHand import VD_to_mano_keypoints, HandType
    from ...manotorch.manotorch.manolayer import ManoLayer, MANOOutput
except ImportError:
    from VDHand import VD_to_mano_keypoints, HandType
    from manotorch.manotorch.manolayer import ManoLayer, MANOOutput

class GloveCSVProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV files not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} frames from {csv_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化 MANO Layer
        self.mano_layer_right = self._init_mano_layer("right")
        self.mano_layer_left = self._init_mano_layer("left")
        
        # 预先分配 Beta
        self.beta = torch.zeros(1, 10).to(self.device)

        # 初始化列缓存以便快速批量访问
        self._init_column_cache()

    def _init_column_cache(self):
        """预计算列名以加速批量访问"""
        self.cols_cache = {}
        for side in ['right', 'left']:
            prefix = 'r' if side == 'right' else 'l'
            kp_cols = []
            for i in range(21):
                kp_cols.extend([f'{prefix}_x{i}', f'{prefix}_y{i}', f'{prefix}_z{i}'])
            
            quat_cols = [f'{prefix}_wrist_qx', f'{prefix}_wrist_qy', f'{prefix}_wrist_qz', f'{prefix}_wrist_qw']
            
            self.cols_cache[side] = {
                'kp': kp_cols,
                'quat': quat_cols,
                'detected': f'{side}_detected'
            }

    def _init_mano_layer(self, side):
        return ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            side=side,
            center_idx=0, 
            mano_assets_root=f"../../manotorch/assets/mano",
            flat_hand_mean=True,
            device=self.device
        )

    def parse_frame(self, index):
        """解析 CSV 中的一帧数据"""
        row = self.df.iloc[index]
        
        data = {}
        for side in ['right', 'left']:
            prefix = 'r' if side == 'right' else 'l'
            
            # 提取关键点 (21, 3)
            keypoints = []
            for i in range(21):
                keypoints.append([
                    row[f'{prefix}_x{i}'], 
                    row[f'{prefix}_y{i}'], 
                    row[f'{prefix}_z{i}']
                ])
            keypoints = np.array(keypoints, dtype=np.float32)
            
            # 提取手腕四元数 [qx, qy, qz, qw] -> Scipy assumes [x, y, z, w]
            # CSV header: l_wrist_qw, l_wrist_qx, l_wrist_qy, l_wrist_qz
            # We need to construct [x, y, z, w] for scipy
            wrist_quat = [
                row[f'{prefix}_wrist_qx'],
                row[f'{prefix}_wrist_qy'],
                row[f'{prefix}_wrist_qz'],
                row[f'{prefix}_wrist_qw']
            ]
            
            # 计算旋转矩阵
            wrist_rot = R.from_quat(wrist_quat).as_matrix()
            
            data[side] = {
                'keypoints': keypoints,
                'wrist_rot': wrist_rot,
                'detected': row[f'{side}_detected']
            }
            
        return data

    def compute_mano_parameter(self, side, raw_kps, wrist_rot):
        """
        计算单只手的 MANO 参数
        
        Args:
            side (str): 'left' or 'right'
            raw_kps (np.ndarray): 原始关键点 (21, 3)
            wrist_rot (np.ndarray): 手腕旋转矩阵 (3, 3)
            
        Returns:
            axis_angles (np.ndarray): 轴角参数
            betas (np.ndarray): 形状参数
        """
        hand_type = HandType.RIGHT if side == 'right' else HandType.LEFT
        mano_layer = self.mano_layer_right if side == 'right' else self.mano_layer_left
        
        # 1. 转换到 MANO 关键点坐标系
        mano_kps = VD_to_mano_keypoints(raw_kps, wrist_rot, hand_type) 
        
        kps_tensor = torch.tensor(mano_kps, device=self.device, dtype=torch.float32).unsqueeze(0) # (1, 21, 3)
        
        # 2. 使用 MANO Layer 的 IK 功能 (joints_to_mano_parameters)
        axis_angle_results = mano_layer.joints_to_mano_parameters(kps_tensor, betas=self.beta)
        
        axis_angles = axis_angle_results["full_poses"][0].cpu().numpy()
        betas = axis_angle_results["betas"][0].cpu().numpy()
        
        return axis_angles, betas

    def compute_mano_mesh(self, side, raw_kps, wrist_rot):
        """
        计算单只手的 MANO Mesh 数据
        
        Args:
            side (str): 'left' or 'right'
            raw_kps (np.ndarray): 原始关键点 (21, 3)
            wrist_rot (np.ndarray): 手腕旋转矩阵 (3, 3)
            
        Returns:
            verts (np.ndarray): Mesh 顶点
            faces (np.ndarray): Mesh 面索引
        """
        hand_type = HandType.RIGHT if side == 'right' else HandType.LEFT
        mano_layer = self.mano_layer_right if side == 'right' else self.mano_layer_left
        
        # 1. 转换到 MANO 关键点坐标系
        mano_kps = VD_to_mano_keypoints(raw_kps, wrist_rot, hand_type) 
        
        kps_tensor = torch.tensor(mano_kps, device=self.device, dtype=torch.float32).unsqueeze(0) # (1, 21, 3)
        
        # 2. 使用 MANO Layer 的 IK 功能 (joints_to_mano_parameters)
        axis_angle_results = mano_layer.joints_to_mano_parameters(kps_tensor, betas=self.beta)
        
        # 3. 生成 MANO Mesh
        mano_output: MANOOutput = mano_layer(axis_angle_results["full_poses"], betas=axis_angle_results["betas"])
        
        # 获取顶点和面
        verts = mano_output.verts[0].cpu().numpy()
        faces = mano_layer.th_faces.cpu().numpy()
        
        return verts, faces

    def process_batch(self, indices):
        """
        批量处理方法，用于 DataLoader 高效调用
        
        Args:
            indices (list/np.ndarray/torch.Tensor): 帧索引列表
            
        Returns:
            dict: 包含 'right' 和 'left' 的批量数据
                'keypoints': (B, 21, 3) Tensor
                'wrist_rot': (B, 3, 3) Tensor
                'mano_pose': (B, 48) Tensor
                'mano_shape': (B, 10) Tensor
                'detected': (B,) Tensor
        """
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        if isinstance(indices, list):
            indices = np.array(indices)
            
        # 确保索引是整数
        indices = indices.astype(int)
            
        batch_size = len(indices)
        sub_df = self.df.iloc[indices]
        batch_res = {}
        
        for side in ['right', 'left']:
            cols = self.cols_cache[side]
            
            # 1. 批量提取基础数据 (Numpy)
            # Detected flag
            detected = sub_df[cols['detected']].values.astype(bool) # (B,)
            
            # Keypoints: Flattened (B, 63) -> (B, 21, 3)
            kps_flat = sub_df[cols['kp']].values.astype(np.float32)
            raw_kps = kps_flat.reshape(batch_size, 21, 3)
            
            # Wrist Quats: (B, 4) [x, y, z, w]
            quats = sub_df[cols['quat']].values.astype(np.float32)
            
            # Convert Quats to Rot Mats (B, 3, 3)
            wrist_rot = R.from_quat(quats).as_matrix().astype(np.float32)
            
            # 2. 转换到 MANO 坐标系
            hand_type = HandType.RIGHT if side == 'right' else HandType.LEFT
            mano_kps = VD_to_mano_keypoints(raw_kps, wrist_rot, hand_type) # (B, 21, 3)
            
            # 3. 准备 Tensor 并进行 IK
            kps_tensor = torch.tensor(mano_kps, device=self.device, dtype=torch.float32)
            
            mano_layer = self.mano_layer_right if side == 'right' else self.mano_layer_left
            
            # 广播 Betas (B, 10)
            betas_expanded = self.beta.repeat(batch_size, 1)
            
            with torch.no_grad():
                ik_res = mano_layer.joints_to_mano_parameters(kps_tensor, betas=betas_expanded)
            
            # 4. 组装结果
            batch_res[side] = {
                'detected': torch.tensor(detected, device=self.device),
                'keypoints': kps_tensor, # MANO space keypoints
                'raw_keypoints': torch.tensor(raw_kps, device=self.device, dtype=torch.float32), # Original VD space
                'wrist_rot': torch.tensor(wrist_rot, device=self.device),
                'mano_pose': ik_res['full_poses'],
                'mano_shape': ik_res['betas']
            }
            
        return batch_res

    def process_and_visualize(self):
        """处理每一帧并在 PyVista 中可视化"""
        pl = pv.Plotter()
        pl.add_camera_orientation_widget()
        pl.set_background('white')
        
        print("Starting visualization...")
        
        # 这里的可视化逻辑是简单的循环播放
        for i in range(len(self.df)):
            data = self.parse_frame(i)
            pl.clear() # 清除上一帧
            pl.add_text(f"Frame: {i}", position='upper_left', color='black')
            
            for side in ['right', 'left']:
                if not data[side]['detected']:
                    continue
                
                try:
                    # 计算 MANO Mesh
                    verts, faces = self.compute_mano_mesh(side, data[side]['keypoints'], data[side]['wrist_rot'])
                    
                    # 创建 PyVista Mesh
                    mesh = Trimesh(vertices=verts, faces=faces)
                    pv_mesh = pv.wrap(mesh)
                    
                    color = 'red' if side == 'right' else 'blue'
                    pl.add_mesh(pv_mesh, color=color, opacity=0.8, smooth_shading=True, name=f"{side}_mesh")
                    
                    # 可选：显示骨架/关键点
                    # pl.add_points(mano_kps, color='green', point_size=10, render_points_as_spheres=True)
                    
                except Exception as e:
                    print(f"Error processing {side} hand at frame {i}: {e}")
        
            pass # 只是占位
            
        print("Visualization complete.")

    def run_interactive(self):
        """交互式播放"""
        pl = pv.Plotter()
        pl.add_camera_orientation_widget()
        pl.set_background('white')
        pl.add_text("Press 'q' to quit", position='upper_right', color='black')

        self.plotter = pl
        self.current_frame = 0
        self.is_playing = True

        pl.show(interactive_update=True)
        
        while True:
            # 检查窗口是否关闭，不同版本 pyvista 检查方式可能不同
            # 使用 try-except block 捕获关闭时的异常通常是最稳健的
            try:
                if self.current_frame >= len(self.df):
                    self.current_frame = 0
                
                data = self.parse_frame(self.current_frame)
                
                for side in ['right', 'left']:
                    mesh_name = f"{side}_mesh"
                    
                    if data[side]['detected']:
                        try:
                            verts, faces = self.compute_mano_mesh(side, data[side]['keypoints'], data[side]['wrist_rot'])
                            
                            # Update Mesh
                            # PyVista 直接替换 Mesh 比较简单
                            mesh = Trimesh(vertices=verts, faces=faces)
                            pv_mesh = pv.wrap(mesh)
                            
                            color = 'orange' if side == 'right' else 'cornflowerblue'
                            pl.add_mesh(pv_mesh, color=color, name=mesh_name, smooth_shading=True, show_edges=False)
                            
                        except Exception as e:
                             # 如果出错（例如IK失败），移除mesh
                            pl.remove_actor(mesh_name)
                    else:
                        pl.remove_actor(mesh_name)
                
                pl.add_text(f"Frame: {self.current_frame}/{len(self.df)}", name="frame_text", position='upper_left', color='black')
                
                pl.update()
                
                self.current_frame += 1
            except AttributeError:
                break
            except Exception as e:
                # print(f"Visualization loop ended: {e}")
                break

if __name__ == "__main__":
    default_csv = "/home/zc/HandMotion/temp_new/temp/robot-data-collector/my_project/resources/logs/captures/2026-01-11/sess_20260111_152148/glove_data.csv"
    
    # 支持命令行参数
    csv_file = sys.argv[1] if len(sys.argv) > 1 else default_csv
    
    try:
        visualizer = GloveCSVProcessor(csv_file)
        visualizer.run_interactive()
    except Exception as e:
        print(f"Error: {e}")
        print(f"Usage: python visualize_glove_csv.py [path_to_csv]")

