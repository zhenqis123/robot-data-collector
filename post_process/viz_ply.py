import open3d as o3d
import glob
import os
import time
import numpy as np

# === 配置区域 ===
DATA_DIR = "/media/zc/500G/tac_hand_data/2026-01-18/sess_20260118_134700/reconstructions_intersection"

# 视觉辅助配置 (根据数据单位调整，如数据是mm，建议设为10或100；如是m，设为0.1)
VIS_SCALE = 0.05       # 坐标轴的长度
GRID_SIZE = 0.2        # 网格总大小 (正负范围)
GRID_STEP = 0.005       # 网格间距
# ================

def create_ground_grid(size=1.0, step=0.1):
    """创建一个位于 XZ 平面 (y=0) 的网格"""
    lines = []
    points = []
    line_color = [0.7, 0.7, 0.7] # 浅灰色

    # 生成网格顶点
    start = -size
    end = size
    num_steps = int((end - start) / step) + 1
    
    # Z轴方向的线 (沿着X轴移动)
    for i in range(num_steps):
        x = start + i * step
        points.append([x, 0, -size])
        points.append([x, 0, size])
        lines.append([len(points)-2, len(points)-1])
        
    # X轴方向的线 (沿着Z轴移动)
    for i in range(num_steps):
        z = start + i * step
        points.append([-size, 0, z])
        points.append([size, 0, z])
        lines.append([len(points)-2, len(points)-1])

    colors = [line_color for _ in range(len(lines))]
    
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(colors)
    return grid

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.ply")))
    if not files:
        print(f"错误: {DATA_DIR} 为空")
        return

    # 1. 初始化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Player with Grid", width=1024, height=768)

    # 2. 添加静态辅助元素 (坐标轴 + 网格)
    # 坐标轴: 红=X, 绿=Y, 蓝=Z
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=VIS_SCALE, origin=[0, 0, 0]
    )
    vis.add_geometry(coord_frame)

    # 网格线
    grid = create_ground_grid(size=GRID_SIZE, step=GRID_STEP)
    vis.add_geometry(grid)

    # 3. 加载首帧
    def load_geom(path):
        return o3d.io.read_point_cloud(path)

    current_geom = load_geom(files[0])
    vis.add_geometry(current_geom)

    print("开始播放... (按 'H' 可查看 Open3D 帮助，按 'Q' 退出)")

    # 4. 循环播放
    for i, file_path in enumerate(files):
        new_geom = load_geom(file_path)

        # 核心：替换几何体，但在 update_renderer 前不重置视角
        vis.remove_geometry(current_geom, reset_bounding_box=False)
        vis.add_geometry(new_geom, reset_bounding_box=False)
        current_geom = new_geom

        vis.poll_events()
        vis.update_renderer()
        
        # 简单控制帧率
        time.sleep(0.05)
        print(f"Frame: {i}/{len(files)}", end="\r")

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()