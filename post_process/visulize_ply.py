import open3d as o3d
import glob
import time
import os
import argparse
import sys

import open3d as o3d
import argparse
import os
import numpy as np

def inspect_file():
    parser = argparse.ArgumentParser(description="PLY 详细观察器")
    parser.add_argument("file_path", help="PLY 文件的路径")
    args = parser.parse_args()

    file_path = args.file_path
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> {file_path}")
        return

    print(f"正在加载: {file_path} ...")
    
    # 尝试读取点云
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 打印基础信息
    print(f"--- 基础信息 ---")
    print(f"点数: {len(pcd.points)}")
    print(f"中心坐标: {pcd.get_center()}")
    print(f"最大边界: {pcd.get_max_bound()}")
    print(f"最小边界: {pcd.get_min_bound()}")

    # 1. 自动计算法线 (如果有必要，这会让渲染更有光影感)
    if not pcd.has_normals():
        print("正在计算法线以增强显示效果...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 2. 添加坐标轴 (红色X, 绿色Y, 蓝色Z), 尺寸为1.0 (根据你的模型单位可能需要调整)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    print("\n--- 操作指南 ---")
    print(" [Shift + 鼠标左键] : 选择一个点 (会在终端打印坐标)")
    print(" [K]              : 锁定/解锁 选择模式")
    print(" [N]              : 打开/关闭 法线显示")
    print(" [+/-]            : 增加/减小 点的大小")
    print(" [Ctrl + C]       : 复制当前视角")
    print(" [Q]              : 退出")
    
    # 3. 启动带编辑功能的视窗
    o3d.visualization.draw_geometries_with_editing([pcd, axes], 
                                                   window_name="Open3D 详细观察模式",
                                                   width=1200, height=900)


def play_sequence():
    # --- 1. 配置命令行参数 ---
    parser = argparse.ArgumentParser(description="PLY 序列播放器")
    parser.add_argument("path", nargs="?", default=".", help="包含 .ply 文件的文件夹路径 (默认: 当前目录)")
    parser.add_argument("--speed", type=float, default=100, help="播放间隔时间(秒), 默认 0.05")
    args = parser.parse_args()

    target_dir = args.path
    
    # --- 2. 获取文件 ---
    # 拼接路径，确保支持绝对路径和相对路径
    search_pattern = os.path.join(target_dir, "*.ply")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"错误: 在路径 '{target_dir}' 下未找到 .ply 文件。")
        print(f"尝试搜索: {search_pattern}")
        return

    print(f"在 '{target_dir}' 发现 {len(files)} 个文件。准备播放...")

    # --- 3. 初始化窗口 ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"播放: {os.path.basename(os.path.abspath(target_dir))}", width=1024, height=768)

    # 加载第一帧
    # 如果是网格文件，请改用 read_triangle_mesh
    current_geometry = o3d.io.read_point_cloud(files[0]) 
    vis.add_geometry(current_geometry)

    # --- 4. 循环播放 ---
    try:
        for i, filename in enumerate(files[1:]):
            next_geometry = o3d.io.read_point_cloud(filename)
            
            # 保持视角不乱跳
            vis.remove_geometry(current_geometry, reset_bounding_box=False)
            vis.add_geometry(next_geometry, reset_bounding_box=False)
            
            current_geometry = next_geometry
            
            vis.poll_events()
            vis.update_renderer()
            
            # 打印相对简洁的日志
            print(f"\r[{i+1}/{len(files)-1}] {os.path.basename(filename)}", end="")
            time.sleep(args.speed)
            
    except KeyboardInterrupt:
        print("\n播放被用户中断")

    print("\n播放结束。")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    inspect_file()