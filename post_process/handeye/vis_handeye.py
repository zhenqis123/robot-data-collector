import argparse
from handeye_utils import CAMERA_IDS
import open3d as o3d
import pyrealsense2 as rs
import numpy as np
import cv2
import time

def get_rs_pipeline(cam_id):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(CAMERA_IDS[cam_id])
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    align_to = rs.stream.color
    align = rs.align(align_to)
    return pipeline, align, pinhole_camera_intrinsic

# 获取对齐的rgb和深度图
def get_aligned_images(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    # 利用中值核进行滤波
    depth_frame = rs.decimation_filter(1).process(aligned_depth_frame)
    # 从深度表示转换为视差表示，反之亦然
    depth_frame = rs.disparity_transform(True).process(depth_frame)
    # 空间滤镜通过使用alpha和delta设置计算帧来平滑图像。
    depth_frame = rs.spatial_filter().process(depth_frame)
    # 时间滤镜通过使用alpha和delta设置计算多个帧来平滑图像。
    depth_frame = rs.temporal_filter().process(depth_frame)
    # 从视差表示转换为深度表示
    depth_frame = rs.disparity_transform(False).process(depth_frame)
    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    color_frame = aligned_frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
    ])
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image, intr_matrix, np.array(intr.coeffs)


if __name__ == "__main__":
    parse = argparse.ArgumentParser() 
    parse.add_argument('--camera', default=4, type=int, help='camera id')
    args = parse.parse_args() 

    # load_path = 'handeye/cam2base_{}.npy'.format(CAMERA_IDS[args.camera])
    # load_path = 'handeye/cam2station_017322074878.npy'
    # load_path = "handeye/cam2station_calib_017322074878.npz"
    load_path = "handeye/cam2station_test_017322074878.npy"
    # load_path = "handeye/cam2right_base_017322074878.npy"
    # RT_cam2base = np.load(load_path)["RT_cam2station"]
    RT_cam2base = np.load(load_path)
    if 'base' in load_path or 'test' in load_path:
        RT_cam2base[0:3, 3] /= 1000.  # mm to m
    # else:
    #     RT_cam2base[0:3, 3] /= 1000.  # mm to m
    print(RT_cam2base)
    # RT_cam2base = np.linalg.inv(RT_cam2base)

    pipeline, align, pinhole_camera_intrinsic = get_rs_pipeline(args.camera)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    pointcloud = o3d.geometry.PointCloud()
    vis.add_geometry(pointcloud)
    # draw coordinate frame
    world_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(world_coord)
    cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    cam_coord.transform(RT_cam2base)
    vis.add_geometry(cam_coord)
    # adjust view point
    view_control = vis.get_view_control()
    view_control.set_lookat([0, 0, 0])  # 观察点设置为世界坐标原点
    view_control.set_front([0.2, -0.2, 1])  # 前视方向
    view_control.set_up([1, 0, 0])  # 上方向设置为x轴方向
    view_control.set_zoom(1)  # 设置缩放比例以适应场景
    while True:
        color_image, depth_image, intr_matrix, intr_coeffs = get_aligned_images(pipeline, align)
        depth = o3d.geometry.Image(depth_image)
        color = o3d.geometry.Image(color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform(RT_cam2base)
        pointcloud.points = pcd.points
        #pointclouds[i].points = o3d.utility.Vector3dVector(pcd)
        vis.update_geometry(pointcloud)
        vis.poll_events()
        vis.update_renderer()

    
