import math
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import transforms3d as tfs
import open3d as o3d
from scipy.spatial.transform import Rotation # 用于做旋转的平均

def estimate_pose_single_markers(corners, marker_size, intr_matrix, intr_coeffs):
    marker_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)

    rvecs = []
    tvecs = []
    
    for c in corners:
        _, r, t = cv2.solvePnP(marker_points, c, intr_matrix, intr_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(r)
        tvecs.append(t)
    
    return np.array(rvecs), np.array(tvecs)

#################### Realsense & Detection #############

CAMERA_IDS = ['239722072965', '247122073084', '243522072934','247122073147','017322074878']

C_rot = np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])

def check_rs_devices():
    ctx = rs.context()
    if len(ctx.devices) > 0:
        for d in ctx.devices:
            print('Found device: ',
              d.get_info(rs.camera_info.name), ' ',
              d.get_info(rs.camera_info.serial_number))

def get_rs_pipeline(cam_id, resolution=(1280,720), fps=6, with_pinhole_intrinsic=False):
    pipeline = rs.pipeline()
    config = rs.config()
    print(cam_id)
    config.enable_device(CAMERA_IDS[cam_id])
    config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, fps)
    config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, fps)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    if with_pinhole_intrinsic:
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
        return pipeline, align, pinhole_camera_intrinsic
    return pipeline, align

# 获取对齐的rgb和深度图
def get_aligned_images(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    # 获取intelrealsense参数
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    # 内参矩阵，转ndarray方便后续opencv直接使用
    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
    ])
    # 深度图-16位
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # 深度图-8位
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    pos = np.where(depth_image_8bit == 0)
    depth_image_8bit[pos] = 255
    # rgb图
    color_image = np.asanyarray(color_frame.get_data())
    # return: rgb图，深度图，相机内参，相机畸变系数(intr.coeffs)
    return color_image, depth_image, intr_matrix, np.array(intr.coeffs)


def get_aruco_detector(sz=aruco.DICT_6X6_100):
    # 获取dictionary, 4x4的码，指示位50个
    aruco_dict = aruco.getPredefinedDictionary(sz)
    # 创建detector parameters
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    return detector


#获取标定板位姿
def get_realsense_mark(detector, rgb, intr_matrix, intr_coeffs):
    corners, ids, rejected_img_points = detector.detectMarkers(rgb)
    #print(corners)
    if len(corners)==0: # not detected
        return None
    rvec, tvec = estimate_pose_single_markers(corners, 0.1, intr_matrix, intr_coeffs)
    #print(rvec, tvec, markerPoints)
    for i in range(rvec.shape[0]):
        aruco.drawDetectedMarkers(rgb, corners)
    if rvec.shape[0]>1: # more than one board
        return None
    return list(np.reshape(tvec,3))+list(np.reshape(rvec,3))


# 标定
def compute_calibrate(R_end2base, T_end2base, R_board2cam, T_board2cam, hand_on_eye=False):
    
    if hand_on_eye:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_end2base, T_end2base, R_board2cam, T_board2cam,
            method=cv2.CALIB_HAND_EYE_TSAI)
        RT_c2g = tfs.affines.compose(np.squeeze(t_cam2gripper), R_cam2gripper, [1, 1, 1])
        #print("RT_c2g：",RT_c2g)
        return RT_c2g
    else:
        #R_base2end = [r.T for r in R_end2base]
        #T_base2end = [-t for t in T_end2base]
        R_base2end, T_base2end = [], []
        for r, t in zip(R_end2base, T_end2base):
            R_base2end.append(r.T)
            T_base2end.append(-r.T@t)

        R_cam2base, t_cam2base =  cv2.calibrateHandEye(R_base2end, T_base2end, R_board2cam, T_board2cam,
            method=cv2.CALIB_HAND_EYE_TSAI)
        RT_cam2base = tfs.affines.compose(np.squeeze(t_cam2base), R_cam2base, [1, 1, 1])
        #print(R_cam2base, t_cam2base)
        return RT_cam2base

def get_aruco_pose_in_cam(detector, rgb, intr_matrix, intr_coeffs):
    corners, ids, rejected_img_points = detector.detectMarkers(rgb)
    if ids is None or len(corners) == 0:
        return None, rgb
    
    # 仅当检测到一个marker时才返回结果，以保证数据质量
    if len(ids) > 1:
        print("警告: 检测到多于一个Aruco码，请确保视野中只有一个。")
        return None, rgb
        
    rvec, tvec = estimate_pose_single_markers(corners, 0.1, intr_matrix, intr_coeffs) # 0.1是marker的边长(米)
    
    # 可视化
    aruco.drawDetectedMarkers(rgb, corners, ids)
    cv2.drawFrameAxes(rgb, intr_matrix, intr_coeffs, rvec[0], tvec[0], 0.05) # 绘制坐标轴

    # 返回第一个检测到的marker的位姿 [tvec, rvec]
    return (list(np.reshape(tvec[0], 3)) + list(np.reshape(rvec[0], 3))), rgb


################# VIVE STATION ###############

# def compute_cam2station_calibration(R_station2tracker, T_station2tracker, R_board2cam, T_board2cam):
#     """
#     接收正确的 R_station2tracker, T_station2tracker 并进行标定
#     """
#     # 直接将数据传入，不再进行求逆操作
#     R_cam2station, t_cam2station = cv2.calibrateHandEye(
#         R_station2tracker, T_station2tracker, 
#         R_board2cam, T_board2cam,
#         method=cv2.CALIB_HAND_EYE_TSAI
#     )
#     RT_cam2station = np.eye(4)
#     RT_cam2station[:3,:3] = R_cam2station
#     RT_cam2station[:3, 3] = np.squeeze(t_cam2station)
#     return RT_cam2station

def compute_cam2station_calibration(R_tracker2station, T_tracker2station, R_board2cam, T_board2cam):
    """
    使用OpenCV的calibrateHandEye函数进行标定。
    这个函数现在用于计算 Cam 到 Station 的变换。
    参数名已更新以反映其物理意义。
    """
    # cv2.calibrateHandEye 需要 Gripper-to-Base 和 Target-to-Camera 的位姿
    # 在我们的类比中:
    # Gripper-to-Base  <=>  Tracker-to-Station
    # Target-to-Camera <=>  Board-to-Camera
    
    # 对于 "eye-to-hand" (固定相机) 的情况, 我们需要提供 Base-to-Gripper 的变换
    # 所以需要对 Tracker-to-Station 求逆
    R_station2tracker, T_station2tracker = [], []
    for r, t in zip(R_tracker2station, T_tracker2station):
        R_inv = r.T
        T_inv = -r.T @ t
        R_station2tracker.append(R_inv)
        T_station2tracker.append(T_inv)

    # 调用标定函数
    # 它将返回 Camera-to-Base 的变换，在我们的场景中即 Camera-to-Station
    R_cam2station, t_cam2station = cv2.calibrateHandEye(
        R_station2tracker, T_station2tracker, 
        R_board2cam, T_board2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # 组合成一个4x4的齐次变换矩阵
    RT_cam2station = tfs.affines.compose(np.squeeze(t_cam2station), R_cam2station, [1, 1, 1])
    return RT_cam2station




############### Robot control ###########

# # from robotic_arm_package.robotic_arm import * 
# from Robotic_Arm.rm_robot_interface import *

# class ArmCalib:
#     def __init__(self, ip):
#         self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#         handle = self.robot.rm_create_robot_arm(ip, 8080)     
#         ret, state = ArmState.rm_get_current_arm_state(self.robot)
#         print("机械臂状态：", state)
#         # self.robot = Arm(RM65, ip)
#         # print(self.get_ee_pose())

#     #获取末端位姿，xyz+弧度制rxryrz
#     def get_ee_pose(self):
#         ret, state = self.robot.rm_get_current_arm_state()
#         return  state['pose'], ret

#     def close(self):
#         self.robot.rm_delete_robot_arm()
#         # self.robot.RM_API_UnInit()
#         # self.robot.Arm_Socket_Close()

