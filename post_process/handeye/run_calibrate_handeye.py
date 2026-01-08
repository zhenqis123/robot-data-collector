import argparse
import os
import sys
import socket
import struct
import threading
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from pathlib import Path
from handeye_utils import *

class VIVEDataReceiver:
    def __init__(self, port=6666, num_floats=36):
        self.port = port
        self.num_floats = num_floats
        # double(8) + num_floats * float(4)
        self.packet_size = 8 + num_floats * 4  
        self.unpack_fmt = f'<d{num_floats}f'
        self.lock = threading.Lock()
        self.latest_data = None
        self.running = False
        self.sock = None
        self.thread = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(('0.0.0.0', self.port))
            self.sock.settimeout(0.5) # 500ms timeout
            self.running = True
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            print(f"[VIVEDataReceiver] Listening on port {self.port}")
        except Exception as e:
            print(f"[VIVEDataReceiver] Failed to start: {e}")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.sock:
            self.sock.close()
        print("[VIVEDataReceiver] Stopped")

    def _receive_loop(self):
        while self.running:
            try:
                # Buffer size > 152 bytes
                data, _ = self.sock.recvfrom(2048) 
                if len(data) == self.packet_size:
                    unpacked = struct.unpack(self.unpack_fmt, data)
                    # unpacked[0] is timestamp, [1:] are floats
                    with self.lock:
                        self.latest_data = np.array(unpacked[1:])
            except socket.timeout:
                continue
            except Exception:
                pass

    def get_latest_data(self):
        with self.lock:
            if self.latest_data is None:
                return None
            return self.latest_data.copy()


def wrapper_trans(trans):
    # return C_rot @ trans
    return trans

def wrapper_rot(rot):
    # return C_rot @ (rot)
    return rot

def VIVE_Station_Calibration():
    parser = argparse.ArgumentParser(description="相机与定位基站的标定与测试程序") 
    parser.add_argument('--camera_idx', default=4, type=int, help='要使用的相机在 CAMERA_IDS 列表中的索引')
    parser.add_argument('--tracker_id', default=0, type=int, help='使用的追踪器ID (0-2), 默认为1(第二个)')
    args = parser.parse_args()
    
    # 假设 VIVEDataReceiver 已经定义并可用
    receiver = VIVEDataReceiver(port=6666, num_floats=36)
    receiver.start()

    # ### 1. 初始化 ###
    cam_serial = CAMERA_IDS[args.camera_idx]
    # 修改保存路径为 .npz 格式
    save_path = f'./handeye/cam2station_calib_{cam_serial}.npz'

    check_rs_devices()
    pipeline, align = get_rs_pipeline(args.camera_idx)
    aruco_detector = get_aruco_detector()
    
    board_poses_in_cam = []
    tracker_poses_in_station = []

    # 用于测试模式的变量
    calib_data = {}
    test_mode = False
    
    # # 启动时尝试加载已有的标定文件
    # if os.path.exists(save_path):
    #     print(f"检测到已存在的标定文件: {save_path}")
    #     try:
    #         calib_data = np.load(save_path)
    #         # 检查文件内容是否完整
    #         if 'RT_cam2station' in calib_data and 'RT_board2tracker' in calib_data:
    #             test_mode = True
    #             print("加载成功，已进入【测试模式】。")
    #             print("按 'n' 键进行单点精度测试。")
    #         else:
    #             print("标定文件不完整，请重新标定。")
    #     except Exception as e:
    #         print(f"加载标定文件失败: {e}，请重新标定。")

    if not test_mode:
        print("\n\n===== 未找到有效标定文件，进入【标定模式】 =====")
        print("操作指南:")
        print(" 1. 移动带有追踪器的标定板到新的位置和姿态。")
        print(" 2. 确保相机视野能清晰地看到Aruco码。")
        print(" 3. 在图像窗口按下 'r' 键，采集数据。")
        print(" 4. 采集足够数据后 (建议20+)，按下 'c' 键进行计算。")
    
    print(" 随时可按 'q' 键退出程序。")
    
    while True:
        try:
            rgb, depth, intr_matrix, intr_coeffs = get_aligned_images(pipeline, align)
            vive_data = receiver.get_latest_data()
            
            board_pose_cam, visualized_rgb = get_aruco_pose_in_cam(aruco_detector, rgb.copy(), intr_matrix, intr_coeffs)
            
        except Exception as e:
            print(f"\r无法获取图像或定位器数据: {e}", end="")
            continue

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break

        # 'r' 键: 记录标定数据
        elif key == ord('r'):
            if test_mode:
                print("\n当前为测试模式，如需重新标定，请删除标定文件后重启程序。")
                continue

            if board_pose_cam is None or vive_data is None:
                print("\n[错误] 未能同时检测到标定板和追踪器! 请调整后重试。")
                continue
            
            # tracker_pose_data = vive_data[12:24]
            start_idx = args.tracker_id * 12
            tracker_pose_data = vive_data[start_idx : start_idx + 12]
            
            if not (tracker_pose_data[0] == 0 and tracker_pose_data[1] == 0 and tracker_pose_data[2] == 0):
                board_poses_in_cam.append(board_pose_cam)
                tracker_poses_in_station.append(tracker_pose_data)
                print(f"\r成功记录第 {len(tracker_poses_in_station)} 组数据。", end="")
            else:
                print("\n[错误] 追踪器位置无效")

        # 'c' 键: 执行标定
        elif key == ord('c'):
            if test_mode:
                print("\n当前为测试模式，无需再次标定。")
                continue

            if len(tracker_poses_in_station) < 10:
                print(f"\n[错误] 数据太少 ({len(tracker_poses_in_station)} 组)，无法进行精确标定。建议至少采集15-20组。")
                # continue
                print("从已经保存的路径加载")
                tracker_poses_in_station = np.load('handeye/raw_poses_cam_station.npz')['tracker_poses_in_station'].tolist()
                board_poses_in_cam = np.load('handeye/raw_poses_cam_station.npz')['board_poses_in_cam'].tolist()

            
            print("\n\n===== 开始计算变换矩阵 =====")
            R_tracker2station, T_tracker2station, R_board2cam, T_board2cam = [], [], [], []

            for pose in tracker_poses_in_station:
                T_tracker2station.append(wrapper_trans(np.array(pose[0:3]).reshape(3, 1)))
                R_tracker2station.append(wrapper_rot(np.array(pose[3:12]).reshape(3, 3)))
            for pose in board_poses_in_cam:
                T_board2cam.append(np.array(pose[0:3]).reshape(3, 1))
                R_board2cam.append(cv2.Rodrigues(np.array(pose[3:6]))[0])
            
            tracker_poses_in_station = np.array(tracker_poses_in_station)
            board_poses_in_cam = np.array(board_poses_in_cam)
            np.savez('handeye/raw_poses_cam_station.npz', tracker_poses_in_station=tracker_poses_in_station, board_poses_in_cam=board_poses_in_cam)

            RT_cam2station = compute_cam2station_calibration(R_tracker2station, T_tracker2station, R_board2cam, T_board2cam)
            print("相机(Cam)到定位基站(Station)的变换矩阵计算完成。")

            RT_board2tracker = np.zeros((4,4))
            
            print(f"\n正在保存结果到: {os.path.abspath(save_path)}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, RT_cam2station=RT_cam2station, RT_board2tracker=RT_board2tracker)
            print("保存成功! 程序将自动切换到测试模式。")

            calib_data['RT_cam2station'] = RT_cam2station
            calib_data['RT_board2tracker'] = RT_board2tracker
            test_mode = True

        if test_mode and 'RT_cam2station' in calib_data:
            # 实时验证
            RT = calib_data['RT_cam2station']
            RT_inv = np.linalg.inv(RT)
            
            # 使用 args.tracker_id
            start_idx = args.tracker_id * 12
            tracker_pose_data = vive_data[start_idx : start_idx + 12]
            
            if not (tracker_pose_data[0] == 0 and tracker_pose_data[1] == 0 and tracker_pose_data[2] == 0):
                T_tracker2station = wrapper_trans(np.array(tracker_pose_data[0:3]).reshape(3, 1))
                R_tracker2station = wrapper_rot(np.array(tracker_pose_data[3:12]).reshape(3, 3))
                RT_tracker2station = np.eye(4)
                RT_tracker2station[:3, :3] = R_tracker2station
                RT_tracker2station[:3, 3] = T_tracker2station.flatten()
                
                RT_cam2tracker = RT_inv @ RT_tracker2station
                
                R_vec, _ = cv2.Rodrigues(RT_cam2tracker[:3, :3])
                T_vec = RT_cam2tracker[:3, 3]
                
                cv2.drawFrameAxes(visualized_rgb, intr_matrix, intr_coeffs, R_vec, T_vec, 0.05) # 0.05米

        # 始终显示图像
        cv2.imshow(f'Camera View (SN: {cam_serial})', visualized_rgb)

    cv2.destroyAllWindows()
    # receiver.stop()
    print("\n程序已退出。")


def RM_ARM_Calibration():
     # 解析命令行参数
    parse = argparse.ArgumentParser() 
    parse.add_argument('--camera', default=4, type=int, help='camera id')
    # 左臂
    parse.add_argument('--arm_ip', default='192.168.101.21', type=str, help='arm ip')
    # #右臂
    # parse.add_argument('--arm_ip', default='192.168.1.20', type=str, help='arm ip')
    args = parse.parse_args() 

    save_path = 'handeye/cam2base_{}.npy'.format(CAMERA_IDS[args.camera])

    check_rs_devices()
    pipeline, align = get_rs_pipeline(args.camera)
    aruco_detector = get_aruco_detector()
    robot = ArmCalib(args.arm_ip)

    hands =[]
    cameras = []
    while True:
        rgb, depth, intr_matrix, intr_coeffs = get_aligned_images(pipeline, align)
        tvec_rvec = get_realsense_mark(aruco_detector,  rgb, intr_matrix, intr_coeffs)
        cv2.imshow('RGB image', rgb)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break

        # 按r记录标定板、末端位姿
        elif key == ord('r'):
            #hands.append(get_jaka_gripper())
            if tvec_rvec is None:
                print("board not detected!!!")
                continue
            pose, flag = robot.get_ee_pose()
            if flag!=0:
                print("error in get robot pose!!!")
            print("board-to-camera pose recorded:", tvec_rvec)
            print("robot ee pose recorded:", pose)
            cameras.append(tvec_rvec)
            hands.append(pose)

        # 计算标定
        elif key ==ord('c'):
            R_Hgs, R_Hcs = [], []
            T_Hgs, T_Hcs = [], []
            for camera in cameras:
                #m-c的旋转矩阵和位移矩阵
                c = camera[3:6]
                # R_Hcs.append(tfs.quaternions.quat2mat((q[3], q[0], q[1], q[2]))) #四元素转旋转矩阵；相机读出x,y,z,w 使用该方法
                camera_mat,j = cv2.Rodrigues((c[0],c[1],c[2])) #旋转矢量到旋转矩阵
                R_Hcs.append(camera_mat)
                T_Hcs.append(np.array(camera[0:3])*1000) # 单位mm
            for hand in hands:
                # g-b的旋转矩阵和位移矩阵
                g = hand[3:6]
                #R_Hgs.append(tfs.euler.euler2mat(math.radians(g[0])... 'sxyz'))#如果读出角度，转弧度再计算
                R_Hgs.append(tfs.euler.euler2mat(g[0], g[1], g[2], 'sxyz'))#欧拉角到旋转矩阵；
                T_Hgs.append(np.array(hand[0:3])*1000) # 单位mm
            print("R_Hcs:",R_Hcs)
            print("T_Hcs:",T_Hcs)
            print("R_Hgs:",R_Hgs)
            print("T_Hgs:",T_Hgs)
            RT_cam2base = compute_calibrate(R_Hgs, T_Hgs, R_Hcs, T_Hcs, hand_on_eye=False)
            print("Calibrated RT_cam2base：",RT_cam2base)

            print("Saving RT_cam2base to", os.path.abspath(save_path))
            if os.path.exists(save_path):
                print("File already exists, overwriting...")
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print("File does not exist, creating new file...")
            np.save(save_path, RT_cam2base)
            break

            # #根据计算 RT_c2g 推算出之前记录数据的机械臂末端相对基地移动矩阵
            # final_pose = []
            # for i in range(len(R_Hgs)):
            #     RT_g2b = tfs.affines.compose(np.squeeze(T_Hgs[i]), R_Hgs[i], [1, 1, 1])
            #     temp = np.dot(RT_g2b, RT_c2g)
            #     RT_t2c = tfs.affines.compose(np.squeeze(T_Hcs[i]), R_Hcs[i], [1, 1, 1])
            #     temp = np.dot(temp, RT_t2c)
            #     tr = temp[0:3, 3:4].T[0]
            #     rot = tfs.euler.mat2euler(temp[0:3, 0:3])
            #     final_pose.append([tr[0], tr[1], tr[2], math.degrees(rot[0]), rot[1], rot[2]])
            # final_pose = np.array(final_pose)
            # print('final_pose\n', final_pose)
            # break
    cv2.destroyAllWindows()
    robot.close()


def One_Calibration():
     # 解析命令行参数
    parse = argparse.ArgumentParser() 
    parse.add_argument('--camera', default=4, type=int, help='camera id')
    parse.add_argument('--tracker_id', default=1, type=int, help='tracker id (0-2)')
    # 左臂
    parse.add_argument('--arm_ip', default='192.168.101.21', type=str, help='arm ip')
    # #右臂
    # parse.add_argument('--arm_ip', default='192.168.1.20', type=str, help='arm ip')
    args = parse.parse_args() 

    save_path = 'handeye/cam2station_test_{}.npy'.format(CAMERA_IDS[args.camera])

    check_rs_devices()
    pipeline, align = get_rs_pipeline(args.camera)
    aruco_detector = get_aruco_detector()

    receiver = VIVEDataReceiver(port=6666, num_floats=36)
    receiver.start()

    hands =[]
    cameras = []
    while True:
        rgb, depth, intr_matrix, intr_coeffs = get_aligned_images(pipeline, align)
        tvec_rvec = get_realsense_mark(aruco_detector,  rgb, intr_matrix, intr_coeffs)
        cv2.imshow('RGB image', rgb)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break

        # 按r记录标定板、末端位姿
        elif key == ord('r'):
            #hands.append(get_jaka_gripper())
            if tvec_rvec is None:
                print("board not detected!!!")
                continue
            vive_data = receiver.get_latest_data()
            if vive_data is None:
                print("VIVE data not received!!!")
                continue
            # tracker_pose_data = vive_data[12:24]
            start_idx = args.tracker_id * 12
            tracker_pose_data = vive_data[start_idx : start_idx + 12]

            if not (tracker_pose_data[0] == 0 and tracker_pose_data[1] == 0 and tracker_pose_data[2] == 0):
                cameras.append(tvec_rvec)
                hands.append(tracker_pose_data)
                print(f"\r成功记录第 {len(hands)} 组数据。", end="")
            else:
                print("\n[错误] 追踪器位置无效")

        # 计算标定
        elif key ==ord('c'):
            if len(hands) < 10:
                print(f"\n[错误] 数据太少 ({len(hands)} 组)，无法进行精确标定。建议至少采集15-20组。")
                # continue
                print("从已经保存的路径加载")
                # hands = np.load('handeye/raw_poses_cam_station_new.npz')['hands'].tolist()
                # cameras = np.load('handeye/raw_poses_cam_station_new.npz')['cameras'].tolist()
                hands = np.load('handeye/raw_poses_cam_station_new_c.npz')['hands'].tolist()
                cameras = np.load('handeye/raw_poses_cam_station_new_c.npz')['cameras'].tolist()
            else:
                hands = np.array(hands)
                cameras = np.array(cameras)
                np.savez('handeye/raw_poses_cam_station_new.npz', hands=hands, cameras=cameras)
            R_Hgs, R_Hcs = [], []
            T_Hgs, T_Hcs = [], []
            for camera in cameras:
                #m-c的旋转矩阵和位移矩阵
                c = camera[3:6]
                # R_Hcs.append(tfs.quaternions.quat2mat((q[3], q[0], q[1], q[2]))) #四元素转旋转矩阵；相机读出x,y,z,w 使用该方法
                camera_mat,j = cv2.Rodrigues((c[0],c[1],c[2])) #旋转矢量到旋转矩阵
                R_Hcs.append(camera_mat)
                T_Hcs.append(np.array(camera[0:3])*1000) # 单位mm
            for hand in hands:
                # g-b的旋转矩阵和位移矩阵
                g = hand[3:6]
                #R_Hgs.append(tfs.euler.euler2mat(math.radians(g[0])... 'sxyz'))#如果读出角度，转弧度再计算
                R_Hgs.append(np.array(hand[3:12]).reshape(3, 3))#欧拉角到旋转矩阵；
                T_Hgs.append(np.array(hand[0:3])*1000) # 单位mm
            print("R_Hcs:",R_Hcs)
            print("T_Hcs:",T_Hcs)
            print("R_Hgs:",R_Hgs)
            print("T_Hgs:",T_Hgs)
            RT_cam2base = compute_calibrate(R_Hgs, T_Hgs, R_Hcs, T_Hcs, hand_on_eye=False)
            print("Calibrated RT_cam2base：",RT_cam2base)

            print("Saving RT_cam2base to", os.path.abspath(save_path))
            if os.path.exists(save_path):
                print("File already exists, overwriting...")
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print("File does not exist, creating new file...")
            np.save(save_path, RT_cam2base)
            break

            # #根据计算 RT_c2g 推算出之前记录数据的机械臂末端相对基地移动矩阵
            # final_pose = []
            # for i in range(len(R_Hgs)):
            #     RT_g2b = tfs.affines.compose(np.squeeze(T_Hgs[i]), R_Hgs[i], [1, 1, 1])
            #     temp = np.dot(RT_g2b, RT_c2g)
            #     RT_t2c = tfs.affines.compose(np.squeeze(T_Hcs[i]), R_Hcs[i], [1, 1, 1])
            #     temp = np.dot(temp, RT_t2c)
            #     tr = temp[0:3, 3:4].T[0]
            #     rot = tfs.euler.mat2euler(temp[0:3, 0:3])
            #     final_pose.append([tr[0], tr[1], tr[2], math.degrees(rot[0]), rot[1], rot[2]])
            # final_pose = np.array(final_pose)
            # print('final_pose\n', final_pose)
            # break
    cv2.destroyAllWindows()
    # robot.close()


def test_calibration():
    """
    加载手眼标定矩阵，并在实时相机图像中渲染追踪器的坐标轴。
    """
    parser = argparse.ArgumentParser(description="手眼标定测试程序")
    parser.add_argument('--camera_idx', default=4, type=int, help='要使用的相机在 CAMERA_IDS 列表中的索引')
    parser.add_argument('--tracker_id', default=0, type=int, help='使用的追踪器ID (0-2)')
    args = parser.parse_args()

    # --- 1. 初始化 ---
    camera_id = args.camera_idx
    tracker_id = args.tracker_id
    cam_serial = CAMERA_IDS[camera_id]
    
    npz_file = f"./handeye/cam2station_calib_{cam_serial}.npz"
    npy_file = f"./handeye/cam2station_test_{cam_serial}.npy"
    
    RT_cam2station = None
    is_mm = False # 标记单位是否为毫米

    if os.path.exists(npz_file):
        print(f"Loading npz calibration file: {os.path.abspath(npz_file)}")
        data = np.load(npz_file)
        RT_cam2station = data['RT_cam2station']
        # .npz from VIVE_Station_Calibration is usually in Meters
        is_mm = False 
    elif os.path.exists(npy_file):
        print(f"Loading npy calibration file: {os.path.abspath(npy_file)}")
        RT_cam2station = np.load(npy_file)
        # .npy from One_Calibration (if that's the source) might be in MM
        # But One_Calibration file name is cam2station_test_... check logic
        # One_Calibration saves cam2station_test_{}.npy WITH *1000. So it is MM.
        is_mm = True
    else:
        print(f"[Error] No calibration file found for camera {cam_serial}")
        print(f"Checked: {npz_file}")
        print(f"Checked: {npy_file}")
        return

    # RT_cam2station usually represents T_station_cam (Points in Cam -> Points in Station)
    # We want T_cam_station (Points in Station -> Points in Cam) to project tracker points to cam.
    # T_cam_tracker = T_cam_station * T_station_tracker
    RT_cam2station_inv = np.linalg.inv(RT_cam2station)
    print("RT_cam2station (loaded):\n", RT_cam2station)
    print("RT_cam2station_inv (used for projection):\n", RT_cam2station_inv)

    # 初始化RealSense相机
    check_rs_devices()
    pipeline, align = get_rs_pipeline(camera_id)

    # 初始化VIVE追踪器数据接收器
    receiver = VIVEDataReceiver(port=6666, num_floats=36)
    receiver.start()

    # 获取一次相机内参
    _, _, intr_matrix, intr_coeffs = get_aligned_images(pipeline, align)
    if intr_matrix is None:
        print("[Error] Cannot get camera intrinsics.")
        pipeline.stop()
        return

    # 定义坐标轴长度
    if is_mm:
        axis_length = 50.0 # 50mm
        print("Assuming calibration unit: MM")
    else:
        axis_length = 0.05 # 50mm in Meters
        print("Assuming calibration unit: Meters")

    axis_points = np.float32([
        [0, 0, 0],         # Origin
        [axis_length, 0, 0], # X
        [0, axis_length, 0], # Y
        [0, 0, axis_length]  # Z
    ]).reshape(-1, 3)

    print("\nTest started. Press 'q' to exit.")

    # --- 2. 主循环 ---
    while True:
        # 获取相机图像
        rgb, _, _, _ = get_aligned_images(pipeline, align)
        if rgb is None:
            continue

        # 获取最新的VIVE追踪器数据
        vive_data = receiver.get_latest_data()
        if vive_data is None:
            continue
        
        start_idx = tracker_id * 12
        tracker_pose_data = vive_data[start_idx : start_idx + 12]
        
        # 检查追踪器数据是否有效
        if not (tracker_pose_data[0] == 0 and tracker_pose_data[1] == 0 and tracker_pose_data[2] == 0):
            # a. 构建 T_station_tracker (Tracker in Station frame)
            # VIVE data is typically in Meters.
            t_tracker = np.array(tracker_pose_data[0:3])
            
            if is_mm:
                t_tracker = t_tracker * 1000.0 # Convert Meters to MM if calibration is in MM
            
            R_tracker = np.array(tracker_pose_data[3:12]).reshape(3, 3)
            T_station_tracker = tfs.affines.compose(t_tracker, R_tracker, [1, 1, 1])

            # b. 计算 T_cam_tracker = T_cam_station * T_station_tracker
            RT_cam2tracker = RT_cam2station_inv @ T_station_tracker
            
            # c. 提取旋转和平移
            R_cam2tracker = RT_cam2tracker[:3, :3]
            t_cam2tracker = RT_cam2tracker[:3, 3]

            # print(t_cam2tracker)
            rvec, _ = cv2.Rodrigues(R_cam2tracker)

            # --- 4. 投影和渲染 ---
            img_points, _ = cv2.projectPoints(axis_points, rvec, t_cam2tracker, intr_matrix, intr_coeffs)
            img_points = np.int32(img_points).reshape(-1, 2)
            
            origin_pt = tuple(img_points[0])
            cv2.drawFrameAxes(rgb, intr_matrix, intr_coeffs, R_cam2tracker, t_cam2tracker, axis_length)

            # Draw lines manually if needed (drawFrameAxes usually sufficient, but keeping lines for visibility)
            if img_points.shape[0] >= 4:
                cv2.line(rgb, origin_pt, tuple(img_points[1]), (0, 0, 255), 3) # X
                cv2.line(rgb, origin_pt, tuple(img_points[2]), (0, 255, 0), 3) # Y
                cv2.line(rgb, origin_pt, tuple(img_points[3]), (255, 0, 0), 3) # Z

        cv2.imshow('Calibration Test - Tracker Axes', rgb)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
            
    # --- 5. 清理 ---
    print("\nTest finished.")
    pipeline.stop()
    receiver.stop()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # 按照需求取消注释所需的函数
    
    # 1. 执行标定（默认推荐）
    VIVE_Station_Calibration()
    
    # 2. 仅测试（需要先有标定文件）
    # test_calibration()
    
    # 3. 旧版/机械臂标定（按需）
    # One_Calibration()
    # RM_ARM_Calibration()
