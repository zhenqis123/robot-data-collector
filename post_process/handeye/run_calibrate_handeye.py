import argparse
import os
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PathConfig import PROJECT_ROOT, TACTASK_DIR, TELEOP_DIR, TACDATA_DIR
from handeye_utils import *
from teleop.mocap.vive_data_receiver import VIVEDataReceiver
import pyrealsense2 as rs

# robot = Arm(RM65, '192.168.1.19')
# for t in range(100):
#     res, joint_now, pose, arm_err, sys_err = robot.Get_Current_Arm_State()
#     print(res, pose)
#     time.sleep(0.5)
# sys.exit(0)

def wrapper_trans(trans):
    # return C_rot @ trans
    return trans

def wrapper_rot(rot):
    # return C_rot @ (rot)
    return rot

def VIVE_Station_Calibration():
    parser = argparse.ArgumentParser(description="相机与定位基站的标定与测试程序") 
    parser.add_argument('--camera_idx', default=4, type=int, help='要使用的相机在 CAMERA_IDS 列表中的索引')
    args = parser.parse_args()
    
    # 假设 VIVEDataReceiver 已经定义并可用
    receiver = VIVEDataReceiver(port=9999, num_floats=36)
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
            
            tracker_pose_data = vive_data[12:24]
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



def temp_test_Calibration():
     # 解析命令行参数
    parse = argparse.ArgumentParser() 
    parse.add_argument('--camera', default=4, type=int, help='camera id')
    # 左臂
    parse.add_argument('--arm_ip', default='192.168.101.21', type=str, help='arm ip')
    # #右臂
    # parse.add_argument('--arm_ip', default='192.168.1.20', type=str, help='arm ip')
    args = parse.parse_args() 

    save_path = 'handeye/cam2station_test_{}.npy'.format(CAMERA_IDS[args.camera])

    check_rs_devices()

    hands =[]
    cameras = []

    hands = np.load('handeye/raw_poses_cam_station.npz')['tracker_poses_in_station'].tolist()
    cameras = np.load('handeye/raw_poses_cam_station.npz')['board_poses_in_cam'].tolist()

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



def One_Calibration():
     # 解析命令行参数
    parse = argparse.ArgumentParser() 
    parse.add_argument('--camera', default=4, type=int, help='camera id')
    # 左臂
    parse.add_argument('--arm_ip', default='192.168.101.21', type=str, help='arm ip')
    # #右臂
    # parse.add_argument('--arm_ip', default='192.168.1.20', type=str, help='arm ip')
    args = parse.parse_args() 

    save_path = 'handeye/cam2station_test_{}.npy'.format(CAMERA_IDS[args.camera])

    check_rs_devices()
    pipeline, align = get_rs_pipeline(args.camera)
    aruco_detector = get_aruco_detector()

    receiver = VIVEDataReceiver(port=9999, num_floats=36)
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
            tracker_pose_data = vive_data[12:24]
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

    Args:
        camera_id (int): 要使用的相机ID。
        calibration_file (str): 保存的RT_cam2base变换矩阵文件路径 (.npy)。
    """
    # --- 1. 初始化 ---
    camera_id = 4
    calibration_file = "./handeye/cam2station_test_017322074878.npy"
    print("正在加载标定文件:", os.path.abspath(calibration_file))
    if not os.path.exists(calibration_file):
        print(f"[错误] 标定文件未找到: {calibration_file}")
        return
    RT_cam2base = np.load(calibration_file)
    RT_cam2base = np.linalg.inv(RT_cam2base)
    print("RT_cam2base 加载成功:\n", RT_cam2base)

    # 初始化RealSense相机
    # 注意：请确保这些辅助函数可用
    check_rs_devices()
    pipeline, align = get_rs_pipeline(camera_id)

    # 初始化VIVE追踪器数据接收器
    receiver = VIVEDataReceiver(port=9999, num_floats=36)
    receiver.start()

    # 获取一次相机内参，假设它们在流式传输期间是恒定的
    _, _, intr_matrix, intr_coeffs = get_aligned_images(pipeline, align)
    if intr_matrix is None:
        print("[错误] 无法获取相机内参。")
        pipeline.stop()
        return

    # 定义要在追踪器上绘制的坐标轴（3D模型点）
    # 长度单位应与标定时使用的单位一致（例如，毫米）
    axis_length = 50.0  # 50mm = 5cm
    axis_points = np.float32([
        [0, 0, 0],         # 原点
        [axis_length, 0, 0], # X轴
        [0, axis_length, 0], # Y轴
        [0, 0, axis_length]  # Z轴
    ]).reshape(-1, 3)

    print("\n测试开始。按 'q' 或 ESC 键退出。")

    # --- 2. 主循环 ---
    while True:
        # 获取相机图像
        rgb, _, _, _ = get_aligned_images(pipeline, align)
        if rgb is None:
            continue

        # 获取最新的VIVE追踪器数据
        vive_data = receiver.get_latest_data()
        tracker_pose_data = vive_data[12:24]
        
        # 检查追踪器数据是否有效
        if not (tracker_pose_data[0] == 0 and tracker_pose_data[1] == 0 and tracker_pose_data[2] == 0):
            # --- 3. 核心变换逻辑 ---
            # a. 从VIVE数据构建"追踪器到基座"的4x4变换矩阵 (RT_tracker2base)
            T_tracker2base = np.array(tracker_pose_data[0:3]) * 1000  # 确保单位是毫米
            R_tracker2base = np.array(tracker_pose_data[3:12]).reshape(3, 3)
            RT_tracker2base = tfs.affines.compose(T_tracker2base, R_tracker2base, [1, 1, 1])

            # b. 计算"追踪器到相机"的变换矩阵
            #    P_cam = RT_cam2base * P_base
            #    P_base = RT_tracker2base * P_tracker
            #    => P_cam = RT_cam2base * RT_tracker2base * P_tracker
            RT_cam2tracker = RT_cam2base @ RT_tracker2base
            
            # c. 从4x4矩阵中提取旋转向量和平移向量，以供cv2.projectPoints使用
            R_cam2tracker = RT_cam2tracker[:3, :3]
            
            t_cam2tracker = RT_cam2tracker[:3, 3]

            print(t_cam2tracker)
            rvec, _ = cv2.Rodrigues(R_cam2tracker)

            # --- 4. 投影和渲染 ---
            # 将3D坐标轴点投影到2D图像平面
            img_points, _ = cv2.projectPoints(axis_points, rvec, t_cam2tracker, intr_matrix, intr_coeffs)
            
            # 将投影点坐标转换为整数
            img_points = np.int32(img_points).reshape(-1, 2)
            
            # 在图像上绘制坐标轴
            # BGR color: X-Red, Y-Green, Z-Blue
            origin_pt = tuple(img_points[0])
            # print(origin_pt)
            cv2.drawFrameAxes(rgb, intr_matrix, intr_coeffs, R_cam2tracker, t_cam2tracker, 0.05) # 绘制坐标轴

            cv2.line(rgb, origin_pt, tuple(img_points[1]), (0, 0, 255), 3) # X轴 (红色)
            cv2.line(rgb, origin_pt, tuple(img_points[2]), (0, 255, 0), 3) # Y轴 (绿色)
            cv2.line(rgb, origin_pt, tuple(img_points[3]), (255, 0, 0), 3) # Z轴 (蓝色)

        # 显示结果图像
        cv2.imshow('Calibration Test - Tracker Axes', rgb)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
            
    # --- 5. 清理 ---
    print("\n测试结束。")
    pipeline.stop()
    receiver.stop()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # VIVE_Station_Calibration()
    # RM_ARM_Calibration()
    # temp_test_Calibration()
    One_Calibration()
    # test_calibration()