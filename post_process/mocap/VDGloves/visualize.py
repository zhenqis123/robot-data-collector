import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import pyvista as pv
import torch
from trimesh import Trimesh
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PathConfig import PROJECT_ROOT, TACTASK_DIR, TELEOP_DIR, TACDATA_DIR
from teleop.hand_inspire_convert.MANO_inspire.manotorch.manotorch import *
from teleop.hand_inspire_convert.MANO_inspire.utils import show_plot

from tactask.tactile_representation.universal_UV_converter import *

from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput

from teleop.VDGloves.VDHand import *
from tactask.glove_hand import HandType

HAND_SKELETON_BONES = [
        # Palm
        [0, 5], [0, 9], [0, 13], [0, 17], [5, 9], [9, 13], [13, 17],
        # Thumb
        [0, 1], [1, 2], [2, 3], [3, 4],
        # Index
        [5, 6], [6, 7], [7, 8],
        # Middle
        [9, 10], [10, 11], [11, 12],
        # Ring
        [13, 14], [14, 15], [15, 16],
        # Pinky
        [17, 18], [18, 19], [19, 20],
    ]


# 定义 MANO 运动学链, 顺序为 Index, Middle, Ring, Pinky, Thumb
# 每个元组是 (子关节索引, 父关节索引)

MANO_KINEMATIC_CHAIN = [
        # Index (MCP, PIP, DIP)
        (4, 0), (5, 4), (6, 5),
        # Middle (MCP, PIP, DIP)
        (8, 0), (9, 8), (10, 9),
        # Ring (MCP, PIP, DIP)
        (12, 0), (13, 12), (14, 13),
        # Pinky (MCP, PIP, DIP)
        (16, 0), (17, 16), (18, 17),
        # Thumb (MCP, PIP, DIP)
        (1, 0), (2, 1), (3, 2),
    ]

MANO_KINEMATIC_CHAIN = [
    # Index (MCP, PIP, DIP)
    (5, 0), (6, 5), (7, 6),
    # Middle (MCP, PIP, DIP)
    (9, 0), (10, 9), (11, 10),
    # Ring (MCP, PIP, DIP)
    (13, 0), (14, 13), (15, 14),
    # Pinky (MCP, PIP, DIP)
    (17, 0), (18, 17), (19, 18),
    # Thumb (MCP, PIP, DIP)
    (2, 0), (3, 2), (4, 3),
]

def average_quaternions(quats_xyzw: np.ndarray) -> R:
    """
    对一组四元数进行稳健的平均。 (已修正 ValueError)
    
    Args:
        quats_xyzw (np.ndarray): 形状为 (N, 4) 的四元数数组 [x, y, z, w]。
    
    Returns:
        scipy.spatial.transform.Rotation: 平均后的旋转对象。
    """
    # 确保数组至少有两个四元数才能进行比较
    if quats_xyzw.shape[0] < 2:
        return R.from_quat(quats_xyzw[0])

    # 为了避免修改原始数据，我们创建一个副本
    quats_copy = np.copy(quats_xyzw)
    
    # 以第一个四元数为参考，检查其他四元数
    # 计算 quats_copy[1:] 中每个四元数与 quats_copy[0] 的点积
    dots = np.dot(quats_copy[1:], quats_copy[0])
    
    # 找到点积为负的四元数的索引
    flip_indices = np.where(dots < 0)
    
    # 将这些四元数翻转 (乘以 -1)，使其与参考四元数位于同一半球
    # 注意：我们是在副本上操作
    quats_copy[1:][flip_indices] *= -1
    
    # 计算平均四元数
    mean_quat = np.mean(quats_copy, axis=0)
    
    # 归一化结果
    mean_quat /= np.linalg.norm(mean_quat)
    
    return R.from_quat(mean_quat)

def calculate_mano_finger_pose_corrected(
    current_quats_wxyz: np.ndarray, 
    t_pose_quats_wxyz: np.ndarray
) -> torch.Tensor:
    """
    最终修正版 V3：
    1. 使用手掌多个关节点的旋转均值来定义一个稳定的手掌参考系。
    2. 将所有关节旋转转换到该局部手掌坐标系下。
    3. 在局部坐标系下进行 T-Pose 校准。
    4. 计算父子相对旋转。
    5. 提供可选的轴向翻转进行最后微调。

    Args:
        current_quats_wxyz (np.ndarray): 当前帧的全局旋转 [w, x, y, z]。
        t_pose_quats_wxyz (np.ndarray): T-Pose 帧的全局旋转 [w, x, y, z]。
    """
    if current_quats_wxyz.shape != t_pose_quats_wxyz.shape:
        raise ValueError("当前帧和 T-Pose 帧的数组形状必须一致。")

    # 定义手掌和运动学链
    PALM_JOINT_IDS = [0, 5, 9, 13, 17] # 你的图片中 Wrist, Index, Middle, Ring, Pinky 的根部

    # 将 wxyz 转换为 scipy 使用的 xyzw 格式
    current_quats_xyzw = current_quats_wxyz[:, [1, 2, 3, 0]]
    t_pose_quats_xyzw = t_pose_quats_wxyz[:, [1, 2, 3, 0]]
    
    # --- 1. 定义稳定的手掌参考系 ---
    palm_rot_current = average_quaternions(current_quats_xyzw[PALM_JOINT_IDS])
    palm_rot_tpose = average_quaternions(t_pose_quats_xyzw[PALM_JOINT_IDS])
    palm_rot_current_inv = palm_rot_current.inv()
    palm_rot_tpose_inv = palm_rot_tpose.inv()

    relative_rotations_axis_angle = []
    
    # 预计算所有关节的 Rotation 对象，避免重复计算
    r_current_globals = R.from_quat(current_quats_xyzw)
    r_tpose_globals = R.from_quat(t_pose_quats_xyzw)

    for child_idx, parent_idx in MANO_KINEMATIC_CHAIN:
        # --- 2. 转换到局部手掌坐标系 ---
        r_child_local_current = palm_rot_current_inv * r_current_globals[child_idx]
        r_parent_local_current = palm_rot_current_inv * r_current_globals[parent_idx]
        
        r_child_local_tpose = palm_rot_tpose_inv * r_tpose_globals[child_idx]
        r_parent_local_tpose = palm_rot_tpose_inv * r_tpose_globals[parent_idx]

        # --- 3. 在局部坐标系下进行 T-Pose 校准 ---
        r_child_pose = r_child_local_tpose.inv() * r_child_local_current
        r_parent_pose = r_parent_local_tpose.inv() * r_parent_local_current
        
        # --- 4. 计算最终的父子相对旋转 ---
        r_relative = r_parent_pose.inv() * r_child_pose
        
        axis_angle_vec = r_relative.as_rotvec()
        
        # --- 5. 最终微调 (根据实际效果取消注释) ---
        # 如果手指弯曲方向依然错误（例如翻向手背），请尝试以下修正
        # axis_angle_vec[0] *= -1  # 翻转 X 轴
        # axis_angle_vec[2] *= -1  # 翻转 Z 轴
        # axis_angle_vec *= -1     # 翻转所有轴
        
        relative_rotations_axis_angle.append(axis_angle_vec)

    finger_pose_numpy = np.concatenate(relative_rotations_axis_angle, axis=0)
    finger_pose_tensor = torch.from_numpy(finger_pose_numpy).float().unsqueeze(0)

    return finger_pose_tensor

def calculate_mano_finger_pose_final_v4(
    current_quats_wxyz: np.ndarray, 
    t_pose_quats_wxyz: np.ndarray
) -> torch.Tensor:
    """
    最终修正版 V4:
    在 V3 的基础上，增加了一个静态坐标系校正旋转，
    以解决弯曲轴向错误（例如横向弯曲）的问题。
    """
    if current_quats_wxyz.shape != t_pose_quats_wxyz.shape:
        raise ValueError("当前帧和 T-Pose 帧的数组形状必须一致。")

    PALM_JOINT_IDS = [0, 5, 9, 13, 17]
    # --- 核心修正点：定义一个从你的传感器坐标系到MANO坐标系的静态校正旋转 ---
    # 这个旋转将 Y 轴映射到 Z 轴，修正横向弯曲问题。
    # 如果-90度不对，可以尝试+90度。
    AXIS_CORRECTION = R.from_euler('x', 90, degrees=True)

    current_quats_xyzw = current_quats_wxyz[:, [1, 2, 3, 0]]
    t_pose_quats_xyzw = t_pose_quats_wxyz[:, [1, 2, 3, 0]]
    
    palm_rot_current = average_quaternions(current_quats_xyzw[PALM_JOINT_IDS])
    palm_rot_tpose = average_quaternions(t_pose_quats_xyzw[PALM_JOINT_IDS])
    palm_rot_current_inv = palm_rot_current.inv()
    palm_rot_tpose_inv = palm_rot_tpose.inv()

    relative_rotations_axis_angle = []
    
    r_current_globals = R.from_quat(current_quats_xyzw)
    r_tpose_globals = R.from_quat(t_pose_quats_xyzw)

    for child_idx, parent_idx in MANO_KINEMATIC_CHAIN:
        r_child_local_current = palm_rot_current_inv * r_current_globals[child_idx]
        r_parent_local_current = palm_rot_current_inv * r_current_globals[parent_idx]
        
        r_child_local_tpose = palm_rot_tpose_inv * r_tpose_globals[child_idx]
        r_parent_local_tpose = palm_rot_tpose_inv * r_tpose_globals[parent_idx]

        r_child_pose = r_child_local_tpose.inv() * r_child_local_current
        r_parent_pose = r_parent_local_tpose.inv() * r_parent_local_current
        
        r_relative = r_parent_pose.inv() * r_child_pose
        
        axis_angle_vec = r_relative.as_rotvec()
        
        # --- 将计算出的轴角向量应用校正旋转，转换到MANO的坐标系下 ---
        axis_angle_vec_corrected = AXIS_CORRECTION.apply(axis_angle_vec)
        
        relative_rotations_axis_angle.append(axis_angle_vec_corrected)

    finger_pose_numpy = np.concatenate(relative_rotations_axis_angle, axis=0)
    finger_pose_tensor = torch.from_numpy(finger_pose_numpy).float().unsqueeze(0)

    return finger_pose_tensor

def calculate_mano_finger_pose_final_v5(
    current_quats_wxyz: np.ndarray, 
    t_pose_quats_wxyz: np.ndarray
) -> torch.Tensor:
    """
    最终修正版 V5:
    1. 修正了第一关节 (MCP) 旋转的计算逻辑。
    2. 为大拇指设置了独立的坐标系校正规则。
    """
    if current_quats_wxyz.shape != t_pose_quats_wxyz.shape:
        raise ValueError("当前帧和 T-Pose 帧的数组形状必须一致。")

    # 注意：请再次确认你的索引与你的数据源完全对应
    PALM_JOINT_IDS = [0, 5, 9, 13, 17]
    MANO_KINEMATIC_CHAIN = [
        (6, 0), (7, 6), (8, 7),       # Index
        (10, 0), (11, 10), (12, 11),    # Middle
        (18, 0), (19, 18), (20, 19),  # Pinky
        (14, 0), (15, 14), (16, 15),  # Ring
        (2, 0), (3, 2), (4, 3),       # Thumb
    ]
    # 定义哪些关节属于大拇指，以便特殊处理
    THUMB_CHILD_JOINT_IDS = [2, 3, 4]
    FINGERS_CHILD_JOINT_IDS = [
        [6,7,8],
        [10,11,12],
        [14,15,16],
        [18,19,20]
    ]

    # 为四个主手指定义的静态坐标系校正
    fingers_angle = [90,90,90,90]
    # FINGER_AXIS_CORRECTION = R.from_euler('x', 90, degrees=True)
    FINGERS_AXIS_CORRECTION = R.from_euler('x', fingers_angle, degrees=True)
    ALL_AXIS_CORRECTION = R.from_euler('z', 180, degrees=True)


    current_quats_xyzw = current_quats_wxyz[:, [1, 2, 3, 0]]
    t_pose_quats_xyzw = t_pose_quats_wxyz[:, [1, 2, 3, 0]]
    
    palm_rot_current = average_quaternions(current_quats_xyzw[PALM_JOINT_IDS])
    palm_rot_tpose = average_quaternions(t_pose_quats_xyzw[PALM_JOINT_IDS])
    palm_rot_current_inv = palm_rot_current.inv()
    palm_rot_tpose_inv = palm_rot_tpose.inv()

    relative_rotations_axis_angle = []
    
    r_current_globals = R.from_quat(current_quats_xyzw)
    r_tpose_globals = R.from_quat(t_pose_quats_xyzw)

    # --- 预计算所有关节校准后的纯净姿态 ---
    all_joint_poses = []
    for i in range(len(current_quats_wxyz)):
        r_local_current = palm_rot_current_inv * r_current_globals[i]
        r_local_tpose = palm_rot_tpose_inv * r_tpose_globals[i]
        r_pose = r_local_tpose.inv() * r_local_current
        all_joint_poses.append(r_pose)

    for child_idx, parent_idx in MANO_KINEMATIC_CHAIN:
        r_child_pose = all_joint_poses[child_idx]
        
        r_parent_pose = all_joint_poses[parent_idx]
        r_relative = r_parent_pose.inv() * r_child_pose
            
        axis_angle_vec = r_relative.as_rotvec()

        if child_idx in THUMB_CHILD_JOINT_IDS:
            # 大拇指使用独立的校正
            # axis_angle_vec_corrected = THUMB_AXIS_CORRECTION.apply(axis_angle_vec)
            axis_angle_vec_corrected = axis_angle_vec
        else:
            for i in range(4):
                if child_idx in FINGERS_CHILD_JOINT_IDS[i]:
                    finger_axis_correction = FINGERS_AXIS_CORRECTION[i]
                    break
            axis_angle_vec_corrected = finger_axis_correction.apply(axis_angle_vec)

        # axis_angle_vec_corrected[0] *= -1 
        # axis_angle_vec_corrected = ALL_AXIS_CORRECTION.apply(axis_angle_vec_corrected)

        relative_rotations_axis_angle.append(axis_angle_vec_corrected)

    finger_pose_numpy = np.concatenate(relative_rotations_axis_angle, axis=0)
    finger_pose_tensor = torch.from_numpy(finger_pose_numpy).float().unsqueeze(0)

    return finger_pose_tensor

def calculate_mano_finger_pose_final_v7(
    current_quats_wxyz: np.ndarray, 
    t_pose_quats_wxyz: np.ndarray
) -> torch.Tensor:
    """
    最终修正版 V7: 基于父子坐标系相对旋转的根本逻辑进行重构。
    1. 补全了15个关节的完整运动学链。
    2. 采用 T-Pose 校准 -> 直接计算相对旋转 -> 最终轴向校正 的清晰流程。
    3. 为主手指和拇指提供独立的、可供最终微调的校正旋转。
    """
    if current_quats_wxyz.shape != t_pose_quats_wxyz.shape:
        raise ValueError("当前帧和 T-Pose 帧的数组形状必须一致。")

    # --- 1. 补全完整的15关节运动学链 (按MANO顺序) ---
    # MANO_KINEMATIC_CHAIN = [
    #     # Index Finger (MCP, PIP, DIP)
    #     (5, 0), (6, 5), (7, 6), (8, 7),       # Index
    #     # Middle Finger
    #     (9, 0), (10, 9), (11, 10),(12,11),
    #     # Ring Finger
    #     (13, 0), (14, 13), (15, 14),(16,15),
    #     # Pinky Finger
    #     (17, 0), (18, 17), (19, 18),(20,19),
    #     # Thumb
    #     (2, 0), (3, 2), (4, 3),
    # ]
    # PALM_JOINT_IDS = [0, 5, 9, 13, 17]
    PALM_JOINT_IDS = [0,6,10,14,18]

    MANO_KINEMATIC_CHAIN = [
        (6, 0), (7, 6), (8, 7),       # Index
        (10, 0), (11, 10), (12, 11),    # Middle
        (18, 0), (19, 18), (20, 19),  # Pinky
        (14, 0), (15, 14), (16, 15),  # Ring
        (2, 0), (3, 2), (4, 3),       # Thumb
    ]

    THUMB_CHILD_JOINT_IDS = []
    THUMB_CHILD_JOINT_IDS = [2, 3, 4]
    THUMB_CHILD_JOINT_IDS = [2]

    # --- 2. 为主手指和拇指提供独立的、可供微调的静态校正旋转 ---
    # 这是最后的微调部分，你需要凭经验调整这里的轴和角度
    # 主手指校正：结合了你代码中的 x(90) 和 z(180)
    # FINGER_AXIS_CORRECTION = R.from_euler('z', 180, degrees=True) * R.from_euler('x', 90, degrees=True)
    THUMB_AXIS_CORRECTION = R.from_euler('x', 180, degrees=True)

    FINGER_ROOT_CORRECTION = R.from_euler('x', 90, degrees=True)

    # 拇指校正：这是一个常见的初始猜测，可能需要你根据实际表现来修改
    # THUMB_AXIS_CORRECTION = R.from_euler('z', -90, degrees=True) * R.from_euler('y', 20, degrees=True)

    current_quats_xyzw = current_quats_wxyz[:, [1, 2, 3, 0]]
    t_pose_quats_xyzw = t_pose_quats_wxyz[:, [1, 2, 3, 0]]
    
    # --- 步骤 1: T-Pose 校准 ---
    # 预计算所有关节纯净的、在MANO世界坐标系下的全局旋转 (R_pose)
    r_current_globals = R.from_quat(current_quats_xyzw)
    r_tpose_inv_list = R.from_quat(t_pose_quats_xyzw).inv()
    all_joint_poses = r_tpose_inv_list * r_current_globals


    relative_rotations_axis_angle = []
    for child_idx, parent_idx in MANO_KINEMATIC_CHAIN:
        r_child_pose = all_joint_poses[child_idx]
        r_parent_pose = all_joint_poses[parent_idx]

        # --- 步骤 2: 计算直接的父子相对旋转 ---
        r_relative = r_parent_pose.inv() * r_child_pose
            
        # --- 步骤 3: 最终的静态轴向校正 ---
        if child_idx in THUMB_CHILD_JOINT_IDS:
            r_final = THUMB_AXIS_CORRECTION * r_relative
        elif child_idx in PALM_JOINT_IDS:
            r_final = FINGER_ROOT_CORRECTION * r_relative
        else:
            r_final = r_relative
        
        axis_angle_vec = r_final.as_rotvec()
        relative_rotations_axis_angle.append(axis_angle_vec)

    finger_pose_numpy = np.concatenate(relative_rotations_axis_angle, axis=0)
    finger_pose_tensor = torch.from_numpy(finger_pose_numpy).float().unsqueeze(0)

    return finger_pose_tensor

def visualize_hand(keypoints, quaternions, bones, title="Hand Visualization"):
        """
        可视化手部关节点、骨架和局部坐标系。
        
        Args:
            keypoints (np.ndarray): (21, 3) 的关节点坐标.
            quaternions (np.ndarray): (21, 4) 的关节点旋转 (w,x,y,z 格式).
            bones (list): 关节点连接关系.
            title (str): 图表标题.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制关节点
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='r', marker='o', label="Keypoints")

        # 绘制骨架
        for start_idx, end_idx in bones:
            ax.plot(
                [keypoints[start_idx, 0], keypoints[end_idx, 0]],
                [keypoints[start_idx, 1], keypoints[end_idx, 1]],
                [keypoints[start_idx, 2], keypoints[end_idx, 2]],
                'b-'
            )

        # 绘制每个关节点的局部坐标系
        axis_length = 0.02  # 坐标轴的显示长度
        for i in range(len(keypoints)):
            pos = keypoints[i]
            quat_wxyz = quaternions[i]
            
            # 将 (w, x, y, z) 转换为 (x, y, z, w) 以适配 scipy
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            
            try:
                rot_matrix = R.from_quat(quat_xyzw).as_matrix()
            except ValueError as e:
                print(f"Warning: Invalid quaternion at index {i}: {quat_wxyz}. Skipping. Error: {e}")
                continue

            # X, Y, Z 轴在世界坐标系下的方向
            x_axis = rot_matrix[:, 0]
            y_axis = rot_matrix[:, 1]
            z_axis = rot_matrix[:, 2]

            # 绘制轴
            ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], length=axis_length, color='red', normalize=True)
            ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], length=axis_length, color='green', normalize=True)
            ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], length=axis_length, color='blue', normalize=True)

        # 设置图表
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        ax.view_init(elev=90, azim=-90) # 俯视图，与你的图片类似 (Top -Y)
        plt.grid(True)
        
        # 保持各轴比例一致
        max_range = np.array([keypoints[:, 0].max()-keypoints[:, 0].min(), 
                            keypoints[:, 1].max()-keypoints[:, 1].min(), 
                            keypoints[:, 2].max()-keypoints[:, 2].min()]).max() / 2.0
        mid_x = (keypoints[:, 0].max()+keypoints[:, 0].min()) * 0.5
        mid_y = (keypoints[:, 1].max()+keypoints[:, 1].min()) * 0.5
        mid_z = (keypoints[:, 2].max()+keypoints[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()

def calculate_mano_finger_pose(global_quats_wxyz: np.ndarray) -> torch.Tensor:
    """
    从全局四元数计算 MANO 的 finger_pose (相对旋转的轴角表示).

    Args:
        global_quats_wxyz (np.ndarray): 形状为 (21, 4) 或 (20, 4) 的 NumPy 数组, 
                                        包含所有关节点的全局旋转四元数。
                                        格式必须为 [w, x, y, z]。

    Returns:
        torch.Tensor: 形状为 (1, 45) 的 PyTorch 张量, 代表 MANO 的 finger_pose。
    """
    if global_quats_wxyz.shape[0] < 20:
        raise ValueError("输入 global_quats_wxyz 必须至少包含 20 个关节点的四元数。")
    if global_quats_wxyz.shape[1] != 4:
        raise ValueError("四元数维度必须为 4。")

    relative_rotations_axis_angle = []

    for child_idx, parent_idx in MANO_KINEMATIC_CHAIN:
        # 提取 w,x,y,z 格式的四元数
        q_child_wxyz = global_quats_wxyz[child_idx]
        q_parent_wxyz = global_quats_wxyz[parent_idx]

        # Scipy 需要 [x, y, z, w] 格式
        q_child_xyzw = q_child_wxyz[[1, 2, 3, 0]]
        q_parent_xyzw = q_parent_wxyz[[1, 2, 3, 0]]

        # 创建 Scipy Rotation 对象
        r_child_global = R.from_quat(q_child_xyzw)
        r_parent_global = R.from_quat(q_parent_xyzw)

        # 计算相对旋转: q_rel = q_parent_inv * q_child
        r_relative = r_parent_global.inv() * r_child_global

        # 转换为轴角表示 (Rotation Vector)
        # as_rotvec() 返回一个3D向量, 方向是旋转轴, 模长是旋转角度(弧度)
        # 这正是 MANO 所需的格式
        axis_angle_vec = r_relative.as_rotvec()
        relative_rotations_axis_angle.append(axis_angle_vec)

    # 将15个 (3,) 向量堆叠成一个 (45,) 向量
    finger_pose_numpy = np.concatenate(relative_rotations_axis_angle, axis=0)

    # 转换为 PyTorch 张量并调整形状为 (1, 45)
    finger_pose_tensor = torch.from_numpy(finger_pose_numpy).float().unsqueeze(0)

    return finger_pose_tensor

def visualize_mano_finger_pose(hand_quats):

    # mock_global_quats = hand_quats
    
    # # 模拟食指弯曲: 假设食指 MCP (索引4) 相对手腕(索引0)旋转了
    # # 绕 X 轴旋转 45 度 (pi/4)
    # r_bend = R.from_euler('x', 45, degrees=True)
    # # 更新食指所有关节点的全局旋转
    # for idx in [4, 5, 6, 7]: # Index MCP, PIP, DIP, Tip
    #     mock_global_quats[idx] = r_bend.as_quat()[[3, 0, 1, 2]] # scipy is xyzw, convert back to wxyz

    # print("输入全局四元数 (手腕和食指MCP):")
    # print(f"Wrist (0): {mock_global_quats[0]}")
    # print(f"Index MCP (4): {mock_global_quats[4]}")
    # print("-" * 30)
    
    # 2. 调用函数计算 finger_pose
    finger_pose = calculate_mano_finger_pose(hand_quats)

    # 3. 打印结果
    print(f"计算得到的 finger_pose 张量形状: {finger_pose.shape}")
    print("finger_pose 内容 (前12维):")
    # 应该只有前3个维度 (对应食指MCP) 有非零值
    print(finger_pose[0, :12])

    return finger_pose


def vis_mano():
    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="left",
        center_idx=0, # 0: 手腕 (根关节)
        mano_assets_root=f"{PROJECT_ROOT}/teleop/hand_inspire_convert/MANO_inspire/manotorch/assets/mano",
        flat_hand_mean=True,
        device = "cuda"
    )
    # root_pose shape torch.Size([1, 3]), xyz, we don't need
    # finger_pose shape torch.Size([1, 45])
    # beta shape torch.Size([1, 10])
    root_pose = torch.zeros(1, 3).to("cuda")
    beta = torch.zeros(1, 10).to("cuda")
    finger_pose = torch.tensor(finger_pose, device="cuda")
    hand_pose = torch.cat([root_pose, finger_pose], dim=1)
    mano_results: MANOOutput = mano_layer(hand_pose, betas=beta)




if __name__ == '__main__':
    print("Starting MOCAP data stream demo...")
    stream = DataStream()
    stream.set_ip("192.168.20.240")
    stream.set_broascast_port(9998)
    stream.set_local_port(9999)

    hand_data_processor = MocapHandDataProcessor()

    if not stream.connect():
        exit()

    print("\nConnection established. Starting to receive data...")
    print("Press Ctrl+C to stop.")

    stream.request_mocap_data()
    mocap_data = stream.get_mocap_data()
    while True:
        if mocap_data.isUpdate:
            print(mocap_data.sensorState_lHand)
            hand_data_dict = hand_data_processor.process(mocap_data)
            # t_pose_quaternions = hand_data_processor.convert_hand_data_to_quaternion(hand_data_dict["right_hand"])
            # np.save("t_pose_right_hand.npy", t_pose_quaternions)

            left_keypoints, left_quaternion, left_wrist_rot, right_keypoints, right_quaternion, right_wrist_rot = hand_data_processor.extract_bimanual_VD_data(hand_data_dict, "both")
            break
    # visualize_hand(right_keypoints, right_quaternion, HAND_SKELETON_BONES)

    # right_keypoints = right_keypoints - right_keypoints[0]  # 将手腕移到原点
    # print(right_wrist_rot)
    # right_keypoints = np.linalg.inv(right_wrist_rot) @ right_keypoints.T
    # right_keypoints1 = right_keypoints.T @ VD_RIGHT_HAND_TO_MANO
    # # right_keypoints2 = VD_RIGHT_HAND_TO_MANO @ right_keypoints
    # # right_keypoints = right_keypoints.T

    # visualize_hand(right_keypoints1, right_quaternion, HAND_SKELETON_BONES)
    # visualize_hand(right_keypoints2, right_quaternion, HAND_SKELETON_BONES)

    # breakpoint()


    # right_mano_keypoints = VD_to_mano_keypoints(right_keypoints=right_keypoints, hand_wrist_rot=right_wrist_rot, hand_type=HandType.RIGHT)

    # left_mano_keypoints = VD_to_mano_keypoints(hand_keypoints=left_keypoints, hand_wrist_rot=left_wrist_rot, hand_type=HandType.LEFT)
    # print(torch.from_numpy(left_mano_keypoints))
    
    # show_plot(left_mano_keypoints)
    # breakpoint()
    
    
    right_mano_keypoints = VD_to_mano_keypoints(hand_keypoints=right_keypoints, hand_wrist_rot=right_wrist_rot, hand_type=HandType.RIGHT)
    show_plot(right_mano_keypoints)
    breakpoint()
    rl_keypoints = swap_lr_keypoints_mano(right_mano_keypoints)
    show_plot(rl_keypoints)
    
    left_mano_keypoints = left_mano_keypoints[np.newaxis, :, :]
    # show_plot(left_mano_keypoints)
    # print(left_mano_keypoints.shape)
    # breakpoint()

    # visualize_keypoints()
    t_pose_data = np.load("t_pose_right_hand.npy")
    # assert right_quaternion is not None
    # assert left_quaternion is not None
    # finger_pose = calculate_mano_finger_pose(right_quaternion)
    # finger_pose = calculate_mano_finger_pose_final_v7(left_quaternion, t_pose_data)


    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="left",
        center_idx=0, # 0: 手腕 (根关节)
        mano_assets_root=f"{PROJECT_ROOT}/teleop/hand_inspire_convert/MANO_inspire/manotorch/assets/mano",
        flat_hand_mean=True,
        device = "cuda"
    )
    # root_pose shape torch.Size([1, 3]), xyz, we don't need
    # finger_pose shape torch.Size([1, 45])
    # beta shape torch.Size([1, 10])
    root_pose = torch.zeros(1, 3).to("cuda")
    beta = torch.zeros(1, 10).to("cuda")
    finger_pose = np.zeros((1, 45))
    finger_pose = torch.tensor(finger_pose, device="cuda")
    hand_pose = torch.cat([root_pose, finger_pose], dim=1)
    print(hand_pose)
    mano_results: MANOOutput = mano_layer(hand_pose, betas=beta)
    right_mano_keypoints = mano_results.joints.cpu().numpy()
    # show_plot(right_mano_keypoints)

    joints = torch.tensor(left_mano_keypoints, device="cuda", dtype=torch.float32)  # (1, 21, 3)
    axis_angle_results = mano_layer.joints_to_mano_parameters(joints, betas=beta)
    mano_results: MANOOutput = mano_layer(axis_angle_results["full_poses"], betas=axis_angle_results["betas"])
   
    tac_converter = UniTacConverter()
    con_mano_results = tac_converter.MANO_keypoints_to_xyz778(
        right_mano_keypoints,
        hand_type=HandType.RIGHT
    )

    # breakpoint()
    joints = mano_results.joints.cpu().numpy()  # (B, 21, 3)
    print(f"joints shape: {joints.shape}")
    print(joints)

    show_plot(joints)

    # joints = mano_results.joints.cpu().numpy()  # (B, 21, 3)
    # t1 = time.time()
    verts = mano_results.verts
    faces = mano_layer.th_faces
    V = verts[0].cpu().numpy()
    F = faces.cpu().numpy()
    tmesh = Trimesh(V, F)
    mesh = pv.wrap(tmesh)
    # joints = mano_results.joints.cpu().numpy()  # (B, 21, 3)
    # print(f"joints shape: {joints.shape}")
    # print(joints)

    # T_g_p = mano_results.transforms_abs  # (B, 16, 4, 4)
    # T_g_a, R, ee = axis_layer(T_g_p)  # ee (B, 16, 3)

    # bul_axes_loc = torch.eye(3).reshape(1, 1, 3, 3).repeat(BS, 16, 1, 1).to(verts.device)
    # bul_axes_glb = torch.matmul(T_g_a[:, :, :3, :3], bul_axes_loc)  # (B, 16, 3, 3)

    # b_axes_dir = bul_axes_glb[:, :, :, 0].cpu().numpy()  # back direction (B, 16, 3)
    # u_axes_dir = bul_axes_glb[:, :, :, 1].cpu().numpy()  # up direction (B, 16, 3)
    # l_axes_dir = bul_axes_glb[:, :, :, 2].cpu().numpy()  # left direction (B, 16, 3)

    # axes_cen = T_g_a[:, :, :3, 3].cpu().numpy()  # center (B, 16, 3)

    pl = pv.Plotter(off_screen=False)
    pl.add_mesh(mesh, opacity=0.4, name="mesh", smooth_shading=True)

    # if args.mode == "axis":
    #     pl.add_arrows(axes_cen, b_axes_dir, color="red", mag=0.02)
    #     pl.add_arrows(axes_cen, u_axes_dir, color="yellow", mag=0.02)
    #     pl.add_arrows(axes_cen, l_axes_dir, color="blue", mag=0.02)
    # elif args.mode == "anchor":
    #     anchors = anchor_layer(verts)[0].numpy()
    #     n_achors = anchors.shape[0]
    #     for i in range(n_achors):
    #         pl.add_mesh(pv.Cube(center=anchors[i], x_length=3e-3, y_length=3e-3, z_length=3e-3),
    #                     color="yellow",
    #                     name=f"anchor{i}")

    # pl.set_background('white')
    pl.add_camera_orientation_widget()
    pl.show(interactive=True)
    pl.close()