import sapien
# from isaacgymenvs.utils.torch_jit_utils import *
import torch
from functools import partial
import pickle
from pathlib import Path
from typing import List
import importlib
import numpy as np
import tqdm
from pynput import keyboard
import time, sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PathConfig import PROJECT_ROOT, TACTASK_DIR, TELEOP_DIR, TACDATA_DIR

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
#from dex_retargeting.robot_wrapper import RobotWrapper
#from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
#import cv2
from copy import deepcopy

import socket
from teleop.noitom.bvh_utils import *
from teleop.VDGloves.VDHand import *


RIGHT_HAND_KEYS = [
    'RightHand', 
    'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb3_end', 
    # 'RightInHandIndex', 
    'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex3_end', 
    # 'RightInHandMiddle', 
    'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle3_end', 
    # 'RightInHandRing', 
    'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing3_end',
    # 'RightInHandPinky', 
    'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky3_end'
]
# RIGHT_HAND_KEYS = [
#     'RightHand', 
#     'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb3', 
#     # 'RightInHandIndex', 
#     'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex3', 
#     # 'RightInHandMiddle', 
#     'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle3', 
#     # 'RightInHandRing', 
#     'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing3',
#     # 'RightInHandPinky', 
#     'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky3'
# ]
LEFT_HAND_KEYS = [
    'LeftHand', 
    'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb3_end', 
    # 'LeftInHandIndex', 
    'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex3_end', 
    # 'LeftInHandMiddle', 
    'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle3_end', 
    # 'LeftInHandRing', 
    'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing3_end', 
    # 'LeftInHandPinky', 
    'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky3_end'
]

ALL_LINK_KEYS = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 'Head',
    
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb3_end', 
    'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex3_end', 
    'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle3_end', 
    'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing3_end',
    'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky3_end',
    
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb3_end',
    'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex3_end',
    'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle3_end',
    'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing3_end',
    'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky3_end'
]

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)
OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)
OPERATOR2AVP_RIGHT = OPERATOR2MANO_RIGHT
OPERATOR2AVP_LEFT = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
)

FACE_WEST_OPERATOR_RIGHT = np.array(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]
)
FACE_WEST_OPERATOR_LEFT = np.array(
    [
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]
)
FACE_EAST_OPERATOR_RIGHT = FACE_WEST_OPERATOR_LEFT
FACE_EAST_OPERATOR_LEFT = FACE_WEST_OPERATOR_RIGHT
FACE_NORTH_OPERATOR_RIGHT = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ]
)
FACE_NORTH_OPERATOR_LEFT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0]
    ]
)
FACE_EAST_OPERATOR_RIGHT_4 = np.eye(4)
FACE_EAST_OPERATOR_RIGHT_4[:3, :3] = FACE_EAST_OPERATOR_RIGHT
FACE_EAST_OPERATOR_LEFT_4 = np.eye(4)
FACE_EAST_OPERATOR_LEFT_4[:3, :3] = FACE_EAST_OPERATOR_LEFT
FACE_WEST_OPERATOR_RIGHT_4 = np.eye(4)
FACE_WEST_OPERATOR_RIGHT_4[:3, :3] = FACE_WEST_OPERATOR_RIGHT
FACE_WEST_OPERATOR_LEFT_4 = np.eye(4)
FACE_WEST_OPERATOR_LEFT_4[:3, :3] = FACE_WEST_OPERATOR_LEFT
FACE_NORTH_OPERATOR_RIGHT_4 = np.eye(4)
FACE_NORTH_OPERATOR_RIGHT_4[:3, :3] = FACE_NORTH_OPERATOR_RIGHT
FACE_NORTH_OPERATOR_LEFT_4 = np.eye(4)
FACE_NORTH_OPERATOR_LEFT_4[:3, :3] = FACE_NORTH_OPERATOR_LEFT


# BVH的右手手腕坐标系是T-pose下的“左上前”；而AVP-keypoints的右手手腕坐标系是逆时针XYZ坐标系（参见zc画的图）
RIGHT_HAND_OPERATOR = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
)
# BVH的左手手腕坐标系是T-pose下的“左上前”；而AVP-keypoints的左手手腕坐标系是逆时针XYZ坐标系（参见zc画的图）
LEFT_HAND_OPERATOR = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
)

# 用于左乘的坐标系变换矩阵
RIGHT_ARM_OPERATOR_1 = np.array(
    [
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]
)
# 用于右臂的平移矩阵
RIGHT_ARM_TRANS_MATRIX = np.array([
    [1, 0, 0, -0.02],
    [0, 1, 0, -0.2],
    [0, 0, 1, -1.0],
    [0, 0, 0, 1]
])
# 用于右乘的坐标系变换矩阵
RIGHT_ARM_OPERATOR_2 = np.array(
    [
        [0, -1, 0, -0.1],
        [0, 0, 1, -0.2],
        [-1, 0, 0, -1.0],
        [0, 0, 0, 1]
    ]
)


# 用于左乘的坐标系变换矩阵
LEFT_ARM_OPERATOR_1 = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]
)
# 用于左臂的平移矩阵
LEFT_ARM_TRANS_MATRIX = np.array([
    [1, 0, 0, -0.1],
    [0, 1, 0, 0.06],
    [0, 0, 1, -1.0],
    [0, 0, 0, 1]
])
# 用于右乘的坐标系变换矩阵
LEFT_ARM_OPERATOR_2 = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
)       # 存疑？


def midpoint_transformation_matrix(mat1: np.ndarray, mat2: np.ndarray):
    assert mat1.shape == (4, 4) and mat2.shape == (4, 4)
    key_times = [0, 1]
    key_rots = R.from_matrix([mat1[:3, :3], mat2[:3, :3]])
    slerp = Slerp(key_times, key_rots)
    mid_rot = slerp(0.5).as_matrix()
    # mid_rot = slerp([0.5]).as_matrix()[0]
    
    # mid_rot = R.from_matrix(mat1[:3, :3]).slerp(R.from_matrix(mat2[:3, :3]), 0.5).as_matrix()
    mid_pos = (mat1[:3, 3] + mat2[:3, 3]) / 2
    mid_mat = np.eye(4)
    mid_mat[:3, :3] = mid_rot
    mid_mat[:3, 3] = mid_pos
    return mid_mat


def three_mat_mul(left_rot: np.ndarray, mat: np.ndarray, right_rot: np.ndarray):
    result = np.eye(4)
    rotation = left_rot @ mat[:3, :3] @ right_rot
    pos = left_rot @ mat[:3, 3]
    result[:3, :3] = rotation
    result[:3, 3] = pos
    return result

def two_mat_batch_mul(batch_mat: np.ndarray, left_rot: np.ndarray):
    result = np.tile(np.eye(4), [batch_mat.shape[0], 1, 1])
    result[:, :3, :3] = np.matmul(left_rot[None, ...], batch_mat[:, :3, :3])
    result[:, :3, 3] = batch_mat[:, :3, 3] @ left_rot.T
    return result

def joint_avp2hand(finger_mat: np.ndarray):
    finger_index = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24])
    finger_mat = finger_mat[finger_index]
    return finger_mat


class HandCosRetargeter:
    """
    一个模块化的类，用于将手部关键点重定向为6-DoF机械手执行器值。

    该类支持两种插值方法：
    1. 'cosine': 直接使用关键点向量之间的余弦值进行线性插值。
                 此模式下的输出与原始代码完全一致。
    2. 'angle':  先计算向量间的夹角（弧度），再对角度进行线性插值。
                 这种方法在物理上更直观。
    """

    def __init__(self, method='angle'):
        """
        初始化HandPoseRetargeter。
        
        参数:
            method (str): 插值方法，'angle' 或 'cosine'。默认为 'angle'。
        """
        if method not in ['angle', 'cosine']:
            raise ValueError("方法必须是 'angle' 或 'cosine'")
        self.method = method
        self._setup_retarget_config()

    def _setup_retarget_config(self):
        """
        配置每个自由度（DOF）的重定向参数。
        
        - 'key_groups': 定义计算特征值所需的向量端点。
        - 'cos_range': 源余弦值范围 [min, max]，使得 min 映射到 0.0，max 映射到 1.0。
        """
        self.config = {
            'thumb_spread': { # 对应 qpos[-1]
                'key_groups': [((0, 4), (0, 2)), ((0, 4), (0, 5))],
                'cos_range': [0.8, 0.99]
            },
            'thumb_knuckle': { # 对应 qpos[-2]
                'key_groups': [(3, 2), (3, 4)], # 标准顶点式计算以点3为顶点
                'cos_range': [0.7, 0.95]
            },
            'index_knuckle': { # 对应 qpos[-3]
                'key_groups': [(6, 5), (6, 8)],
                'cos_range': [0.4, -1.0]
            },
            'middle_knuckle': { # 对应 qpos[-4]
                'key_groups': [(10, 9), (10, 12)],
                'cos_range': [0.45, -1.0]
            },
            'ring_knuckle': { # 对应 qpos[-5]
                'key_groups': [(14, 13), (14, 16)],
                'cos_range': [0.2, -1.0]
            },
            'pinky_knuckle': { # 对应 qpos[-6]
                'key_groups': [(18, 17), (18, 20)],
                'cos_range': [-0.2, -1.0]
            }
        }
        
        for key, params in self.config.items():
            cos_min, cos_max = params['cos_range']
            # np.arccos 是一个递减函数。为了保持映射方向的一致性
            # (即 cos_min -> 0, cos_max -> 1)，角度的范围需要反过来。
            # angle_at_cos_min 应该映射到 0，angle_at_cos_max 应该映射到 1。
            angle_at_cos_min = np.arccos(np.clip(cos_min, -1.0, 1.0))
            angle_at_cos_max = np.arccos(np.clip(cos_max, -1.0, 1.0))
            params['angle_range'] = [angle_at_cos_min, angle_at_cos_max]

    def _calculate_cosine(self, p_mid, p_start, p_end):
        """安全地计算以 p_mid 为顶点的夹角余弦值。"""
        v1 = p_start - p_mid
        v2 = p_end - p_mid
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 * norm_v2 < 1e-8:
            return 1.0
        
        cosine_val = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.clip(cosine_val, -1.0, 1.0)

    def _linear_interpolate(self, value, source_range, target_range=(0.0, 1.0)):
        """将一个值从源范围线性映射到目标范围。"""
        source_min, source_max = source_range
        target_min, target_max = target_range
        
        if abs(source_max - source_min) < 1e-8:
            return target_min

        scale = (value - source_min) / (source_max - source_min)
        return target_min + scale * (target_max - target_min)

    def retarget(self, hand_keypoints):
        """
        执行重定向过程。

        参数:
            hand_keypoints (list or np.ndarray): 手部21个关键点的坐标。

        返回:
            np.ndarray: 6-DoF硬件执行器的目标qpos值，范围在[0, 1]之间。
        """
        joint_pos = np.array(hand_keypoints, dtype=np.float32)
        hardware_qpos = np.zeros(6, dtype=np.float32)
        
        dof_keys = ['thumb_spread', 'thumb_knuckle', 'index_knuckle', 
                    'middle_knuckle', 'ring_knuckle', 'pinky_knuckle']

        for i, key in enumerate(dof_keys):
            params = self.config[key]
            
            if key == 'thumb_spread':
                groups = params['key_groups']
                cos1 = self._calculate_cosine(joint_pos[groups[0][0][0]], joint_pos[groups[0][0][1]], joint_pos[groups[0][1][1]])
                cos2 = self._calculate_cosine(joint_pos[groups[1][0][0]], joint_pos[groups[1][0][1]], joint_pos[groups[1][1][1]])
                feature_cos = (cos1 + cos2) / 2.0
            else:
                p_mid_idx, p_start_idx = params['key_groups'][0]
                _, p_end_idx = params['key_groups'][1]
                feature_cos = self._calculate_cosine(joint_pos[p_mid_idx], joint_pos[p_start_idx], joint_pos[p_end_idx])

                # !!! 关键修正 !!!
                # 原始代码对 thumb_knuckle 的计算方式很特殊 (dot(p3-p2, p4-p3))。
                # 我们的标准顶点式计算结果是 dot(p2-p3, p4-p3)，两者正好差一个负号。
                # 因此，在这里取反以匹配原始代码的逻辑。
                if key == 'thumb_knuckle':
                    feature_cos = -feature_cos

            if self.method == 'angle':
                feature_value = np.arccos(np.clip(feature_cos, -1.0, 1.0))
                source_range = params['angle_range']
            else: # 'cosine'
                feature_value = feature_cos
                source_range = params['cos_range']
            
            # 赋值顺序：i=0 -> qpos[-1], i=1 -> qpos[-2], ...
            hardware_qpos[-(i + 1)] = self._linear_interpolate(feature_value, source_range)

        return np.clip(hardware_qpos, 0.0, 1.0)

class RetargetKeyboardListener:
    def __init__(self, retargetor):
        self.retargetor = retargetor
    
    def rotate_key_press(self, key):
        if key.char == 'x': 
            # self.retargetor.init_right_wrist_pose绕X轴正向旋转90度
            if self.retargetor.use_right_hand:
                tmp_transformation_matrix = np.eye(4)
                tmp_transformation_matrix[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()
                self.retargetor.init_right_wrist_pose = tmp_transformation_matrix @ self.retargetor.init_right_wrist_pose
            if self.retargetor.use_left_hand:
                tmp_transformation_matrix = np.eye(4)
                tmp_transformation_matrix[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()
                self.retargetor.init_left_wrist_pose = tmp_transformation_matrix @ self.retargetor.init_left_wrist_pose
        elif key.char == 'y':
            if self.retargetor.use_right_hand:
                tmp_transformation_matrix = np.eye(4)
                tmp_transformation_matrix[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
                self.retargetor.init_right_wrist_pose = tmp_transformation_matrix @ self.retargetor.init_right_wrist_pose
            if self.retargetor.use_left_hand:
                tmp_transformation_matrix = np.eye(4)
                tmp_transformation_matrix[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
                self.retargetor.init_left_wrist_pose = tmp_transformation_matrix @ self.retargetor.init_left_wrist_pose
        elif key.char == 'z':
            if self.retargetor.use_right_hand:
                tmp_transformation_matrix = np.eye(4)
                tmp_transformation_matrix[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
                self.retargetor.init_right_wrist_pose = tmp_transformation_matrix @ self.retargetor.init_right_wrist_pose
            if self.retargetor.use_left_hand:
                tmp_transformation_matrix = np.eye(4)
                tmp_transformation_matrix[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
                self.retargetor.init_left_wrist_pose = tmp_transformation_matrix @ self.retargetor.init_left_wrist_pose
        elif key.char == 'c':   # calibration, 开始校正——将当前手腕位姿作为初始位姿
            self.retargetor.start_calibration = True
        
        elif key.char == '1':
            tmp_transformation_matrix = np.eye(4)
            tmp_transformation_matrix[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()
            self.retargetor.end_right_wrist_pose = tmp_transformation_matrix @ self.retargetor.end_right_wrist_pose
        elif key.char == '2':
            tmp_transformation_matrix = np.eye(4)
            tmp_transformation_matrix[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
            self.retargetor.end_right_wrist_pose = tmp_transformation_matrix @ self.retargetor.end_right_wrist_pose
        elif key.char == '3':
            tmp_transformation_matrix = np.eye(4)
            tmp_transformation_matrix[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
            self.retargetor.end_right_wrist_pose = tmp_transformation_matrix @ self.retargetor.end_right_wrist_pose



# Retargetor: visionpro human motion -> target end-effector poses and hand joint positions
class VDArmHandRetarget:
    def __init__(
        self, 
        hand_name: RobotName=RobotName.inspiretacofficial, 
        mode="right", # right, left, both
        left_right_base_distance=0.525,
        safe_workspace=None,
        retargeting_type: RetargetingType=RetargetingType.vector,
        use_cosine="consine", # None or "consine" or "angle"
    ):
        self.use_left_hand = (mode=="left" or mode=="both")
        self.use_right_hand = (mode=="right" or mode=="both")

        # Create dexpilot retargeting
        robot_dir = Path(importlib.util.find_spec('dex_retargeting').origin).absolute().parent.parent / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        override = dict(add_dummy_free_joint=True)
        if self.use_right_hand:
            right_config_path = get_default_config_path(hand_name, retargeting_type, HandType.RIGHT, if_vd=True)
            print(f"RIGHT:{right_config_path}")
            self.right_retargeting_config = RetargetingConfig.load_from_file(right_config_path, override=override)
            self.right_retargeting = self.right_retargeting_config.build()
            self.right_retargeting_hand_dof_names = self.right_retargeting.optimizer.robot.dof_joint_names
            print("retargeting right hand dof names:", self.right_retargeting_hand_dof_names)
            self.right_retargeting_hand_active_dof_names = self.right_retargeting.optimizer.target_joint_names
            print("retargeting right hand active dof names:", self.right_retargeting_hand_active_dof_names)
        else:
            self.right_retargeting_hand_dof_names = None
        if self.use_left_hand:
            left_config_path = get_default_config_path(hand_name, retargeting_type, HandType.LEFT, if_vd=True)
            print(f"LEFT:{left_config_path}")
            self.left_retargeting_config = RetargetingConfig.load_from_file(left_config_path, override=override)
            self.left_retargeting = self.left_retargeting_config.build()
            self.left_retargeting_hand_dof_names = self.left_retargeting.optimizer.robot.dof_joint_names
            print("retargeting left hand dof names:", self.left_retargeting_hand_dof_names)
            self.left_retargeting_hand_active_dof_names = self.left_retargeting.optimizer.target_joint_names
            print("retargeting left hand active dof names:", self.left_retargeting_hand_active_dof_names)
        else:
            self.left_retargeting_hand_dof_names = None
        
        # arm configs
        if self.use_right_hand:
            # world->base: 世界坐标系在人的脚底，x约是肩宽的一半，z是右肩的高度(坐着1m), y考虑人手臂比机械臂短多少
            robot_start_pos_right = np.array([left_right_base_distance/2, 0, 0.6]) 
            robot_start_rot_right = R.from_euler('xyz', [0., 0., 0.]).as_matrix()
            self.RT_world2base_right, self.RT_base2world_right = np.eye(4), np.eye(4)
            self.RT_base2world_right[:3,:3] = robot_start_rot_right
            self.RT_base2world_right[:3,3] = robot_start_pos_right
            self.RT_world2base_right[:3,:3] = robot_start_rot_right.T 
            self.RT_world2base_right[:3,3] = -robot_start_rot_right.T @ robot_start_pos_right
            if use_cosine is not None:
                self.hand_cos_retargeter = HandCosRetargeter(method=use_cosine)

        if self.use_left_hand:
            robot_start_pos_left = np.array([-left_right_base_distance/2, 0, 0.6]) 
            robot_start_rot_left = R.from_euler('xyz', [0., 0., -np.pi]).as_matrix()
            self.RT_world2base_left, self.RT_base2world_left = np.eye(4), np.eye(4)
            self.RT_base2world_left[:3,:3] = robot_start_rot_left
            self.RT_base2world_left[:3,3] = robot_start_pos_left
            self.RT_world2base_left[:3,:3] = robot_start_rot_left.T 
            self.RT_world2base_left[:3,3] = -robot_start_rot_left.T @ robot_start_pos_left

            if use_cosine is not None:
                self.hand_cos_retargeter = HandCosRetargeter(method=use_cosine)

            # TODO: support left hand cosine retargeting

        self.safe_workspace = np.array(safe_workspace) if safe_workspace is not None else None
        
        self.start_calibration = False
        if self.use_right_hand:
            self.init_right_wrist_pose = np.eye(4)
            self.end_right_wrist_pose = np.eye(4)
        if self.use_left_hand:
            self.init_left_wrist_pose = np.eye(4)
            self.end_left_wrist_pose = np.eye(4)
        self.use_cosine = use_cosine
    
    def retarget(self, left_hand_keypoints, left_wrist, right_hand_keypoints, right_wrist, return_hand_keypoints=False):
        return self.retarget_left_hand(left_hand_keypoints, left_wrist, return_hand_keypoints), self.retarget_right_hand(right_hand_keypoints, right_wrist, return_hand_keypoints)

    # when including arm, return: arm dof pos, hand joint pos, hardware action for activated joints
    def retarget_right_hand(self, right_hand_keypoints, right_wrist = None, return_hand_keypoints=False, temp_debug=False):
        
        
        if temp_debug:
            pass
        else:
            if not self.use_right_hand:
                if return_hand_keypoints:
                    return None, None, None, None, None
                return None, None, None
        if right_hand_keypoints is None:
            if return_hand_keypoints:
                return None, None, None, None, None
            return None, None, None

        # print(f"right hand keypoints:\n{right_hand_keypoints}")
        joint_pos = np.array(right_hand_keypoints, dtype=np.float32)
        # print(f"right hand keypoints:\n{joint_pos}")

        wrist_pose = right_wrist    # TODO
        hardware_hand_qpos = np.zeros(6, dtype=np.float32)
        qpos = None
        if not self.use_cosine:
            indices = self.right_retargeting.optimizer.target_link_human_indices
            '''
            [[ 8 12 16 20 12 16 20 16 20 20  0  0  0  0  0]
            [ 4  4  4  4  8  8  8 12 12 16  4  8 12 16 20]]
            '''
            origin_indices = indices[0, :]  # [ 8 12 16 20 12 16 20 16 20 20  0  0  0  0  0]
            task_indices = indices[1, :]    # [ 4  4  4  4  8  8  8 12 12 16  4  8 12 16 20]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            qpos = self.right_retargeting.retarget(ref_value)


            hardware_hand_qpos = (qpos[self.right_retargeting.optimizer.idx_pin2target] - self.right_retargeting.joint_limits[:,0]) /\
                    (self.right_retargeting.joint_limits[:,1] - self.right_retargeting.joint_limits[:,0])
            hardware_hand_qpos = np.clip(1.-hardware_hand_qpos, 0., 1.)[6:12] # remove 6-dof dummy free joints in retargeting
            hardware_hand_qpos = np.flip(hardware_hand_qpos) # flip to match the hardware action order, modified 2025/06/24
            
            
            
            a = np.array([1.,         0.95878324, 0.96643896, 0.88189984])
            b = np.array([0.71554245, 0.87360939, 0.96947393, 1.        ]) 
            c = np.array([0.,         0.02875853, 0.15607714, 0.24344852])
            upper_bound = (a + b) / 2.0
            lower_bound = c
            for i in range(4):
                if hardware_hand_qpos[i] < lower_bound[i]:
                    hardware_hand_qpos[i] = 0.0
                elif hardware_hand_qpos[i] > upper_bound[i]:
                    hardware_hand_qpos[i] = 1.0
                else:
                    hardware_hand_qpos[i] = (hardware_hand_qpos[i] - lower_bound[i]) / (upper_bound[i] - lower_bound[i])
        
        # hardware_hand_qpos[-3] = hardware_hand_qpos[-3] * 4.0 - 3.0     # index finger
        else:

            # print(f"{joint_pos[0]=}    {joint_pos[4]=}")
            # 手掌上的关键点：0-手腕，2-拇指根，5-食指根，9-中指根，13-无名指根，17-小指根
            vector_0_5 = joint_pos[5] - joint_pos[0]
            vector_0_17 = joint_pos[17] - joint_pos[0]
            vector_5_17 = joint_pos[17] - joint_pos[5]
            vector_0_2 = joint_pos[2] - joint_pos[0]
            vector_0_4 = joint_pos[4] - joint_pos[0]
            
            # 计算vector_0_4和vector_0_2两个向量夹角余弦值
            cosine_thumb = np.dot(vector_0_4, vector_0_2) / (np.linalg.norm(vector_0_4) * np.linalg.norm(vector_0_2) + 1e-8)
            # print(f"cosine between vector_0_4 and vector_0_2: {cosine_thumb}")
            
            # 计算vector_0_4和vector_0_5两个向量夹角余弦值
            cosine_index = np.dot(vector_0_4, vector_0_5) / (np.linalg.norm(vector_0_4) * np.linalg.norm(vector_0_5) + 1e-8)
            # print(f"cosine between vector_0_4 and vector_0_5: {cosine_index}")
            cosine = (cosine_thumb + cosine_index) / 2.0        # (0.4, 0.9]
            
            vector_3_2 = joint_pos[3] - joint_pos[2]
            vector_3_4 = joint_pos[4] - joint_pos[3]
            cosine_3_2_3_4 = np.dot(vector_3_2, vector_3_4) / (np.linalg.norm(vector_3_2) * np.linalg.norm(vector_3_4) + 1e-8)
            # print(f"cosine between vector_3_2 and vector_3_4: {cosine_3_2_3_4}")        # (0.70, 0.95]
            
            # index: 6->5, 6->8
            vector_6_5 = joint_pos[5] - joint_pos[6]
            vector_6_8 = joint_pos[8] - joint_pos[6]
            cosine_index_6 = np.dot(vector_6_5, vector_6_8) / (np.linalg.norm(vector_6_5) * np.linalg.norm(vector_6_8) + 1e-8)
            # print(f"cosine between vector_6_5 and vector_6_8: {cosine_index_6}")        # (0.4, -1.0]
            
            # mid: 10->9, 10->12
            vector_10_9 = joint_pos[9] - joint_pos[10]
            vector_10_12 = joint_pos[12] - joint_pos[10]
            cosine_mid_10 = np.dot(vector_10_9, vector_10_12) / (np.linalg.norm(vector_10_9) * np.linalg.norm(vector_10_12) + 1e-8)
            # print(f"cosine between vector_10_9 and vector_10_12: {cosine_mid_10}")        # (0.45, -1.0]
            
            # ring: 14->13, 14->16
            vector_14_13 = joint_pos[13] - joint_pos[14]
            vector_14_16 = joint_pos[16] - joint_pos[14]
            cosine_ring_14 = np.dot(vector_14_13, vector_14_16) / (np.linalg.norm(vector_14_13) * np.linalg.norm(vector_14_16) + 1e-8)
            # print(f"cosine between vector_14_13 and vector_14_16: {cosine_ring_14}")        # (0.2, -1.0]
                                    
            
            # pinky: 18->17, 18->20
            vector_18_17 = joint_pos[17] - joint_pos[18]
            vector_18_20 = joint_pos[20] - joint_pos[18]
            cosine_pinky_18 = np.dot(vector_18_17, vector_18_20) / (np.linalg.norm(vector_18_17) * np.linalg.norm(vector_18_20) + 1e-8)
            # print(f"cosine between vector_18_17 and vector_18_20: {cosine_pinky_18}")        # (-0.2, -1.0]
            # print()
        

            hardware_hand_qpos[-1] = hardware_hand_qpos[-1] * 3.0 - 2.0     # thumb finger
            
            hardware_hand_qpos[-1] = (cosine - 0.8) / (0.99 - 0.80)
            hardware_hand_qpos[-2] = cosine_3_2_3_4 * 4.0 - 2.8
            
            hardware_hand_qpos[-3] = (cosine_index_6 - 0.4) / (-1.4)
            hardware_hand_qpos[-4] = (cosine_mid_10 - 0.45) / (-1.45)
            hardware_hand_qpos[-5] = (cosine_ring_14 - 0.2) / (-1.2)
            hardware_hand_qpos[-6] = (cosine_pinky_18 + 0.2) / (-0.8)

            hardware_hand_qpos = self.hand_cos_retargeter.retarget(right_hand_keypoints)
       

        hardware_hand_qpos = np.clip(hardware_hand_qpos, 0., 1.)
        
        # print("hardware_hand_qpos:", hardware_hand_qpos)
        if return_hand_keypoints:
            return wrist_pose, qpos, hardware_hand_qpos, joint_pos, wrist_pose
        else:
            return wrist_pose, qpos, hardware_hand_qpos

    
    # when including arm, return: arm dof pos, hand joint pos, hardware action for activated joints
    def retarget_left_hand(self, left_hand_keypoints, left_wrist, return_hand_keypoints=False):
        return self.retarget_right_hand(left_hand_keypoints, left_wrist, return_hand_keypoints, temp_debug=True)

        if not self.use_left_hand:
            if return_hand_keypoints: 
                return None, None, None, None, None
            return None, None, None
        
        joint_pos = np.array(left_hand_keypoints, dtype=np.float32)
        indices = self.left_retargeting.optimizer.target_link_human_indices
        # print(indices)
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        
        qpos = self.left_retargeting.retarget(ref_value)
        
        
        wrist_pose = left_wrist    # TODO
        
        hardware_hand_qpos = (qpos[self.left_retargeting.optimizer.idx_pin2target] - self.left_retargeting.joint_limits[:,0]) /\
                (self.left_retargeting.joint_limits[:,1] - self.left_retargeting.joint_limits[:,0])
        hardware_hand_qpos = np.clip(1.-hardware_hand_qpos, 0., 1.)[6:12] # remove 6-dof dummy free joints in retargeting
        hardware_hand_qpos = np.flip(hardware_hand_qpos) # flip to match the hardware action order, modified 2025/06/24


        if return_hand_keypoints:
            return wrist_pose, qpos, hardware_hand_qpos, joint_pos, wrist_pose
        else:
            return wrist_pose, qpos, hardware_hand_qpos


# Mimic hardware behavior: wrists poses + hands joint positions -> robot joint positions
class VDHardwareSimulator:
    def __init__(self, arm_urdf_path, eef_name, renderer=None,
                 arm_max_ang_vel=4, low_control_dt=0.01, policy_control_dt=0.01):
        from teleop.pin_robot import PinRobot
        self.arm_model = PinRobot(urdf_path=arm_urdf_path, eef_name=eef_name)
        self.last_left_arm_qpos = np.array([1.57, -0.47, 1.94, 0, 0.70, -1.57])
        self.last_right_arm_qpos = np.array([-1.57, -0.47, 1.94, 3.14, -0.70, -1.57])
        self.max_arm_step_size = low_control_dt * arm_max_ang_vel # 0.02 * 0.4 = 0.008 rad
        self.low_control_dt = low_control_dt
        self.policy_control_dt = policy_control_dt
        self.decimation = int(self.policy_control_dt / self.low_control_dt) # 5 for 0.1/0.02
        self.renderer = renderer

    def policy_step(self, left_wrist_pose=None, left_hand_qpos=None, right_wrist_pose=None, right_hand_qpos=None):
        for t in range(self.decimation):
            dt = time.time()
            start_time = time.time()
            if left_wrist_pose is not None and left_hand_qpos is not None:
                qpos_arm = self.arm_model.solve_ik_local(left_wrist_pose, self.last_left_arm_qpos, max_joint_diff=self.max_arm_step_size)
                #qpos_arm = self.arm_model.solve_ik_damped_pseudo(target_pose=left_wrist_pose, q_init=self.last_left_arm_qpos)
                self.last_left_arm_qpos = qpos_arm
            else:
                self.last_left_arm_qpos = None

            if right_wrist_pose is not None and right_hand_qpos is not None:
                qpos_arm = self.arm_model.solve_ik_local(right_wrist_pose, self.last_right_arm_qpos, max_joint_diff=self.max_arm_step_size)
                #qpos_arm = self.arm_model.solve_ik_damped_pseudo(target_pose=right_wrist_pose, q_init=self.last_right_arm_qpos)
                self.last_right_arm_qpos = qpos_arm
            else:
                self.last_right_arm_qpos = None
        
            rgb = self.renderer.render_data(self.last_left_arm_qpos, left_hand_qpos, self.last_right_arm_qpos, right_hand_qpos)
            #if television:
            #    retargetor.avp_streamer.update_images(rgb, rgb) # render in VisionPro
            
            sleep_time = self.low_control_dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # print(f"dt: {time.time()-dt}")
    
    def policy_step_once(self, left_wrist_pose=None, left_hand_qpos=None, right_wrist_pose=None, right_hand_qpos=None):
        # 此函数外层需要套一层for t in range(self.decimation)
        start_time = time.time()
        if left_wrist_pose is not None and left_hand_qpos is not None:
            qpos_arm = self.arm_model.solve_ik_local(left_wrist_pose, self.last_left_arm_qpos, max_joint_diff=self.max_arm_step_size)
            #qpos_arm = self.arm_model.solve_ik_damped_pseudo(target_pose=left_wrist_pose, q_init=self.last_left_arm_qpos)
            self.last_left_arm_qpos = qpos_arm
        else:
            self.last_left_arm_qpos = None

        if right_wrist_pose is not None and right_hand_qpos is not None:
            qpos_arm = self.arm_model.solve_ik_local(right_wrist_pose, self.last_right_arm_qpos, max_joint_diff=self.max_arm_step_size)
            #qpos_arm = self.arm_model.solve_ik_damped_pseudo(target_pose=right_wrist_pose, q_init=self.last_right_arm_qpos)
            self.last_right_arm_qpos = qpos_arm
        else:
            self.last_right_arm_qpos = None
    
        rgb = self.renderer.render_data(self.last_left_arm_qpos, left_hand_qpos, self.last_right_arm_qpos, right_hand_qpos)
        #if television:
        #    retargetor.avp_streamer.update_images(rgb, rgb) # render in VisionPro
        
        sleep_time = self.low_control_dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        return self.last_left_arm_qpos, left_hand_qpos, self.last_right_arm_qpos, right_hand_qpos


if __name__ == "__main__":
    debug = False # offline avp data or online stream?
    offline_path = "../../assets/debug_offline_avp_stream.pkl"
    television = False # use television
    mode = "both" # left, right, both
    avp_ip="192.168.20.130"

    sys.path.append('../rm65_inspire_tac')
    from render_sapien import SapienArmHandRenderer
    
    retargetor = VDArmHandRetarget(
        hand_name=RobotName.inspiretacofficial, 
        mode=mode, 
        left_right_base_distance=0.525,
        safe_workspace=[[-0.5, -0.5, 0.17], [0.5, 0.5, 0.55]],
        retargeting_type=RetargetingType.dexpilot,
    )
    renderer = SapienArmHandRenderer(
        left_urdf_path="../../assets/rm65_inspire_tac/rm65_inspire_tac_left.urdf" if retargetor.use_left_hand else None,
        left_retargeting_joint_names=retargetor.left_retargeting_hand_dof_names,
        right_urdf_path="../../assets/rm65_inspire_tac/rm65_inspire_tac_right.urdf" if retargetor.use_right_hand else None,
        right_retargeting_joint_names=retargetor.right_retargeting_hand_dof_names,
        arm_joint_names=['arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6'],
        left_right_base_distance=0.525,
    )
    hardware_simulator = VDHardwareSimulator(
        arm_urdf_path="../../assets/rm65_inspire_tac/rm65_arm.urdf",
        eef_name="hand_base",
        renderer=renderer,
    )
    
    
    listener_class = RetargetKeyboardListener(retargetor=retargetor)
    listener = keyboard.Listener(on_press=listener_class.rotate_key_press)
    listener.start()
        
        
    stream = DataStream()
    stream.set_ip("192.168.20.126")  # 广播 IP
    stream.set_broascast_port(9998)  # 广播端口
    stream.set_local_port(9999)

    # [新增] 实例化手部数据处理器
    hand_data_processor = MocapHandDataProcessor()
    if not stream.connect():
        exit()

    while True:
        stream.request_mocap_data()
        mocap_data = stream.get_mocap_data()
        if mocap_data.isUpdate:
            hand_data_dict = hand_data_processor.process(mocap_data)
            if retargetor.use_left_hand:
                left_hand_keypoints = hand_data_processor.convert_left_hand_data_to_keypoints(hand_data_dict["left_hand"])
                left_wrist = np.zeros((4,4))  # TODO
            if retargetor.use_right_hand:
                right_hand_keypoints = hand_data_processor.convert_right_hand_data_to_keypoints(hand_data_dict["right_hand"])
                right_wrist = np.random.randn(4,4)  # TODO
                right_wrist[:3, 3] = np.array([0.3, 0.2, 0.2])
                print("right_wrist:", right_wrist)
                # right_wrist = np.array([[ 0.58531439,  0.34962864,  1.6444376,   0.3       ],
                #                         [ 0.24339061, -1.48387929, -0.57960934,  0.2       ],
                #                         [ 0.81329137, -0.41506045, -0.4002755,   0.2       ],
                #                         [ 0.20899774,  0.48442488,  0.62331055,  0.2311387 ]])
                right_wrist = np.array([[-0.39183619, -0.37813137,  0.06127935,  0.3       ],
                                        [-0.64547154,  1.33318718,  1.79040878,  0.2       ],
                                        [ 2.12662986,  1.51774326, -1.05711954,  0.2       ],
                                        [ 1.75666124,  1.51589867,  1.86485332,  0.48929209]])
        else:
            continue
        
        left, right = retargetor.retarget(left_hand_keypoints, left_wrist, right_hand_keypoints, right_wrist, return_hand_keypoints= True)
        left_ee_pose, left_qpos, left_hardware_hand_qpos, joint_pos, wrist_pose = left
        right_ee_pose, right_qpos, right_hardware_hand_qpos, joint_pos, wrist_pose = right

        # print(right_hardware_hand_qpos)
        # print(f"t1: {time.time()-t1}")
        t2 = time.time()
        hardware_simulator.policy_step(
            left_wrist_pose=left_ee_pose, 
            left_hand_qpos=left_qpos, 
            right_wrist_pose=right_ee_pose, 
            right_hand_qpos=right_qpos
        )
        # print(f"t2: {time.time()-t2} \n")
        
        
        