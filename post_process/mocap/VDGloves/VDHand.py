import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from PathConfig import PROJECT_ROOT, TACTASK_DIR, TELEOP_DIR, TACDATA_DIR
import socket
import threading
import copy
import configs
from time import sleep
import json # 用于美化字典输出
from enum import Enum
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
# 确保从 vdmocapsdk_dataread 中导入 MocapData 结构体
from vdmocapsdk_dataread import *
from data_types_MOCAP import *
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp



class HandType(Enum):
    """Enumeration for left and right hands."""
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    RIGHT1 = "Right"
    LEFT1 = "Left"

    def __eq__(self, other):
        # 如果比较的是同一个枚举类型，使用默认比较
        if isinstance(other, HandType):
            return super().__eq__(other)
        
        # 如果比较的是第一个 HandType 枚举的实例
        try:
            # 检查是否是第一个 HandType 类型
            if hasattr(other, 'name') and hasattr(other, 'value'):
                # 将第一个枚举的名称与第二个枚举的值进行比较（忽略大小写）
                return self.value.lower() == other.name.lower()
        except:
            pass
        
        # 其他情况返回 False
        return False
    
    def __hash__(self):
        # 确保哈希值与默认实现一致
        return hash(self.value)

    def upper(self):
        return self.name.upper()
    
    def lower(self):
        return self.name.lower()


VD_RIGHT_HAND_TO_MANO = np.array([
    [-1,0,0],
    [0,0,1],
    [0,1,0],
])

VD_LEFT_HAND_TO_MANO = np.array([
    [-1,0,0],
    [0,0,1],
    [0,1,0],
])

def VD_to_mano_keypoints(hand_keypoints, hand_wrist_rot, hand_type):
    # 断言输入形状：支持单样本(21,3)或批量样本(N,21,3)
    # print(hand_keypoints.shape)
    assert hand_keypoints.shape[-2:] == (21, 3), \
        f"hand_keypoints 最后两维需为(21,3)，实际为{hand_keypoints.shape[-2:]}"
    assert hand_wrist_rot.shape[-2:] == (3, 3), \
        f"hand_wrist_rot 最后两维需为(3,3)，实际为{hand_wrist_rot.shape[-2:]}"
    
    # 将手腕点移到原点（支持批量维度）
    # hand_keypoints = hand_keypoints - hand_keypoints[..., 0, :, np.newaxis]
    hand_keypoints = hand_keypoints - hand_keypoints[..., 0, np.newaxis, :]
    # print(hand_keypoints.shape) # (13253, 21, 3)

    # 手指关节索引（通用索引方式，支持批量）
    finger_starts = np.array([5, 9, 13, 17])
    s_j1_idx = finger_starts + 1  # [6, 10, 14, 18]
    s_j2_idx = finger_starts + 2  # [7, 11, 15, 19]
    s_j3_idx = finger_starts + 3  # [8, 12, 16, 20]

    # 提取输入关键点（通过...匹配所有前置维度，支持批量）
    A_in = np.expand_dims(hand_keypoints[..., 0, :], axis=-2)  # (..., 1, 3)
    B_in = hand_keypoints[..., s_j1_idx, :]  # (..., 4, 3)
    C_in = hand_keypoints[..., s_j2_idx, :]  # (..., 4, 3)
    D_in = hand_keypoints[..., s_j3_idx, :]  # (..., 4, 3)

    A_in = np.repeat(A_in, repeats=4, axis=-2)  # (..., 4, 3)
    
    # 计算新指尖点
    new_tips = add_finger_tip_point(A_in.reshape((-1,3)), B_in.reshape((-1,3)), C_in.reshape((-1,3)), D_in.reshape((-1,3)))

    # 更新关键点（批量赋值，不干扰原始数据）
    hand_keypoints[..., finger_starts, :] = B_in
    hand_keypoints[..., s_j1_idx, :] = C_in
    hand_keypoints[..., s_j2_idx, :] = D_in
    hand_keypoints[..., s_j3_idx, :] = new_tips.reshape(-1,4,3)
    
    # 旋转调整（支持批量维度的矩阵乘法）
    # 转置为 (..., 3, 21) 以便与 (..., 3, 3) 旋转矩阵相乘
    kp_transposed = hand_keypoints.transpose(*range(hand_keypoints.ndim - 2), -1, -2)
    # 应用旋转逆变换
    rotated_kp = np.linalg.inv(hand_wrist_rot) @ kp_transposed
    # 转置回 (..., 21, 3)
    hand_keypoints = rotated_kp.transpose(*range(hand_keypoints.ndim - 2), -1, -2)

    hand_keypoints = hand_keypoints[...,np.newaxis]

    # 转换到MANO坐标系（保持批量兼容性）
    if hand_type == HandType.RIGHT:
        hand_keypoints =  (hand_keypoints.transpose(*range(hand_keypoints.ndim - 2), -1, -2) @ VD_RIGHT_HAND_TO_MANO).transpose(
            *range(hand_keypoints.ndim - 2), -1, -2
        )
        if hand_keypoints.shape[-1] ==1:
            hand_keypoints = hand_keypoints[...,0]
        return hand_keypoints
    elif hand_type == HandType.LEFT:
        hand_keypoints =  (hand_keypoints.transpose(*range(hand_keypoints.ndim - 2), -1, -2) @ VD_LEFT_HAND_TO_MANO).transpose(
            *range(hand_keypoints.ndim - 2), -1, -2
        )
        if hand_keypoints.shape[-1] ==1:
            hand_keypoints = hand_keypoints[...,0]
        return hand_keypoints
    assert False

def swap_lr_keypoints_mano(keypoints):
    """
    将MANO的关键点表示进行左右手互换，支持ndarray和torch.Tensor
    
    参数:
        keypoints: ndarray 或 torch.Tensor, 形状为 (BS, 21, 3) 的关键点坐标
        
    返回:
        swapped_keypoints: 镜像后的关键点坐标, 形状 (BS, 21, 3)
    """
    is_numpy = isinstance(keypoints, np.ndarray)
    if is_numpy:
        device = 'cpu'
        keypoints = torch.from_numpy(keypoints).float()
    else:
        device = keypoints.device
        keypoints = keypoints.clone().detach()
    
    swapped_keypoints = keypoints.clone()
    swapped_keypoints[..., 0] = -keypoints[..., 0]  # 仅X坐标取反

    return swapped_keypoints.cpu().numpy() if is_numpy else swapped_keypoints.to(device)

def _normalize_vectors(v):
    """
    (..., 3) 向量的批量归一化。
    """
    # 沿最后一个轴 (-1) 计算范数
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    
    # 创建一个掩码来标记零向量
    zero_mask = (norm < 1e-15)
    
    # 避免除以零
    norm[zero_mask] = 1.0
    
    # .squeeze(-1) 移除最后一个维度 (N, 1) -> (N,)
    return v / norm, zero_mask.squeeze(-1)

def _get_rotations_between_vecs(v_from, v_to):
    """
    计算从 v_from 向量到 v_to 向量的批量旋转 (..., 3) -> (...)。
    """
    
    v_from_u, mask_from_zero = _normalize_vectors(v_from)
    v_to_u, mask_to_zero = _normalize_vectors(v_to)

    # '...j,...j->...' 允许任意批量维度
    dots = np.einsum('...j,...j->...', v_from_u, v_to_u)
    dots_clipped = np.clip(dots, -1.0, 1.0)
    
    # angles 形状为 (...)
    angles = np.arccos(dots_clipped)
    
    # axes 形状为 (..., 3)
    axes = np.cross(v_from_u, v_to_u)
    axes_norm, mask_axes_zero = _normalize_vectors(axes)
    
    # rotvecs 形状为 (..., 3)
    rotvecs = np.zeros_like(v_from)

    # --- 处理三种情况 ---
    mask_non_collinear = ~mask_axes_zero
    if np.any(mask_non_collinear):
        # 使用 np.newaxis 将 (...) 扩展为 (..., 1) 以便与 (..., 3) 广播
        rotvecs[mask_non_collinear] = axes_norm[mask_non_collinear] * angles[mask_non_collinear, np.newaxis]

    mask_antiparallel = (dots < -1.0 + 1e-8) & mask_axes_zero
    if np.any(mask_antiparallel):
        # 查找正交轴
        v_from_antiparallel = v_from_u[mask_antiparallel]
        
        ortho_axis_1 = np.cross(v_from_antiparallel, np.array([1.0, 0.0, 0.0]))
        norm_1, mask_1_zero = _normalize_vectors(ortho_axis_1)
        
        if np.any(mask_1_zero):
            ortho_axis_2 = np.cross(v_from_antiparallel[mask_1_zero], np.array([0.0, 1.0, 0.0]))
            norm_2, _ = _normalize_vectors(ortho_axis_2)
            norm_1[mask_1_zero] = norm_2
            
        rotvecs[mask_antiparallel] = norm_1 * np.pi

    return R.from_rotvec(rotvecs)


def add_finger_tip_point(A, B, C, D):
    """
    根据给定的几何约束批量求解点 E。
    此版本经过重构，可处理任意Numpy广播维度 (..., 3)。
    """
    
    # --- 1. 计算向量 ---
    # 假设 A, B, C, D 已经是可广播的 (..., 3)
    v_AB = B - A
    v_BC = C - B
    v_CD = D - C

    # --- 2. 计算 $\vec{DE}$ 的目标长度 ---
    # 沿最后一个轴 (-1) 计算范数，保持维度 (..., 1)
    len_BC = np.linalg.norm(v_BC, axis=-1, keepdims=True)
    len_CD = np.linalg.norm(v_CD, axis=-1, keepdims=True)

    # len_DE 形状为 (..., 1)
    len_DE = (len_BC + len_CD) / 2.0 * 0.8

    # --- 3. 计算旋转 "偏移" ---
    R1 = _get_rotations_between_vecs(v_BC, v_CD)
    R2 = _get_rotations_between_vecs(v_AB, v_BC)

    # --- 4. 计算平均旋转 (delta_mean) ---
    # 使用兼容旧版 scipy 的 Slerp(R2, R1, 0.5) 手动计算
    R_delta = R2.inv() * R1
    R_delta_half = R_delta ** 0.5
    R_mean = R2 * R_delta_half

    # --- 5. 计算 $\vec{DE}$ 的方向和向量 ---
    v_CD_norm, _ = _normalize_vectors(v_CD)
    
    # R_mean (...) 会被逐元素应用到 v_CD_norm (..., 3)
    v_DE_dir = R_mean.apply(v_CD_norm)

    # (..., 3) * (..., 1) -> (..., 3)
    v_DE = v_DE_dir * len_DE

    # --- 6. 计算 E ---
    E = D + v_DE
    
    # 返回 (..., 3) 结果
    return E



def quaternion_average(q1, q2):
    """
    计算两个四元数的平均值
    
    参数:
    q1, q2: 形状为(4,)的numpy数组，表示四元数 [w, x, y, z] 或 [x, y, z, w]
    假设使用 [w, x, y, z] 格式
    
    返回:
    q_avg: 平均四元数，形状为(4,)的numpy数组
    """
    # 确保输入是numpy数组
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    # 步骤1: 处理双重覆盖问题（确保四元数在同一半球）
    # 计算点积来判断它们是否在同一半球
    dot_product = np.dot(q1, q2)
    
    # 如果点积为负，说明它们方向相反，需要将一个取反
    if dot_product < 0:
        q2 = -q2
    
    # 步骤2: 计算平均值并重新归一化
    q_sum = q1 + q2
    q_avg = q_sum / np.linalg.norm(q_sum)
    
    return q_avg

# 更通用的版本，可以处理多个四元数的情况
def quaternion_average_multiple(quaternions, weights=None):
    """
    计算多个四元数的加权平均值
    
    参数:
    quaternions: 四元数列表，每个四元数是形状为(4,)的numpy数组
    weights: 权重列表，如果不提供则使用等权重
    
    返回:
    q_avg: 平均四元数
    """
    quaternions = [np.array(q) for q in quaternions]
    
    if len(quaternions) == 0:
        raise ValueError("四元数列表不能为空")
    
    if len(quaternions) == 1:
        return quaternions[0] / np.linalg.norm(quaternions[0])
    
    # 设置默认权重
    if weights is None:
        weights = [1.0] * len(quaternions)
    
    # 步骤1: 处理双重覆盖问题
    # 以第一个四元数为参考，确保所有四元数在同一半球
    q_ref = quaternions[0]
    aligned_quaternions = [q_ref]
    
    for i in range(1, len(quaternions)):
        q = quaternions[i]
        if np.dot(q_ref, q) < 0:
            aligned_quaternions.append(-q)
        else:
            aligned_quaternions.append(q)
    
    # 步骤2: 计算加权平均并重新归一化
    q_sum = np.zeros(4)
    for q, w in zip(aligned_quaternions, weights):
        q_sum += w * q
    
    q_avg = q_sum / np.linalg.norm(q_sum)
    
    return q_avg


class SensorStatePY(Enum):
    SS_NONE = 0
    SS_Well = 1
    SS_NoData = 2
    SS_UnReady = 3
    SS_BadMag = 4

class HandNodes(Enum):
    HN_Hand = 0
    HN_ThumbFinger = 1
    HN_ThumbFinger1 = 2
    HN_ThumbFinger2 = 3
    HN_IndexFinger = 4
    HN_IndexFinger1 = 5
    HN_IndexFinger2 = 6
    HN_IndexFinger3 = 7
    HN_MiddleFinger = 8
    HN_MiddleFinger1 = 9
    HN_MiddleFinger2 = 10
    HN_MiddleFinger3 = 11
    HN_RingFinger = 12
    HN_RingFinger1 = 13
    HN_RingFinger2 = 14
    HN_RingFinger3 = 15
    HN_PinkyFinger = 16
    HN_PinkyFinger1 = 17
    HN_PinkyFinger2 = 18
    HN_PinkyFinger3 = 19

class Gesture(Enum):
    GESTURE_NONE = 0
    GESTURE_1 = 1
    GESTURE_2 = 2
    GESTURE_3 = 3
    GESTURE_4 = 4
    GESTURE_5 = 5
    GESTURE_6 = 6
    GESTURE_7 = 7
    GESTURE_8 = 8
    GESTURE_9 = 9
    GESTURE_10 = 10
    GESTURE_11 = 11
    GESTURE_12 = 12
    GESTURE_13 = 13
    GESTURE_14 = 14
    GESTURE_15 = 15
    GESTURE_16 = 16
    GESTURE_17 = 17
    GESTURE_18 = 18
    GESTURE_19 = 19
    GESTURE_20 = 20
    GESTURE_21 = 21
    GESTURE_22 = 22

class MocapHandDataProcessor:
    """
    一个用于从 MocapData 结构体中提取和组织手部数据的专用处理类。
    """
    def __init__(self):
        # 将 Enum 转换为 index -> name 的字典，方便快速查找
        self._sensor_state_map = {e.value: e.name for e in SensorStatePY}
        self._hand_node_map = {e.value: e.name for e in HandNodes}
        self._gesture_map = {e.value: e.name for e in Gesture}

    def _extract_single_hand_data(self, mocap_data: MocapData, hand_side: str) -> dict:
        """
        从 MocapData 中提取单只手（左或右）的所有相关数据。
        
        Args:
            mocap_data: 包含所有动捕数据的 MocapData 对象。
            hand_side: 'left' 或 'right'，指定要提取哪只手的数据。
        
        Returns:
            一个包含该手所有信息的字典。
        """
        if hand_side == 'left':
            sensor_state_data = mocap_data.sensorState_lHand
            position_data = mocap_data.position_lHand
            quaternion_data = mocap_data.quaternion_lHand
            gyr_data = mocap_data.gyr_lHand
            acc_data = mocap_data.acc_lHand
            velocity_data = mocap_data.velocity_lHand
            gesture_data = mocap_data.gestureResultL
        elif hand_side == 'right':
            sensor_state_data = mocap_data.sensorState_rHand
            position_data = mocap_data.position_rHand
            quaternion_data = mocap_data.quaternion_rHand
            gyr_data = mocap_data.gyr_rHand
            acc_data = mocap_data.acc_rHand
            velocity_data = mocap_data.velocity_rHand
            gesture_data = mocap_data.gestureResultR
        else:
            raise ValueError("hand_side must be 'left' or 'right'")

        nodes_list = []
        for i in range(len(self._hand_node_map)):
            node_info = {
                "name": self._hand_node_map.get(i, f"UNKNOWN_NODE_{i}"),
                "sensor_state": self._sensor_state_map.get(sensor_state_data[i], "UNKNOWN_STATE"),
                # 将 ctypes 数组转换为普通的 Python 列表
                "position": list(position_data[i]),
                "quaternion": list(quaternion_data[i]),
                "gyroscope": list(gyr_data[i]),
                "accelerometer": list(acc_data[i]),
                "velocity": list(velocity_data[i]),
            }
            nodes_list.append(node_info)

        return {
            "gesture": self._gesture_map.get(gesture_data, "UNKNOWN_GESTURE"),
            "nodes": nodes_list,
        }

    def process(self, mocap_data: MocapData) -> dict:
        """
        处理 MocapData 对象，提取双手的所有信息。
        
        Args:
            mocap_data: 包含所有动捕数据的 MocapData 对象。
        
        Returns:
            一个包含左右手详细信息的字典。
        """
        if not isinstance(mocap_data, MocapData):
            raise TypeError("Input must be a MocapData object")

        all_hand_data = {
            "left_hand": self._extract_single_hand_data(mocap_data, 'left'),
            "right_hand": self._extract_single_hand_data(mocap_data, 'right'),
        }
        return all_hand_data

    def convert_hand_data_to_quaternion(self, hand_data_dict):
        hand_quant = [
            np.array(hand_data_dict["nodes"][HandNodes.HN_Hand.value]["quaternion"]),

            quaternion_average(np.array(hand_data_dict["nodes"][HandNodes.HN_Hand.value]["quaternion"]) , np.array(hand_data_dict["nodes"][HandNodes.HN_ThumbFinger.value]["quaternion"])),
            np.array(hand_data_dict["nodes"][HandNodes.HN_ThumbFinger.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_ThumbFinger1.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_ThumbFinger2.value]["quaternion"]),

            np.array(hand_data_dict["nodes"][HandNodes.HN_IndexFinger.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_IndexFinger1.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_IndexFinger2.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_IndexFinger3.value]["quaternion"]),

            np.array(hand_data_dict["nodes"][HandNodes.HN_MiddleFinger.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_MiddleFinger1.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_MiddleFinger2.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_MiddleFinger3.value]["quaternion"]),

            np.array(hand_data_dict["nodes"][HandNodes.HN_RingFinger.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_RingFinger1.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_RingFinger2.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_RingFinger3.value]["quaternion"]),

            np.array(hand_data_dict["nodes"][HandNodes.HN_PinkyFinger.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_PinkyFinger1.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_PinkyFinger2.value]["quaternion"]),
            np.array(hand_data_dict["nodes"][HandNodes.HN_PinkyFinger3.value]["quaternion"]),
        ]
        hand_quant = np.array(hand_quant)
        return hand_quant

    def convert_left_hand_data_to_keypoints(self, left_hand_data_dict):
        left_hand_keypoints = [
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_Hand.value]["position"]),

            (np.array(left_hand_data_dict["nodes"][HandNodes.HN_Hand.value]["position"]) + np.array(left_hand_data_dict["nodes"][HandNodes.HN_ThumbFinger.value]["position"])) / 2.0,
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_ThumbFinger.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_ThumbFinger1.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_ThumbFinger2.value]["position"]),

            np.array(left_hand_data_dict["nodes"][HandNodes.HN_IndexFinger.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_IndexFinger1.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_IndexFinger2.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_IndexFinger3.value]["position"]),

            np.array(left_hand_data_dict["nodes"][HandNodes.HN_MiddleFinger.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_MiddleFinger1.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_MiddleFinger2.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_MiddleFinger3.value]["position"]),

            np.array(left_hand_data_dict["nodes"][HandNodes.HN_RingFinger.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_RingFinger1.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_RingFinger2.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_RingFinger3.value]["position"]),

            np.array(left_hand_data_dict["nodes"][HandNodes.HN_PinkyFinger.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_PinkyFinger1.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_PinkyFinger2.value]["position"]),
            np.array(left_hand_data_dict["nodes"][HandNodes.HN_PinkyFinger3.value]["position"]),
        ]
        left_hand_keypoints = np.array(left_hand_keypoints)
        return left_hand_keypoints

    def convert_right_hand_data_to_keypoints(self, right_hand_data_dict):
        right_hand_keypoints = [
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_Hand.value]["position"]),

            (np.array(right_hand_data_dict["nodes"][HandNodes.HN_Hand.value]["position"]) + np.array(right_hand_data_dict["nodes"][HandNodes.HN_ThumbFinger.value]["position"])) / 2.0,
            # np.array(right_hand_data_dict["nodes"][HandNodes.HN_ThumbFinger.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_ThumbFinger.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_ThumbFinger1.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_ThumbFinger2.value]["position"]),

            np.array(right_hand_data_dict["nodes"][HandNodes.HN_IndexFinger.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_IndexFinger1.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_IndexFinger2.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_IndexFinger3.value]["position"]),

            np.array(right_hand_data_dict["nodes"][HandNodes.HN_MiddleFinger.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_MiddleFinger1.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_MiddleFinger2.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_MiddleFinger3.value]["position"]),

            np.array(right_hand_data_dict["nodes"][HandNodes.HN_RingFinger.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_RingFinger1.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_RingFinger2.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_RingFinger3.value]["position"]),

            np.array(right_hand_data_dict["nodes"][HandNodes.HN_PinkyFinger.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_PinkyFinger1.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_PinkyFinger2.value]["position"]),
            np.array(right_hand_data_dict["nodes"][HandNodes.HN_PinkyFinger3.value]["position"]),
        ]
        right_hand_keypoints = np.array(right_hand_keypoints)
        return right_hand_keypoints
    
    def convert_bimanual_hand_data_to_keypoints(self, hand_data_dict):
        left_hand_keypoints = self.convert_left_hand_data_to_keypoints(hand_data_dict["left_hand"])
        right_hand_keypoints = self.convert_right_hand_data_to_keypoints(hand_data_dict["right_hand"])
        return left_hand_keypoints, right_hand_keypoints
    
    def extract_right_wrist_rot_from_right_hand_data(self, right_hand_data_dict):
        right_wrist_quat = right_hand_data_dict["nodes"][HandNodes.HN_Hand.value]["quaternion"]
        right_wrist_rot = R.from_quat([right_wrist_quat[1], right_wrist_quat[2], right_wrist_quat[3], right_wrist_quat[0]]).as_matrix()
        return right_wrist_rot      # (3, 3) ndarray
    
    def extract_left_wrist_rot_from_left_hand_data(self, left_hand_data_dict):
        left_wrist_quat = left_hand_data_dict["nodes"][HandNodes.HN_Hand.value]["quaternion"]
        left_wrist_rot = R.from_quat([left_wrist_quat[1], left_wrist_quat[2], left_wrist_quat[3], left_wrist_quat[0]]).as_matrix()
        return left_wrist_rot      # (3, 3) ndarray

    def extract_bimanual_VD_data(self, hand_data_dict, hand_mode):
        '''
        return left_keypoints, left_quaternion, left_wrist_rot, right_keypoints, right_quaternion, right_wrist_rot
        '''
        assert hand_mode in ["left", "right", "both"]
        left_keypoints = None
        left_quaternion = None
        left_wrist_rot = None
        right_keypoints = None
        right_quaternion = None
        right_wrist_rot = None
        if hand_mode in ["left", "both"]:
            left_keypoints = self.convert_left_hand_data_to_keypoints(hand_data_dict["left_hand"])
            left_quaternion = self.convert_hand_data_to_quaternion(hand_data_dict["left_hand"])
            left_wrist_rot = self.extract_left_wrist_rot_from_left_hand_data(hand_data_dict["left_hand"])
        if hand_mode in ["right", "both"]:
            right_keypoints = self.convert_right_hand_data_to_keypoints(hand_data_dict["right_hand"])
            right_quaternion = self.convert_hand_data_to_quaternion(hand_data_dict["right_hand"])
            right_wrist_rot = self.extract_right_wrist_rot_from_right_hand_data(hand_data_dict["right_hand"])
        
        return left_keypoints, left_quaternion, left_wrist_rot, right_keypoints, right_quaternion, right_wrist_rot

if __name__ == '__main__':
    class MockConfigs:
        JOINT_NUM = 12
        HAND = "Inspired"
    configs = MockConfigs()

    print("Starting MOCAP data stream demo...")
    stream = DataStream()
    stream.set_ip("192.168.20.240")  # 广播 IP
    stream.set_broascast_port(9998)  # 广播端口
    stream.set_local_port(9999)

    # [新增] 实例化手部数据处理器
    hand_data_processor = MocapHandDataProcessor()

    if not stream.connect():
        exit()

    print("\nConnection established. Starting to receive data...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            stream.request_mocap_data()
            mocap_data = stream.get_mocap_data()
            time1 = time.time()
            if mocap_data.isUpdate:
                print(mocap_data.sensorState_lHand)
                # [新增] 使用处理器来获取结构化的手部数据
                hand_data_dict = hand_data_processor.process(mocap_data)

                # --- 打印示例 ---
                # 1. 简单打印手势
                # left_gesture = hand_data_dict["left_hand"]["gesture"]
                # right_gesture = hand_data_dict["right_hand"]["gesture"]
                print(f"Frame: {mocap_data.frameIndex} ")

                # 2. 详细打印左手食指指尖的位置
                # HN_IndexFinger3 的索引是 7
                left_index_tip_pos = hand_data_dict["left_hand"]["nodes"][HandNodes.HN_IndexFinger3.value]["position"]
                print(f"  └─ Left Index Tip Position: [x={left_index_tip_pos[0]:.3f}, y={left_index_tip_pos[1]:.3f}, z={left_index_tip_pos[2]:.3f}]")
                
                left_index_tip_pos = hand_data_dict["right_hand"]["nodes"][0]["sensor_state"]
                print(left_index_tip_pos)

                a = hand_data_processor.extract_bimanual_VD_data(hand_data_dict, "both")
                print(a[0].shape, a[1].shape, a[2].shape, a[3].shape, a[4].shape, a[5].shape)
                # breakpoint()
                # 3. 如果需要，可以打印完整的字典（数据量较大）
                # print(json.dumps(hand_data_dict, indent=2))
                time2 = time.time()
                print(f"数据处理耗时: {(time2 - time1)*1000:.3f} ms")
            sleep(1e-5)

    except KeyboardInterrupt:
        print("\nStopping data stream...")
    finally:
        stream.disconnect()
        print("Disconnected successfully. Program terminated.")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# print(hand_data_dict)
# {
#     'left_hand': {
#         'gesture': 'GESTURE_NONE', 
#         'nodes': [
#             {
#                 'name': 'HN_Hand', 
#                 'sensor_state': 'SS_NONE', 
#                 'position': [-0.7480000257492065, 0.0, 1.5169999599456787], 
#                 'quaternion': [1.0, 0.0, 0.0, 0.0], 
#                 'gyroscope': [0.0, 0.0, 0.0], 
#                 'accelerometer': [0.0, 0.0, 0.0], 
#                 'velocity': [0.0, 0.0, 0.0]
#             }, 
#             {
#                 'name': 'HN_ThumbFinger', 
#                 'sensor_state': 'SS_NONE', 
#                 'position': [-0.7820000052452087, 0.041999999433755875, 1.5190000534057617], 
#                 'quaternion': [1.0, 0.0, 0.0, 0.0], 
#                 'gyroscope': [0.0, 0.0, 0.0], 
#                 'accelerometer': [0.0, 0.0, 0.0], 
#                 'velocity': [0.0, 0.0, 0.0]
#             }, 
#             {
#                 'name': 'HN_ThumbFinger1', 
#                 'sensor_state': 'SS_NONE', 
#                 'position': [-0.8169999718666077, 0.07699999958276749, 1.5190000534057617], 
#                 'quaternion': [1.0, 0.0, 0.0, 0.0], 
#                 'gyroscope': [0.0, 0.0, 0.0], 
#                 'accelerometer': [0.0, 0.0, 0.0], 
#                 'velocity': [0.0, 0.0, 0.0]
#             }, 
#             {
#                 'name': 'HN_ThumbFinger2', 
#                 'sensor_state': 'SS_NONE', 
#                 'position': [-0.8420000076293945, 0.10199999809265137, 1.5190000534057617], 
#                 'quaternion': [1.0, 0.0, 0.0, 0.0], 
#                 'gyroscope': [0.0, 0.0, 0.0], 
#                 'accelerometer': [0.0, 0.0, 0.0], 
#                 'velocity': [0.0, 0.0, 0.0]
#             }, 
#             {
#                 'name': 'HN_IndexFinger', 
#                 'sensor_state': 'SS_NONE', 
#                 'position': [-0.7919999957084656, 0.026000000536441803, 1.5230000019073486], 
#                 'quaternion': [1.0, 0.0, 0.0, 0.0], 
#                 'gyroscope': [0.0, 0.0, 0.0], 
#                 'accelerometer': [0.0, 0.0, 0.0], 
#                 'velocity': [0.0, 0.0, 0.0]
#             }, 
#             {'name': 'HN_IndexFinger1', 'sensor_state': 'SS_NONE', 'position': [-0.8619999885559082, 0.03999999910593033, 1.5219999551773071], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_IndexFinger2', 'sensor_state': 'SS_NONE', 'position': [-0.9120000004768372, 0.03999999910593033, 1.5199999809265137], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_IndexFinger3', 'sensor_state': 'SS_NONE', 'position': [-0.9390000104904175, 0.03999999910593033, 1.5180000066757202], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_MiddleFinger', 'sensor_state': 'SS_NONE', 'position': [-0.7940000295639038, 0.009999999776482582, 1.5240000486373901], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_MiddleFinger1', 'sensor_state': 'SS_NONE', 'position': [-0.8640000224113464, 0.014000000432133675, 1.5219999551773071], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_MiddleFinger2', 'sensor_state': 'SS_NONE', 'position': [-0.9169999957084656, 0.014000000432133675, 1.5190000534057617], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_MiddleFinger3', 'sensor_state': 'SS_NONE', 'position': [-0.9509999752044678, 0.014000000432133675, 1.5160000324249268], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_RingFinger', 'sensor_state': 'SS_NONE', 'position': [-0.7929999828338623, -0.0010000000474974513, 1.5240000486373901], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_RingFinger1', 'sensor_state': 'SS_NONE', 'position': [-0.8560000061988831, -0.00800000037997961, 1.5240000486373901], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_RingFinger2', 'sensor_state': 'SS_NONE', 'position': [-0.902999997138977, -0.00800000037997961, 1.5199999809265137], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_RingFinger3', 'sensor_state': 'SS_NONE', 'position': [-0.9350000023841858, -0.00800000037997961, 1.5180000066757202], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_PinkyFinger', 'sensor_state': 'SS_NONE', 'position': [-0.7910000085830688, -0.01600000075995922, 1.5230000019073486], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_PinkyFinger1', 'sensor_state': 'SS_NONE', 'position': [-0.847000002861023, -0.03099999949336052, 1.5230000019073486], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_PinkyFinger2', 'sensor_state': 'SS_NONE', 'position': [-0.8840000033378601, -0.029999999329447746, 1.5199999809265137], 'quaternion': [1.0, 0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, 

#             {
#                 'name': 'HN_PinkyFinger3', 
#                 'sensor_state': 'SS_NONE', 
#                 'position': [-0.9079999923706055, -0.03099999949336052, 1.5190000534057617], 
#                 'quaternion': [1.0, 0.0, 0.0, 0.0], 
#                 'gyroscope': [0.0, 0.0, 0.0], 
#                 'accelerometer': [0.0, 0.0, 0.0], 
#                 'velocity': [0.0, 0.0, 0.0]
#             }
#         ]
#     },
    
    
    
#     'right_hand': {'gesture': 'GESTURE_5', 'nodes': [{'name': 'HN_Hand', 'sensor_state': 'SS_NONE', 'position': [0.7480000257492065, 0.0, 1.5169999599456787], 'quaternion': [-0.1476999968290329, -0.1386999934911728, -0.08030000329017639, 0.9758999943733215], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_ThumbFinger', 'sensor_state': 'SS_NONE', 'position': [0.7293449640274048, -0.049060508608818054, 1.5040243864059448], 'quaternion': [-0.36399999260902405, -0.13300000131130219, -0.062300000339746475, 0.919700026512146], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_ThumbFinger1', 'sensor_state': 'SS_NONE', 'position': [0.728874921798706, -0.09736504405736923, 1.4932526350021362], 'quaternion': [-0.3822000026702881, -0.30079999566078186, -0.1460999995470047, 0.8614000082015991], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_ThumbFinger2', 'sensor_state': 'SS_NONE', 'position': [0.7336173057556152, -0.12759070098400116, 1.4769827127456665], 'quaternion': [-0.4846999943256378, -0.2273000031709671, -0.12080000340938568, 0.8359000086784363], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_IndexFinger', 'sensor_state': 'SS_NONE', 'position': [0.7142103314399719, -0.0374176912009716, 1.5067262649536133], 'quaternion': [-0.1476999968290329, -0.1386999934911728, -0.08030000329017639, 0.9758999943733215], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_IndexFinger1', 'sensor_state': 'SS_NONE', 'position': [0.6545608043670654, -0.06904735416173935, 1.4835466146469116], 'quaternion': [-0.17000000178813934, -0.3084000051021576, -0.09189999848604202, 0.9312999844551086], 'gyroscope': [-0.002040331484749913, -0.01117741595953703, -0.0011029791785404086], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_IndexFinger2', 'sensor_state': 'SS_NONE', 'position': [0.6180594563484192, -0.08149319887161255, 1.451677680015564], 'quaternion': [-0.1664000004529953, -0.3312000036239624, -0.09440000355243683, 0.9239000082015991], 'gyroscope': [-0.005206049885600805, -0.15521180629730225, 0.008542338386178017], 'accelerometer': [0.05443747341632843, 0.025551792234182358, -0.05969419330358505], 'velocity': [-0.0002074211952276528, -5.922000127611682e-05, 0.0005513771902769804]}, {'name': 'HN_IndexFinger3', 'sensor_state': 'SS_NONE', 'position': [0.5996423363685608, -0.08753736317157745, 1.432780385017395], 'quaternion': [-0.16259999573230743, -0.35370001196861267, -0.09690000116825104, 0.9158999919891357], 'gyroscope': [-0.02232312224805355, -0.32351964712142944, 0.010067008435726166], 'accelerometer': [0.05675461143255234, 0.017790336161851883, -0.056382983922958374], 'velocity': [-0.002756302710622549, -0.0015229583950713277, 0.0035624844022095203]}, {'name': 'HN_MiddleFinger', 'sensor_state': 'SS_NONE', 'position': [0.7071589231491089, -0.023053720593452454, 1.508937954902649], 'quaternion': [-0.1476999968290329, -0.1386999934911728, -0.08030000329017639, 0.9758999943733215], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_MiddleFinger1', 'sensor_state': 'SS_NONE', 'position': [0.6446508169174194, -0.04505213350057602, 1.4859673976898193], 'quaternion': [-0.1420000046491623, -0.20260000228881836, -0.08990000188350677, 0.9646000266075134], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_MiddleFinger2', 'sensor_state': 'SS_NONE', 'position': [0.5992485880851746, -0.05694771558046341, 1.461194396018982], 'quaternion': [-0.10480000078678131, -0.5072000026702881, -0.131400004029274, 0.8452000021934509], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_MiddleFinger3', 'sensor_state': 'SS_NONE', 'position': [0.5859826803207397, -0.05745373293757439, 1.4297549724578857], 'quaternion': [-0.0560000017285347, -0.7559000253677368, -0.15850000083446503, 0.6326000094413757], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_RingFinger', 'sensor_state': 'SS_NONE', 'position': [0.7046605944633484, -0.01241080928593874, 1.5105057954788208], 'quaternion': [-0.1476999968290329, -0.1386999934911728, -0.08030000329017639, 0.9758999943733215], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_RingFinger1', 'sensor_state': 'SS_NONE', 'position': [0.6446667909622192, -0.022565679624676704, 1.4927666187286377], 'quaternion': [-0.09350000321865082, -0.29280000925064087, -0.11569999903440475, 0.9444000124931335], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_RingFinger2', 'sensor_state': 'SS_NONE', 'position': [0.609216034412384, -0.02631513774394989, 1.4617491960525513], 'quaternion': [-0.059300001710653305, -0.7110999822616577, -0.16179999709129333, 0.6815999746322632], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_RingFinger3', 'sensor_state': 'SS_NONE', 'position': [0.6126561164855957, -0.02062365598976612, 1.4303065538406372], 'quaternion': [0.00839999970048666, -0.9570000171661377, -0.13840000331401825, 0.25459998846054077], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_PinkyFinger', 'sensor_state': 'SS_NONE', 'position': [0.7020847797393799, 0.0024692302104085684, 1.5118824243545532], 'quaternion': [-0.1476999968290329, -0.1386999934911728, -0.08030000329017639, 0.9758999943733215], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_PinkyFinger1', 'sensor_state': 'SS_NONE', 'position': [0.6460309624671936, 0.0017232412938028574, 1.4971303939819336], 'quaternion': [-0.060100000351667404, -0.16009999811649323, -0.0989999994635582, 0.9801999926567078], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_PinkyFinger2', 'sensor_state': 'SS_NONE', 'position': [0.6121069192886353, -0.0008232367690652609, 1.4822903871536255], 'quaternion': [-0.05310000106692314, -0.36820000410079956, -0.14319999516010284, 0.9169999957084656], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}, {'name': 'HN_PinkyFinger3', 'sensor_state': 'SS_NONE', 'position': [0.595414936542511, -0.00032789522083476186, 1.4650311470031738], 'quaternion': [-0.053599998354911804, -0.5616999864578247, -0.16769999265670776, 0.8083000183105469], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}]}}
