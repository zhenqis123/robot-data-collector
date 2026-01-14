import torch

import pickle
from pathlib import Path
from typing import List
import importlib
import numpy as np
import tqdm
import time, sys
from dex_retargeting.constants import (
    RobotName,
    ROBOT_NAME_MAP,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
#from dex_retargeting.robot_wrapper import RobotWrapper
#from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
#import cv2
from copy import deepcopy
import redis
import tyro
import sapien
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
from pytransform3d import rotations
import os, yaml



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

def visualize_colored_points(points):
    import open3d as o3d
    """
    可交互渲染25个三维点，每5个点一组：
    组颜色：红、蓝、绿、黄、黑；
    每组内颜色从深到浅。
    """
    assert points.shape == (25, 3), "points应为(25,3)的numpy数组"

    base_colors = np.array([
        [1, 0, 0],   # 红
        [0, 0, 1],   # 蓝
        [0, 1, 0],   # 绿
        [1, 1, 0],   # 黄
        [0, 0, 0],   # 黑
    ])
    num_groups = 5
    pts_per_group = 5

    colors = []
    for g in range(num_groups):
        base_color = base_colors[g]
        shades = np.linspace(0.3, 1.0, pts_per_group).reshape(-1, 1)
        group_colors = base_color * shades
        colors.append(group_colors)
    colors = np.vstack(colors)

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window("25 Points Visualization", width=800, height=600)
    vis.add_geometry(pcd)

    # 交互循环
    vis.run()
    vis.destroy_window()

# Retargetor: manus hand motion -> target hand joint positions
class ManusRetarget:
    def __init__(self, 
                 hand_name: RobotName=RobotName.inspiretac, 
                 mode="right", # right, left, both
                 retargeting_type: RetargetingType=RetargetingType.dexpilot,
                 hand_config=None,
                 redis_listener=None,
                ):
        self.redis_listener = redis_listener
        
        self.use_left_hand = (mode=="left" or mode=="both")
        self.use_right_hand = (mode=="right" or mode=="both")
        self.retargeting_type = retargeting_type
        self.hand_config = hand_config
        self.hardware_joint_is_reverse = np.array(self.hand_config["hardware_joint_is_reverse"], dtype=np.int)

        # Create dexpilot retargeting
        #robot_dir = Path(importlib.util.find_spec('dex_retargeting').origin).absolute().parent.parent / "assets" / "robots" / "hands"
        robot_dir = "assets"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        override = dict(add_dummy_free_joint=True)
        if self.use_right_hand:
            right_config_path = get_default_config_path(hand_name, self.retargeting_type, HandType.right)
            self.right_retargeting_config = RetargetingConfig.load_from_file(right_config_path, override=override)
            self.right_retargeting = self.right_retargeting_config.build()
            self.right_retargeting_hand_dof_names = self.right_retargeting.optimizer.robot.dof_joint_names
            print("retargeting right hand dof names:", self.right_retargeting_hand_dof_names)
            self.right_retargeting_hand_active_dof_names = self.right_retargeting.optimizer.target_joint_names
            print("retargeting right hand active dof names:", self.right_retargeting_hand_active_dof_names)
        else:
            self.right_retargeting_hand_dof_names = None
        if self.use_left_hand:
            left_config_path = get_default_config_path(hand_name, self.retargeting_type, HandType.left)
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
            robot_start_pos_right = np.array([0,0.2,-0.5]) 
            robot_start_rot_right = R.from_euler('xyz', [0., 0., -np.pi/2]).as_matrix()
            self.RT_world2base_right = np.eye(4)
            self.RT_world2base_right[:3,:3] = robot_start_rot_right
            self.RT_world2base_right[:3,3] = robot_start_pos_right
        if self.use_left_hand:
            robot_start_pos_left = np.array([0,-0.2,-0.5]) 
            robot_start_rot_left = R.from_euler('xyz', [0., 0., -np.pi/2]).as_matrix()
            self.RT_world2base_left = np.eye(4)
            self.RT_world2base_left[:3,:3] = robot_start_rot_left
            self.RT_world2base_left[:3,3] = robot_start_pos_left

    def _extend_pinky(self, raw_nodes: np.ndarray, extend_factor: float = 1.0) -> np.ndarray:
        """
        raw_nodes: np.ndarray of shape (25,3) representing MANUS raw skeleton nodes
        extend_factor: how much to scale the pinky chain length (>1 to extend)
        Returns a new np.ndarray of shape (25,3) with pinky chain extended.
        """
        new_nodes = raw_nodes.copy()
        # indices for pinky chain (according to docs)
        pinky_idxs = [23, 24]  #20, 21, 22, 
        # For each segment from parent to child, compute vector, scale it, update child pos
        for i in range(1, len(pinky_idxs)):
            parent_idx = pinky_idxs[i-1]
            child_idx  = pinky_idxs[i]
            parent_pos = new_nodes[parent_idx]
            child_pos  = new_nodes[child_idx]
            vec = child_pos - parent_pos
            length = np.linalg.norm(vec)
            if length == 0:
                continue  # avoid zero-length segment
            dir_vec = vec / length
            new_length = length * extend_factor
            new_child_pos = parent_pos + dir_vec * new_length
            new_nodes[child_idx] = new_child_pos
        return new_nodes
    
    def retarget(self, return_raw_data=False):
        data = self.redis_listener.get('manus')
        data = pickle.loads(data)
        data["right_fingers"] = self._extend_pinky(data["right_fingers"], extend_factor=1.8)

        if self.retargeting_type == RetargetingType.position:
            left_hand_data = self.retarget_left_hand_position(data)
            right_hand_data = self.retarget_right_hand_position(data)
        else:
            self.retarget_left_hand(data), self.retarget_right_hand(data)
        
        if return_raw_data:
            
            return self.retarget_left_hand(data), self.retarget_right_hand(data), data
        else:
            return self.retarget_left_hand(data), self.retarget_right_hand(data)

    def retarget_right_hand(self, data):
        if not self.use_right_hand:
            return None, None
        
        if self.retargeting_type == RetargetingType.position:
            joint_pos = joint_avp2hand(data["right_fingers"])
            indices = self.right_retargeting.optimizer.target_link_human_indices
            try:
                task_indices = indices[1, :]
            except Exception:
                task_indices = indices
            
            task_indices = task_indices[task_indices < joint_pos.shape[0]]
            ref_value = joint_pos[task_indices, :]
            qpos = self.right_retargeting.retarget(ref_value)
            hardware_hand_qpos = (qpos[self.right_retargeting.optimizer.idx_pin2target] - self.right_retargeting.joint_limits[:,0]) / \
                                (self.right_retargeting.joint_limits[:,1] - self.right_retargeting.joint_limits[:,0]) 
            hardware_hand_qpos = hardware_hand_qpos[6:]
            hardware_hand_qpos = np.clip(
                hardware_hand_qpos * (1 - self.hardware_joint_is_reverse) + (1. - hardware_hand_qpos) * self.hardware_joint_is_reverse,
                0., 1.
            )
        else: # vector or dexpilot
            #visualize_colored_points(data["right_fingers"])
            joint_pos = joint_avp2hand(data["right_fingers"])
            indices = self.right_retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = self.right_retargeting.retarget(ref_value)
            hardware_hand_qpos = (qpos[self.right_retargeting.optimizer.idx_pin2target] - self.right_retargeting.joint_limits[:,0]) /\
                (self.right_retargeting.joint_limits[:,1] - self.right_retargeting.joint_limits[:,0])
            hardware_hand_qpos = hardware_hand_qpos[6:] # remove 6-dof dummy free joints in retargeting
            hardware_hand_qpos = np.clip(
                hardware_hand_qpos * (1-self.hardware_joint_is_reverse) + (1.-hardware_hand_qpos) * self.hardware_joint_is_reverse, 
                0., 
                1.
            )
        return qpos, hardware_hand_qpos

    def retarget_left_hand(self, data):
        if not self.use_left_hand:
            return None, None
        if self.retargeting_type == RetargetingType.position:
            joint_pos = joint_avp2hand(data["left_fingers"])
            indices = self.left_retargeting.optimizer.target_link_human_indices
            try:
                task_indices = indices[1, :]
            except Exception:
                task_indices = indices
            
            task_indices = task_indices[task_indices < joint_pos.shape[0]]
            ref_value = joint_pos[task_indices, :]
            qpos = self.left_retargeting.retarget(ref_value)
            hardware_hand_qpos = (qpos[self.left_retargeting.optimizer.idx_pin2target] - self.left_retargeting.joint_limits[:,0]) // \
                                (self.left_retargeting.joint_limits[:,1] - self.left_retargeting.joint_limits[:,0]) 
            hardware_hand_qpos = hardware_hand_qpos[6:]
            hardware_hand_qpos = np.clip(
                hardware_hand_qpos * (1 - self.hardware_joint_is_reverse) + (1.-hardware_hand_qpos) * self.hardware_joint_is_reverse,
                0., 1.
            )
        else: # vector or dexpilot
            joint_pos = joint_avp2hand(data["left_fingers"])
            #print(joint_pos)
            indices = self.left_retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = self.left_retargeting.retarget(ref_value)   
            hardware_hand_qpos = (qpos[self.left_retargeting.optimizer.idx_pin2target] - self.left_retargeting.joint_limits[:,0]) /\
                (self.left_retargeting.joint_limits[:,1] - self.left_retargeting.joint_limits[:,0])
            hardware_hand_qpos = hardware_hand_qpos[6:] # remove 6-dof dummy free joints in retargeting
            hardware_hand_qpos = np.clip(
                hardware_hand_qpos * (1-self.hardware_joint_is_reverse) + (1.-hardware_hand_qpos) * self.hardware_joint_is_reverse, 
                0., 
                1.
            )
        return qpos, hardware_hand_qpos


class SapienHandRenderer:
    def __init__(self, left_urdf_path, left_retargeting_joint_names, right_urdf_path, right_retargeting_joint_names, **kwargs):
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
        scene = sapien.Scene()
        scene.set_timestep(1 / 240)
        scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
        scene.add_directional_light(np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True)
        scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
        scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))
        
        # Ground
        visual_material = sapien.render.RenderMaterial()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
        visual_material.set_roughness(0.7)
        visual_material.set_metallic(1)
        visual_material.set_specular(0.04)
        scene.add_ground(-1, render_material=visual_material)

        # Load robot and set it to a good pose to take picture
        loader = scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True
        if left_urdf_path is not None:
            left_robot = loader.load(left_urdf_path)
            self.left_robot = left_robot
        if right_urdf_path is not None:
            right_robot = loader.load(right_urdf_path)#.replace(".urdf", "_glb.urdf"))
            self.right_robot = right_robot

        viewer = Viewer(resolutions=(1200, 900))
        viewer.set_scene(scene)
        viewer.set_camera_xyz(0, 0, 0)
        viewer.set_camera_rpy(0,0,0) #(3.14, 3.14, 0)
        self.viewer = viewer

        if left_urdf_path is not None:
            sapien_left_joint_names = [joint.get_name() for joint in left_robot.get_active_joints()]
            self.left_retargeting_to_sapien = np.array(
                [left_retargeting_joint_names.index(name) for name in sapien_left_joint_names]
            ).astype(int)
        if right_urdf_path is not None:
            sapien_right_joint_names = [joint.get_name() for joint in right_robot.get_active_joints()]
            self.right_retargeting_to_sapien = np.array(
                [right_retargeting_joint_names.index(name) for name in sapien_right_joint_names]
            ).astype(int)
            #print(self.right_retargeting_to_sapien)

    def render_data(self, left_qpos, left_wrist_pose, right_qpos, right_wrist_pose):
        if left_qpos is not None:
            self.left_robot.set_qpos(np.array(left_qpos)[self.left_retargeting_to_sapien])
            self.left_robot.set_pose(sapien.Pose(left_wrist_pose[:3], left_wrist_pose[3:]))
        if right_qpos is not None:
            self.right_robot.set_qpos(np.array(right_qpos)[self.right_retargeting_to_sapien])
            self.right_robot.set_pose(sapien.Pose(right_wrist_pose[:3], right_wrist_pose[3:]))
        
        self.viewer.render()


def main(
    robot_name: RobotName=RobotName.inspiretac, 
    retargeting_type: RetargetingType=RetargetingType.dexpilot, 
    mode: str = "right", # left, right, both
    render_sapien: bool = False,
):
    hand_config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/hand'))
    if "inspire" in ROBOT_NAME_MAP[robot_name].lower():
        hand_config = yaml.safe_load(open(f"{hand_config_dir}/inspire.yaml", 'r'))
    elif ROBOT_NAME_MAP[robot_name] == "linker_hand_l10":
        hand_config = yaml.safe_load(open(f"{hand_config_dir}/linkerl10.yaml", 'r'))
    elif ROBOT_NAME_MAP[robot_name] == "linker_hand_o6":
        hand_config = yaml.safe_load(open(f"{hand_config_dir}/linkero6.yaml", 'r'))
    elif ROBOT_NAME_MAP[robot_name] == "wuji":
        hand_config = yaml.safe_load(open(f"{hand_config_dir}/wuji.yaml", 'r'))
    else:
        raise NotImplementedError

    publisher = redis.Redis(host='localhost', port=6379)
    publisher.set('teleop:hand', 
                  pickle.dumps({'dexhand_qpos': 
                                np.array(hand_config["default_dof_pos"], dtype=np.float32)}))
    #positions = []
    #publisher.set(
    #    "manus",
    #    pickle.dumps({"right_fingers": positions}))
    
    
    retargetor = ManusRetarget(
        hand_name=robot_name, 
        mode=mode, 
        retargeting_type=retargeting_type,
        hand_config=hand_config,
        redis_listener=publisher
    )

    if render_sapien:
        renderer = SapienHandRenderer(
            left_urdf_path=retargetor.left_retargeting_config.urdf_path if retargetor.use_left_hand else None,
            left_retargeting_joint_names=retargetor.left_retargeting_hand_dof_names if retargetor.use_left_hand else None,
            right_urdf_path=retargetor.right_retargeting_config.urdf_path if retargetor.use_right_hand else None,
            right_retargeting_joint_names=retargetor.right_retargeting_hand_dof_names if retargetor.use_right_hand else None,
        )

    while True:
        left, right = retargetor.retarget()
        left_qpos, left_hardware_hand_qpos = left
        right_qpos, right_hardware_hand_qpos = right
        

        # teleoperation publisher
        publisher.set(
            'teleop:hand', 
            pickle.dumps({
                'dexhand_qpos': right_hardware_hand_qpos if mode=="right" else left_hardware_hand_qpos,
            })
        )

        # render
        if render_sapien:
            renderer.render_data(left_qpos, np.array([0,0,0,0,0,0,1]), right_qpos, np.array([0,0,0,0,0,0,1]))


if __name__ == "__main__":
    import sys
    sys.argv.append("--mode=right")
    sys.argv.append("--robot_name=wuji")
    sys.argv.append("--render_sapien")
    sys.argv.append("--retargeting_type=vector")
    
    tyro.cli(main)
    