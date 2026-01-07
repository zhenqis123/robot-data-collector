import os
import shutil
from typing import Tuple
import numpy as np
import mujoco
import configs
from vdmocapsdk_dataread import *


def load_robot_with_scene() -> mujoco.MjModel:
    if not os.path.exists(configs.MODEL_PATH_LEFT):
        print('Model %s is not existed.' % (configs.MODEL_NAME_LEFT))
        return None
    if not os.path.exists(configs.MODEL_PATH_RIGHT):
        print('Model %s is not existed.' % (configs.MODEL_NAME_RIGHT))
        return None
    if not os.path.exists("temp/meshes"):
        shutil.copytree(src="robots/meshes", dst="temp/meshes")

    model = mujoco.MjModel.from_xml_path(configs.MODEL_PATH_LEFT)
    path_temporary_xml = 'temp/%s_temp.xml' % (configs.MODEL_NAME_LEFT)
    mujoco.mj_saveLastXML(filename=path_temporary_xml, m=model)
    with open(path_temporary_xml, "r") as f:
        xml = f.read()
    temporary_xml = xml.replace("<worldbody>",
                                "<worldbody><body name='left_hand' pos='0 0.16 0' quat='0.707 0.707 0 0'>").replace(
                                    "</worldbody>",
                                    "</body></worldbody>"
                                )
    with open(path_temporary_xml, "w") as f:
        f.write(temporary_xml)

    model = mujoco.MjModel.from_xml_path(configs.MODEL_PATH_RIGHT)
    path_temporary_xml = 'temp/%s_temp.xml' % (configs.MODEL_NAME_RIGHT)
    mujoco.mj_saveLastXML(filename=path_temporary_xml, m=model)
    with open(path_temporary_xml, "r") as f:
        xml = f.read()
    temporary_xml = xml.replace("<worldbody>",
                                "<worldbody><body name='right_hand' pos='0 -0.16 0' quat='0.707 -0.707 0 0'>").replace(
                                    "</worldbody>",
                                    "</body></worldbody>"
                                )
    with open(path_temporary_xml, "w") as f:
        f.write(temporary_xml)

    combined_xml = f"""
        <mujoco model="scene">
            <include file="{'%s_temp.xml' % (configs.MODEL_NAME_LEFT)}"/>
            <include file="{'%s_temp.xml' % (configs.MODEL_NAME_RIGHT)}" pos="0.5 0 20"/>

            <statistic center="0 0 0.5" extent="2.0"/>

            <visual>
                <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
                <rgba haze="0.15 0.25 0.35 1"/>
                <global azimuth="-130" elevation="-20"/>
            </visual>

            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
                <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                    markrgb="0.8 0.8 0.8" width="300" height="300"/>
                <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
            </asset>

            <worldbody>
                <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
                <geom name="floor" pos="0 0 -1" size="0 0 0.05" type="plane" material="groundplane"/>
            </worldbody>
        </mujoco>
        """

    os.chdir("temp")    # 将工作目录切换到 temp
    return mujoco.MjModel.from_xml_string(combined_xml)

def initialize_indices(
    model: mujoco.MjModel,
) -> Tuple[np.ndarray, np.ndarray]:
    ids_joint_qpos_hand_left = np.full(configs.JOINT_NUM, -1)
    ids_joint_qpos_hand_right = np.full(configs.JOINT_NUM, -1)
    for i in range(configs.JOINT_NUM):
        for name in configs.NAMES_JOINT_HAND_LEFT[i]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id != -1:
                ids_joint_qpos_hand_left[i] = model.jnt_qposadr[joint_id]
                break
        for name in configs.NAMES_JOINT_HAND_RIGHT[i]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id != -1:
                ids_joint_qpos_hand_right[i] = model.jnt_qposadr[joint_id]
                break
    
    return ids_joint_qpos_hand_left, ids_joint_qpos_hand_right

def drive_robot(data: mujoco.MjData, ids_qpos: np.ndarray,
               eulers: np.ndarray) -> None:
    for i in range(len(eulers)):
        if ids_qpos[i] != -1:
            data.qpos[ids_qpos[i]] = eulers[i]
