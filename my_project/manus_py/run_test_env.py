import pinocchio
import isaacgym
import torch
from isaacgym import gymapi
from isaacgym import gymutil
import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.torch_jit_utils import *
import os
import json
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import numpy as np
import gym
import tasks
import redis
import pickle
import threading
import yaml, time
from manus.manus_dex_retarget import ManusRetarget

from dex_retargeting.constants import (
    RobotName,
    ROBOT_NAME_MAP,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from utils.KeyboardController import KeyboardController


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




@hydra.main(version_base="1.3", config_path="./tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{cfg.task_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    def set_inhand_initial_pose(env, cfg):
        """
        1) 设置物体初始位姿（在手附近）
        2) 设置机械臂关节初始值，使 wrist 处于预定义姿态
        """
        import torch

        # -------- 1. 设置物体初始位姿 --------
        if "inhand" in cfg and "object_pose" in cfg.inhand:
            object_pose = cfg.inhand.object_pose  # [x, y, z, qx, qy, qz, qw]
            env.reset_idx(
                torch.arange(env.num_envs, device=env.device),
                object_init_pose=object_pose,
            )

        # -------- 2. 设置机械臂关节初始姿态（对应 wrist pose） --------
        if "inhand" in cfg and "arm_qpos" in cfg.inhand:
            arm_qpos = torch.tensor(
                cfg.inhand.arm_qpos, dtype=torch.float, device=env.device
            )
            assert arm_qpos.shape[0] == env.num_arm_dofs, \
                f"arm_qpos 长度必须等于 num_arm_dofs={env.num_arm_dofs}"

            # 更新默认关节角（后面你在 manus 分支里用的就是这个）
            env.active_robot_dof_default_pos[:env.num_arm_dofs] = arm_qpos

            # 同时把当前关节位置和 target 也设置一下，避免一开始抖
            # 注意：这里假设 active_robot_dof_indices 是 arm+hand 的索引
            #       我们只改 arm 的那部分
            env.robot_dof_pos[:, env.arm_dof_indices] = arm_qpos
            env.cur_targets[:, env.arm_dof_indices] = arm_qpos

    
    env = create_isaacgym_env()
    if "inhand" in cfg and cfg.inhand.get("enabled", False):
        set_inhand_initial_pose(env, cfg)
    else:
        env.reset_idx(torch.arange(env.num_envs))
    env.reset_idx(torch.arange(env.num_envs))
    
    controller = KeyboardController(
        env=env,
        object_indices=[1],
        robot_dof_indices=[0],
        gravity_enabled=False
    )
    
    # debug the environment
    if "debug" in cfg: 
        if cfg["debug"] == "check_joint":
            per_joint_duration = 100
            for t in range(100000):
                act = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float, device=env.device)
                act[:, :env.num_arm_dofs] = env.active_robot_dof_default_pos[:env.num_arm_dofs]
                act[:, env.hand_dof_start_idx:] = env.active_robot_dof_default_pos[env.num_arm_dofs:]
                i_joint = int(t / per_joint_duration) % env.num_actions
                print(i_joint)
                t_ = t % per_joint_duration
                if t_ < per_joint_duration//2:
                    act[:,i_joint] = env.robot_dof_lower_limits[env.active_robot_dof_indices[i_joint]]
                else:
                    act[:,i_joint] = env.robot_dof_upper_limits[env.active_robot_dof_indices[i_joint]]
                act[:, :env.num_arm_dofs] = unscale(
                    act[:, :env.num_arm_dofs],
                    env.robot_dof_lower_limits[env.arm_dof_indices],
                    env.robot_dof_upper_limits[env.arm_dof_indices],
                )
                act[:, env.hand_dof_start_idx:] = unscale(
                    act[:, env.hand_dof_start_idx:],
                    env.robot_dof_lower_limits[env.active_hand_dof_indices],
                    env.robot_dof_upper_limits[env.active_hand_dof_indices],
                )
                env.step(act)
        
        elif cfg["debug"] == "gello":
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), "../hardware/fr3"))
            from controller.gello_motor_driver import Gello
            import redis, pickle
            from controller.pin_robot import PinRobot
            from pynput import keyboard

            gello_pin_robot = PinRobot(
                urdf_path = env.hand_specific_cfg["GelloURDF"],
                eef_name = "fr3_link8",
                n_arm_dofs = 7
            )
            redis_listener = redis.Redis(host='localhost', port=6379)
            gello = Gello(pin_robot=gello_pin_robot)
            gello.deactivate_motors()

            if env.viewer != None:
                cam_pos = gymapi.Vec3(-0.1, 0.5, 0.8)
                cam_target = gymapi.Vec3(0.5, 0.0, 0.0)
                env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
            
            hardware_active_hand_dof_names = env.hand_specific_cfg["hardware_active_hand_dof_names"]
            #['right_little_1_joint', 'right_ring_1_joint', 'right_middle_1_joint', 
            #'right_index_1_joint', 'right_thumb_2_joint', 'right_thumb_1_joint']
            hardware_to_isaac_hand_dof_indices = [hardware_active_hand_dof_names.index(name) for name in env.active_hand_dof_names]

            
            class KeyboardListener:
                def __init__(self):
                    self.start_recording, self.end_recording, self.reset_flag = False, False, False
                    keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
                    keyboard_listener.start()
                def on_key_press(self, key):
                    if not hasattr(key, 'char'):
                        return
                    if key.char == 'r': 
                        self.reset_flag = True
                        print("Reset flag set to True")
                    elif key.char == 's':
                        print("Start recording episode")
                        self.start_recording = True
                    elif key.char == 'd':
                        print("End recording episode")
                        self.end_recording = True

            keyboard_listener = KeyboardListener()
            initobj_pos = None
            
            for t in range(100000):
                if keyboard_listener.reset_flag:
                    print("reset the environment")
                    env.reset_idx(torch.arange(env.num_envs), object_init_pose=[0.55, -0.1, 0.1, 0, 0, 0, 1])
                    keyboard_listener.reset_flag = False
                    keyboard_listener.start_recording = False
                    keyboard_listener.end_recording = False
                    data = {"wrist_initobj_pos": [], "wrist_quat": [], "hand_qpos": [], "obj_initobj_pos": []}
                
                arm_target = gello.get_franka_qpos() if env.arm_controller=="qpos" else gello.get_franka_pose()
                hand_data_dict = pickle.loads(redis_listener.get('teleop:hand'))
                hand_action = 1. - hand_data_dict['dexhand_qpos'][hardware_to_isaac_hand_dof_indices]
                hand_action = env.robot_dof_lower_limits[env.active_hand_dof_indices] + to_torch(hand_action) * \
                    (env.robot_dof_upper_limits[env.active_hand_dof_indices] - env.robot_dof_lower_limits[env.active_hand_dof_indices])
                if env.arm_controller == "qpos":
                    robot_target = torch.cat([to_torch(arm_target), hand_action]).reshape(1,-1)
                    action = unscale(
                        robot_target,
                        env.robot_dof_lower_limits[env.active_robot_dof_indices],
                        env.robot_dof_upper_limits[env.active_robot_dof_indices],
                    )
                else:
                    hand_action = unscale(
                        hand_action,
                        env.robot_dof_lower_limits[env.active_hand_dof_indices],
                        env.robot_dof_upper_limits[env.active_hand_dof_indices],
                    )
                    arm_target[0:3] *= env.hand_specific_cfg["GelloScale"]
                    action = torch.cat([to_torch(arm_target), hand_action]).reshape(1, -1)

                print(action)
                env.step(action)

                obj_pos = env.object_pos[0]
                wrist_pos = env.rigid_body_states.view(-1, 13)[env.eef_idx, 0:3][0]
                wrist_quat = env.rigid_body_states.view(-1, 13)[env.eef_idx, 3:7][0]
                hand_qpos = env.cur_targets[0, env.active_hand_dof_indices] #env.robot_dof_pos[0, env.active_hand_dof_indices]
                
                if keyboard_listener.start_recording and initobj_pos is None:
                    initobj_pos = obj_pos.clone()
                
                if keyboard_listener.start_recording:
                    data["wrist_initobj_pos"].append(wrist_pos - initobj_pos)
                    data["wrist_quat"].append(wrist_quat)
                    data["hand_qpos"].append(hand_qpos)
                    data["obj_initobj_pos"].append(obj_pos - initobj_pos)
                
                if keyboard_listener.end_recording:
                    for key in data:
                        data[key] = torch.stack(data[key], dim=0).cpu().numpy()
                    with open("grasp_ref.pkl", "wb") as f:
                        pickle.dump(data, f)
                    print("Data saved to grasp_ref.pkl")
                    break

        elif cfg["debug"] == "manus":
            import redis, pickle
            # 创建 ManusRetarget 实例，用于从 Redis 获取数据并进行重定向
            hand_config_dir = f'/home/zbh/yxp/Codespace/DexToolUse/config/hand'
            hand_config = yaml.safe_load(open(f"{hand_config_dir}/wuji.yaml", 'r'))
            mode = "right"  # "right", "left", or "both"
            publisher = redis.Redis(host='localhost', port=6379)
            publisher.set(  'teleop:hand', 
                            pickle.dumps({'dexhand_qpos': 
                                np.array(hand_config["default_dof_pos"], dtype=np.float32)}))
            retargetor = ManusRetarget(
                hand_name=RobotName.wuji,#robot_name, 
                mode=mode,  # 你可以选择 "right", "left", 或 "both"
                retargeting_type=RetargetingType[cfg.get("retargeting_type")], #retargeting_type,#RetargetingType.vector  position
                hand_config=hand_config,
                redis_listener=publisher  # 假设 redis 监听器是从 localhost 获取
            )

            hardware_active_hand_dof_names = env.hand_specific_cfg["hardware_active_hand_dof_names"]
            hardware_to_isaac_hand_dof_indices = [
                hardware_active_hand_dof_names.index(name) for name in env.active_hand_dof_names
            ]
            
            while True:
                
                # 获取 Manus 手套数据并进行重定向
                left_qpos, right_qpos = retargetor.retarget()#return_raw_data=True

                # 获取左手和右手的关节位置
                left_qpos, left_hardware_hand_qpos = left_qpos
                right_qpos, right_hardware_hand_qpos = right_qpos

                # 根据配置选择控制哪只手（右手、左手、或者两只手）
                if right_qpos is not None:
                    hand_norm_01 = right_hardware_hand_qpos[hardware_to_isaac_hand_dof_indices]
                    hand_norm_01 = torch.tensor(hand_norm_01, dtype=torch.float, device=env.device)
                    hand_qpos = env.robot_dof_lower_limits[env.active_hand_dof_indices] + \
                        hand_norm_01 * (env.robot_dof_upper_limits[env.active_hand_dof_indices] -
                                        env.robot_dof_lower_limits[env.active_hand_dof_indices])
                    arm_target = env.active_robot_dof_default_pos[:env.num_arm_dofs]
                    arm_target = arm_target.to(env.device)    
                    robot_target = torch.cat([arm_target, hand_qpos], dim=0).unsqueeze(0)
                    action = unscale(
                        robot_target,
                        env.robot_dof_lower_limits[env.active_robot_dof_indices],
                        env.robot_dof_upper_limits[env.active_robot_dof_indices],
                    )
                    #print(action[0,-1])
                    #[(lower, upper) if (i+1)%4 == 0 else None for i, (lower, upper) in enumerate(zip(env.robot_dof_lower_limits[env.active_robot_dof_indices][-20:],env.robot_dof_upper_limits[env.active_robot_dof_indices][-20:]))]
                    action = torch.cat([torch.zeros([1,1],device=action.device), action], axis=1)# 
                    env.step(action)
                    #right_action = torch.tensor(right_hardware_hand_qpos, dtype=torch.float, device=env.device)
                    #right_action = right_action.unsqueeze(0)  # 增加批次维度 (因为 env.step() 需要批次数据)
                    #env.step(torch.cat([torch.zeros(1, 7, device=env.device), right_action],axis=1))
                    

                if left_qpos is not None:
                    # 将左手的 qpos 数据转换为控制命令 (act)
                    left_action = torch.tensor(left_qpos, dtype=torch.float, device=env.device)
                    left_action = left_action.unsqueeze(0)  # 增加批次维度
                    env.step(left_action)  # 在环境中执行左手控制
                    print(f"Left hand action executed: {left_qpos}")

                # 控制循环的频率
                time.sleep(0.01)  # 每隔 10 毫秒获取一次数据并执行一次动作

        else:
            for t in range(100000):
                action = env.no_op_action
                #action[:, 7:] = 1
                #print(action.shape)
                env.step(action)
    
    else:
        runner = build_runner(cfg, env)
        runner.run()

if __name__ == "__main__":
    import sys
    sys.argv.append("num_envs=1")
    #sys.argv.append("+debug=check_joint")
    sys.argv.append("+debug=manus")
    sys.argv.append("+retargeting_type=dexpilot")# position vector dexpilot
    main()
