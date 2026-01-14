import numpy as np
from pynput import keyboard
from isaacgym import gymapi, gymtorch
import torch
import math

class KeyboardController:
    def __init__(self, env, object_indices, robot_dof_indices, gravity_enabled=True):
        self.gym = env.gym
        self.sim = env.sim
        self.envs = env.envs
        self.object_indices = object_indices
        self.robot_dof_indices = robot_dof_indices
        self.gravity_enabled = gravity_enabled

        # 初始化 root state tensor 和 DOF state tensor
        self._actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.actor_root_state = gymtorch.wrap_tensor(self._actor_root_state_tensor).view(len(self.envs), -1, 13)
        
        self.dof_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor).view(len(self.envs), -1, 13)
        
        # 启动键盘监听器
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def on_key_press(self, key):
        try:
            if key.char == 'q':  # 左移物体
                self.move_object(0.01, 0, 0)
            elif key.char == 'a':  # 右移物体
                self.move_object(-0.01, 0, 0)
            elif key.char == 'w':  # 后移物体
                self.move_object(0, 0.01, 0)
            elif key.char == 's':  # 前移物体
                self.move_object(0, -0.01, 0)
            elif key.char == 'e':  # 下移物体
                self.move_object(0, 0, 0.01)
            elif key.char == 'd':    # 上移物体
                self.move_object(0, 0, -0.01)
            
            
                
            elif key.char == 'r':  # 复位物体
                self.reset_object_position()
            elif key.char == 'g':  # 切换重力
                self.toggle_gravity()
                
            
            elif key.char == 'y':  # 左移机械手
                self.move_hand(0.01, 0, 0)
            elif key.char == 'h':  # 右移机械手
                self.move_hand(-0.01, 0, 0)
            elif key.char == 'u':  # 前移机械手
                self.move_hand(0, 0.01, 0,)
            elif key.char == 'j':  # 后移机械手
                self.move_hand(0, -0.01, 0)
            elif key.char == 'i':  # 上移机械手
                self.move_hand(0, 0, 0.01)
            elif key.char == 'k':  # 下移机械手
                self.move_hand(0, 0, -0.01)
            elif key.char == 'o':  # 机械手旋转
                self.rotate_hand(0.01, 0, 0)
            elif key.char == 'l':  # 机械手旋转
                self.rotate_hand(-0.01, 0, 0)
            elif key.char == 'p':  # 机械手旋转
                self.rotate_hand(0, 0.01, 0)
            elif key.char == ';':  # 机械手旋转
                self.rotate_hand(0, -0.01, 0)
            elif key.char == '[':  # 机械手旋转
                self.rotate_hand(0, 0, 0.01)
            elif key.char == '\'':  # 机械手旋转
                self.rotate_hand(0, 0, -0.01)
        except AttributeError:
            pass

    def move_object(self, dx, dy, dz):
        # 修改物体的位置（通过修改 root state tensor）
        for i in range(len(self.envs)):
            current_pos = self.actor_root_state[i, self.object_indices[i], 0:3].clone()  # 当前物体的位置
            new_pos = current_pos + torch.tensor([dx, dy, dz], dtype=torch.float, device=self.actor_root_state.device)
            self.actor_root_state[i, self.object_indices[i], 0:3] = new_pos  # 更新位置
            self.actor_root_state[i, self.object_indices[i], -6:] = 0  # 清除旋转角度/线速度/角速度
            print(f"Current object position: {((self.actor_root_state[i, self.object_indices[i], 0:7]*100).round()/100).cpu().numpy()}")

        self.gym.set_actor_root_state_tensor(self.sim, self._actor_root_state_tensor)  # 设置新的 root state
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
    def move_hand(self, dx=0, dy=0, dz=0):
        # 机械手的移动，通过修改 DOF 状态来实现
        for i in range(len(self.envs)):
            current_dof_pos = self.dof_state[i, self.robot_dof_indices[i], 0:3].clone() # 获取当前 DOF 位置
            new_dof_pos = current_dof_pos + torch.tensor([dx, dy, dz], dtype=torch.float, device=self.actor_root_state.device)
            self.dof_state[i, self.robot_dof_indices[i], 0:3] = new_dof_pos  # 更新 DOF
            self.dof_state[i, self.robot_dof_indices[i], -6:] = 0  # 清除线速度和角速度
            print(f"Current hand position: {((self.dof_state[i, self.robot_dof_indices[i], 0:3]*100).round()/100).cpu().numpy()}")

        self.gym.set_actor_root_state_tensor(self.sim, self.dof_state_tensor)  # 设置新的 DOF 状态
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
         
    def rotate_hand(self, droll=0, dpitch=0, dyaw=0):
        # 机械手的旋转
        for i in range(len(self.envs)):
            current_dof_pos = self.dof_state[i, self.robot_dof_indices[i], 3:7].clone() # 获取当前 DOF 位置
            current_euler = self._quaternion_to_euler(current_dof_pos)
            new_euler = current_euler + torch.tensor([droll, dpitch, dyaw], dtype=torch.float, device=self.actor_root_state.device)
            new_quat = self._euler_to_quaternion(new_euler[0], new_euler[1], new_euler[2])
            dqw, dqx, dqy, dqz = new_quat - current_dof_pos
            new_dof_pos = current_dof_pos + torch.tensor([dqw, dqx, dqy, dqz], dtype=torch.float, device=self.actor_root_state.device)
            self.dof_state[i, self.robot_dof_indices[i], 3:7] = new_dof_pos  # 更新 DOF
            self.dof_state[i, self.robot_dof_indices[i], -6:] = 0  # 清除线速度和角速度
            print(f"Current hand rotation(euler): {((new_euler*100).round()/100).cpu().numpy()}")
            print(f"Current hand rotation(quaternion): {((new_quat*100).round()/100).cpu().numpy()}")

        self.gym.set_actor_root_state_tensor(self.sim, self.dof_state_tensor)  # 设置新的 DOF 状态
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def reset_object_position(self):
        # 重置物体位置
        for i in range(len(self.envs)):
            self.actor_root_state[i, self.object_indices[i], 0:7] = torch.tensor([0.61, -0.04, 0.52, 0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=self.actor_root_state.device)
            self.actor_root_state[i, self.object_indices[i], -6:] = 0
            self.dof_state[i, self.robot_dof_indices[i], 0:7] = torch.tensor([-0.22, 0.81, 0.57, 0.48, 0.37, -0.29, 0.74], dtype=torch.float, device=self.actor_root_state.device)

        self.gym.set_actor_root_state_tensor(self.sim, self._actor_root_state_tensor)
        self.gym.set_actor_root_state_tensor(self.sim, self.dof_state_tensor)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def toggle_gravity(self):
        # 切换重力
        sim_params = self.gym.get_sim_params(self.sim)
        if self.gravity_enabled:
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)  # 禁用重力
        else:
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # 恢复重力

        # 设置新的仿真参数
        self.gym.set_sim_params(self.sim, sim_params)
        self.gravity_enabled = not self.gravity_enabled
        print(f"Gravity disabled? {not self.gravity_enabled}")

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        q_w = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.cos(yaw / 2) + torch.sin(roll / 2) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
        q_x = torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.cos(yaw / 2) - torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
        q_y = torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.cos(yaw / 2) + torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.sin(yaw / 2)
        q_z = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.sin(yaw / 2) - torch.sin(roll / 2) * torch.sin(pitch / 2) * torch.cos(yaw / 2)

        return torch.stack([q_w, q_x, q_y, q_z], dim=-1)

    @staticmethod
    def _quaternion_to_euler(q):
        q_w, q_x, q_y, q_z = q.unbind(-1)  # 分解四元数

        # 计算欧拉角
        roll = torch.atan2(2.0 * (q_w * q_x + q_y * q_z), 1.0 - 2.0 * (q_x**2 + q_y**2))
        pitch = torch.asin(torch.clamp(2.0 * (q_w * q_y - q_z * q_x), -1.0, 1.0))
        yaw = torch.atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y**2 + q_z**2))
        return torch.stack([roll, pitch, yaw], dim=-1)

if __name__ == "__main__":
    temp_euler = torch.tensor([0.06, -1.09, 3.14], dtype=torch.float)
    temp_quat = KeyboardController._euler_to_quaternion(*temp_euler)
    print(f"Quaternion: {temp_quat.numpy()}")