import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PathConfig import PROJECT_ROOT, TACTASK_DIR, TELEOP_DIR, TACDATA_DIR
import socket
import threading
import copy
import teleop.VDGloves.configs as configs
from time import sleep

# 确保从 vdmocapsdk_dataread 中导入 MocapData 结构体
from teleop.VDGloves.vdmocapsdk_dataread import *


class RobotData:
    """存储各关节欧拉角的结构体
    Attributes:
        eulers_left:    左手各电机需要达到的角度，采用弧度制表示
        eulers_right:   右手各电机需要达到的角度，采用弧度制表示
    """

    def __init__(self):
        self.eulers_left = [0.0 for i in range(configs.JOINT_NUM)]
        self.eulers_right = [0.0 for i in range(configs.JOINT_NUM)]


class DataStream:
    """负责处理 UDP 广播，并获取机器人和 MOCAP 数据

    Attributes:
        ip:             广播 IP
        port_broadcast: 广播端口
        port_local:     本地端口
        index:          连接序号
        lock:           机器人数据的互斥锁
        robot_data:     当前机器人电机数据
        mocap_lock:     MOCAP 数据的互斥锁
        mocap_data:     当前 MOCAP 原始数据
    """

    def __init__(self):
        self.ip = socket.gethostbyname(socket.gethostname())
        self.port_broadcast = 7000
        self.port_local = 9999
        self.index = 0

        # 机器人相关数据和锁
        self.lock = threading.Lock()
        self.robot_data = RobotData()

        # [新增] MOCAP 相关数据和锁
        self.mocap_lock = threading.Lock()
        self.mocap_data = MocapData() # MocapData 是从 SDK 导入的结构体

    def set_ip(self, ip: str) -> None:
        """设置广播 IP

        Args:
            ip  [IN]:   广播 IP
        """
        self.ip = ip

    def set_broascast_port(self, port: int) -> None:
        """设置广播端口

        Args:
            port    [IN]:   广播端口
        """
        self.port_broadcast = port

    def set_local_port(self, port: int) -> None:
        """设置本地端口

        Args:
            port    [IN]:   本地端口
        """
        self.port_local = port

    def set_index(self, index: int) -> None:
        """设置连接序号

        Args:
            index   [IN]:   连接序号
        """
        self.index = index

    def connect(self) -> bool:
        """连接广播

        Returns:
            True:   连接成功
            False:  连接失败
        """
        flag = False
        if not udp_is_open(self.index):
            for i in range(10):
                if udp_open(self.index, self.port_local):
                    flag = True
                    break
            if not flag:
                print("Error: Failed to open UDP port.")
                return False

        if udp_send_request_connect(self.index, self.ip, self.port_broadcast):
            print(f"Successfully connected to {self.ip}:{self.port_broadcast}")
            return True
        
        print(f"Error: Failed to send connect request to {self.ip}:{self.port_broadcast}")
        return False

    def disconnect(self) -> bool:
        """断开广播

        Returns:
            True:   断开成功
            False:  断开失败
        """
        if udp_is_open(self.index):
            udp_close(self.index)
            return udp_remove(self.index, self.ip, self.port_broadcast)
        return True

    def request_robot_data(self) -> bool:
        """获取机器人关节角度数据

        Returns:
            True:   获取成功
            False:  获取失败
        """
        self.lock.acquire()
        try:
            if configs.HAND == "Dex3-1":
                flag = udp_recv_robot_data_dex3_1(self.index, self.ip, self.port_broadcast,
                                    self.robot_data.eulers_left, self.robot_data.eulers_right)
            else:
                flag = udp_recv_robot_data_inspired_hand(self.index, self.ip, self.port_broadcast,
                                    self.robot_data.eulers_left, self.robot_data.eulers_right)
        finally:
            self.lock.release()

        return flag
        
    # [新增] 获取 MOCAP 数据的方法
    def request_mocap_data(self) -> bool:
        """从广播获取完整的 MOCAP 数据

        Returns:
            True:   获取成功
            False:  获取失败
        """
        self.mocap_lock.acquire()
        try:
            # 调用 SDK 函数来接收 MOCAP 数据，并存入 self.mocap_data
            flag = udp_recv_mocap_data(self.index, self.ip, self.port_broadcast, 
                                     self.mocap_data)
        finally:
            self.mocap_lock.release()
        
        return flag

    # [新增] 线程安全地获取 MOCAP 数据
    def get_mocap_data(self) -> MocapData:
        """线程安全地获取 MOCAP 数据副本

        Returns:
            MocapData 的一个深拷贝副本
        """
        self.mocap_lock.acquire()
        try:
            # 返回一个深拷贝，防止数据在被使用时被其他线程修改
            data_copy = copy.deepcopy(self.mocap_data)
        finally:
            self.mocap_lock.release()
            
        return data_copy


### 使用示例 ###
if __name__ == '__main__':
    # 模拟一个 configs 模块，如果你的项目中没有这个文件
    # class MockConfigs:
    #     JOINT_NUM = 12
    #     HAND = "Inspired" # 或 "Dex3-1"
    # configs = MockConfigs()

    print("Starting MOCAP data stream demo...")
    # 1. 实例化数据流对象
    stream = DataStream()

    # 2. 设置连接参数
    stream.set_ip("192.168.20.240") 
    stream.set_broascast_port(9998)
    stream.set_local_port(9999)

    # 3. 连接
    if not stream.connect():
        exit()

    print("\nConnection established. Starting to receive data...")
    print("Press Ctrl+C to stop.")

    try:
        # 4. 在一个循环中持续获取数据
        while True:
            # 请求更新 MOCAP 数据
            stream.request_mocap_data()
            
            # 安全地获取数据副本
            mocap_data = stream.get_mocap_data()

            # MocapData 结构体中的 isUpdate 标志位很重要，
            # 它表示这一帧的数据是否有更新。
            if mocap_data.isUpdate:
                # 打印一些关键信息来验证
                frame_index = mocap_data.frameIndex
                
                # 获取根节点（Hips，索引为0）的位置信息
                # mocap_data.position_body 是一个 ctypes 数组的数组
                # 所以访问方式是 mocap_data.position_body[bone_index][axis_index]
                root_pos = mocap_data.position_body[0]
                pos_x, pos_y, pos_z = root_pos[0], root_pos[1], root_pos[2]

                print(f"Frame: {frame_index:<5} | Root Position: (X: {pos_x: .3f}, Y: {pos_y: .3f}, Z: {pos_z: .3f})")

            # 等待一小段时间，避免 CPU 占用过高
            sleep(0.0001) # 约 100Hz

    except KeyboardInterrupt:
        print("\nStopping data stream...")
    finally:
        # 5. 断开连接，释放资源
        stream.disconnect()
        print("Disconnected successfully. Program terminated.")