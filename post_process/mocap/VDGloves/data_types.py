import socket
import threading
import copy
import configs
from vdmocapsdk_dataread import *


class RobotData:
    """存储各关节欧拉角的结构体
    Attributes:
        eulers_left:    左手各电机需要达到的角度，采用弧度制表示
        eulers_left:    右手各电机需要达到的角度，采用弧度制表示
    """

    def __init__(self):
        self.eulers_left = [0.0 for i in range(configs.JOINT_NUM)]
        self.eulers_right = [0.0 for i in range(configs.JOINT_NUM)]


class DataStream:
    """负责处理 UDP 广播

    Attributes:
        ip:             广播 IP
        port_broadcast: 广播端口
        port_local:     本地端口
        index:          连接序号
        lock:           互斥锁，避免 robot_data 出现写冲突问题
        robot_data:     当前电机数据
    """

    def __init__(self):
        self.ip = socket.gethostbyname(socket.gethostname())
        self.port_broadcast = 7000
        self.port_local = 9999
        self.index = 0
        self.lock = threading.Lock()
        self.robot_data = RobotData()

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
                return False

        if udp_send_request_connect(self.index, self.ip, self.port_broadcast):
            return True
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
        if configs.HAND == "Dex3-1":
            flag = udp_recv_robot_data_dex3_1(self.index, self.ip, self.port_broadcast,
                                self.robot_data.eulers_left, self.robot_data.eulers_right)
        else:
            flag = udp_recv_robot_data_inspired_hand(self.index, self.ip, self.port_broadcast,
                                self.robot_data.eulers_left, self.robot_data.eulers_right)
        self.lock.release()

        return flag
