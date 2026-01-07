import mujoco
import mujoco.viewer
import time
from data_types import *
from functions import *


def main() -> None:
    # 加载模型
    model = load_robot_with_scene()
    if model is None:
        return
    data = mujoco.MjData(model)

    # 获取模型各关节的下标
    ids_qpos_hand_left, ids_qpos_hand_right = initialize_indices(model)

    # 初始化被动查看器（无法暂停和运行）
    viewer = mujoco.viewer.launch_passive(model, data)

    # 设置场景相机
    camera = viewer.cam
    camera.lookat[:] = [0, 0, -0.2]
    camera.elevation = 0
    camera.azimuth = 180  # 角度制

    # 仿真参数
    timestep = model.opt.timestep  # 仿真时间步长

    # 主循环
    
    start_time = time.time()

    # 获取广播数据相关
    data_stream = DataStream()
    data_stream.set_ip("192.168.20.240")  # 广播 IP
    data_stream.set_broascast_port(9998)  # 广播端口
    data_stream.set_local_port(9999)  # 本地端口
    if not data_stream.connect():
        print("连接失败")
        return

    while viewer.is_running():
        if data_stream.request_robot_data():
            # 消解因瞬间设置角度导致的短时间内速度变化过快，速度和加速度都超过阈值，系统不合法导致的重置
            for i in range(len(data.qvel)):
                data.qvel[i] = 0
            # 获取数据并驱动模型
            data_stream.request_robot_data()
            drive_robot(data, ids_qpos_hand_left, data_stream.robot_data.eulers_left)
            drive_robot(data, ids_qpos_hand_right, data_stream.robot_data.eulers_right)

        # 步进仿真
        mujoco.mj_step(model, data)

        # 更新查看器
        viewer.sync()

        # 控制帧率
        time_until_next_step = timestep - (time.time() - start_time) % timestep
        time.sleep(time_until_next_step)

    data_stream.disconnect()

    # 关闭查看器
    viewer.close()


if __name__ == "__main__":
    main()