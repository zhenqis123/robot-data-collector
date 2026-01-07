from triad_openvr import triad_openvr
from triad_openvr.triad_openvr import matrix_to_flat_list
import time
import sys
import socket
import struct
import numpy as np

# --- 配置 ---
UBUNTU_IP = "192.168.20.123" 
PORT = 6666              
NUM_TRACKERS = 3
FLOATS_PER_TRACKER = 12 # 3x4 矩阵展平
TOTAL_FLOATS = NUM_TRACKERS * FLOATS_PER_TRACKER # 36

# 创建 UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"开始向 {UBUNTU_IP}:{PORT} 发送 Vive Tracker 数据...")

# < 表示小端序
# d 表示 double (8字节时间戳)
# 36f 表示 36个 float (数据)
# 总大小 = 8 + 36*4 = 152 字节
format_string = f'<d{TOTAL_FLOATS}f'

v = triad_openvr.triad_openvr()
v.print_discovered_objects()

print("Sending VIVE data...")    

try:
    i = 0
    while True:
        data_to_send = []
        zero_pad = [0.] * FLOATS_PER_TRACKER # 12个0
        
        # 记录采集时间戳
        capture_time = time.time()
        
        # 遍历 3 个 Tracker
        for tracker_name in ["tracker_1", "tracker_2", "tracker_3"]:
            try:
                # 获取 3x4 矩阵并展平为列表 (长度12)
                # 格式通常为: [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]
                pose_matrix = v.devices[tracker_name].get_pose_matrix()
                flat_data = matrix_to_flat_list(pose_matrix)
                data_to_send.extend(flat_data)
            except Exception:
                # 如果设备丢失，发送全0
                data_to_send.extend(zero_pad)

        # 打包: 时间戳 + 36个浮点数
        packed_data = struct.pack(format_string, capture_time, *data_to_send)
        
        sock.sendto(packed_data, (UBUNTU_IP, PORT))
        
        # 打印调试信息 (每500帧)
        i += 1
        if i % 500 == 0:
            print(f"Sent frame {i}, TS: {capture_time:.4f}")
            # 简单打印 tracker_0 的位置 (tx, ty, tz -> index 3, 7, 11)
            # 注意: matrix_to_flat_list 的具体顺序取决于 triad_openvr 实现，
            # 通常 OpenVR 是 Row-Major: Row0(3 rot, 1 pos), Row1...
            print(f"T0 Pos: {data_to_send[3]:.2f}, {data_to_send[7]:.2f}, {data_to_send[11]:.2f}")
            print("-" * 30)
            print(f"T1 Pos: {data_to_send[15]:.2f}, {data_to_send[19]:.2f}, {data_to_send[23]:.2f}")
            print("-" * 30)
            print(f"T2 Pos: {data_to_send[27]:.2f}, {data_to_send[31]:.2f}, {data_to_send[35]:.2f}")

        # 100Hz 发送频率
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n程序停止。")
finally:
    sock.close()
    print("Socket已关闭。")