import pyrealsense2 as rs
import json

def get_all_realsense_intrinsics(width=1280, height=720, fps=30):
    """
    获取所有连接的 RealSense 相机的 Color 和 Depth 内参。
    指定分辨率(width, height)非常重要，因为不同分辨率下内参不同。
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    all_intrinsics = []

    if len(devices) == 0:
        print("未检测到 RealSense 设备 / No RealSense devices found.")
        return []

    print(f"检测到 {len(devices)} 台设备 / Found {len(devices)} devices.")

    # 遍历每一台连接的设备
    for dev in devices:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 获取序列号用于唯一标识设备
        serial_number = dev.get_info(rs.camera_info.serial_number)
        device_name = dev.get_info(rs.camera_info.name)
        print(f"正在处理: {device_name} (S/N: {serial_number})...")

        # 关键：指定要开启的设备序列号，避免多相机冲突
        config.enable_device(serial_number)
        
        # 配置 Depth 和 Color 流 (分辨率需一致或根据需求调整)
        try:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        except Exception as e:
            print(f"配置流失败，该设备可能不支持 {width}x{height}: {e}")
            continue

        # 启动管道
        try:
            pipeline_profile = pipeline.start(config)
        except RuntimeError as e:
            print(f"启动设备 {serial_number} 失败: {e}")
            continue

        try:
            # 需要获取的流类型列表
            streams_to_fetch = [
                (rs.stream.depth, "depth"), 
                (rs.stream.color, "color")
            ]

            for stream_type, stream_name in streams_to_fetch:
                # 获取对应的流配置
                stream_profile = pipeline_profile.get_stream(stream_type)
                intr = stream_profile.as_video_stream_profile().get_intrinsics()

                # 构建数据字典
                intr_data = {
                    "coeffs": intr.coeffs,      # [k1, k2, p1, p2, k3]
                    "cx": intr.ppx,
                    "cy": intr.ppy,
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "height": intr.height,
                    "width": intr.width,
                    "identifier": f"RealSense#{serial_number}",
                    "stream": stream_name
                }
                all_intrinsics.append(intr_data)

        finally:
            # 必须停止管道，释放资源，以便下一个循环或其他程序使用
            pipeline.stop()

    return all_intrinsics

if __name__ == "__main__":
    # 这里设置您想要的分辨率，必须是相机支持的（例如 1280x720, 640x480, 848x480 等）
    intrinsics_list = get_all_realsense_intrinsics(width=1280, height=720)
    
    # 输出最终的 JSON 列表
    print("\n--- 最终结果 (Final JSON) ---")
    print(json.dumps(intrinsics_list, indent=4))