# Data Collector (Multi-Device)

## 概览
- Qt Widgets GUI，支持多设备并行采集、显示、写盘。每个设备独立三线程：采集（设备 SDK）、显示（送 Preview）、写盘（设备自带 writer）。
- 音频提示仅支持 IndexTTS，默认中文；Piper 已移除。
- 数据目录由设备 writer 决定，内置 PNG writer 会写 `<base>/<deviceId>/color|depth` 与 `timestamps.csv`。

## 构建与运行
```bash
cmake -S my_project -B my_project/build -DCMAKE_BUILD_TYPE=Release
cmake --build my_project/build -j
./my_project/build/DataCollectorApp
```
运行前确认依赖：Qt5 Widgets/Core/Multimedia、OpenCV、librealsense2（如用 RealSense）。

## 设备配置
`resources/config.json` 中的 `cameras` 数组定义设备，例如：
```json
{
  "type": "RealSense",
  "id": 0,
  "serial": "151322070562", 
  "color": { "resolution": "1280x720", "frame_rate": 30 },
  "depth": { "resolution": "1280x720", "frame_rate": 30 }
},
{
  "type": "{device_type}",
  "id": 1,
  "endpoint": "http://10.0.0.2:9000",
  "params": { ... }
}
```

## 新设备接入指南
1) 实现 `CameraInterface` 子类（可参考 `NetworkDevice`）：
   - `initialize(const CameraConfig&)` 读取配置。
   - `captureFrame()` 返回 `FrameData`（必填 `image` 或其他自定义字段）。
   - `makeWriter(basePath, logger)` 返回设备的 writer（可复用 `makePngWriter` 或自定义二进制/CSV writer）。
2) 在 `createCamera` 工厂中注册新 `type` 字符串。
3) 在配置中为新设备添加一条记录，包含所需参数（可放入 `endpoint` 等字段）。
4) 如需专用 UI/配置编辑器，可在 MainWindow 中扩展，但核心采集/写盘无需改动。

## 数据流
- 采集线程：SDK 拉帧 → enqueue 到显示队列 / 写盘队列 → 可选 ArUco 提交。
- 显示线程：消费显示队列，调用 Preview 显示并统计显示 FPS。
- 写盘线程：消费写盘队列，调用设备 writer 落盘并统计写盘 FPS。

## 已知注意事项
- RealSense 多机需确保 USB3 带宽/供电；若超时频繁可调低分辨率/FPS。
- DataStorage 仅处理元数据/事件/注释；设备数据由各自 writer 负责。
