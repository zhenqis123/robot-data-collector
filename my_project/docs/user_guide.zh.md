# 使用指南

## 环境需求
- CMake ≥ 3.16，支持 C++17 的编译器
- Qt 5 Widgets
- OpenCV
- librealsense2（通过包管理或源码安装）
- 运行/调试所需的 X11 或本地图形环境

## 构建步骤
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/Qt -DOpenCV_DIR=/path/to/opencv
cmake --build build
```
执行 `cmake --build build --target unit_tests && ctest --test-dir build` 运行 GoogleTest。

## 运行程序
1. 根据实际硬件编辑 `resources/config.json`（将摄像头 `type` 设为 `RealSense` 可启用真实设备）。
2. 确认 `resources/logs` 可写。
3. 运行 `./build/DataCollectorApp`（Windows 执行 `.exe`），先填写“采集名称/被试信息/保存路径”，点击“打开摄像头”建立预览，再用“开始/停止/暂停/继续采集”控制录制；界面会为每台相机生成独立预览（RealSense 同时显示 RGB 与深度）。左侧的“相机设置”区域可以实时调整每台设备的彩色/深度分辨率与帧率。

## 配置说明
`resources/config.json` 中的 RealSense 相机条目示例：
```json
{
  "type": "RealSense",
  "id": 0,
  "color": {
    "resolution": "1920x1080",
    "frame_rate": 30
  },
  "depth": {
    "resolution": "1280x720",
    "frame_rate": 30
  }
}
```
RGB 相机可仅提供 `color` 段，也可以沿用旧版 `resolution`/`frame_rate`。支持多条记录，用于创建多路相机。

### 标记配置（ArUco / AprilTag）
在 `aruco_targets` 数组中填写需要识别的标定板：
```json
"aruco_targets": [
  {
    "type": "aruco",
    "dictionary": "DICT_4X4_50",
    "marker_ids": [0, 1, 2]
  }
]
```
程序只会检测这些字典与 ID 的标记，并将结果写入 `<采集路径>/aruco/<camera>.csv`。

如需切换到 AprilTag，将 `type` 设为 `"apriltag"` 并提供家族名称（缺省时默认 `tagStandard41h12`）：
```json
"aruco_targets": [
  {
    "type": "apriltag",
    "family": "tagStandard41h12",
    "marker_ids": [0, 1, 2]
  }
]
```

## 数据存储结构
录制文件采用金字塔式层级：
```
<保存路径>/<采集名称>_<被试>/
├── meta.json
├── <camera_id>/
│   ├── timestamps.csv            # timestamp_iso,timestamp_ms,color_path,depth_path
│   ├── color/<时间戳>.png
│   └── depth/<时间戳>.png
└── ...
```
`meta.json` 记录采集名称、被试信息与开始时间，每台相机有独立的 `timestamps.csv`，方便后续按时间戳对齐彩色与深度帧。

## 被试提示窗口
点击“Show Prompt Window”按钮会弹出一个独立窗口，用于向被试展示提示信息，可拖拽到单独屏幕。

## 扩展建议
- 继承 `CameraInterface` 并在 `createCamera` 中注册新类型。
- 在 `::setOnSynchronized` 增加新的分析回调。
- 添加新测试到 `tests/` 目录，确保核心逻辑覆盖完整。
