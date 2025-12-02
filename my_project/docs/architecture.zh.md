# 架构概述

## 分层设计
1. **配置层**：`ConfigManager` 解析 `resources/config.json`，生成包含彩色/深度独立分辨率与帧率的 `CameraConfig`，所有模块统一从此处读取参数，避免 GUI 保存状态。
2. **采集层**：`CameraInterface` 定义相机契约，`RealSense`、`RGB` 等实现封装具体硬件，向外暴露 `FrameData`（对齐的彩色/深度 `cv::Mat`、时间戳、相机 ID）。
3. **处理层**：`DataCapture` 管理相机与线程，将帧送往 ``、`DataStorage`、`ArucoTracker` 与 `Preview`。`` 根据时间戳容差匹配多路帧，并保留回调钩子；`DataStorage` 以“采集→相机→流”金字塔目录保存 `color/`、`depth/`、`timestamps.csv`（同时记录系统/设备时间戳）以及顶层 `meta.json`；`ArucoTracker` 在独立线程中完成 ArUco 检测与 CSV 输出。
4. **展示层**：`Preview` 将 `cv::Mat` 转换成 `QImage`/`QPixmap`，根据相机类型动态构建预览面板，RealSense 同时展示 RGB+深度；`MainWindow` 负责控制面板、状态提示以及“采集名称/被试信息/保存路径”输入，并提供打开/关闭摄像头、开始/停止/暂停/继续采集等操作。
5. **诊断层**：`Logger` 提供 INFO/WARN/ERROR 级别日志，写入 `resources/logs/app.log`，辅助定位问题。

## 可扩展性
- 新相机：实现 `CameraInterface` 并在 `createCamera` 注册即可。
- 同步策略：扩展 `` 的容差逻辑或添加新的回调处理。
- 存储方式：在 `DataStorage` 中替换写入策略（如视频编码、云端 API）。
- GUI：可在 `MainWindow` 增加调试面板、直方图等组件，对后端影响 minimal。

## 线程模型
- `DataCapture` 为每路相机创建线程，线程安全的队列由 `std::mutex`/`std::atomic` 保护。
- `` 在 `submitFrame` 中加锁，计算时间差，必要时触发回调。
- Qt GUI 运行于主线程，跨线程更新控件通过 `QMetaObject::invokeMethod` 完成，确保线程安全。
