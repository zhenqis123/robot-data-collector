# 数据采集应用

本项目提供模块化的数据采集框架，可同时处理多路相机流、在 Qt Widgets 界面中实时预览、对齐帧并通过 OpenCV 保存。每个子系统（相机抽象、采集、预览、存储、配置、日志）都有独立接口，便于扩展与测试。

## 功能概览
- `CameraInterface` 抽象两类相机（RealSense、RGB），支持 librealsense2 实时采集对齐后的彩色+深度帧，并可分别配置彩色/深度分辨率与帧率。
- Qt 控制面板会根据相机动态创建预览窗口，RealSense 额外呈现深度图，并提供“打开/关闭摄像头”“开始/停止/暂停/继续采集”等按钮。
- 每个相机的面板下方会实时显示捕获/显示/写盘帧率，方便监控性能瓶颈。
- GUI 还包含“采集名称/被试信息/保存路径”表单，便于在录制前输入元数据并指定存储位置。
- 控制面板内置相机设置编辑器，可直接调整彩色/深度分辨率与帧率，无需手写配置文件。
- 支持根据 `config.json` 中配置的 ArUco 目标实时检测，并在采集目录下输出检测 CSV；另提供用于向被试展示提示信息的独立窗口。
- 数据以金字塔结构存储：`meta.json`、每个相机各自的 `color/`、`depth/` 子目录以及 `timestamps.csv`（记录时间戳与文件路径）。
- `DataCapture` 多线程捕获，将帧传递给同步器、存储器与预览层。
- `ConfigManager` 解析 `resources/config.json`，集中管理相机参数。
- `Logger` 统一写入 `resources/logs/app.log`，便于复现问题。
- Qt 控制面板包含启动/停止按钮、状态文本和预览画面。
- GoogleTest 覆盖核心模块，支持快速回归。

详细设计请见 `architecture.md` / `architecture.zh.md`，构建与使用指南见 `user_guide.md` / `user_guide.zh.md`。
