# Robot Data Collector

基于 Qt Widgets 的多设备数据采集应用仓库，核心 C++ 程序位于 `my_project/`，语音提示使用 IndexTTS（`index-tts-vllm` 子模块）。

## 仓库结构
- `my_project/`：Data Collector GUI 应用（C++/Qt/OpenCV），详细说明见 `my_project/README.md` 与 `my_project/docs`。
- `index-tts-vllm/`：IndexTTS 语音模型子模块，仅作为依赖引用。
- `my_project/resources/`：运行时配置、标定、日志、提示文案、语音模型等。
- `build/`、`my_project/build/`：本地 CMake 构建目录（不应提交到 Git）。
- 根目录下的脚本与音频文件：用于本地调试与示例。

## 环境与依赖（Ubuntu 示例）

运行前建议在 Ubuntu 上安装以下依赖：

- Qt5 Widgets/Core/Multimedia
- OpenCV
- librealsense2（如需使用 RealSense 摄像头）
- CMake、Git、curl、nlohmann/json 等基础库

示例安装命令：

```bash
sudo apt update
sudo apt install build-essential cmake git pkg-config

# Qt5 开发包
sudo apt install qtbase5-dev qttools5-dev qtdeclarative5-dev qtmultimedia5-dev

# OpenCV
sudo apt install libopencv-dev python3-opencv

# 网络与 JSON
sudo apt install libcurl4-openssl-dev nlohmann-json3-dev

# RealSense SDK（如需）
sudo apt install librealsense2-utils librealsense2-dev librealsense2-dbg
```

## 获取代码与子模块

```bash
git clone https://github.com/zhenqis123/robot-data-collector.git
cd robot-data-collector

# 初始化 IndexTTS 子模块
git submodule update --init --recursive
```

## 构建与运行 C++ 应用

在仓库根目录执行 Debug 构建：

```bash
cmake -S my_project -B my_project/build -DCMAKE_BUILD_TYPE=Debug
cmake --build my_project/build -j
./my_project/build/DataCollectorApp
```

如果需要一键重新构建 Release 版本，可在仓库根目录运行：

```bash
./build_release.sh
```

该脚本会清理并重新生成 `my_project/build_release` 目录，并以 `Release` 配置构建可执行文件。

更多关于设备配置、数据流和扩展方式，参见 `my_project/README.md` 与 `my_project/docs`。

## IndexTTS 子模块与权重

本仓库使用 IndexTTS 作为语音提示引擎，对应子模块位于 `index-tts-vllm/`。

将模型权重放在：

```text
index-tts-vllm/checkpoints/Index-TTS-1.5-vLLM
```

在仓库根目录启动 IndexTTS API 服务（示例）：

```bash
cd index-tts-vllm
python3 api_server.py \
  --model_dir ./checkpoints/Index-TTS-1.5-vLLM \
  --host 127.0.0.1 \
  --port 6006
```

权重文件和缓存不会被提交到 Git，仅在本地存在。

## 带触觉手套的使用说明

先启动python端：

```bash
python my_project/tac_glove_py/TacDataCollector.py
```

再启动C++端：

```bash
./build_release.sh
./my_project/build_release/DataCollectorApp
```

## 数据存储结构

录制数据的目录结构现在如下：

```text
basePath/
├── RealSense_xxx/          # 相机数据
│   ├── color/
│   │   ├── 000001.png
│   │   └── ...
│   ├── depth/
│   │   ├── 000001.png
│   │   └── ...
│   └── timestamps.csv
├── tactile_left/           # 左手触觉手套数据
│   ├── tactile_data.bin    # N×137 二进制矩阵 (float32)
│   ├── tactile_data.csv    # N×137 CSV 矩阵（可读格式）
│   ├── timestamps.csv      # 时间戳文件
│   └── meta.json           # 元数据
└── tactile_right/          # 右手触觉手套数据
    ├── tactile_data.bin
    ├── tactile_data.csv
    ├── timestamps.csv
    └── meta.json
```

文件格式:

- ```tactile_data.bin``` - 二进制矩阵文件
  - 格式：```float32_le```（小端 32 位浮点数）
  - 大小：```N × 137 × 4``` 字节
  - 可直接用 numpy 读取：```np.fromfile("tactile_data.bin", dtype=np.float32).reshape(-1, 137)```
  
- ```tactile_data.csv``` - CSV 矩阵文件
  - 每行 137 个值，逗号分隔
  - 无表头，纯数据，方便可读性检查
  - 可用 pandas 读取：```pd.read_csv("tactile_data.csv", header=None)```
  
- ```timestamps.csv``` - 时间戳文件
  - 包含列：```frame_index, timestamp_iso, timestamp_ms, device_timestamp_ms```
  - 与相机帧时间戳对齐
  
- ```meta.json``` - 元数据文件

## 新设备接入概览

如果需要在 Data Collector 中接入新的相机/传感器类型，典型步骤如下（概览）：

1. 在 `my_project` 中实现新的 `CameraInterface` 子类（可参考 `NetworkDevice`），完成设备初始化、拉流和帧数据封装。
2. 在相机工厂函数中注册新的 `type` 字符串，使配置文件中的设备类型能映射到该实现。
3. 在 `my_project/resources/config.json` 的 `cameras` 数组中，为新设备添加一条配置记录（包含分辨率、帧率、连接参数等）。
4. 如需要专用显示或配置界面，可在 `MainWindow` / Preview 相关代码中扩展，但核心采集与写盘逻辑不需要改动。

更多细节请参考 `my_project/README.md` 与 `my_project/docs` 中的设计与示例。

## Git 使用与协作

- 建议默认分支使用 `main`，通过分支 + PR 工作流协作开发。
- 提交前请确认：
  - 构建通过：`cmake --build my_project/build -j`
  - 测试通过：`ctest --test-dir my_project/build --output-on-failure`
- 仓库已忽略构建目录、日志、外部 SDK 和本地模型权重，避免推送大文件。

GitHub 仓库地址：https://github.com/zhenqis123/robot-data-collector.git
