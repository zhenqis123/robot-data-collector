# Robot Data Collector

基于 Qt Widgets 的多设备数据采集应用仓库，核心 C++ 程序位于 `my_project/`，语音提示使用 IndexTTS（`index-tts-vllm` 子模块）。

## 仓库结构
- `my_project/`：Data Collector GUI 应用（C++/Qt/OpenCV），详细说明见 `my_project/README.md` 与 `my_project/docs`。
- `index-tts-vllm/`：IndexTTS 语音模型子模块，仅作为依赖引用。
- `my_project/resources/`：运行时配置、标定、日志、提示文案、语音模型等。
- `build/`、`my_project/build/`：本地 CMake 构建目录（不应提交到 Git）。
- 根目录下的脚本与音频文件：用于本地调试与示例。

## 快速开始

```bash
git clone https://github.com/zhenqis123/robot-data-collector.git
cd robot-data-collector

# 如需本地 TTS，可初始化子模块（可选）
git submodule update --init --recursive

# Debug 构建并运行
cmake -S my_project -B my_project/build -DCMAKE_BUILD_TYPE=Debug
cmake --build my_project/build -j
./my_project/build/DataCollectorApp
```

## 环境与依赖（Ubuntu 示例）

运行前在 Ubuntu 上安装以下依赖：

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

建议使用较新的 Ubuntu LTS（例如 20.04/22.04），并确保 CMake ≥ 3.16、GCC ≥ 9。

## 获取代码与子模块

如果不需要在本地运行tts服务，可以不初始化indetts子模块，访问我这边4090主机的tts服务就可以。ip是192.168.20.173。需要在config.json中修改endpoint为这个地址，然后将audio_paths设置为项目根目录下的jay_promptvn.wav文件路径或者其他示例音频文件路径。
```bash
git clone https://github.com/zhenqis123/robot-data-collector.git
cd robot-data-collector

# 初始化 IndexTTS 子模块
git submodule update --init --recursive
```


可以通过运行根目录下的test_index_tts.py脚本测试tts服务是否可用：
```bash
python3 test_index_tts.py
```
我这边尽量保证4090主机的tts服务是在线的。

## 构建与运行 C++ 应用

在仓库根目录执行 Debug 构建：

```bash
cmake -S my_project -B my_project/build -DCMAKE_BUILD_TYPE=Debug
cmake --build my_project/build -j
./my_project/build/DataCollectorApp
```

如果需要一键重新构建 Release 版本，在仓库根目录运行：

```bash
./build_release.sh
```

该脚本会清理并重新生成 `my_project/build_release` 目录，并以 `Release` 配置构建可执行文件。

更多关于设备配置、数据流和扩展方式，参见 `my_project/README.md` 与 `my_project/docs`。

## VLM 任务生成

本项目支持通过多模态大模型自动生成任务模板（用于指导录制过程），相关配置位于 `my_project/resources/config.json` 的 `vlm` 字段：

- `model`：后端使用的模型名称（取决于实际服务实现）。
- `endpoint`：兼容 OpenAI Chat Completions 的 HTTP 接口地址。
- `prompt_path`：用于构造系统提示词的文本文件路径（通常为 `resources/prompts/vlm_task_prompt.txt`）。
- `api_key`：访问 VLM 服务的密钥（推荐留空，改用环境变量）。

运行时，应用会按以下顺序获取 API key：
- 首选 `config.json` 中的 `vlm.api_key`；
- 若为空，则尝试读取环境变量 `GPT_API_KEY`。

出于安全考虑，协作开发时推荐：
- 在 `config.json` 中使用占位符（例如 `"api_key": ""`），
- 实际密钥通过本地环境变量 `GPT_API_KEY` 提供，并避免提交到 Git。

VLM 生成的任务模板会被保存为：
- `my_project/resources/logs/vlm_output.json`

GUI 中的 “Generate Task (VLM)” 按钮会调用该流程，成功后即可在任务选择面板中使用生成的模板。

## 测试

在完成构建后，可以在仓库根目录执行：

```bash
ctest --test-dir my_project/build --output-on-failure
```

用于运行 C++ 单元测试，建议在提交前保持测试通过。

## IndexTTS 子模块与权重

如果不需要在本地运行tts服务，可以跳过本节。

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

## 新设备接入概览

如果需要在 Data Collector 中接入新的相机/传感器类型，典型步骤如下（概览）：

1. 在 `my_project` 中实现新的 `CameraInterface` 子类（可参考 `NetworkDevice`），完成设备初始化、拉流和帧数据封装。
2. 在相机工厂函数中注册新的 `type` 字符串，使配置文件中的设备类型能映射到该实现。
3. 在 `my_project/resources/config.json` 的 `cameras` 数组中，为新设备添加一条配置记录（包含分辨率、帧率、连接参数等）。
4. 如需要专用显示或配置界面，可在 `MainWindow` / Preview 相关代码中扩展，但核心采集与写盘逻辑不需要改动。

更多细节请参考 `my_project/README.md` 与 `my_project/docs` 中的设计与示例。

## VIVE tracker 和 VD Glove

使用局域网发送数据, 分别设置接收端口, 如6666和9999, 然后开启端口防火墙:
```bash
sudo ufw allow 9999/udp
sudo ufw allow 6666/udp
sudo ufw reload
```

将 libVDMocapSDK_DataRead.so 放到 my_project/resources/libVDMocapSDK_DataRead.so

## Git 使用与协作

- 建议默认分支使用 `main`，通过分支 + PR 工作流协作开发。
- C++ 代码风格：C++17、4 空格缩进，命名与包含顺序等细节请参考根目录 `AGENTS.md`。
- 第三方子模块（如 `index-tts-vllm/`）与 SDK 目录（如 `librealsense/`）通常不直接改动，如需修改请单独讨论。
- 提交前请确认：
  - 构建通过：`cmake --build my_project/build -j`
  - 测试通过：`ctest --test-dir my_project/build --output-on-failure`
- 仓库已忽略构建目录、日志、外部 SDK 和本地模型权重，避免推送大文件。

GitHub 仓库地址：https://github.com/zhenqis123/robot-data-collector.git
