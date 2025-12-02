# Robot Data Collector

基于 Qt Widgets 的多设备数据采集应用仓库，核心 C++ 程序位于 `my_project/`，语音提示使用 IndexTTS（`index-tts-vllm` 子模块）。

## 仓库结构
- `my_project/`：Data Collector GUI 应用（C++/Qt/OpenCV），详细说明见 `my_project/README.md`。
- `index-tts-vllm/`：IndexTTS 语音模型子模块，仅作为依赖引用。
- `resources/`（在 `my_project` 内）：运行时配置、标定、日志、提示文案、语音模型等。
- `build/`、`my_project/build/`：本地 CMake 构建目录（不应提交到 Git）。
- 其它根目录文件：脚本、示例音频等，仅用于本地调试。

## 构建与运行 C++ 应用
在仓库根目录执行：

```bash
cmake -S my_project -B my_project/build -DCMAKE_BUILD_TYPE=Debug
cmake --build my_project/build -j
./my_project/build/DataCollectorApp
```

运行前请确保已安装：
- Qt5 Widgets/Core/Multimedia
- OpenCV
- librealsense2（仅当使用 RealSense 摄像头时需要）

更多关于设备配置、数据流和扩展方式，参见 `my_project/README.md`。

## IndexTTS 子模块与权重
本仓库使用 IndexTTS 作为语音提示引擎，对应子模块位于 `index-tts-vllm/`。

首次克隆仓库后，初始化子模块：

```bash
git submodule update --init --recursive
```

将模型权重放在：

```text
index-tts-vllm/checkpoints/Index-TTS-1.5-vLLM
```

然后在仓库根目录启动 IndexTTS API 服务（示例）：

```bash
cd index-tts-vllm
python3 api_server.py \
  --model_dir ./checkpoints/Index-TTS-1.5-vLLM \
  --host 127.0.0.1 \
  --port 6006
```

权重文件和缓存不会被提交到 Git，仅在本地存在。

## Git 使用与协作
- 建议默认分支使用 `main`，通过分支 + PR 工作流协作开发。
- 提交前请确认：
  - 构建通过：`cmake --build my_project/build -j`
  - 测试通过：`ctest --test-dir my_project/build --output-on-failure`
- 仓库已忽略构建目录、日志、外部 SDK 和本地模型权重，避免推送大文件。

