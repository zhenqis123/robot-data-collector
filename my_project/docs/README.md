# Data Collector Application

This project provides a modular data-acquisition framework that can multiplex multiple cameras, display live previews built with Qt Widgets, synchronize frames, and persist them with OpenCV. Each subsystem (camera abstraction, capture pipeline, preview, storage, configuration, logging) is isolated behind an interface to simplify extension and testing.

## Features
- Abstract `CameraInterface` with pluggable implementations (RealSense, RGB).
- Native librealsense2 integration providing aligned color + depth frames with per-stream configuration (color/depth resolution + FPS).
- Qt control panel auto-builds per-camera preview panes (RealSense gets side-by-side RGB/Depth) and exposes Open/Close + Start/Stop/Pause/Resume capture controls。
- 每个相机面板附带捕获/显示/写盘帧率信息，便于实时监控性能。
- Built-in camera settings editor lets you tweak color/depth resolution and FPS per device without editing JSON.
- Real-time ArUco marker detection (based on targets defined in `config.json`) with detection CSV logs per capture and a detachable participant prompt window for instructions.
- Metadata form in the GUI captures session name, subject details, and destination path to keep recordings organized.
- Pyramid storage layout per capture: `meta.json`, per-camera directories (`color/`, `depth/`), and `timestamps.csv` for precise frame-time logging.
- Multithreaded `DataCapture` that streams to storage, synchronizer, and real-time preview.
- `ConfigManager` that reads JSON camera definitions and exposes typed settings.
- File-based logging with levels (INFO/WARN/ERROR) for reproducibility.
- Qt control panel featuring start/stop controls, preview canvases, and status text.
- GoogleTest suite validating core behaviors.

Refer to `architecture.md` for detailed design and `user_guide.md` for build and usage guidance.
