# Architecture Overview

## Layers
1. **Configuration Layer** – `ConfigManager` parses `resources/config.json` into typed `CameraConfig` objects (per-stream color/depth resolution + FPS). All subsystems pull their settings from this interface to avoid GUI-owned state.
2. **Acquisition Layer** – `CameraInterface` defines the hardware contract. Concrete classes (`RealSenseCamera`, `RGBCamera`) encapsulate vendor-specific logic but expose a consistent API returning `FrameData` objects (aligned color + depth `cv::Mat`, timestamp, camera id).
3. **Processing Layer** – `DataCapture` owns cameras and spins a worker thread per device. Each thread emits frames to ``, `DataStorage`, `ArucoTracker`, and `Preview`. `` aligns frames using timestamp tolerances and exposes callbacks for future analytics hooks. `DataStorage` enforces a capture → camera → stream（金字塔式）层级，并维护 `meta.json`、每相机的 `timestamps.csv`（同时写入系统/设备时间戳）以及 ArUco 结果 CSV。
4. **Presentation Layer** – `Preview` converts `cv::Mat` frames into `QImage` objects, builds per-device panels inside the Qt UI, and renders RealSense RGB+depth streams side-by-side while `MainWindow` orchestrates the controls (Open/Close camera lifecycle + Start/Stop/Pause/Resume capture toggles) and exposes metadata inputs (session name, subject, storage path).
5. **Diagnostics Layer** – `Logger` timestamps log entries per level and writes to `resources/logs/app.log`, giving every subsystem simple logging macros.

## Extensibility Points
- Add new camera types by implementing `CameraInterface::initialize/captureFrame/close` and registering in `createCamera`.
- Extend synchronization strategies by replacing `::synchronize` (e.g., add temporal filters or network distribution).
- Additional storage formats can plug into `DataStorage` by swapping writers or piping to cloud SDKs.
- GUI allows more widgets (histograms, telemetry) without touching non-UI modules.

## Threading Model
- A capture thread per camera (managed by `DataCapture`) writes to thread-safe queues guarded by `std::mutex`.
- `` batches frames using timestamps and uses a `std::condition_variable` to wake consumers.
- Qt GUI runs on the main thread; cross-thread communications use `QMetaObject::invokeMethod`/signals from `Preview`.
