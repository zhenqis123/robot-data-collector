# User Guide

## Requirements
- CMake ≥ 3.16
- Compiler with C++17 support
- Qt 6 (or Qt 5) Widgets
- OpenCV
- librealsense2 (installed via package manager or from source)
- Python optional (for auxiliary scripts)

## Build Instructions
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/qt -DOpenCV_DIR=/path/to/opencv
cmake --build build
```
`cmake --build build --target unit_tests && ctest --test-dir build` will run GoogleTest suites.

## Running
1. Ensure `resources/config.json` describes your cameras (set `"type": "RealSense"` to use actual hardware).
2. Create `resources/logs` (already tracked) and ensure the process can write to it.
3. Execute `./build/DataCollectorApp`. Click **Open Cameras** to initialize streams, fill in **Capture Name / Subject Info / Save Path**, then use **Start/Stop/Pause/Resume Capture** to control recording while previews stay live (each RealSense view shows RGB + depth). Use the **Camera Settings** section to adjust color/depth resolution and FPS per device as needed.

## Configuration
Override runtime parameters by editing `resources/config.json`. RealSense entries can define separate color/depth stream settings:
```json
{
  "type": "RealSense",
  "id": 1,
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
Simple RGB cameras can omit the `depth` block and fall back to the `color` parameters.

### ArUco Targets
Add detection targets under `aruco_targets`:
```json
"aruco_targets": [
  {
    "dictionary": "DICT_4X4_50",
    "marker_ids": [0, 1, 2]
  }
]
```
Only the listed markers from the specified dictionary will be searched in each RGB frame. Detection results are written per capture under `<capture_path>/aruco/<camera>.csv`.

## Storage Layout
Recordings are organized hierarchically:
```
<save_path>/<capture_name>_<subject>/
├── meta.json
├── <camera_id>/
│   ├── timestamps.csv            # timestamp_iso,timestamp_ms,color_path,depth_path
│   ├── color/000001.png
│   └── depth/000001.png
└── ...
```
`meta.json` logs capture metadata and start time. Each camera maintains its own timestamp file so downstream tooling can align frames precisely.

## Participant Prompt Window
Use the **Show Prompt Window** button to open a secondary window for displaying instructions to participants. The window is detachable and can be dragged to a different screen.

## Extending
- Add a new camera implementation by subclassing `CameraInterface` and registering it in `createCamera`.
- Connect additional processing callbacks inside `::setOnSynchronized`.
- Update tests under `tests/` to cover new behaviors.
