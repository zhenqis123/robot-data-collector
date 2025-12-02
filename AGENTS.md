# Repository Guidelines

## Project Structure & Module Organization
- Core app lives in `my_project/src` with Qt widget logic, device handling, and capture/storage components; public headers sit in `my_project/include`.
- `resources/` holds runtime config (`config.json`), calibration data (`intrinsics.json`), and log output directory (`logs/`).
- Tests reside in `my_project/tests` (GoogleTest, see `test_*.cpp`); supplementary notes or diagrams go in `my_project/docs`.
- CMake builds are placed in `my_project/build` (create it via CMake; do not check it in). A small RealSense probe binary is built as `rs_test` alongside the main executable.

## Build, Test, and Development Commands
```
cmake -S my_project -B my_project/build -DCMAKE_BUILD_TYPE=Debug   # configure
cmake --build my_project/build -j                                  # build app + tests
./my_project/build/DataCollectorApp                                # run GUI
ctest --test-dir my_project/build --output-on-failure              # run unit tests
./my_project/build/rs_test                                         # optional RealSense smoke test
```
- Ensure OpenCV, Qt5 (Widgets/Core), librealsense2, and Threads are available; CMake will fail fast if headers/libs are missing.

## Coding Style & Naming Conventions
- C++17, 4-space indentation, braces on new lines for functions/classes, `#pragma once` for headers.
- Classes and Qt types use `PascalCase`; functions/methods and variables use `camelCase` (e.g., `captureFrame`, `deviceTimestampMs`).
- Prefer RAII and smart pointers (`std::unique_ptr`), and keep includes ordered as standard -> third-party -> project.
- Respect compile-time paths (`APP_CONFIG_PATH`, `APP_LOG_DIR`, `APP_INTRINSICS_PATH`) and avoid hard-coding alternates in sources.

## Testing Guidelines
- Framework: GoogleTest via CMake; test files follow `test_<Component>.cpp` and `TEST(Component, Behavior)` naming.
- Keep tests deterministic and use temp directories for artifacts (see `tempDir()` in `test_CameraInterface.cpp`).
- After modifying capture, storage, or synchronization logic, run `ctest` to verify both functional and timing-sensitive expectations.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (`Add camera stats averaging`, `Fix preview frame sync`). Group related changes; avoid mixed concerns.
- For PRs, include a short summary of behavior changes, key commands run (build/tests), dependencies introduced, and any screenshots or logs relevant to the GUI.
- Reference related issues or tasks when available, and call out known limitations or follow-ups to aid reviewers. 

## Security & Configuration Tips
- Protect device credentials and calibration artifacts; do not commit user-specific RealSense serials or large recorded datasets.
- Verify `resources/logs` exists and is writable before running the GUI; keep `config.json` minimal and checked in so defaults remain reproducible.
