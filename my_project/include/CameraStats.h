#pragma once

#include <chrono>
#include <mutex>
#include <optional>

struct FrameData;

struct CameraFps
{
    double captureColor{0.0};
    double captureDepth{0.0};
    double displayColor{0.0};
    double displayDepth{0.0};
    double writeColor{0.0};
    double writeDepth{0.0};
};

class CameraStats
{
public:
    CameraStats();

    std::optional<CameraFps> recordCapture(const FrameData &frame);
    std::optional<CameraFps> recordDisplay(const FrameData &frame);
    std::optional<CameraFps> recordWrite(const FrameData &frame);
    CameraFps current() const;

private:
    mutable std::mutex _mutex;
    CameraFps _latest;
    size_t _captureColorCounter{0};
    size_t _captureDepthCounter{0};
    size_t _displayColorCounter{0};
    size_t _displayDepthCounter{0};
    size_t _writeColorCounter{0};
    size_t _writeDepthCounter{0};
    std::chrono::steady_clock::time_point _captureColorLast;
    std::chrono::steady_clock::time_point _captureDepthLast;
    std::chrono::steady_clock::time_point _displayColorLast;
    std::chrono::steady_clock::time_point _displayDepthLast;
    std::chrono::steady_clock::time_point _writeColorLast;
    std::chrono::steady_clock::time_point _writeDepthLast;
    int64_t _captureColorLastFrame{-1};
    int64_t _captureDepthLastFrame{-1};
    int64_t _displayColorLastFrame{-1};
    int64_t _displayDepthLastFrame{-1};
    int64_t _writeColorLastFrame{-1};
    int64_t _writeDepthLastFrame{-1};

    bool updateStreamFps(bool hasStream,
                         int64_t frameNumber,
                         size_t &counter,
                         std::chrono::steady_clock::time_point &last,
                         int64_t &lastFrameNumber,
                         double CameraFps::*field);
};
