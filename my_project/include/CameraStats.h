#pragma once

#include <chrono>
#include <mutex>
#include <optional>

struct CameraFps
{
    double capture{0.0};
    double display{0.0};
    double write{0.0};
};

class CameraStats
{
public:
    CameraStats();

    std::optional<CameraFps> recordCapture();
    std::optional<CameraFps> recordDisplay();
    std::optional<CameraFps> recordWrite();
    CameraFps current() const;

private:
    mutable std::mutex _mutex;
    CameraFps _latest;
    size_t _captureCounter{0};
    size_t _displayCounter{0};
    size_t _writeCounter{0};
    std::chrono::steady_clock::time_point _captureLast;
    std::chrono::steady_clock::time_point _displayLast;
    std::chrono::steady_clock::time_point _writeLast;

    std::optional<CameraFps> updateFps(size_t &counter,
                                       std::chrono::steady_clock::time_point &last,
                                       double CameraFps::*field);
};
