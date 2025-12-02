#include "CameraStats.h"

CameraStats::CameraStats()
{
    auto now = std::chrono::steady_clock::now();
    _captureLast = now;
    _displayLast = now;
    _writeLast = now;
}

std::optional<CameraFps> CameraStats::recordCapture()
{
    return updateFps(_captureCounter, _captureLast, &CameraFps::capture);
}

std::optional<CameraFps> CameraStats::recordDisplay()
{
    return updateFps(_displayCounter, _displayLast, &CameraFps::display);
}

std::optional<CameraFps> CameraStats::recordWrite()
{
    return updateFps(_writeCounter, _writeLast, &CameraFps::write);
}

CameraFps CameraStats::current() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _latest;
}

std::optional<CameraFps> CameraStats::updateFps(size_t &counter,
                                                std::chrono::steady_clock::time_point &last,
                                                double CameraFps::*field)
{
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(_mutex);
    ++counter;
    const auto elapsed = now - last;
    if (elapsed >= std::chrono::milliseconds(500))
    {
        const double seconds = std::chrono::duration<double>(elapsed).count();
        _latest.*field = counter / seconds;
        counter = 0;
        last = now;
        return _latest;
    }
    return std::nullopt;
}
