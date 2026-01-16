#include "CameraStats.h"

#include "CameraInterface.h"

CameraStats::CameraStats()
{
    auto now = std::chrono::steady_clock::now();
    _captureColorLast = now;
    _captureDepthLast = now;
    _displayColorLast = now;
    _displayDepthLast = now;
    _writeColorLast = now;
    _writeDepthLast = now;
}

std::optional<CameraFps> CameraStats::recordCapture(const FrameData &frame)
{
    const bool hasAux =
        frame.gloveData.has_value() || frame.viveData.has_value() ||
        frame.manusData.has_value() || frame.tacGloveData.has_value();
    const bool hasColor = frame.hasImage() || hasAux;
    const bool hasDepth = frame.hasDepth();

    bool updated = false;
    updated |= updateStreamFps(hasColor,
                               frame.colorFrameNumber,
                               _captureColorCounter,
                               _captureColorLast,
                               _captureColorLastFrame,
                               &CameraFps::captureColor);
    updated |= updateStreamFps(hasDepth,
                               frame.depthFrameNumber,
                               _captureDepthCounter,
                               _captureDepthLast,
                               _captureDepthLastFrame,
                               &CameraFps::captureDepth);
    if (updated)
        return _latest;
    return std::nullopt;
}

std::optional<CameraFps> CameraStats::recordDisplay(const FrameData &frame)
{
    const bool hasAux =
        frame.gloveData.has_value() || frame.viveData.has_value() ||
        frame.manusData.has_value() || frame.tacGloveData.has_value();
    const bool hasColor = frame.hasImage() || hasAux;
    const bool hasDepth = frame.hasDepth();

    bool updated = false;
    updated |= updateStreamFps(hasColor,
                               frame.colorFrameNumber,
                               _displayColorCounter,
                               _displayColorLast,
                               _displayColorLastFrame,
                               &CameraFps::displayColor);
    updated |= updateStreamFps(hasDepth,
                               frame.depthFrameNumber,
                               _displayDepthCounter,
                               _displayDepthLast,
                               _displayDepthLastFrame,
                               &CameraFps::displayDepth);
    if (updated)
        return _latest;
    return std::nullopt;
}

std::optional<CameraFps> CameraStats::recordWrite(const FrameData &frame)
{
    const bool hasColor = frame.hasImage();
    const bool hasDepth = frame.hasDepth();

    bool updated = false;
    updated |= updateStreamFps(hasColor,
                               frame.colorFrameNumber,
                               _writeColorCounter,
                               _writeColorLast,
                               _writeColorLastFrame,
                               &CameraFps::writeColor);
    updated |= updateStreamFps(hasDepth,
                               frame.depthFrameNumber,
                               _writeDepthCounter,
                               _writeDepthLast,
                               _writeDepthLastFrame,
                               &CameraFps::writeDepth);
    if (updated)
        return _latest;
    return std::nullopt;
}

CameraFps CameraStats::current() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _latest;
}

bool CameraStats::updateStreamFps(bool hasStream,
                                  int64_t frameNumber,
                                  size_t &counter,
                                  std::chrono::steady_clock::time_point &last,
                                  int64_t &lastFrameNumber,
                                  double CameraFps::*field)
{
    const auto now = std::chrono::steady_clock::now();
    if (!hasStream)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        const auto elapsed = now - last;
        if (lastFrameNumber >= 0 && elapsed >= std::chrono::milliseconds(500))
        {
            _latest.*field = 0.0;
            counter = 0;
            last = now;
            return true;
        }
        return false;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    if (frameNumber >= 0 && frameNumber == lastFrameNumber)
    {
        const auto elapsed = now - last;
        if (elapsed >= std::chrono::milliseconds(500))
        {
            _latest.*field = 0.0;
            counter = 0;
            last = now;
            return true;
        }
        return false;
    }
    if (frameNumber >= 0)
        lastFrameNumber = frameNumber;
    ++counter;
    const auto elapsed = now - last;
    if (elapsed >= std::chrono::milliseconds(500))
    {
        const double seconds = std::chrono::duration<double>(elapsed).count();
        _latest.*field = counter / seconds;
        counter = 0;
        last = now;
        return true;
    }
    return false;
}
