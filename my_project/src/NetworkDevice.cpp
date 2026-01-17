#include "NetworkDevice.h"

#include <thread>

#include <opencv2/imgproc.hpp>

#include "Logger.h"

NetworkDevice::NetworkDevice(Logger &logger)
    : _logger(logger)
{
}

bool NetworkDevice::initialize(const CameraConfig &config)
{
    _config = config;
    _label = "Network#" + std::to_string(config.id);
    if (_config.width <= 0)
        _config.width = 320;
    if (_config.height <= 0)
        _config.height = 240;
    if (_config.frameRate <= 0)
        _config.frameRate = 10;
    if (_config.endpoint.empty())
        _logger.warn("Network device id=%d has empty endpoint", config.id);
    _running = true;
    return true;
}

FrameData NetworkDevice::captureFrame()
{
    FrameData data;
    if (!_running)
        return data;

    cv::Mat frame(_config.height, _config.width, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::putText(frame, _label, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {200, 200, 200}, 2);
    if (!_config.endpoint.empty())
        cv::putText(frame, _config.endpoint, {20, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {150, 200, 150}, 2);
    data.image = frame;
    data.timestamp = std::chrono::system_clock::now();
    data.deviceTimestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
    data.cameraId = _label;
    data.colorFormat = "BGR";
    if (_config.frameRate > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / _config.frameRate));
    return data;
}

void NetworkDevice::close()
{
    _running = false;
}

std::string NetworkDevice::name() const
{
    return _label;
}

CaptureMetadata NetworkDevice::captureMetadata() const
{
    CaptureMetadata meta;
    meta.deviceId = _label;
    meta.model = "Network";
    meta.aligned = false;
    meta.colorFormat = "BGR8";
    meta.colorIntrinsics.stream = "color";
    meta.colorIntrinsics.width = _config.width;
    meta.colorIntrinsics.height = _config.height;
    meta.colorFps = _config.frameRate;
    return meta;
}

std::unique_ptr<FrameWriter> NetworkDevice::makeWriter(const std::string &basePath, Logger &logger)
{
    return makeGstHdf5Writer(_label,
                             basePath,
                             logger,
                             _config.frameRate,
                             _config.depth.chunkSize,
                             _config.color.bitrateKbps,
                             _config.color.rateControl);
}
