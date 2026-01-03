#pragma once

#include <array>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <chrono>
#include <optional>
#include <variant>

#include "ConfigManager.h"
#include "IntrinsicsManager.h"
#include "VDGloveDataTypes.h"
#include "ViveDataTypes.h"

class Logger;

struct FrameData
{
    cv::Mat image;
    cv::Mat depth;
    std::chrono::system_clock::time_point timestamp;
    int64_t deviceTimestampMs{0};
    std::string cameraId;
    
    // Optional sensor data
    std::optional<VDGloveFrameData> gloveData;
    std::optional<ViveFrameData> viveData;
};
struct ArucoDetection
{
    std::string cameraId;
    std::chrono::system_clock::time_point timestamp;
    int64_t deviceTimestampMs{0};
    int markerId{0};
    std::vector<cv::Point2f> corners;
};

struct CaptureExtrinsics
{
    std::array<float, 9> rotation{};
    std::array<float, 3> translation{};
};

struct CaptureMetadata
{
    std::string model;
    std::string serial;
    std::string deviceId;
    bool aligned{true};
    double depthScale{0.0};
    int colorFps{0};
    int depthFps{0};
    std::string colorFormat;
    std::string depthFormat;
    StreamIntrinsics colorIntrinsics;
    StreamIntrinsics depthIntrinsics;
    CaptureExtrinsics depthToColor;
};

struct FrameWriter
{
    virtual ~FrameWriter() = default;
    virtual bool write(const FrameData &frame) = 0;
};

std::unique_ptr<FrameWriter> makePngWriter(const std::string &deviceId,
                                           const std::string &basePath,
                                           Logger &logger);
std::unique_ptr<FrameWriter> makeGstHdf5Writer(const std::string &deviceId,
                                               const std::string &basePath,
                                               Logger &logger,
                                               int colorFps,
                                               int depthChunkSize);

class CameraInterface
{
public:
    virtual ~CameraInterface() = default;

    virtual bool initialize(const CameraConfig &config) = 0;
    virtual FrameData captureFrame() = 0;
    virtual void close() = 0;
    virtual std::string name() const = 0;
    virtual std::vector<CameraConfig::StreamConfig> getAvailableResolutions() const { return {}; }
    virtual CaptureMetadata captureMetadata() const { return {}; }
    virtual std::unique_ptr<FrameWriter> makeWriter(const std::string &basePath, Logger &logger) = 0;
};

std::unique_ptr<CameraInterface> createCamera(const CameraConfig &config, Logger &logger);
