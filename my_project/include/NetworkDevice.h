#pragma once

#include "CameraInterface.h"

class NetworkDevice : public CameraInterface
{
public:
    explicit NetworkDevice(Logger &logger);

    bool initialize(const CameraConfig &config) override;
    FrameData captureFrame() override;
    void close() override;
    std::string name() const override;
    std::unique_ptr<FrameWriter> makeWriter(const std::string &basePath, Logger &logger) override;

private:
    CameraConfig _config;
    std::string _label;
    bool _running{false};
    Logger &_logger;
};
