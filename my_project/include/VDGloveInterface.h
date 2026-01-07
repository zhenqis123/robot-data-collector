#pragma once
#include "CameraInterface.h"
#include "VDGloveDataTypes.h"
#include "Logger.h"
#include <memory>
#include <string>

// 继承 CameraInterface 以便集成到 DataCapture 框架中
class VDGloveInterface : public CameraInterface {
public:
    virtual ~VDGloveInterface() = default;
    
    // 继承自 CameraInterface 的纯虚函数需要被实现
    // initialize(const CameraConfig&)
    // captureFrame()
    // close()
    // name()
    // makeWriter()

    // 保留特定类型的接口（如果需要）
    virtual VDGloveFrameData captureGloveData() = 0;
};

std::unique_ptr<VDGloveInterface> createGloveDevice(const std::string& type, Logger& logger);
