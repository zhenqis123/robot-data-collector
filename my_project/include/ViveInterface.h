#pragma once
#include "CameraInterface.h"
#include "ViveDataTypes.h"
#include "Logger.h"
#include <memory>
#include <string>

class ViveInterface : public CameraInterface {
public:
    virtual ~ViveInterface() = default;
    
    virtual ViveFrameData captureViveData() = 0;
};

std::unique_ptr<ViveInterface> createViveDevice(Logger& logger);
