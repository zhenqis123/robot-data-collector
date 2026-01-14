#pragma once
#include "CameraInterface.h"
#include "ManusDataTypes.h"
#include "Logger.h"
#include <memory>
#include <string>

class ManusInterface : public CameraInterface {
public:
    virtual ~ManusInterface() = default;
    
    virtual ManusFrameData captureManusData() = 0;
};

std::unique_ptr<ManusInterface> createManusDevice(Logger& logger);
