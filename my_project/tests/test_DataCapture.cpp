#include <gtest/gtest.h>

#include <filesystem>
#include <thread>

#include "ConfigManager.h"
#include "DataCapture.h"
#include "DataStorage.h"
#include "Logger.h"
#include "Preview.h"

static std::string captureDir()
{
    auto dir = (std::filesystem::temp_directory_path() / "data_capture_tests").string();
    std::filesystem::create_directories(dir);
    return dir;
}

TEST(DataCaptureTest, StartsAndStopsThreads)
{
    Logger logger(captureDir());
    Preview preview(logger);
    DataStorage storage(captureDir(), logger);

    CameraConfig cfg;
    cfg.type = "RGB";
    cfg.serial = "";
    cfg.id = 1;
    cfg.width = 160;
    cfg.height = 120;
    cfg.frameRate = 5;
    cfg.color.width = 160;
    cfg.color.height = 120;
    cfg.color.frameRate = 5;

    auto device = createCamera(cfg, logger);
    ASSERT_TRUE(device != nullptr);
    ASSERT_TRUE(device->initialize(cfg));
    DeviceSpec spec;
    spec.device = std::move(device);
    spec.type = cfg.type;
    std::vector<DeviceSpec> devices;
    devices.push_back(std::move(spec));

    DataCapture capture(std::move(devices), storage, preview, logger, nullptr);
    EXPECT_FALSE(capture.isRunning());
    EXPECT_TRUE(capture.start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    capture.stop();
    EXPECT_FALSE(capture.isRunning());
}
