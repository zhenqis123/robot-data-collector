#include <gtest/gtest.h>

#include <filesystem>

#include "CameraInterface.h"
#include "Logger.h"

static std::string tempDir()
{
    auto dir = (std::filesystem::temp_directory_path() / "data_collector_tests").string();
    std::filesystem::create_directories(dir);
    return dir;
}

TEST(CameraInterfaceTest, SimulatedCameraProducesFrames)
{
    Logger logger(tempDir());
    CameraConfig config;
    config.type = "RGB";
    config.id = 0;
    config.width = 320;
    config.height = 240;
    config.frameRate = 15;
    config.color.width = 320;
    config.color.height = 240;
    config.color.frameRate = 15;
    auto camera = createCamera(config, logger);
    ASSERT_TRUE(camera->initialize(config));
    FrameData frame = camera->captureFrame();
    camera->close();
    EXPECT_FALSE(frame.image.empty());
    EXPECT_FALSE(frame.cameraId.empty());
}
