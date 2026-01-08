#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <thread>

#include "ConfigManager.h"
#include "DataCapture.h"
#include "DataStorage.h"
#include "Logger.h"
#include "Preview.h"
#include "TacGlove.h"

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

    std::vector<TacGloveSpec> tacGloves; // 空的 TacGlove 列表

    DataCapture capture(std::move(devices), std::move(tacGloves), storage, preview, logger, nullptr, 30.0);
    EXPECT_FALSE(capture.isRunning());
    EXPECT_TRUE(capture.start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    capture.stop();
    EXPECT_FALSE(capture.isRunning());
}

TEST(DataCaptureTest, TacGloveDataSavedDuringRecording)
{
    // 创建临时录制目录
    auto baseDir = captureDir() + "/tacglove_integration_test";
    std::filesystem::remove_all(baseDir);
    std::filesystem::create_directories(baseDir);

    Logger logger(baseDir);
    Preview preview(logger);
    DataStorage storage(baseDir, logger);

    // 创建模拟相机
    CameraConfig cfg;
    cfg.type = "RGB";
    cfg.serial = "";
    cfg.id = 1;
    cfg.width = 160;
    cfg.height = 120;
    cfg.frameRate = 10;  // 10 FPS
    cfg.color.width = 160;
    cfg.color.height = 120;
    cfg.color.frameRate = 10;

    auto device = createCamera(cfg, logger);
    ASSERT_TRUE(device != nullptr);
    ASSERT_TRUE(device->initialize(cfg));
    DeviceSpec camSpec;
    camSpec.device = std::move(device);
    camSpec.type = cfg.type;
    std::vector<DeviceSpec> devices;
    devices.push_back(std::move(camSpec));

    // 创建 TacGlove（使用 Both 模式）
    std::vector<TacGloveSpec> tacGloves;

    auto tacGlove = createTacGlove("Simulated", logger);
    ASSERT_TRUE(tacGlove != nullptr);
    ASSERT_TRUE(tacGlove->initialize("TacGlove#0", TacGloveMode::Both));
    TacGloveSpec tacSpec;
    tacSpec.device = std::move(tacGlove);
    tacSpec.deviceId = "TacGlove#0";
    tacSpec.mode = TacGloveMode::Both;
    tacGloves.push_back(std::move(tacSpec));

    // 创建 DataCapture
    DataCapture capture(std::move(devices), std::move(tacGloves), storage, preview, logger, nullptr, 30.0);

    // 启动采集
    ASSERT_TRUE(capture.start());
    EXPECT_TRUE(capture.isRunning());

    // 开始录制
    auto recordPath = baseDir + "/session_001";
    capture.startRecording("test_session", "test_subject", recordPath);
    EXPECT_TRUE(capture.isRecording());

    // 等待一段时间让数据采集（约 500ms，预期 5 帧左右）
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 停止录制
    capture.stopRecording();
    EXPECT_FALSE(capture.isRecording());

    // 停止采集
    capture.stop();
    EXPECT_FALSE(capture.isRunning());

    // 验证 TacGlove 数据文件是否创建
    auto leftDir = std::filesystem::path(recordPath) / "tactile_left";
    auto rightDir = std::filesystem::path(recordPath) / "tactile_right";

    EXPECT_TRUE(std::filesystem::exists(leftDir)) << "Left tactile directory should exist";
    EXPECT_TRUE(std::filesystem::exists(rightDir)) << "Right tactile directory should exist";

    // 检查文件是否存在
    EXPECT_TRUE(std::filesystem::exists(leftDir / "tactile_data.bin"));
    EXPECT_TRUE(std::filesystem::exists(leftDir / "tactile_data.csv"));
    EXPECT_TRUE(std::filesystem::exists(leftDir / "timestamps.csv"));
    EXPECT_TRUE(std::filesystem::exists(leftDir / "meta.json"));

    EXPECT_TRUE(std::filesystem::exists(rightDir / "tactile_data.bin"));
    EXPECT_TRUE(std::filesystem::exists(rightDir / "tactile_data.csv"));
    EXPECT_TRUE(std::filesystem::exists(rightDir / "timestamps.csv"));
    EXPECT_TRUE(std::filesystem::exists(rightDir / "meta.json"));

    // 验证二进制文件大小 > 0（至少有一帧数据）
    auto leftBinSize = std::filesystem::file_size(leftDir / "tactile_data.bin");
    auto rightBinSize = std::filesystem::file_size(rightDir / "tactile_data.bin");
    EXPECT_GT(leftBinSize, 0u) << "Left tactile data should have content";
    EXPECT_GT(rightBinSize, 0u) << "Right tactile data should have content";

    // 验证左右手数据大小相同（同步捕获）
    EXPECT_EQ(leftBinSize, rightBinSize) << "Left and right should have same amount of data";

    // 验证 meta.json 内容
    {
        std::ifstream metaFile((leftDir / "meta.json").string());
        ASSERT_TRUE(metaFile.is_open());
        std::string content((std::istreambuf_iterator<char>(metaFile)),
                            std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("\"hand\": \"left\"") != std::string::npos);
        EXPECT_TRUE(content.find("\"vector_dimension\": 137") != std::string::npos);
    }

    {
        std::ifstream metaFile((rightDir / "meta.json").string());
        ASSERT_TRUE(metaFile.is_open());
        std::string content((std::istreambuf_iterator<char>(metaFile)),
                            std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("\"hand\": \"right\"") != std::string::npos);
    }

    // 清理
    std::filesystem::remove_all(baseDir);
}
