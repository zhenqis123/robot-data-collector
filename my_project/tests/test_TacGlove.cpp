#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "TacGlove.h"
#include "Logger.h"

static std::string testDir()
{
    auto dir = (std::filesystem::temp_directory_path() / "tacglove_tests").string();
    std::filesystem::create_directories(dir);
    return dir;
}

TEST(TacGloveTest, SimulatedTacGloveInitializesWithMode)
{
    Logger logger(testDir());

    // 测试 Both 模式
    auto bothGlove = createTacGlove("Simulated", logger);
    ASSERT_TRUE(bothGlove != nullptr);
    EXPECT_TRUE(bothGlove->initialize("TacGlove#Test", TacGloveMode::Both));
    EXPECT_EQ(bothGlove->name(), "TacGlove#Test");
    EXPECT_EQ(bothGlove->mode(), TacGloveMode::Both);
    bothGlove->close();

    // 测试 LeftOnly 模式
    auto leftGlove = createTacGlove("Simulated", logger);
    ASSERT_TRUE(leftGlove != nullptr);
    EXPECT_TRUE(leftGlove->initialize("TacGlove_left#Test", TacGloveMode::LeftOnly));
    EXPECT_EQ(leftGlove->mode(), TacGloveMode::LeftOnly);
    leftGlove->close();

    // 测试 RightOnly 模式
    auto rightGlove = createTacGlove("Simulated", logger);
    ASSERT_TRUE(rightGlove != nullptr);
    EXPECT_TRUE(rightGlove->initialize("TacGlove_right#Test", TacGloveMode::RightOnly));
    EXPECT_EQ(rightGlove->mode(), TacGloveMode::RightOnly);
    rightGlove->close();
}

TEST(TacGloveTest, BothModeProducesTwoValidVectors)
{
    Logger logger(testDir());

    auto glove = createTacGlove("Simulated", logger);
    ASSERT_TRUE(glove != nullptr);
    ASSERT_TRUE(glove->initialize("TacGlove#Both", TacGloveMode::Both));

    auto now = std::chrono::system_clock::now();
    int64_t deviceTs = 12345;
    TacGloveDualFrameData frame = glove->captureFrame(now, deviceTs);

    // 验证双手数据都存在且有效
    EXPECT_EQ(frame.leftFrame.data.size(), 137u);
    EXPECT_EQ(frame.rightFrame.data.size(), 137u);
    EXPECT_FALSE(frame.leftFrame.isMissing);
    EXPECT_FALSE(frame.rightFrame.isMissing);

    // 检查数据范围 (0.0 ~ 1.0)
    for (const auto &val : frame.leftFrame.data)
    {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
    for (const auto &val : frame.rightFrame.data)
    {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }

    glove->close();
}

TEST(TacGloveTest, LeftOnlyModeProducesLeftDataAndMissingRight)
{
    Logger logger(testDir());

    auto glove = createTacGlove("Simulated", logger);
    ASSERT_TRUE(glove != nullptr);
    ASSERT_TRUE(glove->initialize("TacGlove#LeftOnly", TacGloveMode::LeftOnly));

    auto now = std::chrono::system_clock::now();
    int64_t deviceTs = 12345;
    TacGloveDualFrameData frame = glove->captureFrame(now, deviceTs);

    // 左手有有效数据
    EXPECT_EQ(frame.leftFrame.data.size(), 137u);
    EXPECT_FALSE(frame.leftFrame.isMissing);
    for (const auto &val : frame.leftFrame.data)
    {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }

    // 右手标记为缺失，数据全为 -1
    EXPECT_EQ(frame.rightFrame.data.size(), 137u);
    EXPECT_TRUE(frame.rightFrame.isMissing);
    for (const auto &val : frame.rightFrame.data)
    {
        EXPECT_EQ(val, SimulatedTacGlove::kMissingValue);
    }

    glove->close();
}

TEST(TacGloveTest, RightOnlyModeProducesRightDataAndMissingLeft)
{
    Logger logger(testDir());

    auto glove = createTacGlove("Simulated", logger);
    ASSERT_TRUE(glove != nullptr);
    ASSERT_TRUE(glove->initialize("TacGlove#RightOnly", TacGloveMode::RightOnly));

    auto now = std::chrono::system_clock::now();
    int64_t deviceTs = 12345;
    TacGloveDualFrameData frame = glove->captureFrame(now, deviceTs);

    // 左手标记为缺失，数据全为 -1
    EXPECT_EQ(frame.leftFrame.data.size(), 137u);
    EXPECT_TRUE(frame.leftFrame.isMissing);
    for (const auto &val : frame.leftFrame.data)
    {
        EXPECT_EQ(val, SimulatedTacGlove::kMissingValue);
    }

    // 右手有有效数据
    EXPECT_EQ(frame.rightFrame.data.size(), 137u);
    EXPECT_FALSE(frame.rightFrame.isMissing);
    for (const auto &val : frame.rightFrame.data)
    {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }

    glove->close();
}

TEST(TacGloveTest, DualWriterCreatesMatrixFilesForBothHands)
{
    auto dir = testDir() + "/writer_test_dual";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    Logger logger(dir);

    // 使用 Both 模式创建单个手套实例
    auto glove = createTacGlove("Simulated", logger);
    ASSERT_TRUE(glove != nullptr);
    ASSERT_TRUE(glove->initialize("TacGlove#0", TacGloveMode::Both));

    auto writer = glove->makeWriter(dir, logger);
    ASSERT_TRUE(writer != nullptr);

    // 写入几帧数据
    const int numFrames = 5;
    for (int i = 0; i < numFrames; ++i)
    {
        auto now = std::chrono::system_clock::now();
        int64_t deviceTs = 1000 + i;

        TacGloveDualFrameData frame = glove->captureFrame(now, deviceTs);
        EXPECT_TRUE(writer->write(frame));
    }

    // 释放 writer 以触发 finalize
    writer.reset();

    // 检查左手目录结构
    auto leftDir = std::filesystem::path(dir) / "tactile_left";
    EXPECT_TRUE(std::filesystem::exists(leftDir));
    EXPECT_TRUE(std::filesystem::exists(leftDir / "tactile_data.bin"));
    EXPECT_TRUE(std::filesystem::exists(leftDir / "tactile_data.csv"));
    EXPECT_TRUE(std::filesystem::exists(leftDir / "timestamps.csv"));
    EXPECT_TRUE(std::filesystem::exists(leftDir / "meta.json"));

    // 检查右手目录结构
    auto rightDir = std::filesystem::path(dir) / "tactile_right";
    EXPECT_TRUE(std::filesystem::exists(rightDir));
    EXPECT_TRUE(std::filesystem::exists(rightDir / "tactile_data.bin"));
    EXPECT_TRUE(std::filesystem::exists(rightDir / "tactile_data.csv"));
    EXPECT_TRUE(std::filesystem::exists(rightDir / "timestamps.csv"));
    EXPECT_TRUE(std::filesystem::exists(rightDir / "meta.json"));

    // 验证二进制文件大小（N * 137 * sizeof(float)）
    auto leftBinSize = std::filesystem::file_size(leftDir / "tactile_data.bin");
    EXPECT_EQ(leftBinSize, numFrames * 137 * sizeof(float));

    auto rightBinSize = std::filesystem::file_size(rightDir / "tactile_data.bin");
    EXPECT_EQ(rightBinSize, numFrames * 137 * sizeof(float));

    // 验证 CSV 文件行数（N 行数据）
    {
        std::ifstream csvFile((leftDir / "tactile_data.csv").string());
        int lineCount = 0;
        std::string line;
        while (std::getline(csvFile, line))
        {
            if (!line.empty())
                ++lineCount;
        }
        EXPECT_EQ(lineCount, numFrames);
    }

    // 验证时间戳文件行数（1 行表头 + N 行数据）
    {
        std::ifstream tsFile((leftDir / "timestamps.csv").string());
        int lineCount = 0;
        std::string line;
        while (std::getline(tsFile, line))
        {
            if (!line.empty())
                ++lineCount;
        }
        EXPECT_EQ(lineCount, numFrames + 1);  // 表头 + 数据行
    }

    // 读取并验证 meta.json（左手有效帧）
    {
        std::ifstream metaFile((leftDir / "meta.json").string());
        ASSERT_TRUE(metaFile.is_open());
        std::string content((std::istreambuf_iterator<char>(metaFile)),
                            std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("\"hand\": \"left\"") != std::string::npos);
        EXPECT_TRUE(content.find("\"num_frames\": 5") != std::string::npos);
        EXPECT_TRUE(content.find("\"missing_frames\": 0") != std::string::npos);  // Both 模式无缺失
        EXPECT_TRUE(content.find("\"vector_dimension\": 137") != std::string::npos);
        EXPECT_TRUE(content.find("\"data_shape\": [5, 137]") != std::string::npos);
    }

    glove->close();
    std::filesystem::remove_all(dir);
}

TEST(TacGloveTest, LeftOnlyModeWritesMissingDataToRightFolder)
{
    auto dir = testDir() + "/writer_test_leftonly";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    Logger logger(dir);

    auto glove = createTacGlove("Simulated", logger);
    ASSERT_TRUE(glove != nullptr);
    ASSERT_TRUE(glove->initialize("TacGlove#LeftOnly", TacGloveMode::LeftOnly));

    auto writer = glove->makeWriter(dir, logger);
    ASSERT_TRUE(writer != nullptr);

    // 写入 3 帧
    const int numFrames = 3;
    for (int i = 0; i < numFrames; ++i)
    {
        auto now = std::chrono::system_clock::now();
        TacGloveDualFrameData frame = glove->captureFrame(now, 1000 + i);
        EXPECT_TRUE(writer->write(frame));
    }

    writer.reset();

    // 验证左手 meta.json（无缺失）
    auto leftDir = std::filesystem::path(dir) / "tactile_left";
    {
        std::ifstream metaFile((leftDir / "meta.json").string());
        ASSERT_TRUE(metaFile.is_open());
        std::string content((std::istreambuf_iterator<char>(metaFile)),
                            std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("\"missing_frames\": 0") != std::string::npos);
        EXPECT_TRUE(content.find("\"valid_frames\": 3") != std::string::npos);
    }

    // 验证右手 meta.json（全部缺失）
    auto rightDir = std::filesystem::path(dir) / "tactile_right";
    {
        std::ifstream metaFile((rightDir / "meta.json").string());
        ASSERT_TRUE(metaFile.is_open());
        std::string content((std::istreambuf_iterator<char>(metaFile)),
                            std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("\"missing_frames\": 3") != std::string::npos);
        EXPECT_TRUE(content.find("\"valid_frames\": 0") != std::string::npos);
    }

    // 验证右手数据全为 -1
    {
        std::ifstream binFile((rightDir / "tactile_data.bin").string(), std::ios::binary);
        ASSERT_TRUE(binFile.is_open());
        for (int f = 0; f < numFrames; ++f)
        {
            for (int i = 0; i < 137; ++i)
            {
                float val;
                binFile.read(reinterpret_cast<char*>(&val), sizeof(float));
                EXPECT_EQ(val, SimulatedTacGlove::kMissingValue);
            }
        }
    }

    glove->close();
    std::filesystem::remove_all(dir);
}

TEST(TacGloveTest, HandEnumConversion)
{
    // 测试枚举转字符串
    EXPECT_EQ(tacGloveHandToString(TacGloveHand::Left), "left");
    EXPECT_EQ(tacGloveHandToString(TacGloveHand::Right), "right");

    // 测试字符串转枚举
    EXPECT_EQ(tacGloveHandFromString("left"), TacGloveHand::Left);
    EXPECT_EQ(tacGloveHandFromString("Left"), TacGloveHand::Left);
    EXPECT_EQ(tacGloveHandFromString("LEFT"), TacGloveHand::Left);
    EXPECT_EQ(tacGloveHandFromString("L"), TacGloveHand::Left);
    EXPECT_EQ(tacGloveHandFromString("l"), TacGloveHand::Left);
    EXPECT_EQ(tacGloveHandFromString("right"), TacGloveHand::Right);
    EXPECT_EQ(tacGloveHandFromString("Right"), TacGloveHand::Right);
    EXPECT_EQ(tacGloveHandFromString("unknown"), TacGloveHand::Right);  // 默认右手
}

TEST(TacGloveTest, ModeEnumConversion)
{
    // 测试枚举转字符串
    EXPECT_EQ(tacGloveModeToString(TacGloveMode::LeftOnly), "left_only");
    EXPECT_EQ(tacGloveModeToString(TacGloveMode::RightOnly), "right_only");
    EXPECT_EQ(tacGloveModeToString(TacGloveMode::Both), "both");
}

TEST(TacGloveTest, UnknownTypeDefaultsToSimulated)
{
    Logger logger(testDir());
    auto tacGlove = createTacGlove("UnknownType", logger);
    ASSERT_TRUE(tacGlove != nullptr);
    EXPECT_TRUE(tacGlove->initialize("TacGlove#Default", TacGloveMode::Both));
    tacGlove->close();
}

// ============================================================================
// LocalTacGlove Tests
// ============================================================================

TEST(LocalTacGloveTest, CreateLocalTacGloveType)
{
    Logger logger(testDir());

    // 测试创建 Local 类型
    auto glove = createTacGlove("Local", logger);
    ASSERT_TRUE(glove != nullptr);

    // 初始化（即使没有 Python 进程也应该成功）
    EXPECT_TRUE(glove->initialize("LocalTacGlove#Test", TacGloveMode::Both));
    EXPECT_EQ(glove->name(), "LocalTacGlove#Test");
    EXPECT_EQ(glove->mode(), TacGloveMode::Both);

    glove->close();
}

TEST(LocalTacGloveTest, LocalTacGloveReturnsMissingWhenNotConnected)
{
    Logger logger(testDir());

    auto glove = createTacGlove("Local", logger);
    ASSERT_TRUE(glove != nullptr);
    ASSERT_TRUE(glove->initialize("LocalTacGlove#NoConnection", TacGloveMode::Both));

    // 没有 Python 进程连接时，应该返回缺失数据
    auto now = std::chrono::system_clock::now();
    TacGloveDualFrameData frame = glove->captureFrame(now, 12345);

    // 两只手都应该标记为缺失
    EXPECT_TRUE(frame.leftFrame.isMissing);
    EXPECT_TRUE(frame.rightFrame.isMissing);

    // 数据应该是 137 维的 -1
    EXPECT_EQ(frame.leftFrame.data.size(), 137u);
    EXPECT_EQ(frame.rightFrame.data.size(), 137u);

    for (const auto &val : frame.leftFrame.data)
    {
        EXPECT_EQ(val, LocalTacGlove::kMissingValue);
    }
    for (const auto &val : frame.rightFrame.data)
    {
        EXPECT_EQ(val, LocalTacGlove::kMissingValue);
    }

    glove->close();
}

TEST(LocalTacGloveTest, LocalTacGloveDualWriter)
{
    auto dir = testDir() + "/local_writer_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    Logger logger(dir);

    auto glove = createTacGlove("Local", logger);
    ASSERT_TRUE(glove != nullptr);
    ASSERT_TRUE(glove->initialize("LocalTacGlove#WriterTest", TacGloveMode::Both));

    auto writer = glove->makeWriter(dir, logger);
    ASSERT_TRUE(writer != nullptr);

    // 写入几帧（都是缺失数据，因为没有 Python 连接）
    for (int i = 0; i < 3; ++i)
    {
        auto now = std::chrono::system_clock::now();
        TacGloveDualFrameData frame = glove->captureFrame(now, 1000 + i);
        EXPECT_TRUE(writer->write(frame));
    }

    writer.reset();

    // 验证文件创建
    auto leftDir = std::filesystem::path(dir) / "tactile_left";
    auto rightDir = std::filesystem::path(dir) / "tactile_right";

    EXPECT_TRUE(std::filesystem::exists(leftDir / "tactile_data.bin"));
    EXPECT_TRUE(std::filesystem::exists(rightDir / "tactile_data.bin"));

    glove->close();
    std::filesystem::remove_all(dir);
}
