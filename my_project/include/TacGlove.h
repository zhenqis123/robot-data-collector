#pragma once

#include <iostream>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

class Logger;

/**
 * @brief 手套工作模式
 */
enum class TacGloveMode
{
    LeftOnly,   // 仅左手（右手数据填充 -1）
    RightOnly,  // 仅右手（左手数据填充 -1）
    Both        // 双手模式
};

/**
 * @brief 手套手性（用于标识单个数据帧）
 */
enum class TacGloveHand
{
    Left,   // 左手
    Right   // 右手
};

/**
 * @brief 将 TacGloveMode 转换为字符串
 */
inline std::string tacGloveModeToString(TacGloveMode mode)
{
    switch (mode)
    {
    case TacGloveMode::LeftOnly:
        return "left_only";
    case TacGloveMode::RightOnly:
        return "right_only";
    case TacGloveMode::Both:
        return "both";
    }
    return "unknown";
}

/**
 * @brief 从字符串解析 TacGloveMode
 */
inline TacGloveMode tacGloveModeFromString(const std::string &str)
{
    if (str == "left_only" || str == "LeftOnly" || str == "left")
        return TacGloveMode::LeftOnly;
    if (str == "right_only" || str == "RightOnly" || str == "right")
        return TacGloveMode::RightOnly;
    return TacGloveMode::Both;
}

/**
 * @brief 将 TacGloveHand 转换为字符串
 */
inline std::string tacGloveHandToString(TacGloveHand hand)
{
    return hand == TacGloveHand::Left ? "left" : "right";
}

/**
 * @brief 从字符串解析 TacGloveHand
 */
inline TacGloveHand tacGloveHandFromString(const std::string &str)
{
    if (str == "left" || str == "Left" || str == "LEFT" || str == "L" || str == "l")
        return TacGloveHand::Left;
    return TacGloveHand::Right;
}

/**
 * @brief TacGlove 单手帧数据结构
 *
 * 包含 137 维传感器向量和同步时间戳
 */
struct TacGloveFrameData
{
    std::vector<float> data;  // 137 维向量
    std::chrono::system_clock::time_point timestamp;
    int64_t deviceTimestampMs{0};
    std::string deviceId;
    TacGloveHand hand{TacGloveHand::Left};  // 手套手性（左/右手）
    bool isMissing{false};  // 是否为缺失数据（填充 -1）

    void printData(size_t k = 137) const  // 调试用，打印数据内容
    {
        std::cout << "[" << tacGloveHandToString(hand) << "] ";
        if (isMissing)
        {
            std::cout << "(missing)";
        }
        else
        {
            for (size_t i = 0; i < k && i < data.size(); ++i)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << data[i];
            }
        }
        std::cout << std::endl;
    }
};

/**
 * @brief TacGlove 双手帧数据结构
 *
 * 包含左手和右手各 137 维向量
 */
struct TacGloveDualFrameData
{
    TacGloveFrameData leftFrame;   // 左手数据
    TacGloveFrameData rightFrame;  // 右手数据
    std::chrono::system_clock::time_point timestamp;
    int64_t deviceTimestampMs{0};
    std::string deviceId;

    void printData(size_t k = 5) const
    {
        std::cout << "=== TacGlove Frame (ts=" << deviceTimestampMs << "ms) ===" << std::endl;
        leftFrame.printData(k);
        rightFrame.printData(k);
    }
};

/**
 * @brief TacGlove 数据写入器接口（单手）
 */
struct TacGloveFrameWriter
{
    virtual ~TacGloveFrameWriter() = default;
    virtual bool write(const TacGloveFrameData &frame) = 0;
};

/**
 * @brief TacGlove 双手数据写入器
 *
 * 内部维护两个 TacGloveFrameWriter，分别写入左手和右手数据
 */
class TacGloveDualWriter
{
public:
    TacGloveDualWriter(std::unique_ptr<TacGloveFrameWriter> leftWriter,
                       std::unique_ptr<TacGloveFrameWriter> rightWriter);

    bool write(const TacGloveDualFrameData &frame);

private:
    std::unique_ptr<TacGloveFrameWriter> _leftWriter;
    std::unique_ptr<TacGloveFrameWriter> _rightWriter;
};

/**
 * @brief 创建单手 TacGlove 数据写入器
 */
std::unique_ptr<TacGloveFrameWriter> makeTacGloveWriter(const std::string &deviceId,
                                                        const std::string &basePath,
                                                        TacGloveHand hand,
                                                        Logger &logger);

/**
 * @brief 创建双手 TacGlove 数据写入器
 */
std::unique_ptr<TacGloveDualWriter> makeTacGloveDualWriter(const std::string &deviceId,
                                                           const std::string &basePath,
                                                           Logger &logger);

// 保持向后兼容的别名
inline std::unique_ptr<TacGloveFrameWriter> makeTacGloveCsvWriter(const std::string &deviceId,
                                                                  const std::string &basePath,
                                                                  TacGloveHand hand,
                                                                  Logger &logger)
{
    return makeTacGloveWriter(deviceId, basePath, hand, logger);
}

/**
 * @brief TacGlove 设备接口（支持双手模式）
 *
 * 提供触觉手套数据采集的抽象接口，每帧采集左右手各 137 维向量数据
 */
class TacGloveInterface
{
public:
    virtual ~TacGloveInterface() = default;

    /**
     * @brief 初始化设备
     * @param deviceId 设备标识符
     * @param mode 工作模式（左手/右手/双手）
     * @return 初始化成功返回 true
     */
    virtual bool initialize(const std::string &deviceId, TacGloveMode mode) = 0;

    /**
     * @brief 使用给定时间戳捕获一帧双手数据
     *
     * 时间戳由外部（DataCapture）传入，确保与相机帧同步
     * 根据模式，缺失的手会用 -1 填充
     *
     * @param timestamp 系统时间戳
     * @param deviceTimestampMs 设备时间戳（毫秒）
     * @return TacGloveDualFrameData 包含左右手各 137 维向量的帧数据
     */
    virtual TacGloveDualFrameData captureFrame(
        const std::chrono::system_clock::time_point &timestamp,
        int64_t deviceTimestampMs) = 0;

    /**
     * @brief 关闭设备
     */
    virtual void close() = 0;

    /**
     * @brief 获取设备名称
     */
    virtual std::string name() const = 0;

    /**
     * @brief 获取工作模式
     */
    virtual TacGloveMode mode() const = 0;

    /**
     * @brief 创建双手数据写入器
     */
    virtual std::unique_ptr<TacGloveDualWriter> makeWriter(const std::string &basePath,
                                                           Logger &logger) = 0;
};

/**
 * @brief 模拟 TacGlove 设备实现
 *
 * 使用随机数生成 137 维向量，用于测试和开发
 * 支持三种模式：仅左手、仅右手、双手
 */
class SimulatedTacGlove : public TacGloveInterface
{
public:
    explicit SimulatedTacGlove(Logger &logger);

    bool initialize(const std::string &deviceId, TacGloveMode mode) override;
    TacGloveDualFrameData captureFrame(
        const std::chrono::system_clock::time_point &timestamp,
        int64_t deviceTimestampMs) override;
    void close() override;
    std::string name() const override;
    TacGloveMode mode() const override;
    std::unique_ptr<TacGloveDualWriter> makeWriter(const std::string &basePath,
                                                   Logger &logger) override;

    static constexpr int kVectorDimension = 137;
    static constexpr float kMissingValue = -1.0f;

private:
    /**
     * @brief 生成随机 137 维向量
     */
    std::vector<float> generateRandomVector();

    /**
     * @brief 生成缺失数据向量（全 -1）
     */
    std::vector<float> generateMissingVector();

    Logger &_logger;
    std::string _deviceId;
    TacGloveMode _mode{TacGloveMode::Both};
    bool _initialized{false};
};

/**
 * @brief 创建 TacGlove 设备实例
 *
 * @param type 设备类型，支持 "Simulated" 或 "Local"
 * @param logger 日志记录器
 * @return TacGlove 设备实例
 */
std::unique_ptr<TacGloveInterface> createTacGlove(const std::string &type, Logger &logger);

// ============================================================================
// LocalTacGlove: 通过共享内存从 Python 进程接收真实手套数据
// ============================================================================

/**
 * @brief IPC 共享内存协议常量
 */
namespace TacGloveIPC
{
    constexpr int kVectorDimension = 137;
    constexpr int kQueueSize = 64;
    constexpr int kFrameSize = 8 + kVectorDimension * 4;  // timestamp + data
    constexpr int kQueueDataSize = 8 + kQueueSize * kFrameSize;
    constexpr int kHeaderSize = 8;
    constexpr int kShmSize = kHeaderSize + 2 * kQueueDataSize;
    constexpr uint32_t kMagic = 0x54414347;  // "TACG"
    constexpr uint16_t kVersion = 1;
    constexpr float kMissingValue = -1.0f;

    extern const char *kShmName;
    extern const char *kSemLeftName;
    extern const char *kSemRightName;
}

/**
 * @brief 本地 TacGlove 设备实现
 *
 * 通过共享内存从 Python 进程（TacDataCollector.py）接收真实手套数据
 * 使用环形队列缓冲，支持左右手独立的数据流
 */
class LocalTacGlove : public TacGloveInterface
{
public:
    explicit LocalTacGlove(Logger &logger);
    ~LocalTacGlove();

    bool initialize(const std::string &deviceId, TacGloveMode mode) override;
    TacGloveDualFrameData captureFrame(
        const std::chrono::system_clock::time_point &timestamp,
        int64_t deviceTimestampMs) override;
    void close() override;
    std::string name() const override;
    TacGloveMode mode() const override;
    std::unique_ptr<TacGloveDualWriter> makeWriter(const std::string &basePath,
                                                   Logger &logger) override;

    /**
     * @brief 检查共享内存是否可用
     */
    bool isConnected() const;

    /**
     * @brief 获取队列中待读取的帧数
     */
    size_t queuedFrames(bool isLeft) const;

    static constexpr int kVectorDimension = 137;
    static constexpr float kMissingValue = -1.0f;

private:
    /**
     * @brief 从共享内存队列读取一帧数据
     * @param isLeft true 读取左手队列，false 读取右手队列
     * @param outFrame 输出帧数据
     * @return 是否成功读取
     */
    bool readFrameFromQueue(bool isLeft, TacGloveFrameData &outFrame);

    /**
     * @brief 生成缺失数据向量（全 -1）
     */
    std::vector<float> generateMissingVector();

    /**
     * @brief 打开共享内存
     */
    bool openSharedMemory();

    /**
     * @brief 关闭共享内存
     */
    void closeSharedMemory();

    Logger &_logger;
    std::string _deviceId;
    TacGloveMode _mode{TacGloveMode::Both};
    bool _initialized{false};

    // 共享内存相关
    int _shmFd{-1};
    void *_shmPtr{nullptr};
    void *_semLeft{nullptr};   // sem_t*
    void *_semRight{nullptr};  // sem_t*
};
