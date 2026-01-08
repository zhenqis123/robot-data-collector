#include "TacGlove.h"

#include <cctype>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>

// POSIX IPC headers
#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Logger.h"

// IPC 常量定义
namespace TacGloveIPC
{
    const char *kShmName = "/tacglove_ipc";
    const char *kSemLeftName = "/tacglove_left_sem";
    const char *kSemRightName = "/tacglove_right_sem";
}

namespace
{
/**
 * @brief 矩阵格式的 TacGlove 数据写入器
 *
 * 将所有帧数据存储为 N×137 矩阵文件
 * 目录结构：basePath/tactile_left/ 或 basePath/tactile_right/
 * 文件格式：tactile_data.bin (二进制) 和 tactile_data.csv (可读)
 */
class TacGloveMatrixWriter : public TacGloveFrameWriter
{
public:
    TacGloveMatrixWriter(const std::string &deviceId, const std::string &basePath,
                         TacGloveHand hand, Logger &logger)
        : _deviceId(deviceId), _basePath(basePath), _hand(hand), _logger(logger)
    {
    }

    ~TacGloveMatrixWriter() override
    {
        finalize();
    }

    bool write(const TacGloveFrameData &frame) override
    {
        if (frame.data.empty() || _basePath.empty())
            return false;

        std::lock_guard<std::mutex> lock(_mutex);

        // 创建手性目录：tactile_left 或 tactile_right（与相机文件夹同级）
        const std::string handDir = "tactile_" + tacGloveHandToString(_hand);
        const auto tactileDir = std::filesystem::path(_basePath) / handDir;
        std::error_code ec;
        std::filesystem::create_directories(tactileDir, ec);
        if (ec)
        {
            _logger.error("Failed to create TacGlove directory: %s", ec.message().c_str());
            return false;
        }

        // 初始化文件（首次写入时）
        if (!_initialized)
        {
            // 二进制文件
            auto binPath = tactileDir / "tactile_data.bin";
            _binStream.open(binPath.string(), std::ios::out | std::ios::binary | std::ios::trunc);

            // CSV 文件（可读格式）
            auto csvPath = tactileDir / "tactile_data.csv";
            _csvStream.open(csvPath.string(), std::ios::out | std::ios::trunc);

            // 时间戳文件
            auto tsPath = tactileDir / "timestamps.csv";
            _tsStream.open(tsPath.string(), std::ios::out | std::ios::trunc);
            if (_tsStream.is_open())
            {
                _tsStream << "frame_index,timestamp_iso,timestamp_ms,device_timestamp_ms,is_missing\n";
            }

            if (!_binStream.is_open() || !_csvStream.is_open())
            {
                _logger.error("Failed to open TacGlove data files");
                return false;
            }

            _initialized = true;
            _logger.info("TacGloveMatrixWriter initialized: %s -> %s",
                         _deviceId.c_str(), tactileDir.string().c_str());
        }

        // 仅当数据有效（非缺失）时才写入矩阵文件
        if (!frame.isMissing)
        {
            // 写入二进制数据（直接写入 137 个 float）
            _binStream.write(reinterpret_cast<const char *>(frame.data.data()),
                             frame.data.size() * sizeof(float));

            // 写入 CSV 数据（一行 137 个值，逗号分隔）
            for (size_t i = 0; i < frame.data.size(); ++i)
            {
                if (i > 0)
                    _csvStream << ",";
                _csvStream << std::fixed << std::setprecision(6) << frame.data[i];
            }
            _csvStream << "\n";
        }

        // 写入时间戳
        if (_tsStream.is_open())
        {
            const auto iso = toIso(frame.timestamp);
            const auto tsMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  frame.timestamp.time_since_epoch())
                                  .count();
            _tsStream << _frameCount << "," << iso << "," << tsMs << ","
                      << frame.deviceTimestampMs << "," << (frame.isMissing ? "true" : "false") << "\n";
        }

        ++_frameCount;
        if (frame.isMissing)
            ++_missingCount;
        return true;
    }

    void finalize()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_initialized)
            return;

        // 刷新并关闭所有文件
        if (_binStream.is_open())
        {
            _binStream.flush();
            _binStream.close();
        }
        if (_csvStream.is_open())
        {
            _csvStream.flush();
            _csvStream.close();
        }
        if (_tsStream.is_open())
        {
            _tsStream.flush();
            _tsStream.close();
        }

        // 写入元数据文件
        const std::string handDir = "tactile_" + tacGloveHandToString(_hand);
        const auto tactileDir = std::filesystem::path(_basePath) / handDir;
        auto metaPath = tactileDir / "meta.json";
        std::ofstream metaStream(metaPath.string());
        if (metaStream.is_open())
        {
            metaStream << "{\n";
            metaStream << "  \"device_id\": \"" << _deviceId << "\",\n";
            metaStream << "  \"hand\": \"" << tacGloveHandToString(_hand) << "\",\n";
            metaStream << "  \"num_frames\": " << _frameCount << ",\n";
            metaStream << "  \"missing_frames\": " << _missingCount << ",\n";
            metaStream << "  \"valid_frames\": " << (_frameCount - _missingCount) << ",\n";
            metaStream << "  \"full_sequence_length\": " << _frameCount << ",\n";
            metaStream << "  \"vector_dimension\": " << SimulatedTacGlove::kVectorDimension << ",\n";
            metaStream << "  \"missing_value\": " << SimulatedTacGlove::kMissingValue << ",\n";
            // 数据形状改为 [valid_frames, dimension]，因为只存储有效帧
            metaStream << "  \"data_shape\": [" << (_frameCount - _missingCount) << ", "
                       << SimulatedTacGlove::kVectorDimension << "],\n";
            metaStream << "  \"is_sparse\": true,\n";
            metaStream << "  \"binary_format\": \"float32_le\",\n";
            metaStream << "  \"files\": {\n";
            metaStream << "    \"binary\": \"tactile_data.bin\",\n";
            metaStream << "    \"csv\": \"tactile_data.csv\",\n";
            metaStream << "    \"timestamps\": \"timestamps.csv\"\n";
            metaStream << "  }\n";
            metaStream << "}\n";
            metaStream.close();
        }

        _logger.info("TacGloveMatrixWriter finalized: %s, frames=%zu (valid=%zu, missing=%zu)",
                     _deviceId.c_str(), _frameCount, _frameCount - _missingCount, _missingCount);
        _initialized = false;
    }

private:
    static std::string toIso(const std::chrono::system_clock::time_point &ts)
    {
        const auto time = std::chrono::system_clock::to_time_t(ts);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &time);
#else
        localtime_r(&time, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(ts.time_since_epoch()) % 1000;
        oss << "." << std::setw(3) << std::setfill('0') << ms.count() << "Z";
        return oss.str();
    }

    std::string _deviceId;
    std::string _basePath;
    TacGloveHand _hand;
    Logger &_logger;
    std::ofstream _binStream;
    std::ofstream _csvStream;
    std::ofstream _tsStream;
    size_t _frameCount{0};
    size_t _missingCount{0};
    bool _initialized{false};
    std::mutex _mutex;
};

} // namespace

// ============================================================================
// Writer Factory Functions
// ============================================================================

std::unique_ptr<TacGloveFrameWriter> makeTacGloveWriter(const std::string &deviceId,
                                                        const std::string &basePath,
                                                        TacGloveHand hand,
                                                        Logger &logger)
{
    return std::make_unique<TacGloveMatrixWriter>(deviceId, basePath, hand, logger);
}

std::unique_ptr<TacGloveDualWriter> makeTacGloveDualWriter(const std::string &deviceId,
                                                           const std::string &basePath,
                                                           Logger &logger)
{
    auto leftWriter = makeTacGloveWriter(deviceId, basePath, TacGloveHand::Left, logger);
    auto rightWriter = makeTacGloveWriter(deviceId, basePath, TacGloveHand::Right, logger);
    return std::make_unique<TacGloveDualWriter>(std::move(leftWriter), std::move(rightWriter));
}

// ============================================================================
// TacGloveDualWriter Implementation
// ============================================================================

TacGloveDualWriter::TacGloveDualWriter(std::unique_ptr<TacGloveFrameWriter> leftWriter,
                                       std::unique_ptr<TacGloveFrameWriter> rightWriter)
    : _leftWriter(std::move(leftWriter)), _rightWriter(std::move(rightWriter))
{
}

bool TacGloveDualWriter::write(const TacGloveDualFrameData &frame)
{
    bool leftOk = _leftWriter && _leftWriter->write(frame.leftFrame);
    bool rightOk = _rightWriter && _rightWriter->write(frame.rightFrame);
    return leftOk && rightOk;
}

// ============================================================================
// SimulatedTacGlove Implementation
// ============================================================================

SimulatedTacGlove::SimulatedTacGlove(Logger &logger)
    : _logger(logger)
{
}

bool SimulatedTacGlove::initialize(const std::string &deviceId, TacGloveMode mode)
{
    _mode = mode;
    const std::string modeStr = tacGloveModeToString(mode);
    _deviceId = deviceId.empty() ? ("TacGlove_" + modeStr + "#0") : deviceId;
    _initialized = true;
    _logger.info("SimulatedTacGlove initialized: %s (mode=%s, dimension=%d)",
                 _deviceId.c_str(), modeStr.c_str(), kVectorDimension);
    return true;
}

std::vector<float> SimulatedTacGlove::generateRandomVector()
{
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(kVectorDimension);
    for (int i = 0; i < kVectorDimension; ++i)
    {
        data[i] = dist(gen);
    }
    return data;
}

std::vector<float> SimulatedTacGlove::generateMissingVector()
{
    return std::vector<float>(kVectorDimension, kMissingValue);
}

TacGloveDualFrameData SimulatedTacGlove::captureFrame(
    const std::chrono::system_clock::time_point &timestamp,
    int64_t deviceTimestampMs)
{
    TacGloveDualFrameData dualFrame;
    if (!_initialized)
    {
        _logger.warn("SimulatedTacGlove captureFrame called before initialize");
        return dualFrame;
    }

    dualFrame.timestamp = timestamp;
    dualFrame.deviceTimestampMs = deviceTimestampMs;
    dualFrame.deviceId = _deviceId;

    // 左手数据
    dualFrame.leftFrame.timestamp = timestamp;
    dualFrame.leftFrame.deviceTimestampMs = deviceTimestampMs;
    dualFrame.leftFrame.deviceId = _deviceId;
    dualFrame.leftFrame.hand = TacGloveHand::Left;

    // 右手数据
    dualFrame.rightFrame.timestamp = timestamp;
    dualFrame.rightFrame.deviceTimestampMs = deviceTimestampMs;
    dualFrame.rightFrame.deviceId = _deviceId;
    dualFrame.rightFrame.hand = TacGloveHand::Right;

    // 根据模式生成数据
    switch (_mode)
    {
    case TacGloveMode::Both:
        // 双手模式：两只手都有真实数据
        dualFrame.leftFrame.data = generateRandomVector();
        dualFrame.leftFrame.isMissing = false;
        dualFrame.rightFrame.data = generateRandomVector();
        dualFrame.rightFrame.isMissing = false;
        break;

    case TacGloveMode::LeftOnly:
        // 仅左手模式：左手有数据，右手填充 -1
        dualFrame.leftFrame.data = generateRandomVector();
        dualFrame.leftFrame.isMissing = false;
        dualFrame.rightFrame.data = generateMissingVector();
        dualFrame.rightFrame.isMissing = true;
        break;

    case TacGloveMode::RightOnly:
        // 仅右手模式：右手有数据，左手填充 -1
        dualFrame.leftFrame.data = generateMissingVector();
        dualFrame.leftFrame.isMissing = true;
        dualFrame.rightFrame.data = generateRandomVector();
        dualFrame.rightFrame.isMissing = false;
        break;
    }

    return dualFrame;
}

void SimulatedTacGlove::close()
{
    _initialized = false;
    _logger.info("SimulatedTacGlove closed: %s", _deviceId.c_str());
}

std::string SimulatedTacGlove::name() const
{
    return _deviceId;
}

TacGloveMode SimulatedTacGlove::mode() const
{
    return _mode;
}

std::unique_ptr<TacGloveDualWriter> SimulatedTacGlove::makeWriter(const std::string &basePath,
                                                                  Logger &logger)
{
    return makeTacGloveDualWriter(_deviceId, basePath, logger);
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<TacGloveInterface> createTacGlove(const std::string &type, Logger &logger)
{
    if (type == "Simulated" || type.empty())
    {
        return std::make_unique<SimulatedTacGlove>(logger);
    }
    if (type == "Local" || type == "local")
    {
        return std::make_unique<LocalTacGlove>(logger);
    }
    logger.warn("Unknown TacGlove type '%s', defaulting to Simulated", type.c_str());
    return std::make_unique<SimulatedTacGlove>(logger);
}

// ============================================================================
// LocalTacGlove Implementation
// ============================================================================

LocalTacGlove::LocalTacGlove(Logger &logger)
    : _logger(logger)
{
    resetOffsets();
}

LocalTacGlove::~LocalTacGlove()
{
    close();
}

bool LocalTacGlove::openSharedMemory()
{
    using namespace TacGloveIPC;

    // 打开共享内存（只读模式，由 Python 端创建）
    _shmFd = shm_open(kShmName, O_RDWR, 0666);
    if (_shmFd == -1)
    {
        _logger.warn("LocalTacGlove: Cannot open shared memory '%s': %s",
                     kShmName, strerror(errno));
        return false;
    }

    // 映射共享内存
    _shmPtr = mmap(nullptr, kShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, _shmFd, 0);
    if (_shmPtr == MAP_FAILED)
    {
        _logger.error("LocalTacGlove: Failed to mmap shared memory: %s", strerror(errno));
        ::close(_shmFd);
        _shmFd = -1;
        _shmPtr = nullptr;
        return false;
    }

    // 验证 magic 和 version
    const auto *header = static_cast<const uint8_t *>(_shmPtr);
    uint32_t magic;
    uint16_t version;
    std::memcpy(&magic, header, sizeof(magic));
    std::memcpy(&version, header + 4, sizeof(version));

    if (magic != kMagic)
    {
        _logger.error("LocalTacGlove: Invalid magic number: 0x%08X (expected 0x%08X)",
                      magic, kMagic);
        closeSharedMemory();
        return false;
    }
    if (version != kVersion)
    {
        _logger.warn("LocalTacGlove: Version mismatch: %d (expected %d)", version, kVersion);
    }

    // 打开信号量
    _semLeft = sem_open(kSemLeftName, 0);
    if (_semLeft == SEM_FAILED)
    {
        _logger.warn("LocalTacGlove: Cannot open left semaphore '%s': %s",
                     kSemLeftName, strerror(errno));
        _semLeft = nullptr;
    }

    _semRight = sem_open(kSemRightName, 0);
    if (_semRight == SEM_FAILED)
    {
        _logger.warn("LocalTacGlove: Cannot open right semaphore '%s': %s",
                     kSemRightName, strerror(errno));
        _semRight = nullptr;
    }

    _logger.info("LocalTacGlove: Shared memory connected: %s", kShmName);
    return true;
}

void LocalTacGlove::closeSharedMemory()
{
    if (_semLeft != nullptr)
    {
        sem_close(static_cast<sem_t *>(_semLeft));
        _semLeft = nullptr;
    }
    if (_semRight != nullptr)
    {
        sem_close(static_cast<sem_t *>(_semRight));
        _semRight = nullptr;
    }
    if (_shmPtr != nullptr && _shmPtr != MAP_FAILED)
    {
        munmap(_shmPtr, TacGloveIPC::kShmSize);
        _shmPtr = nullptr;
    }
    if (_shmFd != -1)
    {
        ::close(_shmFd);
        _shmFd = -1;
    }
}

bool LocalTacGlove::initialize(const std::string &deviceId, TacGloveMode mode)
{
    _mode = mode;
    const std::string modeStr = tacGloveModeToString(mode);
    _deviceId = deviceId.empty() ? ("LocalTacGlove_" + modeStr + "#0") : deviceId;

    // 重置偏移量
    resetOffsets();

    // 尝试打开共享内存
    if (!openSharedMemory())
    {
        _logger.warn("LocalTacGlove: Shared memory not available, will retry on capture");
    }

    _initialized = true;
    _logger.info("LocalTacGlove initialized: %s (mode=%s, dimension=%d)",
                 _deviceId.c_str(), modeStr.c_str(), kVectorDimension);
    return true;
}

bool LocalTacGlove::isConnected() const
{
    return _shmPtr != nullptr;
}

size_t LocalTacGlove::queuedFrames(bool isLeft) const
{
    if (_shmPtr == nullptr)
        return 0;

    using namespace TacGloveIPC;
    size_t queueOffset = kHeaderSize + (isLeft ? 0 : kQueueDataSize);
    const auto *queuePtr = static_cast<const uint8_t *>(_shmPtr) + queueOffset;

    uint32_t writeIdx, readIdx;
    std::memcpy(&writeIdx, queuePtr, sizeof(writeIdx));
    std::memcpy(&readIdx, queuePtr + 4, sizeof(readIdx));

    return static_cast<size_t>(writeIdx - readIdx);
}

bool LocalTacGlove::readFrameFromQueue(bool isLeft, TacGloveFrameData &outFrame)
{
    using namespace TacGloveIPC;

    if (_shmPtr == nullptr)
    {
        // 尝试重新连接
        if (!const_cast<LocalTacGlove *>(this)->openSharedMemory())
        {
            return false;
        }
    }

    sem_t *sem = static_cast<sem_t *>(isLeft ? _semLeft : _semRight);
    size_t queueOffset = kHeaderSize + (isLeft ? 0 : kQueueDataSize);
    auto *queuePtr = static_cast<uint8_t *>(_shmPtr) + queueOffset;

    // 尝试获取信号量（非阻塞）
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_nsec += 10000000;  // 10ms timeout
    if (timeout.tv_nsec >= 1000000000)
    {
        timeout.tv_sec += 1;
        timeout.tv_nsec -= 1000000000;
    }

    int semResult = 0;
    if (sem != nullptr)
    {
        semResult = sem_timedwait(sem, &timeout);
    }

    if (semResult == -1 && errno == ETIMEDOUT)
    {
        return false;  // 超时，稍后重试
    }

    // 读取队列索引
    uint32_t writeIdx, readIdx;
    std::memcpy(&writeIdx, queuePtr, sizeof(writeIdx));
    std::memcpy(&readIdx, queuePtr + 4, sizeof(readIdx));

    // 检查是否有数据
    if (readIdx >= writeIdx)
    {
        if (sem != nullptr)
            sem_post(sem);
        return false;  // 队列为空
    }

    // 计算帧位置
    size_t frameIdx = readIdx % kQueueSize;
    size_t frameOffset = 8 + frameIdx * kFrameSize;
    const auto *framePtr = queuePtr + frameOffset;

    // 读取时间戳
    int64_t timestampMs;
    std::memcpy(&timestampMs, framePtr, sizeof(timestampMs));

    // 读取数据
    outFrame.data.resize(kVectorDimension);
    std::memcpy(outFrame.data.data(), framePtr + 8, kVectorDimension * sizeof(float));

    // 更新读索引
    uint32_t newReadIdx = readIdx + 1;
    std::memcpy(queuePtr + 4, &newReadIdx, sizeof(newReadIdx));

    if (sem != nullptr)
        sem_post(sem);

    // 设置帧元数据
    outFrame.deviceTimestampMs = timestampMs;
    outFrame.hand = isLeft ? TacGloveHand::Left : TacGloveHand::Right;
    outFrame.deviceId = _deviceId;

    // 检查是否为缺失数据（所有值都是 -1）
    bool allMissing = true;
    for (const auto &val : outFrame.data)
    {
        if (val != kMissingValue)
        {
            allMissing = false;
            break;
        }
    }
    outFrame.isMissing = allMissing;

    return true;
}

std::vector<float> LocalTacGlove::generateMissingVector()
{
    return std::vector<float>(kVectorDimension, kMissingValue);
}

TacGloveDualFrameData LocalTacGlove::captureFrame(
    const std::chrono::system_clock::time_point &timestamp,
    int64_t deviceTimestampMs)
{
    TacGloveDualFrameData dualFrame;
    if (!_initialized)
    {
        _logger.warn("LocalTacGlove captureFrame called before initialize");
        return dualFrame;
    }

    dualFrame.timestamp = timestamp;
    dualFrame.deviceTimestampMs = deviceTimestampMs;
    dualFrame.deviceId = _deviceId;

    // 初始化帧结构
    dualFrame.leftFrame.timestamp = timestamp;
    dualFrame.leftFrame.deviceTimestampMs = deviceTimestampMs;
    dualFrame.leftFrame.deviceId = _deviceId;
    dualFrame.leftFrame.hand = TacGloveHand::Left;

    dualFrame.rightFrame.timestamp = timestamp;
    dualFrame.rightFrame.deviceTimestampMs = deviceTimestampMs;
    dualFrame.rightFrame.deviceId = _deviceId;
    dualFrame.rightFrame.hand = TacGloveHand::Right;

    // 根据模式读取数据
    bool needLeft = (_mode == TacGloveMode::Both || _mode == TacGloveMode::LeftOnly);
    bool needRight = (_mode == TacGloveMode::Both || _mode == TacGloveMode::RightOnly);

    // 读取左手数据
    if (needLeft)
    {
        if (!readFrameFromQueue(true, dualFrame.leftFrame))
        {
            // 队列为空或读取失败，使用缺失数据
            dualFrame.leftFrame.data = generateMissingVector();
            dualFrame.leftFrame.isMissing = true;
        }
    }
    else
    {
        dualFrame.leftFrame.data = generateMissingVector();
        dualFrame.leftFrame.isMissing = true;
    }

    // 读取右手数据
    if (needRight)
    {
        if (!readFrameFromQueue(false, dualFrame.rightFrame))
        {
            dualFrame.rightFrame.data = generateMissingVector();
            dualFrame.rightFrame.isMissing = true;
        }
    }
    else
    {
        dualFrame.rightFrame.data = generateMissingVector();
        dualFrame.rightFrame.isMissing = true;
    }

    // 应用偏移量校准（仅对非缺失数据）
    applyOffsets(dualFrame);

    return dualFrame;
}

void LocalTacGlove::close()
{
    closeSharedMemory();
    _initialized = false;
    _logger.info("LocalTacGlove closed: %s", _deviceId.c_str());
}

std::string LocalTacGlove::name() const
{
    return _deviceId;
}

TacGloveMode LocalTacGlove::mode() const
{
    return _mode;
}

std::unique_ptr<TacGloveDualWriter> LocalTacGlove::makeWriter(const std::string &basePath,
                                                              Logger &logger)
{
    return makeTacGloveDualWriter(_deviceId, basePath, logger);
}

bool LocalTacGlove::calibrateOffsets()
{
    // 读取当前原始帧（不减偏移量的原始数据）
    TacGloveFrameData leftRaw;
    TacGloveFrameData rightRaw;
    bool leftUpdated = false;
    bool rightUpdated = false;

    const bool needLeft = (_mode == TacGloveMode::Both || _mode == TacGloveMode::LeftOnly);
    const bool needRight = (_mode == TacGloveMode::Both || _mode == TacGloveMode::RightOnly);

    // 尝试读取左手数据
    if (needLeft && readFrameFromQueue(true, leftRaw) && !leftRaw.isMissing &&
        leftRaw.data.size() == static_cast<size_t>(kVectorDimension))
    {
        leftUpdated = true;
    }

    // 尝试读取右手数据
    if (needRight && readFrameFromQueue(false, rightRaw) && !rightRaw.isMissing &&
        rightRaw.data.size() == static_cast<size_t>(kVectorDimension))
    {
        rightUpdated = true;
    }

    if (leftUpdated || rightUpdated)
    {
        std::lock_guard<std::mutex> lock(_offsetMutex);
        if (leftUpdated)
        {
            // 新偏移量 = 当前原始值 + 旧偏移量（因为读到的是已减偏移的值，需还原）
            // 但这里 readFrameFromQueue 读到的是未减偏移的原始值，所以直接用
            _offsetLeft = leftRaw.data;
        }
        if (rightUpdated)
        {
            _offsetRight = rightRaw.data;
        }
        _logger.info("LocalTacGlove: offsets calibrated (left=%s, right=%s)",
                     leftUpdated ? "yes" : "no",
                     rightUpdated ? "yes" : "no");
        return true;
    }

    _logger.warn("LocalTacGlove: calibration skipped (no valid non-missing frames available)");
    return false;
}

void LocalTacGlove::applyOffsets(TacGloveDualFrameData &frame)
{
    std::vector<float> leftOffset;
    std::vector<float> rightOffset;
    {
        std::lock_guard<std::mutex> lock(_offsetMutex);
        leftOffset = _offsetLeft;
        rightOffset = _offsetRight;
    }

    // 对非缺失的左手数据应用偏移量
    if (!frame.leftFrame.isMissing && 
        frame.leftFrame.data.size() == leftOffset.size())
    {
        for (size_t i = 0; i < frame.leftFrame.data.size(); ++i)
        {
            frame.leftFrame.data[i] -= leftOffset[i];
            if (frame.leftFrame.data[i] < 0.0f)
                frame.leftFrame.data[i] = 0.0f;
        }
    }

    // 对非缺失的右手数据应用偏移量
    if (!frame.rightFrame.isMissing && 
        frame.rightFrame.data.size() == rightOffset.size())
    {
        for (size_t i = 0; i < frame.rightFrame.data.size(); ++i)
        {
            frame.rightFrame.data[i] -= rightOffset[i];
            if (frame.rightFrame.data[i] < 0.0f)
                frame.rightFrame.data[i] = 0.0f;
        }
    }
}

void LocalTacGlove::resetOffsets()
{
    std::lock_guard<std::mutex> lock(_offsetMutex);
    _offsetLeft.assign(kVectorDimension, 0.0f);
    _offsetRight.assign(kVectorDimension, 0.0f);
}
