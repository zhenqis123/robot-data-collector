#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>
#include <chrono>

#include "CameraInterface.h"
#include "CameraStats.h"
#include "TacGlove.h"

class ArucoTracker;
class Preview;
class DataStorage;
class Logger;

struct DeviceSpec
{
    std::unique_ptr<CameraInterface> device;
    std::string type;
};

struct TacGloveSpec
{
    std::unique_ptr<TacGloveInterface> device;
    std::string deviceId;
    TacGloveMode mode{TacGloveMode::Both};
};

class DataCapture
{
public:
    DataCapture(std::vector<DeviceSpec> devices,
                std::vector<TacGloveSpec> tacGloves,
                DataStorage &storage,
                Preview &preview,
                Logger &logger,
                ArucoTracker *arucoTracker,
                double displayFpsLimit = 0.0);
    ~DataCapture();

    bool start();
    void stop();
    bool isRunning() const { return _running.load(); }
    void startRecording(const std::string &captureName,
                        const std::string &subject,
                        const std::string &basePath,
                        const std::unordered_set<std::string> &recordTypes = {});
    void stopRecording();
    void pauseRecording();
    void resumeRecording();
    bool isRecording() const { return _recording.load(); }
    bool isPaused() const { return _paused.load(); }
    std::vector<std::string> deviceIds() const;

    /**
     * @brief 校准 TacGlove 偏移量
     *
     * 以当前原始数据为新的偏移量基准，用于消除手套噪声
     * @return 至少有一个设备成功校准返回 true
     */
    bool calibrateTacGloveOffsets();

private:
    struct CaptureItem
    {
        FrameData frame;
        std::shared_ptr<CameraStats> stats;
    };

    struct DisplayItem
    {
        FrameData frame;
        std::shared_ptr<CameraStats> stats;
    };

    struct TacGloveItem
    {
        TacGloveDualFrameData frame;
    };

    struct TacGloveContext
    {
        std::unique_ptr<TacGloveInterface> device;
        std::thread storageThread;
        std::shared_ptr<TacGloveDualWriter> writer;
        std::queue<TacGloveItem> storageQueue;
        std::condition_variable storageCv;
        std::mutex storageMutex;
        bool storageRunning{false};
        bool dropWarned{false};
    };

    struct DeviceContext
    {
        std::unique_ptr<CameraInterface> device;
        std::string type;
        std::shared_ptr<CameraStats> stats;
        std::thread captureThread;
        std::thread displayThread;
        std::thread storageThread;
        std::shared_ptr<FrameWriter> writer;
        std::queue<DisplayItem> displayQueue;
        std::queue<CaptureItem> storageQueue;
        std::condition_variable storageCv;
        std::condition_variable displayCv;
        std::mutex storageMutex;
        std::mutex displayMutex;
        bool storageRunning{false};
        bool displayRunning{false};
        bool dropWarned{false};
        std::chrono::steady_clock::time_point lastDisplay;
    };

    std::vector<std::unique_ptr<DeviceContext>> _devices;
    std::unordered_map<std::string, std::shared_ptr<CameraStats>> _stats;
    size_t _maxStorageQueue{200};
    size_t _maxDisplayQueue{100};
    std::vector<std::unique_ptr<TacGloveContext>> _tacGloves;
    size_t _maxTacGloveQueue{500};
    std::atomic<bool> _running{false};
    std::atomic<bool> _recording{false};
    std::atomic<bool> _paused{false};
    DataStorage &_storage;
    Preview &_preview;
    Logger &_logger;
    ArucoTracker *_arucoTracker{nullptr};
    double _displayFpsLimit{0.0};
    std::chrono::steady_clock::duration _displayInterval{};

    void captureLoop(DeviceContext *ctx);
    void displayLoop(DeviceContext *ctx);
    void storageLoop(DeviceContext *ctx);
    void enqueueForStorage(DeviceContext *ctx, const FrameData &frame);
    void enqueueForDisplay(DeviceContext *ctx, const FrameData &frame);

    // TacGlove 相关方法
    void tacGloveStorageLoop(TacGloveContext *ctx);
    void enqueueForTacGloveStorage(TacGloveContext *ctx, const TacGloveDualFrameData &frame);
    void captureTacGloveFrame(const std::chrono::system_clock::time_point &timestamp,
                              int64_t deviceTimestampMs);
};
