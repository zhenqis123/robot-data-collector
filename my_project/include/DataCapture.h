#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>
#include <functional>
#include <unordered_set>

#include "CameraInterface.h"
#include "CameraStats.h"

class ArucoTracker;

class Preview;
class DataStorage;
class Logger;

struct DeviceSpec
{
    std::unique_ptr<CameraInterface> device;
    std::string type;
};

class DataCapture
{
public:
    DataCapture(std::vector<DeviceSpec> devices,
                DataStorage &storage,
                Preview &preview,
                Logger &logger,
                ArucoTracker *arucoTracker);
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

    struct DeviceContext
    {
        std::unique_ptr<CameraInterface> device;
        std::string type;
        std::shared_ptr<CameraStats> stats;
        std::thread captureThread;
        std::thread displayThread;
        std::thread storageThread;
        std::unique_ptr<FrameWriter> writer;
        std::queue<DisplayItem> displayQueue;
        std::queue<CaptureItem> storageQueue;
        std::condition_variable storageCv;
        std::condition_variable displayCv;
        std::mutex storageMutex;
        std::mutex displayMutex;
        bool storageRunning{false};
        bool displayRunning{false};
        bool dropWarned{false};
    };

    std::vector<std::unique_ptr<DeviceContext>> _devices;
    std::unordered_map<std::string, std::shared_ptr<CameraStats>> _stats;
    size_t _maxStorageQueue{200};
    size_t _maxDisplayQueue{100};
    std::atomic<bool> _running{false};
    std::atomic<bool> _recording{false};
    std::atomic<bool> _paused{false};
    DataStorage &_storage;
    Preview &_preview;
    Logger &_logger;
    ArucoTracker *_arucoTracker{nullptr};

    void captureLoop(DeviceContext *ctx);
    void displayLoop(DeviceContext *ctx);
    void storageLoop(DeviceContext *ctx);
    void enqueueForStorage(DeviceContext *ctx, const FrameData &frame);
    void enqueueForDisplay(DeviceContext *ctx, const FrameData &frame);
};
