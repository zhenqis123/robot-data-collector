#include "DataCapture.h"

#include <utility>

#include "DataStorage.h"
#include "Logger.h"
#include "Preview.h"
#include "ArucoTracker.h"

DataCapture::DataCapture(std::vector<DeviceSpec> devices,
                         DataStorage &storage,
                         Preview &preview,
                         Logger &logger,
                         ArucoTracker *arucoTracker)
    : _storage(storage),
      _preview(preview),
      _logger(logger),
      _arucoTracker(arucoTracker)
{
    for (auto &spec : devices)
    {
        auto ctx = std::make_unique<DeviceContext>();
        ctx->device = std::move(spec.device);
        ctx->type = spec.type;
        if (ctx->device)
        {
            auto cameraId = ctx->device->name();
            auto stats = std::make_shared<CameraStats>();
            _stats[cameraId] = stats;
            ctx->stats = stats;
            _preview.registerCameraView(cameraId, spec.type);
            _preview.updateCameraStats(cameraId, stats->current());
            _devices.push_back(std::move(ctx));
        }
    }
}

DataCapture::~DataCapture()
{
    stop();
}

std::vector<std::string> DataCapture::deviceIds() const
{
    std::vector<std::string> ids;
    ids.reserve(_devices.size());
    for (const auto &ctxPtr : _devices)
    {
        if (ctxPtr && ctxPtr->device)
            ids.push_back(ctxPtr->device->name());
    }
    return ids;
}

bool DataCapture::start()
{
    if (_running.exchange(true))
    {
        _logger.warn("DataCapture already running");
        return false;
    }

    for (auto &ctxPtr : _devices)
    {
        auto *ctx = ctxPtr.get();
        ctx->storageRunning = true;
        ctx->displayRunning = true;
        // Writer is created on startRecording, not here.
        ctx->writer = nullptr;
        ctx->captureThread = std::thread(&DataCapture::captureLoop, this, ctx);
        ctx->displayThread = std::thread(&DataCapture::displayLoop, this, ctx);
        ctx->storageThread = std::thread(&DataCapture::storageLoop, this, ctx);
    }

    _logger.info("DataCapture started with %zu devices", _devices.size());
    return true;
}

void DataCapture::stop()
{
    if (!_running.exchange(false))
        return;

    _recording = false;
    _paused = false;
    _logger.info("Stopping DataCapture");

    for (auto &ctxPtr : _devices)
    {
        auto *ctx = ctxPtr.get();
        {
            std::lock_guard<std::mutex> lock(ctx->storageMutex);
            ctx->storageRunning = false;
            std::queue<CaptureItem> empty;
            ctx->storageQueue.swap(empty); // drop pending frames to accelerate shutdown
        }
        ctx->storageCv.notify_all();
        {
            std::lock_guard<std::mutex> lock(ctx->displayMutex);
            ctx->displayRunning = false;
            std::queue<DisplayItem> empty;
            ctx->displayQueue.swap(empty);
        }
        ctx->displayCv.notify_all();
    }

    for (auto &ctxPtr : _devices)
    {
        auto *ctx = ctxPtr.get();
        // Close device first to interrupt any blocking capture calls
        if (ctx->device)
            ctx->device->close();

        if (ctx->captureThread.joinable())
            ctx->captureThread.join();
        if (ctx->displayThread.joinable())
            ctx->displayThread.join();
        if (ctx->storageThread.joinable())
            ctx->storageThread.join();
        
        ctx->writer.reset();
    }
}

void DataCapture::captureLoop(DeviceContext *ctx)
{
    while (_running.load())
    {
        FrameData frame = ctx->device->captureFrame();
        const bool hasImage = !frame.image.empty();
        // Allow frames with glove or vive data even if image is empty
        if (!hasImage && !frame.gloveData.has_value() && !frame.viveData.has_value())
            continue;
        frame.cameraId = ctx->device->name();
        auto stats = ctx->stats;
        if (stats)
        {
            if (auto fps = stats->recordCapture())
                _preview.updateCameraStats(frame.cameraId, *fps);
        }
        enqueueForDisplay(ctx, frame);
        if (_recording.load() && !_paused.load())
        {
            enqueueForStorage(ctx, frame);
        }
        if (_arucoTracker && hasImage)
            _arucoTracker->submit(frame);
    }
}

void DataCapture::startRecording(const std::string &captureName,
                                 const std::string &subject,
                                 const std::string &basePath,
                                 const std::unordered_set<std::string> &recordTypes)
{
    if (!_running.load())
        return;
    for (auto &ctxPtr : _devices)
        ctxPtr->dropWarned = false;
    
    std::vector<CaptureMetadata> metas;
    // Only include metadata for devices we are actually recording
    for (const auto &ctxPtr : _devices)
    {
        if (ctxPtr->device)
        {
            bool shouldRecord = recordTypes.empty() || recordTypes.count(ctxPtr->type);
            if (shouldRecord) {
                metas.push_back(ctxPtr->device->captureMetadata());
            }
        }
    }
    
    _storage.beginRecording(captureName, subject, basePath, metas);
    
    for (auto &ctxPtr : _devices)
    {
        if (ctxPtr->device)
        {
            std::lock_guard<std::mutex> lock(ctxPtr->storageMutex); // Protect writer assignment
            bool shouldRecord = recordTypes.empty() || recordTypes.count(ctxPtr->type);
            if (shouldRecord) {
                ctxPtr->writer = ctxPtr->device->makeWriter(_storage.basePath(), _logger);
            } else {
                ctxPtr->writer = nullptr;
                _logger.info("Skipping recording for device %s (type: %s)", ctxPtr->device->name().c_str(), ctxPtr->type.c_str());
            }
        }
    }
    _recording = true;
    _paused = false;
    _logger.info("Recording started");
}

void DataCapture::stopRecording()
{
    _recording = false;
    _paused = false;
    _storage.endRecording();
    _logger.info("Recording stopped");
    
    // Clean up writers
    for (auto &ctxPtr : _devices)
    {
        std::lock_guard<std::mutex> lock(ctxPtr->storageMutex); // Protect writer reset
        ctxPtr->writer.reset();
    }
}

void DataCapture::pauseRecording()
{
    if (!_recording.load())
        return;
    _paused = true;
    _logger.info("Recording paused");
}

void DataCapture::resumeRecording()
{
    if (!_recording.load())
        return;
    _paused = false;
    _logger.info("Recording resumed");
}

void DataCapture::enqueueForStorage(DeviceContext *ctx, const FrameData &frame)
{
    std::unique_lock<std::mutex> lock(ctx->storageMutex);
    
    // If no writer is configured (e.g. recording disabled for this device), skip queuing
    if (!ctx->writer)
        return;

    if (ctx->storageQueue.size() >= _maxStorageQueue)
    {
        if (!ctx->dropWarned)
        {
            _logger.warn("Storage queue full for %s; dropping oldest frames", frame.cameraId.c_str());
            ctx->dropWarned = true;
        }
        ctx->storageQueue.pop();
    }
    ctx->storageQueue.push({frame, ctx->stats});
    lock.unlock();
    ctx->storageCv.notify_one();
}

void DataCapture::enqueueForDisplay(DeviceContext *ctx, const FrameData &frame)
{
    std::unique_lock<std::mutex> lock(ctx->displayMutex);
    if (ctx->displayQueue.size() >= _maxDisplayQueue)
    {
        ctx->displayQueue.pop();
    }
    ctx->displayQueue.push({frame, ctx->stats});
    lock.unlock();
    ctx->displayCv.notify_one();
}

void DataCapture::displayLoop(DeviceContext *ctx)
{
    while (true)
    {
        DisplayItem item;
        {
            std::unique_lock<std::mutex> lock(ctx->displayMutex);
            ctx->displayCv.wait(lock, [ctx]() { return !ctx->displayQueue.empty() || !ctx->displayRunning; });
            if (!ctx->displayRunning && ctx->displayQueue.empty())
                break;
            item = std::move(ctx->displayQueue.front());
            ctx->displayQueue.pop();
        }
        _preview.showFrame(item.frame);
        if (item.stats)
        {
            if (auto fps = item.stats->recordDisplay())
                _preview.updateCameraStats(item.frame.cameraId, *fps);
        }
    }
}

void DataCapture::storageLoop(DeviceContext *ctx)
{
    while (true)
    {
        CaptureItem item;
        std::shared_ptr<FrameWriter> currentWriter; // Keep writer alive during write operation
        {
            std::unique_lock<std::mutex> lock(ctx->storageMutex);
            ctx->storageCv.wait(lock, [ctx]() { return !ctx->storageQueue.empty() || !ctx->storageRunning; });
            if (!ctx->storageRunning && ctx->storageQueue.empty())
                break;
            item = std::move(ctx->storageQueue.front());
            ctx->storageQueue.pop();
            
            // Capture the writer shared_ptr while holding the lock
            currentWriter = ctx->writer;
        }

        const bool writeOk = (currentWriter && currentWriter->write(item.frame));
        if (writeOk && item.stats)
        {
            if (auto fps = item.stats->recordWrite())
                _preview.updateCameraStats(item.frame.cameraId, *fps);
        }
    }
}
