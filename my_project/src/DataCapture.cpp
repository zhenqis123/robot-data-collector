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
                         ArucoTracker *arucoTracker,
                         double displayFpsLimit)
    : _storage(storage),
      _preview(preview),
      _logger(logger),
      _arucoTracker(arucoTracker),
      _displayFpsLimit(displayFpsLimit)
{
    if (_displayFpsLimit > 0.0)
    {
        _displayInterval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / _displayFpsLimit));
    }
    for (auto &spec : devices)
    {
        auto ctx = std::make_unique<DeviceContext>();
        ctx->device = std::move(spec.device);
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
        ctx->writer = ctx->device ? ctx->device->makeWriter(_storage.basePath(), _logger) : nullptr;
        if (ctx->writer && ctx->device && ctx->stats)
        {
            auto stats = ctx->stats;
            auto cameraId = ctx->device->name();
            ctx->writer->setWriteCallback([this, stats, cameraId]() {
                if (auto fps = stats->recordWrite())
                    _preview.updateCameraStats(cameraId, *fps);
            });
        }
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
        if (ctx->captureThread.joinable())
            ctx->captureThread.join();
        if (ctx->displayThread.joinable())
            ctx->displayThread.join();
        if (ctx->storageThread.joinable())
            ctx->storageThread.join();
        if (ctx->device)
            ctx->device->close();
        ctx->writer.reset();
    }
}

void DataCapture::captureLoop(DeviceContext *ctx)
{
    while (_running.load())
    {
        FrameData frame = ctx->device->captureFrame();
        if (!frame.hasImage())
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
        if (_arucoTracker)
            _arucoTracker->submit(frame);
    }
}

void DataCapture::startRecording(const std::string &captureName,
                                 const std::string &subject,
                                 const std::string &basePath)
{
    if (!_running.load())
        return;
    for (auto &ctxPtr : _devices)
        ctxPtr->dropWarned = false;
    std::vector<CaptureMetadata> metas;
    metas.reserve(_devices.size());
    for (const auto &ctxPtr : _devices)
    {
        if (ctxPtr->device)
            metas.push_back(ctxPtr->device->captureMetadata());
    }
    _storage.beginRecording(captureName, subject, basePath, metas);
    for (auto &ctxPtr : _devices)
    {
        if (ctxPtr->device)
            ctxPtr->writer = ctxPtr->device->makeWriter(_storage.basePath(), _logger);
        if (ctxPtr->writer && ctxPtr->device && ctxPtr->stats)
        {
            auto stats = ctxPtr->stats;
            auto cameraId = ctxPtr->device->name();
            ctxPtr->writer->setWriteCallback([this, stats, cameraId]() {
                if (auto fps = stats->recordWrite())
                    _preview.updateCameraStats(cameraId, *fps);
            });
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
        if (_displayFpsLimit > 0.0)
        {
            const auto now = std::chrono::steady_clock::now();
            if (ctx->lastDisplay.time_since_epoch().count() != 0 &&
                now - ctx->lastDisplay < _displayInterval)
            {
                continue;
            }
            ctx->lastDisplay = now;
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
        {
            std::unique_lock<std::mutex> lock(ctx->storageMutex);
            ctx->storageCv.wait(lock, [ctx]() { return !ctx->storageQueue.empty() || !ctx->storageRunning; });
            if (!ctx->storageRunning && ctx->storageQueue.empty())
                break;
            item = std::move(ctx->storageQueue.front());
            ctx->storageQueue.pop();
        }

        if (ctx->writer)
            ctx->writer->write(item.frame);
    }
}
