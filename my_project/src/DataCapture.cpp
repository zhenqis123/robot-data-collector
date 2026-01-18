#include "DataCapture.h"

#include <utility>

#include "ArucoTracker.h"
#include "DataStorage.h"
#include "Logger.h"
#include "Preview.h"

DataCapture::DataCapture(std::vector<DeviceSpec> devices,
                         std::vector<TacGloveSpec> tacGloves,
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

    for (auto &spec : tacGloves)
    {
        auto ctx = std::make_unique<TacGloveContext>();
        ctx->device = std::move(spec.device);
        if (ctx->device)
        {
            auto deviceId = ctx->device->name();
            auto stats = std::make_shared<CameraStats>();
            _stats[deviceId] = stats;
            ctx->stats = stats;
            _preview.registerCameraView(deviceId, "TacGlove");
            _preview.updateCameraStats(deviceId, stats->current());
            
            _logger.info("TacGlove device added: %s", ctx->device->name().c_str());
            _tacGloves.push_back(std::move(ctx));
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
        {
            std::lock_guard<std::mutex> lock(ctx->storageMutex);
            ctx->writer.reset();
        }
        ctx->captureThread = std::thread(&DataCapture::captureLoop, this, ctx);
        ctx->displayThread = std::thread(&DataCapture::displayLoop, this, ctx);
        ctx->storageThread = std::thread(&DataCapture::storageLoop, this, ctx);
    }

    for (auto &ctxPtr : _tacGloves)
    {
        auto *ctx = ctxPtr.get();
        ctx->storageRunning = true;
        {
            std::lock_guard<std::mutex> lock(ctx->storageMutex);
            ctx->writer.reset();
        }
        ctx->storageThread = std::thread(&DataCapture::tacGloveStorageLoop, this, ctx);
    }

    _logger.info("DataCapture started with %zu cameras and %zu TacGloves",
                 _devices.size(), _tacGloves.size());
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
            ctx->storageQueue.swap(empty);
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

    for (auto &ctxPtr : _tacGloves)
    {
        auto *ctx = ctxPtr.get();
        {
            std::lock_guard<std::mutex> lock(ctx->storageMutex);
            ctx->storageRunning = false;
            std::queue<TacGloveItem> empty;
            ctx->storageQueue.swap(empty);
        }
        ctx->storageCv.notify_all();
    }

    for (auto &ctxPtr : _devices)
    {
        auto *ctx = ctxPtr.get();
        if (ctx->device)
            ctx->device->close();

        if (ctx->captureThread.joinable())
            ctx->captureThread.join();
        if (ctx->displayThread.joinable())
            ctx->displayThread.join();
        if (ctx->storageThread.joinable())
            ctx->storageThread.join();

        std::lock_guard<std::mutex> lock(ctx->storageMutex);
        ctx->writer.reset();
    }

    for (auto &ctxPtr : _tacGloves)
    {
        auto *ctx = ctxPtr.get();
        if (ctx->storageThread.joinable())
            ctx->storageThread.join();
        if (ctx->device)
            ctx->device->close();
        std::lock_guard<std::mutex> lock(ctx->storageMutex);
        ctx->writer.reset();
    }
}

void DataCapture::captureLoop(DeviceContext *ctx)
{
    while (_running.load())
    {
        FrameData frame = ctx->device->captureFrame();
        const bool hasImage = frame.hasImage();
        if (!hasImage && !frame.gloveData.has_value() && !frame.viveData.has_value() && !frame.manusData.has_value())
            continue;
        frame.cameraId = ctx->device->name();

        auto stats = ctx->stats;
        if (stats)
        {
            if (auto fps = stats->recordCapture(frame))
                _preview.updateCameraStats(frame.cameraId, *fps);
        }

        enqueueForDisplay(ctx, frame);

        captureTacGloveFrame(frame.timestamp, frame.deviceTimestampMs);

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
    for (auto &ctxPtr : _tacGloves)
        ctxPtr->dropWarned = false;

    std::vector<CaptureMetadata> metas;
    metas.reserve(_devices.size());
    for (const auto &ctxPtr : _devices)
    {
        if (ctxPtr->device)
        {
            const bool shouldRecord = recordTypes.empty() || recordTypes.count(ctxPtr->type) > 0;
            if (shouldRecord)
                metas.push_back(ctxPtr->device->captureMetadata());
        }
    }

    _storage.beginRecording(captureName, subject, basePath, metas);

    for (auto &ctxPtr : _devices)
    {
        if (!ctxPtr->device)
            continue;

        const bool shouldRecord = recordTypes.empty() || recordTypes.count(ctxPtr->type) > 0;
        std::shared_ptr<FrameWriter> writer;
        if (shouldRecord)
        {
            writer = std::move(ctxPtr->device->makeWriter(_storage.basePath(), _logger));
            if (writer && ctxPtr->stats)
            {
                auto stats = ctxPtr->stats;
                writer->setWriteCallback([this, stats](const FrameData &frame) {
                    if (auto fps = stats->recordWrite(frame))
                        _preview.updateCameraStats(frame.cameraId, *fps);
                });
            }
        }
        else
        {
            _logger.info("Skipping recording for device %s (type: %s)",
                         ctxPtr->device->name().c_str(), ctxPtr->type.c_str());
        }

        std::lock_guard<std::mutex> lock(ctxPtr->storageMutex);
        ctxPtr->writer = std::move(writer);
    }

    for (auto &ctxPtr : _tacGloves)
    {
        if (!ctxPtr->device)
            continue;
        std::shared_ptr<TacGloveDualWriter> writer =
            std::move(ctxPtr->device->makeWriter(_storage.basePath(), _logger));
        std::lock_guard<std::mutex> lock(ctxPtr->storageMutex);
        ctxPtr->writer = std::move(writer);
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

    for (auto &ctxPtr : _devices)
    {
        std::lock_guard<std::mutex> lock(ctxPtr->storageMutex);
        ctxPtr->writer.reset();
    }

    for (auto &ctxPtr : _tacGloves)
    {
        std::lock_guard<std::mutex> lock(ctxPtr->storageMutex);
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
            if (auto fps = item.stats->recordDisplay(item.frame))
                _preview.updateCameraStats(item.frame.cameraId, *fps);
        }
    }
}

void DataCapture::storageLoop(DeviceContext *ctx)
{
    while (true)
    {
        CaptureItem item;
        std::shared_ptr<FrameWriter> writer;
        {
            std::unique_lock<std::mutex> lock(ctx->storageMutex);
            ctx->storageCv.wait(lock, [ctx]() { return !ctx->storageQueue.empty() || !ctx->storageRunning; });
            if (!ctx->storageRunning && ctx->storageQueue.empty())
                break;
            item = std::move(ctx->storageQueue.front());
            ctx->storageQueue.pop();
            writer = ctx->writer;
        }

        if (writer)
            writer->write(item.frame);
    }
}

void DataCapture::captureTacGloveFrame(const std::chrono::system_clock::time_point &timestamp,
                                       int64_t deviceTimestampMs)
{
    std::lock_guard<std::mutex> lock(_tacGloveCaptureMutex);
    for (auto &ctxPtr : _tacGloves)
    {
        auto *ctx = ctxPtr.get();
        if (ctx->device)
        {
            TacGloveDualFrameData frame = ctx->device->captureFrame(timestamp, deviceTimestampMs);

            FrameData fd;
            fd.timestamp = timestamp;
            fd.deviceTimestampMs = deviceTimestampMs;
            fd.cameraId = frame.deviceId;
            fd.tacGloveData = frame;
            if (ctx->stats)
            {
                if (auto fps = ctx->stats->recordCapture(fd))
                    _preview.updateCameraStats(fd.cameraId, *fps);
            }
            _preview.showFrame(fd);

            if (_recording.load() && !_paused.load())
            {
                if (!frame.leftFrame.data.empty() || !frame.rightFrame.data.empty())
                {
                    enqueueForTacGloveStorage(ctx, frame);
                }
            }
        }
    }
}

void DataCapture::enqueueForTacGloveStorage(TacGloveContext *ctx, const TacGloveDualFrameData &frame)
{
    std::unique_lock<std::mutex> lock(ctx->storageMutex);
    if (ctx->storageQueue.size() >= _maxTacGloveQueue)
    {
        if (!ctx->dropWarned)
        {
            _logger.warn("TacGlove storage queue full for %s; dropping oldest frames",
                         frame.deviceId.c_str());
            ctx->dropWarned = true;
        }
        ctx->storageQueue.pop();
    }
    ctx->storageQueue.push({frame});
    lock.unlock();
    ctx->storageCv.notify_one();
}

void DataCapture::tacGloveStorageLoop(TacGloveContext *ctx)
{
    while (true)
    {
        TacGloveItem item;
        std::shared_ptr<TacGloveDualWriter> writer;
        {
            std::unique_lock<std::mutex> lock(ctx->storageMutex);
            ctx->storageCv.wait(lock, [ctx]() {
                return !ctx->storageQueue.empty() || !ctx->storageRunning;
            });
            if (!ctx->storageRunning && ctx->storageQueue.empty())
                break;
            item = std::move(ctx->storageQueue.front());
            ctx->storageQueue.pop();
            writer = ctx->writer;
        }

        if (writer)
            writer->write(item.frame);
    }
}

bool DataCapture::calibrateTacGloveOffsets()
{
    bool updated = false;
    for (auto &ctxPtr : _tacGloves)
    {
        if (ctxPtr && ctxPtr->device)
        {
            if (ctxPtr->device->calibrateOffsets())
                updated = true;
        }
    }
    if (updated)
    {
        _logger.info("TacGlove calibration: offsets refreshed");
    }
    else
    {
        _logger.warn("TacGlove calibration: no valid data available");
    }
    return updated;
}
