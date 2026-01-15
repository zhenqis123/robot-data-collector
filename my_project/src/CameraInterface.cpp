#include "CameraInterface.h"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <hdf5.h>
#include <librealsense2/rs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "IntrinsicsManager.h"
#include "Logger.h"
#include "NetworkDevice.h"
#include "VDGloveInterface.h"
#include "ViveInterface.h"
#include "ManusInterface.h"

namespace
{
int ensurePositive(int value, int fallback)
{
    return value > 0 ? value : fallback;
}

class SimulatedCamera : public CameraInterface
{
public:
    SimulatedCamera(const std::string &model, Logger &logger)
        : _model(model), _logger(logger)
    {
    }

    bool initialize(const CameraConfig &config) override
    {
        _config = config;
        _initialized = true;
        _label = _model + "#" + std::to_string(config.id);
        _logger.info("%s initialized (id=%d, %dx%d @ %d FPS)",
                     _model.c_str(), config.id, config.width, config.height, config.frameRate);
        return true;
    }

    FrameData captureFrame() override
    {
        FrameData data;
        if (!_initialized)
        {
            _logger.warn("%s captureFrame called before initialize", _model.c_str());
            return data;
        }

        cv::Mat frame(ensurePositive(_config.color.height, ensurePositive(_config.height, 480)),
                      ensurePositive(_config.color.width, ensurePositive(_config.width, 640)),
                      CV_8UC3,
                      _model == "RealSense" ? cv::Scalar(20, 40, 200) : cv::Scalar(40, 200, 20));
        cv::putText(frame, _label, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {255, 255, 255}, 2);

        data.image = frame;
        data.timestamp = std::chrono::system_clock::now();
        data.deviceTimestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count();
        data.cameraId = _label;
        data.colorFormat = "BGR";

        if (_config.frameRate > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / _config.frameRate));
        }

        return data;
    }

    void close() override
    {
        _initialized = false;
        _logger.info("%s closed", _model.c_str());
    }

    std::string name() const override { return _label; }

    std::vector<CameraConfig::StreamConfig> getAvailableResolutions() const override
    {
        return {
            {640, 480, 30, 0, 8000, CameraConfig::StreamConfig::StreamType::Color},
            {1280, 720, 30, 0, 8000, CameraConfig::StreamConfig::StreamType::Color},
            {1920, 1080, 30, 0, 8000, CameraConfig::StreamConfig::StreamType::Color}
        };
    }

    std::unique_ptr<FrameWriter> makeWriter(const std::string &basePath, Logger &logger) override;
    CaptureMetadata captureMetadata() const override;

private:
    std::string _model;
    std::string _label;
    CameraConfig _config;
    bool _initialized{false};
    Logger &_logger;
};

class RealSenseCamera : public CameraInterface
{
public:
    explicit RealSenseCamera(Logger &logger)
        : _logger(logger), _align(RS2_STREAM_COLOR)
    {
    }

    bool initialize(const CameraConfig &config) override
    {
        _config = config;
        _alignEnabled = config.alignDepth;
        const int colorWidth = ensurePositive(config.color.width, ensurePositive(config.width, 640));
        const int colorHeight = ensurePositive(config.color.height, ensurePositive(config.height, 480));
        const int colorFps = ensurePositive(config.color.frameRate, ensurePositive(config.frameRate, 30));
        const int depthWidth = ensurePositive(config.depth.width, colorWidth);
        const int depthHeight = ensurePositive(config.depth.height, colorHeight);
        const int depthFps = ensurePositive(config.depth.frameRate, colorFps);

        try
        {
            // Ensure any previous pipeline is stopped before starting a new one.
            try
            {
                _pipeline.stop();
            }
            catch (const rs2::error &)
            {
                // ignore stop errors
            }

            _rsConfig.disable_all_streams();
            if (!config.serial.empty())
            {
                _rsConfig.enable_device(config.serial);
                _logger.info("Binding RealSense to serial %s", config.serial.c_str());
            }
            _rsConfig.enable_stream(RS2_STREAM_COLOR, colorWidth, colorHeight, RS2_FORMAT_YUYV, colorFps);
            _logger.info("RealSense color stream: %dx%d @ %d FPS", colorWidth, colorHeight, colorFps);
            _rsConfig.enable_stream(RS2_STREAM_DEPTH, depthWidth, depthHeight, RS2_FORMAT_Z16, depthFps);
            _logger.info("RealSense depth stream: %dx%d @ %d FPS", depthWidth, depthHeight, depthFps);
            auto profile = _pipeline.start(_rsConfig);
            _logger.info("RealSense pipeline started");
            _device = profile.get_device();
            auto dev = _device;
            if (dev.supports(RS2_CAMERA_INFO_NAME))
                _logger.info("RealSense device connected: %s", dev.get_info(RS2_CAMERA_INFO_NAME));
            for (auto &&sensor : dev.query_sensors())
            {
                if (sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED))
                {
                    sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 1.f);
                    _logger.info("Enabled global time sync for sensor %s",
                                 sensor.get_info(RS2_CAMERA_INFO_NAME));
                }
                if (auto depthSensor = sensor.as<rs2::depth_sensor>())
                {
                    _depthScale = depthSensor.get_depth_scale();
                }
            }
            if (dev.supports(RS2_CAMERA_INFO_NAME))
                _cameraName = dev.get_info(RS2_CAMERA_INFO_NAME);
            if (dev.supports(RS2_CAMERA_INFO_SERIAL_NUMBER))
                _serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
            if (_serial.empty())
                _identifier = _cameraName + "#" + std::to_string(config.id);
            else
                _identifier = "RealSense#" + _serial;
            _running = true;
            _logger.info("RealSense started: %s (serial=%s)", _cameraName.c_str(), _serial.c_str());
            auto colorProfile = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
            auto depthProfile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            auto &intrMgr = IntrinsicsManager::instance();
            if (auto existing = intrMgr.find(_identifier, "color", colorWidth, colorHeight))
            {
                _logger.info("Loaded cached intrinsics for %s color stream (%dx%d)",
                             _identifier.c_str(), colorWidth, colorHeight);
            }
            else
            {
                auto intr = colorProfile.get_intrinsics();
                StreamIntrinsics data;
                data.stream = "color";
                data.width = intr.width;
                data.height = intr.height;
                data.fx = intr.fx;
                data.fy = intr.fy;
                data.cx = intr.ppx;
                data.cy = intr.ppy;
                data.coeffs.assign(intr.coeffs, intr.coeffs + 5);
                intrMgr.save(_identifier, data);
            }
            if (auto existingDepth = intrMgr.find(_identifier, "depth", depthWidth, depthHeight))
            {
                _logger.info("Loaded cached intrinsics for %s depth stream (%dx%d)",
                             _identifier.c_str(), depthWidth, depthHeight);
            }
            else
            {
                auto intr = depthProfile.get_intrinsics();
                StreamIntrinsics data;
                data.stream = "depth";
                data.width = intr.width;
                data.height = intr.height;
                data.fx = intr.fx;
                data.fy = intr.fy;
                data.cx = intr.ppx;
                data.cy = intr.ppy;
                data.coeffs.assign(intr.coeffs, intr.coeffs + 5);
                intrMgr.save(_identifier, data);
            }
            rs2_extrinsics extrinsics = depthProfile.get_extrinsics_to(colorProfile);
            _metadata = {};
            _metadata.deviceId = _identifier;
            _metadata.model = _cameraName;
            _metadata.serial = _serial;
            _metadata.aligned = _alignEnabled;
            _metadata.depthScale = _depthScale;
            _metadata.colorFps = colorFps;
            _metadata.depthFps = depthFps;
            _metadata.colorFormat = "YUYV";
            _metadata.depthFormat = "Z16";
            auto colorIntr = colorProfile.get_intrinsics();
            _metadata.colorIntrinsics.stream = "color";
            _metadata.colorIntrinsics.width = colorIntr.width;
            _metadata.colorIntrinsics.height = colorIntr.height;
            _metadata.colorIntrinsics.fx = colorIntr.fx;
            _metadata.colorIntrinsics.fy = colorIntr.fy;
            _metadata.colorIntrinsics.cx = colorIntr.ppx;
            _metadata.colorIntrinsics.cy = colorIntr.ppy;
            _metadata.colorIntrinsics.coeffs.assign(colorIntr.coeffs, colorIntr.coeffs + 5);
            auto depthIntr = depthProfile.get_intrinsics();
            _metadata.depthIntrinsics.stream = "depth";
            _metadata.depthIntrinsics.width = depthIntr.width;
            _metadata.depthIntrinsics.height = depthIntr.height;
            _metadata.depthIntrinsics.fx = depthIntr.fx;
            _metadata.depthIntrinsics.fy = depthIntr.fy;
            _metadata.depthIntrinsics.cx = depthIntr.ppx;
            _metadata.depthIntrinsics.cy = depthIntr.ppy;
            _metadata.depthIntrinsics.coeffs.assign(depthIntr.coeffs, depthIntr.coeffs + 5);
            std::copy(std::begin(extrinsics.rotation), std::end(extrinsics.rotation),
                      _metadata.depthToColor.rotation.begin());
            std::copy(std::begin(extrinsics.translation), std::end(extrinsics.translation),
                      _metadata.depthToColor.translation.begin());
            return true;
        }
        catch (const rs2::error &e)
        {
            _logger.error("RealSense init failed: %s", e.what());
            _running = false;
            return false;
        }
    }

    FrameData captureFrame() override
    {
        FrameData data;
        if (!_running)
            return data;

        try
        {
            rs2::frameset frames = _pipeline.wait_for_frames(15000);
            rs2::frameset processed = _alignEnabled ? _align.process(frames) : frames;
            rs2::video_frame color = processed.get_color_frame();
            rs2::depth_frame depth = processed.get_depth_frame();

            if (!color || !depth)
            {
                _logger.warn("RealSense: missing color or depth frame");
                return data;
            }

            // Check for stuck frames
            const auto currentDepthFrameNumber = depth.get_frame_number();
            if (currentDepthFrameNumber == _lastDepthFrameNumber && _lastDepthFrameNumber != 0)
            {
                _stuckDepthCount++;
                if (_stuckDepthCount % 30 == 0) // Log every ~1s (assuming 30fps)
                {
                    _logger.warn("RealSense depth frame stuck on %llu for %d frames", 
                        _lastDepthFrameNumber, _stuckDepthCount);
                }
                
                if (_stuckDepthCount > 90) // 3 seconds of stuck frames
                {
                    restartPipeline();
                    return data; // Return empty data to skip this frame
                }
            }
            else
            {
                _stuckDepthCount = 0;
                _lastDepthFrameNumber = currentDepthFrameNumber;
            }

            auto colorHolder = std::make_shared<rs2::video_frame>(color);
            auto depthHolder = std::make_shared<rs2::depth_frame>(depth);
            cv::Mat colorMat(cv::Size(colorHolder->get_width(), colorHolder->get_height()), CV_8UC2,
                             const_cast<void *>(colorHolder->get_data()), cv::Mat::AUTO_STEP);
            cv::Mat depthMat(cv::Size(depthHolder->get_width(), depthHolder->get_height()), CV_16U,
                             const_cast<void *>(depthHolder->get_data()), cv::Mat::AUTO_STEP);

            data.image = colorMat;
            // Perform deep copy for depth to ensure data stability during storage queueing.
            // This prevents issues where the SDK might reuse the underlying buffer for subsequent frames
            // while this frame is still waiting in the write queue.
            data.depth = depthMat.clone();
            data.imageOwner = colorHolder;
            data.depthOwner = nullptr; // Depth data is now owned by cv::Mat
            data.timestamp = std::chrono::system_clock::now();
            data.deviceTimestampMs = static_cast<int64_t>(color.get_timestamp());
            data.cameraId = _identifier;
            data.colorFormat = "YUYV";
        }
        catch (const rs2::error &e)
        {
            _logger.error("RealSense capture failed: %s", e.what());
            // Attempt a restart (with optional hardware reset) on timeouts or pipeline errors if still running
            if (_running)
            {
                const std::string msg = e.what();
                const bool timeout = msg.find("Frame didn't arrive") != std::string::npos ||
                                     msg.find("Timeout") != std::string::npos ||
                                     msg.find("timeout") != std::string::npos;
                try
                {
                    _pipeline.stop();
                }
                catch (const rs2::error &)
                {
                }
                try
                {
                    if (timeout && _device)
                    {
                        _logger.info("RealSense: hardware reset before restart for %s", name().c_str());
                        _device.hardware_reset();
                        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                    }
                    auto profile = _pipeline.start(_rsConfig);
                    _device = profile.get_device();
                    _logger.info("RealSense pipeline restarted for %s (timeout=%d)", name().c_str(), timeout ? 1 : 0);
                }
                catch (const rs2::error &re)
                {
                    _logger.warn("RealSense restart failed: %s", re.what());
                }
            }
        }
        return data;
    }

    void close() override
    {
        if (!_running)
            return;
        try
        {
            _pipeline.stop();
        }
        catch (const rs2::error &e)
        {
            _logger.warn("RealSense stop failed: %s", e.what());
        }
        _running = false;
        _logger.info("RealSense camera closed");
    }

    std::string name() const override { return _identifier.empty() ? "RealSense" : _identifier; }

    std::vector<CameraConfig::StreamConfig> getAvailableResolutions() const override
    {
        std::vector<CameraConfig::StreamConfig> configs;
        rs2::context ctx;
        auto list = ctx.query_devices();
        if (list.size() == 0)
            return configs;
        try
        {
            for (auto &&dev : list)
            {
                if (!_config.serial.empty() &&
                    (!dev.supports(RS2_CAMERA_INFO_SERIAL_NUMBER) ||
                     _config.serial != dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)))
                    continue;
                for (auto &&sensor : dev.query_sensors())
                {
                    for (auto &&profile : sensor.get_stream_profiles())
                    {
                        if (auto video = profile.as<rs2::video_stream_profile>())
                        {
                            CameraConfig::StreamConfig cfg;
                            cfg.width = video.width();
                            cfg.height = video.height();
                            cfg.frameRate = profile.fps();
                            cfg.streamType = profile.stream_type() == RS2_STREAM_DEPTH
                                                 ? CameraConfig::StreamConfig::StreamType::Depth
                                                 : CameraConfig::StreamConfig::StreamType::Color;
                            configs.push_back(cfg);
                        }
                    }
                }
            }
        }
        catch (const rs2::error &e)
        {
            _logger.warn("Failed to enumerate RealSense profiles: %s", e.what());
        }
        return configs;
    }

    std::unique_ptr<FrameWriter> makeWriter(const std::string &basePath, Logger &logger) override;
    CaptureMetadata captureMetadata() const override;

private:
    void restartPipeline()
    {
        _logger.warn("Restarting RealSense pipeline due to stuck frames...");
        try
        {
            _pipeline.stop();
        }
        catch (...) {}
        
        try 
        {
            if (_device)
            {
                // Optional: hardware reset
                // _device.hardware_reset(); 
                // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            }
            _pipeline.start(_rsConfig);
            _lastDepthFrameNumber = 0;
            _stuckDepthCount = 0;
            _logger.info("RealSense pipeline restarted successfully");
        }
        catch (const rs2::error &e)
        {
            _logger.error("Failed to restart pipeline: %s", e.what());
        }
    }

    Logger &_logger;
    CameraConfig _config;
    bool _alignEnabled{true};
    rs2::pipeline _pipeline;
    rs2::config _rsConfig;
    rs2::align _align;
    rs2::device _device;
    double _depthScale{0.0};
    bool _running{false};
    std::string _cameraName{"RealSense"};
    std::string _serial;
    std::string _identifier;
    CaptureMetadata _metadata;
    unsigned long long _lastDepthFrameNumber{0};
    int _stuckDepthCount{0};
};

class WebcamCamera : public CameraInterface
{
public:
    explicit WebcamCamera(Logger &logger)
        : _logger(logger)
    {
    }

    bool initialize(const CameraConfig &config) override
    {
        _config = config;
        const int index = config.id;
        const int colorWidth = ensurePositive(config.color.width, ensurePositive(config.width, 640));
        const int colorHeight = ensurePositive(config.color.height, ensurePositive(config.height, 480));
        const int colorFps = ensurePositive(config.color.frameRate, ensurePositive(config.frameRate, 30));
        _config.color.width = colorWidth;
        _config.color.height = colorHeight;
        _config.color.frameRate = colorFps;
        _capture.open(index);
        if (!_capture.isOpened())
        {
            _logger.error("Webcam %d failed to open", index);
            return false;
        }
        if (colorWidth > 0)
            _capture.set(cv::CAP_PROP_FRAME_WIDTH, colorWidth);
        if (colorHeight > 0)
            _capture.set(cv::CAP_PROP_FRAME_HEIGHT, colorHeight);
        if (colorFps > 0)
            _capture.set(cv::CAP_PROP_FPS, colorFps);
        _label = "Webcam#" + std::to_string(index);
        _running = true;
        return true;
    }

    FrameData captureFrame() override
    {
        FrameData data;
        if (!_running)
            return data;
        cv::Mat frame;
        if (!_capture.read(frame))
        {
            _logger.warn("Webcam %s read failed", _label.c_str());
            return data;
        }
        data.image = frame;
        data.timestamp = std::chrono::system_clock::now();
        data.deviceTimestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count();
        data.cameraId = _label;
        data.colorFormat = "BGR";
        return data;
    }

    void close() override
    {
        if (_capture.isOpened())
            _capture.release();
        _running = false;
    }

    std::string name() const override { return _label.empty() ? "Webcam" : _label; }

    std::unique_ptr<FrameWriter> makeWriter(const std::string &basePath, Logger &logger) override
    {
        return makeGstHdf5Writer(name(),
                                 basePath,
                                 logger,
                                 ensurePositive(_config.color.frameRate, ensurePositive(_config.frameRate, 30)),
                                 _config.depth.chunkSize,
                                 _config.color.bitrateKbps);
    }

    CaptureMetadata captureMetadata() const override
    {
        CaptureMetadata meta;
        meta.deviceId = _label;
        meta.model = "Webcam";
        meta.aligned = false;
        meta.colorFormat = "BGR8";
        meta.colorIntrinsics.stream = "color";
        meta.colorIntrinsics.width = ensurePositive(_config.color.width, ensurePositive(_config.width, 640));
        meta.colorIntrinsics.height = ensurePositive(_config.color.height, ensurePositive(_config.height, 480));
        meta.colorFps = ensurePositive(_config.color.frameRate, ensurePositive(_config.frameRate, 30));
        return meta;
    }

private:
    Logger &_logger;
    CameraConfig _config;
    cv::VideoCapture _capture;
    bool _running{false};
    std::string _label;
};
} // namespace

class GstHdf5FrameWriter : public FrameWriter
{
public:
    GstHdf5FrameWriter(const std::string &deviceId,
                       const std::string &basePath,
                       Logger &logger,
                       int colorFps,
                       int depthChunkSize,
                       int colorBitrateKbps)
        : _deviceId(deviceId),
          _basePath(basePath),
          _logger(logger),
          _colorFps(colorFps > 0 ? colorFps : 30),
          _depthChunkSize(depthChunkSize > 0 ? depthChunkSize : 0),
          _colorBitrateKbps(colorBitrateKbps > 0 ? colorBitrateKbps : 8000)
    {
        const auto sanitized = sanitize(_deviceId);
        _cameraDir = std::filesystem::path(_basePath) / sanitized;
        std::error_code ec;
        std::filesystem::create_directories(_cameraDir, ec);
        _rgbPath = _cameraDir / "rgb.mkv";
        _depthPath = _cameraDir / "depth.h5";
        _worker = std::thread(&GstHdf5FrameWriter::worker, this);
    }

    ~GstHdf5FrameWriter() override
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopped = true;
        }
        _cv.notify_all();
        if (_worker.joinable())
            _worker.join();
        closeStreams();
    }

    void setWriteCallback(std::function<void()> callback) override
    {
        std::lock_guard<std::mutex> lock(_callbackMutex);
        _writeCallback = std::move(callback);
    }

    bool write(const FrameData &frame) override
    {
        if (!frame.hasImage() || _basePath.empty())
            return false;

        FrameData copy;
        copy.image = frame.image;
        copy.depth = frame.depth;
        copy.imageOwner = frame.imageOwner;
        copy.depthOwner = frame.depthOwner;
        copy.timestamp = frame.timestamp;
        copy.deviceTimestampMs = frame.deviceTimestampMs;
        copy.cameraId = frame.cameraId;
        copy.colorFormat = frame.colorFormat;

        {
            std::lock_guard<std::mutex> lock(_mutex);
            _queue.push(std::move(copy));
        }
        _cv.notify_one();
        return true;
    }

private:
    void worker()
    {
        while (true)
        {
            FrameData frame;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _cv.wait(lock, [&]() { return !_queue.empty() || _stopped; });
                if (_stopped && _queue.empty())
                    break;
                frame = std::move(_queue.front());
                _queue.pop();
            }

            const uint64_t idx = ++_frameIndex;
            const auto t0 = std::chrono::steady_clock::now();
            const bool rgbOk = writeRgb(frame);
            const auto t1 = std::chrono::steady_clock::now();
            const bool depthOk = writeDepth(frame);
            const auto t2 = std::chrono::steady_clock::now();

            _perfRgbMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
            _perfDepthMs += std::chrono::duration<double, std::milli>(t2 - t1).count();
            _perfTotalMs += std::chrono::duration<double, std::milli>(t2 - t0).count();
            ++_perfCount;
            if (_perfCount % _perfLogEvery == 0)
            {
                const double denom = static_cast<double>(_perfLogEvery);
                _logger.info("Writer perf %s: avg total %.2f ms (rgb %.2f ms, depth %.2f ms) over %llu frames",
                             _deviceId.c_str(),
                             _perfTotalMs / denom,
                             _perfRgbMs / denom,
                             _perfDepthMs / denom,
                             static_cast<unsigned long long>(_perfLogEvery));
                _perfRgbMs = 0.0;
                _perfDepthMs = 0.0;
                _perfTotalMs = 0.0;
            }
            if (!rgbOk && !depthOk)
            {
                _logger.error("Failed to store frame for %s", _deviceId.c_str());
                continue;
            }

            const auto iso = toIso(frame.timestamp);
            const auto tsMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  frame.timestamp.time_since_epoch())
                                  .count();
            {
                std::lock_guard<std::mutex> lock(_tsMutex);
                if (!_tsStream.is_open())
                {
                    auto filePath = _cameraDir / "timestamps.csv";
                    const bool existed = std::filesystem::exists(filePath);
                    _tsStream.open(filePath.string(), std::ios::out | std::ios::app);
                    if (!existed)
                        _tsStream << "frame_index,timestamp_iso,timestamp_ms,device_timestamp_ms,rgb_path,depth_path\n";
                }
                _tsStream << idx << "," << iso << "," << tsMs << "," << frame.deviceTimestampMs << ","
                          << "rgb.mkv"
                          << "," << (frame.hasDepth() ? "depth.h5" : "") << "\n";
                ++_timestampLineCount;
                if (_timestampLineCount % _timestampFlushEvery == 0)
                    _tsStream.flush();
            }
            std::function<void()> callback;
            {
                std::lock_guard<std::mutex> lock(_callbackMutex);
                callback = _writeCallback;
            }
            if (callback)
                callback();
        }
    }

    bool writeRgb(const FrameData &frame)
    {
        if (!frame.hasImage())
            return false;
        if (_inputGstFormat.empty())
        {
            if (frame.colorFormat == "YUYV")
                _inputGstFormat = "YUY2";
            else if (frame.colorFormat.empty() && frame.imageRef().channels() == 2)
                _inputGstFormat = "YUY2";
            else
                _inputGstFormat = "BGR";
        }
        if (!_gstReady && !initGst(frame.imageRef().cols, frame.imageRef().rows))
            return false;

        cv::Mat image = frame.imageRef();
        const size_t bufferSize = image.total() * image.elemSize();
        GstBuffer *buffer = nullptr;

        if (frame.imageOwner && image.isContinuous())
        {
            auto holder = new std::shared_ptr<void>(frame.imageOwner);
            buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
                                                 const_cast<unsigned char *>(image.data),
                                                 bufferSize,
                                                 0,
                                                 bufferSize,
                                                 holder,
                                                 [](gpointer data) {
                                                     delete static_cast<std::shared_ptr<void> *>(data);
                                                 });
        }
        else
        {
            if (!image.isContinuous())
                image = image.clone();

            buffer = gst_buffer_new_allocate(nullptr, bufferSize, nullptr);
            if (!buffer)
                return false;

            GstMapInfo map;
            if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE))
            {
                gst_buffer_unref(buffer);
                return false;
            }
            std::memcpy(map.data, image.data, bufferSize);
            gst_buffer_unmap(buffer, &map);
        }

        if (!_ptsInitialized)
        {
            _firstTimestamp = frame.timestamp;
            _ptsInitialized = true;
        }
        auto delta = frame.timestamp - _firstTimestamp;
        auto ptsNs = std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count();
        if (ptsNs < 0)
            ptsNs = 0;
        GstClockTime pts = static_cast<GstClockTime>(ptsNs);
        GstClockTime dur = _colorFps > 0 ? gst_util_uint64_scale_int(1, GST_SECOND, _colorFps) : GST_CLOCK_TIME_NONE;
        GST_BUFFER_PTS(buffer) = pts;
        GST_BUFFER_DTS(buffer) = pts;
        GST_BUFFER_DURATION(buffer) = dur;

        const GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(_appsrc), buffer);
        if (ret != GST_FLOW_OK)
        {
            _logger.warn("GStreamer push failed for %s: %d", _deviceId.c_str(), ret);
            return false;
        }
        return true;
    }

    bool writeDepth(const FrameData &frame)
    {
        if (!frame.hasDepth())
            return false;
        if (!_hdf5Ready && !initHdf5(frame.depthRef().cols, frame.depthRef().rows))
            return false;

        cv::Mat depthMat = frame.depthRef();
        if (depthMat.type() != CV_16U)
        {
            if (!_depthTypeWarned)
            {
                _logger.warn("Depth frame type %d, converting to CV_16U for %s", depthMat.type(), _deviceId.c_str());
                _depthTypeWarned = true;
            }
            depthMat.convertTo(depthMat, CV_16U);
        }

        if (_depthChunkSizeUsed <= 1)
            return writeDepthBatch(depthMat.data, 1);

        if (!appendDepthToBuffer(depthMat))
            return false;
        if (_depthBuffered < static_cast<size_t>(_depthChunkSizeUsed))
            return true;

        const bool ok = writeDepthBatch(_depthChunkBuffer.data(), _depthBuffered);
        _depthBuffered = 0;
        return ok;
    }

    bool appendDepthToBuffer(const cv::Mat &depthMat)
    {
        if (_depthFramePixels == 0 || _depthChunkBuffer.empty())
            return false;
        const size_t frameBytes = _depthFramePixels * sizeof(uint16_t);
        uint16_t *dest = _depthChunkBuffer.data() + (_depthBuffered * _depthFramePixels);
        if (depthMat.isContinuous())
        {
            std::memcpy(dest, depthMat.data, frameBytes);
        }
        else
        {
            const int rowBytes = static_cast<int>(_depthWidth * sizeof(uint16_t));
            for (int row = 0; row < _depthHeight; ++row)
            {
                const uint8_t *src = depthMat.ptr<uint8_t>(row);
                std::memcpy(reinterpret_cast<uint8_t *>(dest) + static_cast<size_t>(row) * rowBytes, src, rowBytes);
            }
        }
        ++_depthBuffered;
        return true;
    }

    bool ensureDepthCapacity(uint64_t needed)
    {
        if (needed <= _depthAllocSize)
            return true;
        uint64_t target = _depthAllocSize;
        while (target < needed)
            target += _depthExtendStep;
        const hsize_t extendDims[3] = {static_cast<hsize_t>(target),
                                       static_cast<hsize_t>(_depthHeight),
                                       static_cast<hsize_t>(_depthWidth)};
        if (H5Dset_extent(_h5Dataset, extendDims) < 0)
            return false;
        _depthAllocSize = target;
        return true;
    }

    bool writeDepthBatch(const void *data, size_t countFrames)
    {
        if (countFrames == 0)
            return true;
        const uint64_t targetIndex = _depthIndex + static_cast<uint64_t>(countFrames);
        if (!ensureDepthCapacity(targetIndex))
            return false;

        hid_t filespace = H5Dget_space(_h5Dataset);
        if (filespace < 0)
            return false;
        const hsize_t start[3] = {static_cast<hsize_t>(_depthIndex), 0, 0};
        const hsize_t count[3] = {static_cast<hsize_t>(countFrames),
                                  static_cast<hsize_t>(_depthHeight),
                                  static_cast<hsize_t>(_depthWidth)};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);

        hid_t memspace = H5Screate_simple(3, count, nullptr);
        const herr_t status = H5Dwrite(_h5Dataset, H5T_NATIVE_UINT16, memspace, filespace, H5P_DEFAULT, data);
        H5Sclose(memspace);
        H5Sclose(filespace);

        if (status < 0)
            return false;

        _depthIndex += static_cast<uint64_t>(countFrames);
        if (_depthChunkSizeUsed > 0 && (_depthIndex % static_cast<uint64_t>(_depthChunkSizeUsed) == 0))
            H5Fflush(_h5File, H5F_SCOPE_LOCAL);
        return true;
    }

    void flushDepthBuffer()
    {
        if (!_hdf5Ready || _depthBuffered == 0)
            return;
        if (writeDepthBatch(_depthChunkBuffer.data(), _depthBuffered))
            _depthBuffered = 0;
    }

    GstElement *makeEncoder()
    {
        const std::vector<const char *> candidates = {
            "vaapih264enc",
            "nvh264enc",
            "x264enc",
            "openh264enc",
            "avenc_h264"
        };
        for (const auto *name : candidates)
        {
            if (gst_element_factory_find(name))
            {
                _encoderName = name;
                _useHwEncoder = (_encoderName == "vaapih264enc" || _encoderName == "nvh264enc");
                _logger.info("Selected encoder for %s: %s", _deviceId.c_str(), _encoderName.c_str());
                return gst_element_factory_make(name, "encoder");
            }
        }
        return nullptr;
    }

    bool initGst(int width, int height)
    {
        static std::once_flag gstInitFlag;
        std::call_once(gstInitFlag, []() { gst_init(nullptr, nullptr); });

        _pipeline = gst_pipeline_new(nullptr);
        if (!_pipeline)
            return false;

        _appsrc = gst_element_factory_make("appsrc", "src");
        GstElement *convert = gst_element_factory_make("videoconvert", "convert");
        GstElement *encoder = makeEncoder();
        GstElement *parse = gst_element_factory_make("h264parse", "parse");
        GstElement *mux = gst_element_factory_make("matroskamux", "mux");
        _filesink = gst_element_factory_make("filesink", "sink");

        GstElement *capsfilter = nullptr;
        if (_useHwEncoder)
        {
            capsfilter = gst_element_factory_make("capsfilter", "capsfilter");
        }

        if (!_appsrc || !convert || !encoder || !parse || !mux || !_filesink ||
            (_useHwEncoder && !capsfilter))
        {
            if (!_appsrc)
                _logger.error("Missing GStreamer element appsrc for %s", _deviceId.c_str());
            if (!convert)
                _logger.error("Missing GStreamer element videoconvert for %s", _deviceId.c_str());
            if (!encoder)
                _logger.error("Missing GStreamer element encoder (vaapih264enc/nvh264enc/x264enc/openh264enc/avenc_h264) for %s",
                              _deviceId.c_str());
            if (!parse)
                _logger.error("Missing GStreamer element h264parse for %s", _deviceId.c_str());
            if (!mux)
                _logger.error("Missing GStreamer element matroskamux for %s", _deviceId.c_str());
            if (!_filesink)
                _logger.error("Missing GStreamer element filesink for %s", _deviceId.c_str());
            if (_useHwEncoder && !capsfilter)
                _logger.error("Missing GStreamer element capsfilter for %s", _deviceId.c_str());
            closeGst();
            return false;
        }

        g_object_set(G_OBJECT(_filesink), "location", _rgbPath.string().c_str(), nullptr);
        if (_encoderName == "x264enc")
        {
            g_object_set(G_OBJECT(encoder),
                         "tune", "zerolatency",
                         "speed-preset", "fast",
                         "bitrate", _colorBitrateKbps,
                         nullptr);
        }
        else
        {
            g_object_set(G_OBJECT(encoder), "bitrate", _colorBitrateKbps, nullptr);
        }

        GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                            "format", G_TYPE_STRING, _inputGstFormat.c_str(),
                                            "width", G_TYPE_INT, width,
                                            "height", G_TYPE_INT, height,
                                            "framerate", GST_TYPE_FRACTION, _colorFps, 1,
                                            nullptr);
        gst_app_src_set_caps(GST_APP_SRC(_appsrc), caps);
        gst_caps_unref(caps);
        g_object_set(G_OBJECT(_appsrc),
                     "is-live", TRUE,
                     "block", TRUE,
                     "format", GST_FORMAT_TIME,
                     nullptr);

        if (_useHwEncoder)
        {
            GstCaps *hwCaps = gst_caps_new_simple("video/x-raw",
                                                  "format", G_TYPE_STRING, "NV12",
                                                  nullptr);
            g_object_set(G_OBJECT(capsfilter), "caps", hwCaps, nullptr);
            gst_caps_unref(hwCaps);

            gst_bin_add_many(GST_BIN(_pipeline), _appsrc, convert, capsfilter, encoder, parse, mux, _filesink, nullptr);
            if (!gst_element_link_many(_appsrc, convert, capsfilter, encoder, parse, mux, _filesink, nullptr))
            {
                _logger.error("Failed to link GStreamer pipeline for %s", _deviceId.c_str());
                closeGst();
                return false;
            }
        }
        else
        {
            gst_bin_add_many(GST_BIN(_pipeline), _appsrc, convert, encoder, parse, mux, _filesink, nullptr);
            if (!gst_element_link_many(_appsrc, convert, encoder, parse, mux, _filesink, nullptr))
            {
                _logger.error("Failed to link GStreamer pipeline for %s", _deviceId.c_str());
                closeGst();
                return false;
            }
        }

        const GstStateChangeReturn stateRet = gst_element_set_state(_pipeline, GST_STATE_PLAYING);
        if (stateRet == GST_STATE_CHANGE_FAILURE)
        {
            _logger.error("Failed to start GStreamer pipeline for %s", _deviceId.c_str());
            closeGst();
            return false;
        }
        _gstReady = true;
        return true;
    }

    bool initHdf5(int width, int height)
    {
        _depthWidth = width;
        _depthHeight = height;
        _depthFramePixels = static_cast<size_t>(width) * static_cast<size_t>(height);

        _h5File = H5Fcreate(_depthPath.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (_h5File < 0)
            return false;

        hsize_t dims[3] = {0, static_cast<hsize_t>(height), static_cast<hsize_t>(width)};
        hsize_t maxDims[3] = {H5S_UNLIMITED, static_cast<hsize_t>(height), static_cast<hsize_t>(width)};
        hid_t dataspace = H5Screate_simple(3, dims, maxDims);
        if (dataspace < 0)
        {
            closeHdf5();
            return false;
        }

        hid_t props = H5Pcreate(H5P_DATASET_CREATE);
        int chunkSize = _depthChunkSize > 0 ? _depthChunkSize : 1;
        chunkSize = std::min(chunkSize, 8);
        _depthChunkSizeUsed = chunkSize;
        hsize_t chunkDims[3] = {static_cast<hsize_t>(_depthChunkSizeUsed),
                                static_cast<hsize_t>(height),
                                static_cast<hsize_t>(width)};
        H5Pset_chunk(props, 3, chunkDims);
        // H5Pset_shuffle(props);
        // H5Pset_deflate(props, 1);

        _h5Dataset = H5Dcreate2(_h5File, "/depth", H5T_NATIVE_UINT16, dataspace, H5P_DEFAULT, props, H5P_DEFAULT);
        H5Pclose(props);
        H5Sclose(dataspace);

        if (_h5Dataset < 0)
        {
            closeHdf5();
            return false;
        }
        _depthAllocSize = 0;
        if (_depthChunkSizeUsed > 1 && _depthFramePixels > 0)
            _depthChunkBuffer.resize(_depthChunkSizeUsed * _depthFramePixels);
        _hdf5Ready = true;
        return true;
    }

    void closeStreams()
    {
        if (_tsStream.is_open())
        {
            _tsStream.flush();
            _tsStream.close();
        }
        closeGst();
        closeHdf5();
    }

    void closeGst()
    {
        if (!_pipeline)
            return;
        if (_appsrc)
            gst_app_src_end_of_stream(GST_APP_SRC(_appsrc));
        GstBus *bus = gst_element_get_bus(_pipeline);
        if (bus)
        {
            GstMessage *msg = gst_bus_timed_pop_filtered(bus,
                                                         2 * GST_SECOND,
                                                         static_cast<GstMessageType>(GST_MESSAGE_ERROR |
                                                                                     GST_MESSAGE_EOS));
            if (msg)
                gst_message_unref(msg);
            gst_object_unref(bus);
        }
        gst_element_set_state(_pipeline, GST_STATE_NULL);
        gst_object_unref(_pipeline);
        _pipeline = nullptr;
        _appsrc = nullptr;
        _filesink = nullptr;
        _gstReady = false;
    }

    void closeHdf5()
    {
        flushDepthBuffer();
        if (_h5Dataset >= 0)
        {
            if (_depthIndex > 0)
            {
                const hsize_t finalDims[3] = {static_cast<hsize_t>(_depthIndex),
                                              static_cast<hsize_t>(_depthHeight),
                                              static_cast<hsize_t>(_depthWidth)};
                H5Dset_extent(_h5Dataset, finalDims);
            }
            H5Dclose(_h5Dataset);
            _h5Dataset = H5I_INVALID_HID;
        }
        if (_h5File >= 0)
        {
            H5Fclose(_h5File);
            _h5File = H5I_INVALID_HID;
        }
        _hdf5Ready = false;
    }

    static std::string sanitize(std::string value)
    {
        for (auto &ch : value)
        {
            if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_')
                ch = '_';
        }
        return value;
    }

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
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(ts.time_since_epoch()) % 1000;
        oss << "." << std::setw(3) << std::setfill('0') << ms.count() << "Z";
        return oss.str();
    }

    std::string _deviceId;
    std::string _basePath;
    Logger &_logger;
    std::ofstream _tsStream;
    uint64_t _frameIndex{0};
    std::mutex _tsMutex;
    std::mutex _mutex;
    std::mutex _callbackMutex;
    std::condition_variable _cv;
    std::queue<FrameData> _queue;
    bool _stopped{false};
    std::thread _worker;
    std::filesystem::path _cameraDir;
    std::filesystem::path _rgbPath;
    std::filesystem::path _depthPath;
    int _colorFps{30};
    int _depthChunkSize{0};
    int _colorBitrateKbps{8000};
    bool _gstReady{false};
    bool _hdf5Ready{false};
    bool _ptsInitialized{false};
    bool _depthTypeWarned{false};
    int _depthChunkSizeUsed{0};
    size_t _depthFramePixels{0};
    size_t _depthBuffered{0};
    std::vector<uint16_t> _depthChunkBuffer;
    std::chrono::system_clock::time_point _firstTimestamp;
    std::string _encoderName;
    std::string _inputGstFormat;
    bool _useHwEncoder{false};
    GstElement *_pipeline{nullptr};
    GstElement *_appsrc{nullptr};
    GstElement *_filesink{nullptr};
    hid_t _h5File{H5I_INVALID_HID};
    hid_t _h5Dataset{H5I_INVALID_HID};
    int _depthWidth{0};
    int _depthHeight{0};
    uint64_t _depthIndex{0};
    uint64_t _depthAllocSize{0};
    static constexpr uint64_t _depthExtendStep = 100;
    static constexpr uint64_t _perfLogEvery = 100;
    uint64_t _perfCount{0};
    double _perfRgbMs{0.0};
    double _perfDepthMs{0.0};
    double _perfTotalMs{0.0};
    std::function<void()> _writeCallback;
    uint64_t _timestampLineCount{0};
    static constexpr uint64_t _timestampFlushEvery = 30;
};

std::unique_ptr<FrameWriter> makePngWriter(const std::string &deviceId,
                                           const std::string &basePath,
                                           Logger &logger)
{
    return std::make_unique<GstHdf5FrameWriter>(deviceId, basePath, logger, 30, 0, 8000);
}

std::unique_ptr<FrameWriter> makeGstHdf5Writer(const std::string &deviceId,
                                               const std::string &basePath,
                                               Logger &logger,
                                               int colorFps,
                                               int depthChunkSize,
                                               int colorBitrateKbps)
{
    return std::make_unique<GstHdf5FrameWriter>(deviceId, basePath, logger, colorFps, depthChunkSize, colorBitrateKbps);
}

std::unique_ptr<CameraInterface> createCamera(const CameraConfig &config, Logger &logger)
{
    if (config.type == "RealSense")
        return std::make_unique<RealSenseCamera>(logger);
    if (config.type == "RGB")
        return std::make_unique<SimulatedCamera>("RGB", logger);
    if (config.type == "Network")
        return std::make_unique<NetworkDevice>(logger);
    if (config.type == "Webcam")
        return std::make_unique<WebcamCamera>(logger);
    if (config.type == "VDGlove")
        return createGloveDevice(config.type, logger);
    if (config.type == "Vive" || config.type == "ViveTracker")
        return createViveDevice(logger);
    if (config.type == "Manus" || config.type == "ManusGloves")
        return createManusDevice(logger);
    logger.warn("Unknown camera type '%s', defaulting to RGB", config.type.c_str());
    return std::make_unique<SimulatedCamera>("RGB", logger);
}

std::unique_ptr<FrameWriter> SimulatedCamera::makeWriter(const std::string &basePath, Logger &logger)
{
    return makeGstHdf5Writer(_label,
                             basePath,
                             logger,
                             ensurePositive(_config.color.frameRate, ensurePositive(_config.frameRate, 30)),
                             _config.depth.chunkSize,
                             _config.color.bitrateKbps);
}

CaptureMetadata SimulatedCamera::captureMetadata() const
{
    CaptureMetadata meta;
    meta.deviceId = _label;
    meta.model = _model;
    meta.aligned = false;
    meta.colorFormat = "BGR8";
    meta.colorIntrinsics.stream = "color";
    meta.colorIntrinsics.width = ensurePositive(_config.color.width, ensurePositive(_config.width, 640));
    meta.colorIntrinsics.height = ensurePositive(_config.color.height, ensurePositive(_config.height, 480));
    meta.colorFps = ensurePositive(_config.color.frameRate, ensurePositive(_config.frameRate, 30));
    return meta;
}

std::unique_ptr<FrameWriter> RealSenseCamera::makeWriter(const std::string &basePath, Logger &logger)
{
    return makeGstHdf5Writer(name(),
                             basePath,
                             logger,
                             ensurePositive(_config.color.frameRate, ensurePositive(_config.frameRate, 30)),
                             _config.depth.chunkSize,
                             _config.color.bitrateKbps);
}

CaptureMetadata RealSenseCamera::captureMetadata() const
{
    CaptureMetadata meta = _metadata;
    meta.aligned = _alignEnabled;
    return meta;
}
