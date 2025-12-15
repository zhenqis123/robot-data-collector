#include "CameraInterface.h"

#include <librealsense2/rs.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <utility>
#include <cctype>

#include "Logger.h"
#include "IntrinsicsManager.h"
#include "NetworkDevice.h"

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
            {640, 480, 30, CameraConfig::StreamConfig::StreamType::Color},
            {1280, 720, 30, CameraConfig::StreamConfig::StreamType::Color},
            {1920, 1080, 30, CameraConfig::StreamConfig::StreamType::Color}
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
            _rsConfig.enable_stream(RS2_STREAM_COLOR, colorWidth, colorHeight, RS2_FORMAT_BGR8, colorFps);
            _rsConfig.enable_stream(RS2_STREAM_DEPTH, depthWidth, depthHeight, RS2_FORMAT_Z16, depthFps);
            auto profile = _pipeline.start(_rsConfig);
            _device = profile.get_device();
            auto dev = _device;
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
            _metadata.colorFormat = "BGR8";
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

            cv::Mat colorMat(cv::Size(color.get_width(), color.get_height()), CV_8UC3,
                             const_cast<void *>(color.get_data()), cv::Mat::AUTO_STEP);
            cv::Mat depthMat(cv::Size(depth.get_width(), depth.get_height()), CV_16U,
                             const_cast<void *>(depth.get_data()), cv::Mat::AUTO_STEP);

            data.image = colorMat.clone();
            data.depth = depthMat.clone();
            data.timestamp = std::chrono::system_clock::now();
            data.deviceTimestampMs = static_cast<int64_t>(color.get_timestamp());
            data.cameraId = _identifier;
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
        data.image = frame.clone();
        data.timestamp = std::chrono::system_clock::now();
        data.deviceTimestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count();
        data.cameraId = _label;
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
        return makePngWriter(name(), basePath, logger);
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

class PngFrameWriter : public FrameWriter
{
public:
    PngFrameWriter(const std::string &deviceId,
                   const std::string &basePath,
                   Logger &logger,
                   size_t threadCount = std::thread::hardware_concurrency())
        : _deviceId(deviceId),
          _basePath(basePath),
          _logger(logger),
          _threadCount(threadCount == 0 ? 1 : threadCount)
    {
        const auto sanitized = sanitize(_deviceId);
        _cameraDir = std::filesystem::path(_basePath) / sanitized;
        std::error_code ec;
        std::filesystem::create_directories(_cameraDir / "color", ec);
        std::filesystem::create_directories(_cameraDir / "depth", ec);
        _workers.reserve(_threadCount);
        for (size_t i = 0; i < _threadCount; ++i)
            _workers.emplace_back(&PngFrameWriter::worker, this);
    }

    ~PngFrameWriter()
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopped = true;
        }
        _cv.notify_all();
        for (auto &t : _workers)
        {
            if (t.joinable())
                t.join();
        }
        if (_tsStream.is_open())
        {
            _tsStream.flush();
            _tsStream.close();
        }
    }

    bool write(const FrameData &frame) override
    {
        if (frame.image.empty() || _basePath.empty())
            return false;

        FrameData copy;
        copy.image = frame.image.clone();
        copy.depth = frame.depth.clone();
        copy.timestamp = frame.timestamp;
        copy.deviceTimestampMs = frame.deviceTimestampMs;
        copy.cameraId = frame.cameraId;

        {
            std::lock_guard<std::mutex> lock(_mutex);
            _queue.push(std::move(copy));
        }
        _cv.notify_one();
        return true;
    }

private:
    struct WorkItem
    {
        FrameData frame;
        uint64_t index{0};
    };

    void worker()
    {
        while (true)
        {
            FrameData frame;
            uint64_t idx = 0;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _cv.wait(lock, [&]() { return !_queue.empty() || _stopped; });
                if (_stopped && _queue.empty())
                    break;
                frame = std::move(_queue.front());
                _queue.pop();
                idx = ++_frameIndex;
            }

            std::ostringstream ss;
            ss << std::setw(6) << std::setfill('0') << idx;
            const auto frameName = ss.str();
            const auto colorPath = (_cameraDir / "color" / (frameName + ".png")).string();
            std::string depthRel;

            std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
            bool ok = cv::imwrite(colorPath, frame.image, params);
            if (!frame.depth.empty())
            {
                const auto depthPath = (_cameraDir / "depth" / (frameName + ".png")).string();
                ok = ok && cv::imwrite(depthPath, frame.depth, params);
                depthRel = (std::filesystem::path("depth") / (frameName + ".png")).string();
            }
            if (!ok)
            {
                _logger.error("Failed to store frame for %s", _deviceId.c_str());
                continue;
            }

            const auto iso = toIso(frame.timestamp);
            const auto tsMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  frame.timestamp.time_since_epoch())
                                  .count();
            const auto colorRel = (std::filesystem::path("color") / (frameName + ".png")).string();
            {
                std::lock_guard<std::mutex> lock(_tsMutex);
                if (!_tsStream.is_open())
                {
                    auto filePath = _cameraDir / "timestamps.csv";
                    const bool existed = std::filesystem::exists(filePath);
                    _tsStream.open(filePath.string(), std::ios::out | std::ios::app);
                    if (!existed)
                        _tsStream << "timestamp_iso,timestamp_ms,device_timestamp_ms,color_path,depth_path\n";
                }
                _tsStream << iso << "," << tsMs << "," << frame.deviceTimestampMs << "," << colorRel << "," << depthRel
                          << "\n";
            }
        }
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
    std::condition_variable _cv;
    std::queue<FrameData> _queue;
    std::vector<std::thread> _workers;
    bool _stopped{false};
    size_t _threadCount{1};
    std::filesystem::path _cameraDir;
};

std::unique_ptr<FrameWriter> makePngWriter(const std::string &deviceId,
                                           const std::string &basePath,
                                           Logger &logger)
{
    return std::make_unique<PngFrameWriter>(deviceId, basePath, logger);
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
    logger.warn("Unknown camera type '%s', defaulting to RGB", config.type.c_str());
    return std::make_unique<SimulatedCamera>("RGB", logger);
}

std::unique_ptr<FrameWriter> SimulatedCamera::makeWriter(const std::string &basePath, Logger &logger)
{
    return makePngWriter(_label, basePath, logger);
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
    return makePngWriter(name(), basePath, logger);
}

CaptureMetadata RealSenseCamera::captureMetadata() const
{
    CaptureMetadata meta = _metadata;
    meta.aligned = _alignEnabled;
    return meta;
}
