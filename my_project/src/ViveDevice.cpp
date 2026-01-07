#include <atomic>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

#include "ViveInterface.h"

// Linux Socket
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

class ViveWriter : public FrameWriter {
public:
    ViveWriter(const std::string& basePath, Logger& logger) : _logger(logger) {
        std::filesystem::path path = std::filesystem::path(basePath) / "vive_data.csv";
        _csvFile.open(path.string());
        if (!_csvFile.is_open()) {
            _logger.error("Failed to open vive data file: %s", path.string().c_str());
        } else {
            // Header
            _csvFile << "timestamp,python_timestamp,";
            for (int i = 0; i < NUM_TRACKERS; ++i) {
                _csvFile << "t" << i << "_valid,"
                         << "t" << i << "_px," << "t" << i << "_py," << "t" << i << "_pz,"
                         << "t" << i << "_qw," << "t" << i << "_qx," << "t" << i << "_qy," << "t" << i << "_qz,";
            }
            _csvFile << "\n";
        }
    }
    
    ~ViveWriter() {
        if (_csvFile.is_open()) _csvFile.close();
    }

    bool write(const FrameData& frame) override {
        if (!_csvFile.is_open()) return false;
        if (!frame.viveData) return true;

        const auto& data = *frame.viveData;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame.timestamp.time_since_epoch()).count();
        
        _csvFile << ms << "," << std::fixed << std::setprecision(6) << data.python_timestamp << ",";

        for (const auto& tracker : data.trackers) {
            _csvFile << (tracker.valid ? 1 : 0) << ",";
            if (tracker.valid) {
                _csvFile << tracker.position.x() << "," << tracker.position.y() << "," << tracker.position.z() << ",";
                Eigen::Quaternionf q(tracker.rotation);
                _csvFile << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ",";
            } else {
                _csvFile << "0,0,0,0,0,0,0,";
            }
        }
        _csvFile << "\n";
        return true;
    }

private:
    std::ofstream _csvFile;
    Logger& _logger;
};

class ViveDevice : public ViveInterface {
public:

    ViveDevice(Logger& logger) : _logger(logger) {}

    ~ViveDevice() override {
        close();
    }

    bool initialize(const CameraConfig& config) override {
        _config = ViveConfig();
        if (config.extraSettings.count("port")) {
            _config.port = std::stoi(config.extraSettings.at("port"));
        }
        
        // Rate limiting
        _targetFps = 0.0;
        if (config.extraSettings.count("target_fps")) {
            _targetFps = std::stod(config.extraSettings.at("target_fps"));
        }
        if (_targetFps > 0.0) {
            _minFrameInterval = 1.0 / _targetFps;
            _logger.info("Vive: Target FPS set to %.1f (Interval: %.3fs)", _targetFps, _minFrameInterval);
        } else {
            _minFrameInterval = 0.0;
        }

        return initializeInternal();
    }

    bool initializeInternal() {
        // 1. 创建 UDP Socket
        _sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (_sockfd < 0) {
            _logger.error("Vive: Failed to create socket");
            return false;
        }

        // 允许地址重用
        int opt = 1;
        setsockopt(_sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms
        if (setsockopt(_sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
            _logger.error("Vive: Set timeout failed");
            return false;
        }

        // 绑定端口
        struct sockaddr_in servaddr;
        memset(&servaddr, 0, sizeof(servaddr));
        servaddr.sin_family = AF_INET;
        servaddr.sin_addr.s_addr = INADDR_ANY; 
        servaddr.sin_port = htons(_config.port);

        if (bind(_sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
            _logger.error("Vive: Bind failed on port %d (Is it already used?)", _config.port);
            ::close(_sockfd); // 绑定失败要关掉 socket
            _sockfd = -1;
            return false;
        }

        // 启动接收线程
        _running = true;
        _receiverThread = std::thread(&ViveDevice::receiveLoop, this);
        _logger.info("Vive listening on port %d", _config.port);
        
        return true;
    }

    FrameData captureFrame() override {
        FrameData frame;
        frame.viveData = captureViveData();
        // If timed out or empty, timestamp might be old.
        frame.timestamp = frame.viveData->host_timestamp;
        // Use system clock as device timestamp if no better option, or python timestamp cast to ms
        frame.deviceTimestampMs = static_cast<int64_t>(frame.viveData->python_timestamp * 1000); 
        frame.cameraId = "ViveTracker";
        return frame;
    }
    
    std::unique_ptr<FrameWriter> makeWriter(const std::string &basePath, Logger &logger) override {
        return std::make_unique<ViveWriter>(basePath, logger);
    }

    ViveFrameData captureViveData() override {
        std::unique_lock<std::mutex> lock(_dataMutex);
        // Block wait
        _frameCv.wait(lock, [this]{ return _newFrameReady || !_running; });
        
        if (!_running) return {};

        _newFrameReady = false;
        return _currentFrame;
    }

    void close() override {
        _running = false;
        _frameCv.notify_all(); // Wake up waiter

        if (_sockfd >= 0) {
            // shutdown 可以确保中断读写操作，比直接 close 更安全
            shutdown(_sockfd, SHUT_RDWR); 
            ::close(_sockfd);
            _sockfd = -1;
        }

        if (_receiverThread.joinable()) {
            _receiverThread.join();
        }
        _logger.info("Vive closed successfully.");
    }

    std::string name() const override { return "ViveTrackerDevice"; }

    CaptureMetadata captureMetadata() const override {
        CaptureMetadata meta;
        meta.deviceId = "ViveTracker";
        meta.model = "ViveTracker";
        meta.aligned = false;
        return meta;
    }

private:
    Logger& _logger;
    ViveConfig _config;
    int _sockfd = -1;
    std::thread _receiverThread;
    std::atomic<bool> _running{false};
    
    std::mutex _dataMutex;
    std::condition_variable _frameCv;
    bool _newFrameReady{false};
    ViveFrameData _currentFrame;

    // FPS Control
    double _targetFps{0.0};
    double _minFrameInterval{0.0};
    std::chrono::steady_clock::time_point _lastFrameTime;

    // 解析逻辑：将 12个 float 转换为 Rotation 和 Position
    ViveTrackerPose parseSingleTracker(const float* raw_data) {
        ViveTrackerPose pose;
        
        // 简单判断有效性：如果所有数据都是0，认为无效
        bool all_zeros = true;
        for (int i=0; i<12; ++i) {
            if (std::abs(raw_data[i]) > 1e-6) {
                all_zeros = false;
                break;
            }
        }
        if (all_zeros) {
            pose.valid = false;
            pose.rotation = Eigen::Matrix3f::Identity();
            pose.position = Eigen::Vector3f::Zero();
            return pose;
        }

        pose.valid = true;
        
        // 填充 Position (第 3, 7, 11 位)
        pose.position << raw_data[3], raw_data[7], raw_data[11];

        // 填充 Rotation (第 0,1,2; 4,5,6; 8,9,10 位)
        pose.rotation << raw_data[0], raw_data[1], raw_data[2],
                         raw_data[4], raw_data[5], raw_data[6],
                         raw_data[8], raw_data[9], raw_data[10];

        return pose;
    }

    void receiveLoop() {
        ViveRawPacket packet;
        while (_running) {
            struct sockaddr_in cliaddr;
            socklen_t len = sizeof(cliaddr);
            
            // 接收数据
            int n = recvfrom(_sockfd, &packet, sizeof(ViveRawPacket), 0, (struct sockaddr *)&cliaddr, &len);
            
            if (n == sizeof(ViveRawPacket)) {
                 // Check rate limit
                bool shouldProcess = true;
                if (_minFrameInterval > 0.0) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(now - _lastFrameTime).count();
                    if (elapsed < _minFrameInterval) {
                        shouldProcess = false;
                    } else {
                        _lastFrameTime = now;
                    }
                }

                if (!shouldProcess) continue;

                // 解析数据
                ViveFrameData frame;
                frame.python_timestamp = packet.timestamp; // 直接读取 double
                frame.host_timestamp = std::chrono::system_clock::now();
                frame.trackers.reserve(NUM_TRACKERS);

                // 循环处理 3 个 Tracker
                for (int i = 0; i < NUM_TRACKERS; ++i) {
                    // 指针偏移：data + i * 12
                    frame.trackers.push_back(parseSingleTracker(packet.data + (i * 12)));
                }

                // 线程安全更新
                {
                    std::lock_guard<std::mutex> lock(_dataMutex);
                    _currentFrame = frame;
                    _newFrameReady = true;
                }
                _frameCv.notify_one();

            } else if (n < 0) {
                // 如果是超时 (EAGAIN / EWOULDBLOCK)，说明只是没数据，继续循环检查 _running 即可
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue; 
                }
                // 如果是其他错误，退出
                else {
                    if (_running) {
                        // _logger.warn("Vive recv error: %s", strerror(errno));
                    }
                    break;
                }
            }
        }
    }
};

std::unique_ptr<ViveInterface> createViveDevice(Logger& logger) {
    return std::make_unique<ViveDevice>(logger);
}
