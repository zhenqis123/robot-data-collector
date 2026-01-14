#include <atomic>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "ManusInterface.h"

// Linux Socket
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

class ManusWriter : public FrameWriter {
public:
    ManusWriter(const std::string& basePath, Logger& logger) : _logger(logger) {
        std::filesystem::path path = std::filesystem::path(basePath) / "manus_data.csv";
        _csvFile.open(path.string());
        if (!_csvFile.is_open()) {
            _logger.error("Failed to open manus data file: %s", path.string().c_str());
        } else {
            // Header
            _csvFile << "timestamp,python_timestamp,";
            // Left Hand
            for (int i = 0; i < MANUS_JOINTS_PER_HAND; ++i) {
                _csvFile << "lh_j" << i << "_px,lh_j" << i << "_py,lh_j" << i << "_pz,";
                _csvFile << "lh_j" << i << "_ox,lh_j" << i << "_oy,lh_j" << i << "_oz,lh_j" << i << "_ow,";
            }
            // Right Hand
            for (int i = 0; i < MANUS_JOINTS_PER_HAND; ++i) {
                _csvFile << "rh_j" << i << "_px,rh_j" << i << "_py,rh_j" << i << "_pz,";
                _csvFile << "rh_j" << i << "_ox,rh_j" << i << "_oy,rh_j" << i << "_oz,rh_j" << i << "_ow,";
            }
            _csvFile << "\n";
        }
    }
    
    ~ManusWriter() {
        if (_csvFile.is_open()) _csvFile.close();
    }

    bool write(const FrameData& frame) override {
        if (!_csvFile.is_open()) return false;
        if (!frame.manusData) return true;

        const auto& data = *frame.manusData;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame.timestamp.time_since_epoch()).count();
        
        _csvFile << ms << "," << std::fixed << std::setprecision(6) << data.python_timestamp << ",";

        // Helper to write hand
        auto writeHand = [&](const ManusHandData& hand) {
            for (const auto& joint : hand.joints) {
                _csvFile << joint.position.x() << "," << joint.position.y() << "," << joint.position.z() << ",";
                _csvFile << joint.orientation.x() << "," << joint.orientation.y() << "," << joint.orientation.z() << "," << joint.orientation.w() << ",";
            }
        };

        writeHand(data.left_hand);
        writeHand(data.right_hand);

        _csvFile << "\n";
        return true;
    }

private:
    std::ofstream _csvFile;
    Logger& _logger;
};

class ManusDevice : public ManusInterface {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:

    ManusDevice(Logger& logger) : _logger(logger) {}

    ~ManusDevice() override {
        close();
    }

    bool initialize(const CameraConfig& config) override {
        _config = ManusConfig();
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
            _logger.info("Manus: Target FPS set to %.1f (Interval: %.3fs)", _targetFps, _minFrameInterval);
        } else {
            _minFrameInterval = 0.0;
        }

        return initializeInternal();
    }

    bool initializeInternal() {
        _sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (_sockfd < 0) {
            _logger.error("Manus: Failed to create socket");
            return false;
        }

        int opt = 1;
        setsockopt(_sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms
        if (setsockopt(_sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
             _logger.warn("Manus: Failed to set socket timeout");
        }

        struct sockaddr_in servaddr;
        memset(&servaddr, 0, sizeof(servaddr));
        servaddr.sin_family = AF_INET;
        servaddr.sin_addr.s_addr = INADDR_ANY; 
        servaddr.sin_port = htons(_config.port);

        if (bind(_sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
            _logger.error("Manus: Bind failed on port %d", _config.port);
            return false;
        }

        _running = true;
        _receiverThread = std::thread(&ManusDevice::receiveLoop, this);
        _logger.info("Manus listening on port %d", _config.port);
        
        return true;
    }

    FrameData captureFrame() override {
        FrameData frame;
        frame.manusData = captureManusData();
        frame.timestamp = frame.manusData->host_timestamp;
        frame.deviceTimestampMs = static_cast<int64_t>(frame.manusData->python_timestamp * 1000); 
        frame.cameraId = "ManusGloves";
        return frame;
    }
    
    std::unique_ptr<FrameWriter> makeWriter(const std::string &basePath, Logger &logger) override {
        return std::make_unique<ManusWriter>(basePath, logger);
    }

    ManusFrameData captureManusData() override {
        std::unique_lock<std::mutex> lock(_dataMutex);
        _frameCv.wait(lock, [this]{ return _newFrameReady || !_running; });
        
        if (!_running) return ManusFrameData();

        _newFrameReady = false;
        return _currentFrame;
    }

    void close() override {
        _running = false;
        _frameCv.notify_all();

        if (_sockfd >= 0) {
            ::close(_sockfd);
            _sockfd = -1;
        }

        if (_receiverThread.joinable()) {
            _receiverThread.join();
        }
        _logger.info("Manus closed successfully.");
    }

    std::string name() const override { return "ManusGloves"; }

    CaptureMetadata captureMetadata() const override {
        CaptureMetadata meta;
        meta.deviceId = "ManusGloves";
        meta.model = "ManusGloves";
        meta.aligned = false;
        return meta;
    }

private:
    Logger& _logger;
    ManusConfig _config;
    int _sockfd = -1;
    std::thread _receiverThread;
    std::atomic<bool> _running{false};
    
    std::mutex _dataMutex;
    std::condition_variable _frameCv;
    bool _newFrameReady{false};
    ManusFrameData _currentFrame;

    double _targetFps{0.0};
    double _minFrameInterval{0.0};
    std::chrono::steady_clock::time_point _lastFrameTime;

    void receiveLoop() {
        ManusRawPacket packet;
        while (_running) {
            struct sockaddr_in cliaddr;
            socklen_t len = sizeof(cliaddr);
            int n = recvfrom(_sockfd, &packet, sizeof(packet), 0, (struct sockaddr *) &cliaddr, &len);

            if (n < 0) {
                 continue; // timeout or error
            }
            if (n != sizeof(ManusRawPacket)) {
                _logger.warn("Manus: Received packet size mismatch: %d expected %d", n, (int)sizeof(ManusRawPacket));
                continue;
            }

            // Rate limiting
            auto now = std::chrono::steady_clock::now();
            if (_minFrameInterval > 0.0) {
                std::chrono::duration<double> diff = now - _lastFrameTime;
                if (diff.count() < _minFrameInterval) continue;
            }
            _lastFrameTime = now;

            {
                std::lock_guard<std::mutex> lock(_dataMutex);
                _currentFrame.host_timestamp = std::chrono::system_clock::now();
                _currentFrame.python_timestamp = packet.timestamp;
                
                // Parse Left Hand
                _currentFrame.left_hand.joints.resize(MANUS_JOINTS_PER_HAND);
                for(int i=0; i<MANUS_JOINTS_PER_HAND; ++i) {
                    _currentFrame.left_hand.joints[i].position = Eigen::Vector3f(packet.left_fingers[i][0], packet.left_fingers[i][1], packet.left_fingers[i][2]);
                    // Assuming packet orientation is sent as [x, y, z, w], construct Eigen Quaternion (w, x, y, z)
                    _currentFrame.left_hand.joints[i].orientation = Eigen::Quaternionf(packet.left_orientations[i][3], packet.left_orientations[i][0], packet.left_orientations[i][1], packet.left_orientations[i][2]); 
                }

                // Parse Right Hand
                _currentFrame.right_hand.joints.resize(MANUS_JOINTS_PER_HAND);
                for(int i=0; i<MANUS_JOINTS_PER_HAND; ++i) {
                    _currentFrame.right_hand.joints[i].position = Eigen::Vector3f(packet.right_fingers[i][0], packet.right_fingers[i][1], packet.right_fingers[i][2]);
                    _currentFrame.right_hand.joints[i].orientation = Eigen::Quaternionf(packet.right_orientations[i][3], packet.right_orientations[i][0], packet.right_orientations[i][1], packet.right_orientations[i][2]);
                }
                
                _newFrameReady = true;
            }
            _frameCv.notify_all();
        }
    }
};

std::unique_ptr<ManusInterface> createManusDevice(Logger& logger) {
    return std::make_unique<ManusDevice>(logger);
}
