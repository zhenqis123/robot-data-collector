#include "VDGloveInterface.h"
#include "GloveMath.h"
#include "VDSDKLoader.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstring>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>   // Windows
#else
#include <unistd.h>    // Linux readlink
#include <limits.h>    // PATH_MAX
#endif

std::string getExecutableDirectory(){
    std::filesystem::path exePath;
#ifdef _WIN32
    char buffer[MAX_PATH] = {0};
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    exePath = std::filesystem::path(buffer).parent_path();
#else
    char buffer[PATH_MAX] = {0};
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        exePath = std::filesystem::path(buffer).parent_path();
    } else {
        exePath = std::filesystem::current_path();
    }
#endif
    return exePath.string();
}


// CSV Writer Implementation
class GloveWriter : public FrameWriter {
public:
    GloveWriter(const std::string& basePath, Logger& logger) : _logger(logger) {
        std::filesystem::path path = std::filesystem::path(basePath) / "glove_data.csv";
        _csvFile.open(path.string());
        if (!_csvFile.is_open()) {
            _logger.error("Failed to open glove data file: %s", path.string().c_str());
        } else {
            // Write header
            _csvFile << "timestamp,device_timestamp_ms,";
            // Left Hand
            _csvFile << "left_detected,left_gesture,";
            for (int i = 0; i < 21; ++i) _csvFile << "l_x" << i << ",l_y" << i << ",l_z" << i << ",";
            _csvFile << "l_wrist_qw,l_wrist_qx,l_wrist_qy,l_wrist_qz,";
            // Right Hand
            _csvFile << "right_detected,right_gesture,";
            for (int i = 0; i < 21; ++i) _csvFile << "r_x" << i << ",r_y" << i << ",r_z" << i << ",";
            _csvFile << "r_wrist_qw,r_wrist_qx,r_wrist_qy,r_wrist_qz";
            _csvFile << "\n";
        }
    }

    ~GloveWriter() {
        if (_csvFile.is_open()) _csvFile.close();
    }

    bool write(const FrameData& frame) override {
        if (!_csvFile.is_open()) return false;
        if (!frame.gloveData) return true; // Not a glove frame, ignore or handle gracefully

        const auto& data = *frame.gloveData;
        
        // Convert timestamp to string or ms
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame.timestamp.time_since_epoch()).count();

        _csvFile << ms << "," << frame.deviceTimestampMs << ",";

        // Helper to write hand
        auto writeHand = [&](const HandData& hand) {
            _csvFile << (hand.detected ? 1 : 0) << "," << hand.gesture_id << ",";
            if (hand.keypoints.size() >= 21) {
                for (const auto& p : hand.keypoints) {
                    _csvFile << p.x() << "," << p.y() << "," << p.z() << ",";
                }
            } else {
                for (int i=0; i<21; ++i) _csvFile << "0,0,0,";
            }
            _csvFile << hand.wrist_quaternion.w() << "," 
                     << hand.wrist_quaternion.x() << "," 
                     << hand.wrist_quaternion.y() << "," 
                     << hand.wrist_quaternion.z();
        };

        writeHand(data.left_hand);
        _csvFile << ",";
        writeHand(data.right_hand);
        _csvFile << "\n";
        
        return true;
    }

private:
    std::ofstream _csvFile;
    Logger& _logger;
};

class VDGloveDevice : public VDGloveInterface {

    std::chrono::system_clock::time_point _lastPacketTime;
    bool _is_connected = false;

public:
    VDGloveDevice(Logger& logger) : _logger(logger) {}

    ~VDGloveDevice() override {
            close();
        }

    // Adapt CameraConfig to VDGloveConfig
    bool initialize(const CameraConfig& config) override {
        // Default config
        _config = VDGloveConfig();
        
        // Parse extraSettings
        if (config.extraSettings.count("server_ip")) 
            _config.server_ip = config.extraSettings.at("server_ip");
        if (config.extraSettings.count("server_port")) 
            _config.server_port = std::stoi(config.extraSettings.at("server_port"));
        if (config.extraSettings.count("local_port")) 
            _config.local_port = std::stoi(config.extraSettings.at("local_port"));
        if (config.extraSettings.count("device_index")) 
            _config.device_index = std::stoi(config.extraSettings.at("device_index"));
        if (config.extraSettings.count("process_mano")) 
            _config.process_mano = (config.extraSettings.at("process_mano") == "true");

        // Rate limiting
        _targetFps = 0.0;
        if (config.extraSettings.count("target_fps")) {
            _targetFps = std::stod(config.extraSettings.at("target_fps"));
        }
        if (_targetFps > 0.0) {
            _minFrameInterval = 1.0 / _targetFps;
            _logger.info("VDGlove: Target FPS set to %.1f (Interval: %.3fs)", _targetFps, _minFrameInterval);
        } else {
            _minFrameInterval = 0.0;
             _logger.info("VDGlove: No FPS limit set.");
        }

        // Use original initialize logic
        return initializeInternal();
    }

    bool initializeInternal() {
        std::string libPath;

        _lastPacketTime = std::chrono::system_clock::now();
        _is_connected = true;

// #ifdef _WIN32
//         libPath = "VDMocapSDK_DataRead.dll";
// #else
//         libPath = "/home/zc/robot-data-collector/libVDMocapSDK_DataRead.so";
// #endif


        std::string exeDir = getExecutableDirectory();

#ifdef _WIN32
        libPath = std::filesystem::path(exeDir) / "libVDMocapSDK_DataRead.dll";
#else
        libPath = std::filesystem::path(exeDir) / "../resources/libVDMocapSDK_DataRead.so";
#endif

        libPath = std::filesystem::canonical(libPath).string();

        if (!_sdk.loadLibrary(libPath)) {
            _logger.error("Could not load VD SDK library from %s", libPath.c_str());
            return false;
        }

        if (!_sdk.UdpOpen(_config.device_index, _config.local_port)) {
            _logger.error("SDK UdpOpen failed on port %d", _config.local_port);
            return false;
        }
        _logger.info("SDK UdpOpen success. Port: %d", _config.local_port);

        char ip_buffer[32];
        strncpy(ip_buffer, _config.server_ip.c_str(), 31);
        
        if (!_sdk.UdpSendRequestConnect(_config.device_index, ip_buffer, (unsigned short)_config.server_port)) {
            _logger.warn("SDK UdpSendRequestConnect returned false (check server IP/Port)");
        } else {
            _logger.info("Connection request sent to %s:%d", _config.server_ip.c_str(), _config.server_port);
        }

        _running = true;
        _receiverThread = std::thread(&VDGloveDevice::receiveLoop, this);

        return true;
    }

    // Required by CameraInterface
    FrameData captureFrame() override {
        FrameData frame;
        frame.gloveData = captureGloveData();
        // If captureGloveData returns empty/invalid due to timeout or stop, handle it?
        // Current implementation will return last valid or empty, let's ensure timestamp is updated only if valid
        frame.timestamp = frame.gloveData->timestamp;
        frame.deviceTimestampMs = frame.gloveData->deviceTimestampMs;
        frame.cameraId = "VDGlove"; // Virtual ID
        return frame;
    }

    std::unique_ptr<FrameWriter> makeWriter(const std::string &basePath, Logger &logger) override {
        return std::make_unique<GloveWriter>(basePath, logger);
    }

    VDGloveFrameData captureGloveData() override {
        auto now = std::chrono::system_clock::now();
        if (_is_connected && std::chrono::duration_cast<std::chrono::seconds>(now - _lastPacketTime).count() > 2) {
            _logger.warn("VDGlove: Data timeout (>2s). Attempting to reconnect...");
            reconnect();
        }

        std::unique_lock<std::mutex> lock(_dataMutex);
        // Block until new frame arrives or stop requested
        _frameCv.wait(lock, [this]{ return _newFrameReady || !_running; });

        if (!_running) {
            return {};
        }

        _newFrameReady = false; // Consume the frame
        return _currentFrame;
    }

    void close() override {
        if (!_running) return;

        _logger.info("VDGlove: Closing...");
        
        _running = false;
        _frameCv.notify_all(); // Wake up any waiting capture threads

        if (_sdk.UdpClose) {
            _logger.info("VDGlove: Calling SDK UdpClose...");
            _sdk.UdpClose(_config.device_index);
        }

        if (_receiverThread.joinable()) {
            _logger.info("VDGlove: Waiting for thread to join...");
            _receiverThread.join(); 
            _logger.info("VDGlove: Thread joined.");
        }
        
        _is_connected = false;
    }

    std::string name() const override { return "VDGloveDevice"; }

    CaptureMetadata captureMetadata() const override {
        CaptureMetadata meta;
        meta.deviceId = "VDGlove";
        meta.model = "VirtualGlove";
        meta.aligned = false;
        return meta;
    }

private:
    Logger& _logger;
    VDGloveConfig _config;
    VDSDKLoader _sdk;
    
    std::thread _receiverThread;
    std::atomic<bool> _running{true};
    std::mutex _dataMutex;
    std::condition_variable _frameCv;
    bool _newFrameReady{false};
    VDGloveFrameData _currentFrame;
    
    // FPS Control
    double _targetFps{0.0};
    double _minFrameInterval{0.0};
    std::chrono::steady_clock::time_point _lastFrameTime;

    void receiveLoop() {
        MocapData raw_data;
        char ip_buffer[32];
        strncpy(ip_buffer, _config.server_ip.c_str(), 31);
        
        _logger.info("VDGlove: Receive thread started.");

        while (_running) {
            bool res = false;
            if (_sdk.UdpRecvMocapData) {
                res = _sdk.UdpRecvMocapData(
                    _config.device_index, 
                    ip_buffer, 
                    (unsigned short)_config.server_port, 
                    &raw_data
                );
            }
            
            if (!_running) break;

            if (res && raw_data.isUpdate) {
                _lastPacketTime = std::chrono::system_clock::now();
                
                // Rate Limiting Logic
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

                if (shouldProcess) {
                    processRawData(raw_data);
                }
                // If not processed, we just drop it effectively
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        _logger.info("VDGlove: Receive thread exiting loop.");
    }

    void reconnect() {
        char ip_buffer[32];
        strncpy(ip_buffer, _config.server_ip.c_str(), 31);
        if (_sdk.UdpSendRequestConnect) {
             _sdk.UdpSendRequestConnect(_config.device_index, ip_buffer, (unsigned short)_config.server_port);
             _logger.info("VDGlove: Re-sent connect request.");
             _lastPacketTime = std::chrono::system_clock::now(); 
        }
    }

    void processRawData(const MocapData& data) {
        VDGloveFrameData frame;
        frame.timestamp = std::chrono::system_clock::now();
        frame.deviceTimestampMs = data.frameIndex;

        // --- 右手处理 ---
        {
            std::vector<Eigen::Vector3f> raw_pts;
            for(int i=0; i<LENGTH_HAND; i++) {
                float x = data.position_rHand[i*3 + 0];
                float y = data.position_rHand[i*3 + 1];
                float z = data.position_rHand[i*3 + 2];
                raw_pts.push_back(Eigen::Vector3f(x, y, z));
            }
            raw_pts.push_back(Eigen::Vector3f::Zero());

            Eigen::Quaternionf wrist_q(
                data.quaternion_rHand[0*4 + 0],
                data.quaternion_rHand[0*4 + 1],
                data.quaternion_rHand[0*4 + 2],
                data.quaternion_rHand[0*4 + 3]
            );
            
            frame.right_hand.wrist_quaternion = Eigen::Vector4f(wrist_q.w(), wrist_q.x(), wrist_q.y(), wrist_q.z());
            frame.right_hand.wrist_rotation = wrist_q.toRotationMatrix();
            frame.right_hand.gesture_id = data.gestureResultR;
            frame.right_hand.detected = true;

            if (_config.process_mano && frame.right_hand.detected) {
                frame.right_hand.keypoints = GloveMath::process_hand_to_mano(
                    raw_pts, frame.right_hand.wrist_rotation, true
                );
            } else {
                frame.right_hand.keypoints = raw_pts;
            }
        }

        // --- 左手处理 ---
        {
            std::vector<Eigen::Vector3f> raw_pts;
            for(int i=0; i<LENGTH_HAND; i++) {
                float x = data.position_lHand[i*3 + 0];
                float y = data.position_lHand[i*3 + 1];
                float z = data.position_lHand[i*3 + 2];
                raw_pts.push_back(Eigen::Vector3f(x, y, z));
            }
            raw_pts.push_back(Eigen::Vector3f::Zero());

            Eigen::Quaternionf wrist_q(
                data.quaternion_lHand[0], data.quaternion_lHand[1], 
                data.quaternion_lHand[2], data.quaternion_lHand[3]
            );
            
            frame.left_hand.wrist_quaternion = Eigen::Vector4f(wrist_q.w(), wrist_q.x(), wrist_q.y(), wrist_q.z());
            frame.left_hand.wrist_rotation = wrist_q.toRotationMatrix();
            frame.left_hand.gesture_id = data.gestureResultL;
            frame.left_hand.detected = true;

            if (_config.process_mano && frame.left_hand.detected) {
                frame.left_hand.keypoints = GloveMath::process_hand_to_mano(
                    raw_pts, frame.left_hand.wrist_rotation, false
                );
            } else {
                frame.left_hand.keypoints = raw_pts;
            }
        }

        {
            std::lock_guard<std::mutex> lock(_dataMutex);
            _currentFrame = frame;
            _newFrameReady = true;
        }
        _frameCv.notify_one();
    }
};

std::unique_ptr<VDGloveInterface> createGloveDevice(const std::string& type, Logger& logger) {
    if (type == "VDGlove") {
        return std::make_unique<VDGloveDevice>(logger);
    }
    return nullptr;
}
