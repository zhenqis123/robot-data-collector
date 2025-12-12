#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>

#include "CameraInterface.h"
#include "ConfigManager.h"

struct apriltag_detector;
struct apriltag_family;

class ArucoTracker
{
public:
    explicit ArucoTracker(const std::vector<ArucoTarget> &targets);
    ~ArucoTracker();

    void startSession(const std::string &basePath);
    void endSession();
    void submit(const FrameData &frame);
    std::vector<ArucoDetection> getLatestDetections(const std::string &cameraId) const;
    bool isAvailable() const { return _detectorType == FiducialType::Aruco ? static_cast<bool>(_dictionary)
                                                                           : _aprilDetector != nullptr; }
    std::string detectorName() const { return _detectorLabel; }

private:
    struct Job
    {
        std::string cameraId;
        cv::Mat image;
        std::chrono::system_clock::time_point timestamp;
        int64_t deviceTimestampMs{0};
    };

    void worker();
    void writeDetections(const std::string &cameraId,
                         const std::vector<ArucoDetection> &detections);
    static std::string sanitize(const std::string &value);
    std::vector<ArucoDetection> detectWithAruco(const Job &job);
    std::vector<ArucoDetection> detectWithAprilTag(const Job &job);
    void destroyAprilTag();

    std::thread _thread;
    mutable std::mutex _mutex;
    std::condition_variable _cv;
    std::queue<Job> _queue;
    bool _running{true};
    bool _sessionActive{false};
    std::string _sessionPath;
    std::unordered_map<std::string, std::ofstream> _files;
    std::unordered_map<std::string, std::vector<ArucoDetection>> _latestDetections;

    FiducialType _detectorType{FiducialType::Aruco};
    cv::Ptr<cv::aruco::Dictionary> _dictionary;
    std::vector<int> _markerIds;

    apriltag_detector *_aprilDetector{nullptr};
    apriltag_family *_aprilFamily{nullptr};
    void (*_aprilFamilyDestroy)(apriltag_family *){nullptr};
    std::string _detectorLabel{"ArUco"};
};
