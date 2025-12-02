#include "ArucoTracker.h"

#include <QDir>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

using namespace cv;

namespace
{
Ptr<aruco::Dictionary> dictionaryFromString(const std::string &name)
{
    static const std::vector<std::pair<std::string, aruco::PREDEFINED_DICTIONARY_NAME>> dicts = {
        {"DICT_4X4_50", aruco::DICT_4X4_50},
        {"DICT_4X4_100", aruco::DICT_4X4_100},
        {"DICT_5X5_50", aruco::DICT_5X5_50},
        {"DICT_5X5_100", aruco::DICT_5X5_100},
        {"DICT_6X6_50", aruco::DICT_6X6_50},
        {"DICT_6X6_100", aruco::DICT_6X6_100},
        {"DICT_7X7_50", aruco::DICT_7X7_50},
        {"DICT_ARUCO_ORIGINAL", aruco::DICT_ARUCO_ORIGINAL}};
    for (const auto &d : dicts)
    {
        if (d.first == name)
            return aruco::getPredefinedDictionary(d.second);
    }
    return nullptr;
}
} // namespace

ArucoTracker::ArucoTracker(const std::vector<ArucoTarget> &targets)
{
    if (!targets.empty())
    {
        _dictionary = dictionaryFromString(targets.front().dictionary);
        if (_dictionary)
            _markerIds = targets.front().markerIds;
    }
    if (_dictionary)
        _thread = std::thread(&ArucoTracker::worker, this);
    else
        _running = false;
}

ArucoTracker::~ArucoTracker()
{
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _running = false;
        _sessionActive = false;
    }
    _cv.notify_all();
    if (_thread.joinable())
        _thread.join();
}

void ArucoTracker::startSession(const std::string &basePath)
{
    if (!_dictionary)
        return;
    std::lock_guard<std::mutex> lock(_mutex);
    _sessionPath = basePath + "/aruco";
    std::filesystem::create_directories(_sessionPath);
    _sessionActive = true;
}

void ArucoTracker::endSession()
{
    std::lock_guard<std::mutex> lock(_mutex);
    _sessionActive = false;
    for (auto &kv : _files)
    {
        if (kv.second.is_open())
            kv.second.close();
    }
    _files.clear();
}

void ArucoTracker::submit(const FrameData &frame)
{
    if (!_dictionary)
        return;
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_sessionActive)
        return;
    Job job;
    job.cameraId = frame.cameraId;
    job.timestamp = frame.timestamp;
    job.deviceTimestampMs = frame.deviceTimestampMs;
    job.image = frame.image.clone();
    _queue.push(std::move(job));
    _cv.notify_one();
}

void ArucoTracker::worker()
{
    while (true)
    {
        Job job;
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _cv.wait(lock, [&] { return !_queue.empty() || !_running; });
            if (!_running)
                break;
            job = std::move(_queue.front());
            _queue.pop();
            if (!_sessionActive)
                continue;
        }

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        aruco::detectMarkers(job.image, _dictionary, corners, ids);

        std::vector<ArucoDetection> detections;
        for (size_t i = 0; i < ids.size(); ++i)
        {
            if (!_markerIds.empty() &&
                std::find(_markerIds.begin(), _markerIds.end(), ids[i]) == _markerIds.end())
                continue;
            ArucoDetection det;
            det.cameraId = job.cameraId;
            det.timestamp = job.timestamp;
            det.deviceTimestampMs = job.deviceTimestampMs;
            det.markerId = ids[i];
            det.corners = corners[i];
            detections.push_back(std::move(det));
        }
        if (!detections.empty())
            writeDetections(job.cameraId, detections);
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _latestDetections[job.cameraId] = detections;
        }
    }
}

void ArucoTracker::writeDetections(const std::string &cameraId,
                                   const std::vector<ArucoDetection> &detections)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_sessionActive)
        return;
    auto key = sanitize(cameraId);
    auto &stream = _files[key];
    if (!stream.is_open())
    {
        std::filesystem::create_directories(_sessionPath);
        auto path = std::filesystem::path(_sessionPath) / (key + ".csv");
        stream.open(path.string(), std::ios::out | std::ios::trunc);
        stream << "timestamp_iso,timestamp_ms,device_timestamp_ms,marker_id,corners\n";
    }
    for (const auto &det : detections)
    {
        const auto tsMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                              det.timestamp.time_since_epoch())
                              .count();
        const auto timeT = std::chrono::system_clock::to_time_t(det.timestamp);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &timeT);
#else
        localtime_r(&timeT, &tm);
#endif
        std::ostringstream iso;
        iso << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            det.timestamp.time_since_epoch()) %
                        1000;
        iso << "." << std::setw(3) << std::setfill('0') << ms.count() << "Z";
        stream << iso.str() << "," << tsMs << "," << det.deviceTimestampMs << "," << det.markerId
               << ",";
        for (size_t i = 0; i < det.corners.size(); ++i)
        {
            stream << det.corners[i].x << ";" << det.corners[i].y;
            if (i + 1 < det.corners.size())
                stream << ";";
        }
        stream << "\n";
    }
}

std::string ArucoTracker::sanitize(const std::string &value)
{
    std::string sanitized = value;
    for (auto &ch : sanitized)
    {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_')
            ch = '_';
    }
    return sanitized;
}

std::vector<ArucoDetection> ArucoTracker::getLatestDetections(const std::string &cameraId) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _latestDetections.find(cameraId);
    if (it == _latestDetections.end())
        return {};
    return it->second;
}
