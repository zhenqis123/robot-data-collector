#include "ArucoTracker.h"

#include <QDir>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

extern "C"
{
#include "apriltag.h"
#include "tag36h11.h"
#include "tagStandard41h12.h"
}

#include <opencv2/imgproc.hpp>

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

struct AprilFamilyHandle
{
    apriltag_family *family{nullptr};
    void (*destroy)(apriltag_family *){nullptr};
};

std::string toLower(const std::string &value)
{
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
    return out;
}

AprilFamilyHandle aprilFamilyFromString(const std::string &name)
{
    const auto lower = toLower(name);
    if (lower == "tag36h11")
        return {tag36h11_create(), tag36h11_destroy};
    if (lower == "tagstandard41h12")
        return {tagStandard41h12_create(), tagStandard41h12_destroy};
    return {};
}
} // namespace

ArucoTracker::ArucoTracker(const std::vector<ArucoTarget> &targets)
{
    if (!targets.empty())
    {
        const auto &target = targets.front();
        _detectorType = target.type;
        _markerIds = target.markerIds;
        if (_detectorType == FiducialType::Aruco)
        {
            _dictionary = dictionaryFromString(target.dictionary);
            _detectorLabel = target.dictionary.empty() ? "ArUco" : "ArUco (" + target.dictionary + ")";
        }
        else
        {
            auto fam = aprilFamilyFromString(target.dictionary);
            _aprilFamily = fam.family;
            _aprilFamilyDestroy = fam.destroy;
            if (_aprilFamily)
            {
                _aprilDetector = apriltag_detector_create();
                apriltag_detector_add_family(_aprilDetector, _aprilFamily);
                _aprilDetector->nthreads = std::max(1u, std::thread::hardware_concurrency());
                _detectorLabel = target.dictionary.empty() ? "AprilTag" : "AprilTag (" + target.dictionary + ")";
            }
        }
    }
    if (isAvailable())
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
    destroyAprilTag();
}

std::vector<ArucoDetection> ArucoTracker::detectWithAruco(const Job &job)
{
    std::vector<ArucoDetection> detections;
    if (!_dictionary)
        return detections;

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    aruco::detectMarkers(job.image, _dictionary, corners, ids);

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
    return detections;
}

std::vector<ArucoDetection> ArucoTracker::detectWithAprilTag(const Job &job)
{
    std::vector<ArucoDetection> detections;
    if (!_aprilDetector || !_aprilFamily)
        return detections;

    cv::Mat gray;
    if (job.image.channels() == 3)
        cv::cvtColor(job.image, gray, cv::COLOR_BGR2GRAY);
    else if (job.image.channels() == 4)
        cv::cvtColor(job.image, gray, cv::COLOR_BGRA2GRAY);
    else
        gray = job.image;

    if (gray.empty())
        return detections;

    image_u8_t img_header{gray.cols, gray.rows, static_cast<int>(gray.step), gray.data};
    zarray_t *detectionsRaw = apriltag_detector_detect(_aprilDetector, &img_header);
    if (!detectionsRaw)
        return detections;

    for (int i = 0; i < zarray_size(detectionsRaw); ++i)
    {
        apriltag_detection_t *det = nullptr;
        zarray_get(detectionsRaw, i, &det);
        if (!det)
            continue;

        if (!_markerIds.empty() &&
            std::find(_markerIds.begin(), _markerIds.end(), det->id) == _markerIds.end())
            continue;

        ArucoDetection out;
        out.cameraId = job.cameraId;
        out.timestamp = job.timestamp;
        out.deviceTimestampMs = job.deviceTimestampMs;
        out.markerId = det->id;
        out.corners = {
            cv::Point2f(static_cast<float>(det->p[0][0]), static_cast<float>(det->p[0][1])),
            cv::Point2f(static_cast<float>(det->p[1][0]), static_cast<float>(det->p[1][1])),
            cv::Point2f(static_cast<float>(det->p[2][0]), static_cast<float>(det->p[2][1])),
            cv::Point2f(static_cast<float>(det->p[3][0]), static_cast<float>(det->p[3][1]))};
        detections.push_back(std::move(out));
    }

    apriltag_detections_destroy(detectionsRaw);
    return detections;
}

void ArucoTracker::destroyAprilTag()
{
    if (_aprilDetector && _aprilFamily)
        apriltag_detector_remove_family(_aprilDetector, _aprilFamily);
    if (_aprilDetector)
    {
        apriltag_detector_destroy(_aprilDetector);
        _aprilDetector = nullptr;
    }
    if (_aprilFamily && _aprilFamilyDestroy)
        _aprilFamilyDestroy(_aprilFamily);
    _aprilFamily = nullptr;
    _aprilFamilyDestroy = nullptr;
}

void ArucoTracker::startSession(const std::string &basePath)
{
    if (!isAvailable())
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
    if (!isAvailable())
        return;
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_sessionActive)
        return;
    Job job;
    job.cameraId = frame.cameraId;
    job.timestamp = frame.timestamp;
    job.deviceTimestampMs = frame.deviceTimestampMs;
    if (frame.colorFormat == "YUYV")
    {
        cv::cvtColor(frame.imageRef(), job.image, cv::COLOR_YUV2BGR_YUY2);
    }
    else
    {
        job.image = frame.image ? frame.image->clone() : cv::Mat();
    }
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

        std::vector<ArucoDetection> detections =
            _detectorType == FiducialType::AprilTag ? detectWithAprilTag(job) : detectWithAruco(job);
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
