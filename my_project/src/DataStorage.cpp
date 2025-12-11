#include "DataStorage.h"

#include <iomanip>
#include <sstream>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "Logger.h"

using json = nlohmann::json;

DataStorage::DataStorage(const std::string &basePath, Logger &logger)
    : _basePath(basePath), _logger(logger)
{
    std::filesystem::create_directories(_basePath);
}

void DataStorage::beginRecording(const std::string &captureName,
                                 const std::string &subject,
                                 const std::string &basePath,
                                 const std::vector<CaptureMetadata> &cameraMetas)
{
    _basePath = basePath;
    _cameraMetas = cameraMetas;
    if (std::filesystem::exists(_basePath))
    {
        std::filesystem::remove_all(_basePath);
    }
    _captureName = captureName;
    _subject = subject;
    _sessionId = std::filesystem::path(_basePath).filename().string();
    _sessionStart = std::chrono::system_clock::now();
    std::filesystem::create_directories(_basePath);
    writeMetadataFile();
    if (!_taskTemplatePath.empty())
    {
        std::error_code ec;
        auto target = std::filesystem::path(_basePath) / "task_used.json";
        std::filesystem::copy_file(_taskTemplatePath, target,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        if (ec)
            _logger.warn("Failed to copy task template: %s", ec.message().c_str());
    }
    {
        auto eventPath = std::filesystem::path(_basePath) / "events.jsonl";
        _eventStream.open(eventPath.string(), std::ios::out | std::ios::app);
    }
    logEvent("start_recording");
    _sessionActive = true;
    _logger.info("Recording path set to %s", _basePath.c_str());
}

void DataStorage::endRecording()
{
    logEvent("stop_recording");
    if (_eventStream.is_open())
        _eventStream.close();
    if (_annotationStream.is_open())
        _annotationStream.close();
    _sessionActive = false;
}

bool DataStorage::ensureDirectory(const std::string &path) const
{
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return !ec;
}

std::string DataStorage::timestampToIso(const std::chrono::system_clock::time_point &ts)
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

void DataStorage::writeMetadataFile() const
{
    std::filesystem::path metaPath = std::filesystem::path(_basePath) / "meta.json";
    std::ofstream meta(metaPath.string());
    if (!meta.is_open())
    {
        _logger.error("Failed to write metadata file at %s", metaPath.string().c_str());
        return;
    }

    auto intrinsicsToJson = [](const StreamIntrinsics &intr) {
        json j;
        if (intr.width > 0)
            j["width"] = intr.width;
        if (intr.height > 0)
            j["height"] = intr.height;
        j["fx"] = intr.fx;
        j["fy"] = intr.fy;
        j["cx"] = intr.cx;
        j["cy"] = intr.cy;
        if (!intr.coeffs.empty())
            j["coeffs"] = intr.coeffs;
        return j;
    };

    json root;
    root["session_id"] = _sessionId;
    root["capture_name"] = _captureName;
    root["subject"] = _subject;
    root["base_path"] = _basePath;
    root["start_time"] = timestampToIso(_sessionStart);
    root["scene_id"] = _sceneId;
    root["task_id"] = _taskId;
    root["task_template_path"] = _taskTemplatePath;
    root["task_template_version"] = _taskTemplateVersion;
    root["task_source"] = _taskSource;
    if (!_vlmPromptPath.empty())
    {
        json prompt;
        prompt["path"] = _vlmPromptPath;
        if (!_vlmPromptContent.empty())
            prompt["content"] = _vlmPromptContent;
        root["vlm_prompt"] = prompt;
    }
    json cameras = json::array();
    for (const auto &cam : _cameraMetas)
    {
        json camObj;
        camObj["id"] = cam.deviceId;
        if (!cam.model.empty())
            camObj["model"] = cam.model;
        if (!cam.serial.empty())
            camObj["serial"] = cam.serial;

        json streams;
        if (!cam.colorFormat.empty())
            streams["color"]["format"] = cam.colorFormat;
        if (cam.colorFps > 0)
            streams["color"]["fps"] = cam.colorFps;
        if (cam.colorIntrinsics.width > 0 || cam.colorIntrinsics.height > 0)
            streams["color"]["intrinsics"] = intrinsicsToJson(cam.colorIntrinsics);
        if (!cam.depthFormat.empty())
            streams["depth"]["format"] = cam.depthFormat;
        if (cam.depthFps > 0)
            streams["depth"]["fps"] = cam.depthFps;
        if (cam.depthIntrinsics.width > 0 || cam.depthIntrinsics.height > 0)
            streams["depth"]["intrinsics"] = intrinsicsToJson(cam.depthIntrinsics);
        if (!streams.empty())
            camObj["streams"] = streams;

        json alignment;
        alignment["aligned"] = cam.aligned;
        if (cam.depthScale > 0.0)
            alignment["depth_scale_m"] = cam.depthScale;
        alignment["depth_to_color"] = {
            {"rotation", cam.depthToColor.rotation},
            {"translation", cam.depthToColor.translation}
        };
        camObj["alignment"] = alignment;
        cameras.push_back(camObj);
    }
    if (!cameras.empty())
        root["cameras"] = cameras;

    meta << root.dump(2);
}

void DataStorage::setTaskSelection(const std::string &sceneId,
                                   const std::string &taskId,
                                   const std::string &templatePath,
                                   const std::string &templateVersion,
                                   const std::string &source)
{
    _sceneId = sceneId;
    _taskId = taskId;
    _taskTemplatePath = templatePath;
    _taskTemplateVersion = templateVersion;
    _taskSource = source;
}

void DataStorage::setVlmPrompt(const std::string &path, const std::string &content)
{
    _vlmPromptPath = path;
    _vlmPromptContent = content;
}

void DataStorage::logEvent(const std::string &eventName)
{
    if (!_eventStream.is_open())
        return;
    const auto now = std::chrono::system_clock::now();
    const auto tsMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    _eventStream << "{"
                 << "\"ts_ms\":" << tsMs << ","
                 << "\"event\":\"" << eventName << "\""
                 << "}\n";
    _eventStream.flush();
}

void DataStorage::logAnnotation(const AnnotationEntry &entry)
{
    if (!_sessionActive)
        return;
    if (!_annotationStream.is_open())
    {
        auto path = std::filesystem::path(_basePath) / "annotations.jsonl";
        _annotationStream.open(path.string(), std::ios::out | std::ios::app);
    }
    if (!_annotationStream.is_open())
    {
        _logger.error("Failed to open annotations file");
        return;
    }
    json j;
    j["schema_version"] = "1.0";
    j["session_id"] = _sessionId;
    j["source"] = entry.source;
    j["task_ref"] = {
        {"scene_id", entry.sceneId},
        {"task_id", entry.taskId},
        {"template_path", entry.templatePath},
        {"template_version", entry.templateVersion}
    };
    j["timestamp_ms"] = entry.timestampMs;
    j["state"] = entry.state;
    if (!entry.currentStepId.empty())
        j["current_step_id"] = entry.currentStepId;
    if (!entry.triggerType.empty())
        j["trigger"] = {{"type", entry.triggerType}};
    if (!entry.notes.empty())
        j["notes"] = entry.notes;

    if (!entry.stepOverrides.empty())
    {
        json arr = json::array();
        for (const auto &ov : entry.stepOverrides)
        {
            json o;
            o["step_id"] = ov.stepId;
            o["done"] = ov.done;
            o["attempts"] = ov.attempts;
            arr.push_back(o);
        }
        j["step_overrides"] = arr;
    }

    _annotationStream << j.dump() << "\n";
    _annotationStream.flush();
}

void DataStorage::logStepTiming(const std::string &stepId, int64_t startMs, int64_t endMs)
{
    if (!_sessionActive)
        return;
    if (!_annotationStream.is_open())
    {
        auto path = std::filesystem::path(_basePath) / "annotations.jsonl";
        _annotationStream.open(path.string(), std::ios::out | std::ios::app);
    }
    if (!_annotationStream.is_open())
    {
        _logger.error("Failed to open annotations file");
        return;
    }
    json j;
    j["schema_version"] = "1.0";
    j["session_id"] = _sessionId;
    j["source"] = _taskSource;
    j["task_ref"] = {
        {"scene_id", _sceneId},
        {"task_id", _taskId},
        {"template_path", _taskTemplatePath},
        {"template_version", _taskTemplateVersion}
    };
    j["timestamp_ms"] = endMs;
    j["state"] = "step_timing";
    j["current_step_id"] = stepId;
    json timing;
    timing["start_ms"] = startMs;
    timing["end_ms"] = endMs;
    timing["duration_ms"] = std::max<int64_t>(0, endMs - startMs);
    j["timing"] = timing;
    _annotationStream << j.dump() << "\n";
    _annotationStream.flush();
}
