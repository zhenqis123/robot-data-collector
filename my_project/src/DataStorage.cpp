#include "DataStorage.h"

#include <iomanip>
#include <sstream>

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
                                 const std::string &basePath)
{
    _basePath = basePath;
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
    auto escape = [](const std::string &text) {
        std::string out;
        for (char c : text)
        {
            if (c == '\"')
                out += "\\\"";
            else if (c == '\\')
                out += "\\\\";
            else
                out += c;
        }
        return out;
    };
    meta << "{\n";
    meta << "  \"session_id\": \"" << escape(_sessionId) << "\",\n";
    meta << "  \"capture_name\": \"" << escape(_captureName) << "\",\n";
    meta << "  \"subject\": \"" << escape(_subject) << "\",\n";
    meta << "  \"base_path\": \"" << escape(_basePath) << "\",\n";
    meta << "  \"start_time\": \"" << timestampToIso(_sessionStart) << "\",\n";
    meta << "  \"scene_id\": \"" << escape(_sceneId) << "\",\n";
    meta << "  \"task_id\": \"" << escape(_taskId) << "\",\n";
    meta << "  \"task_template_path\": \"" << escape(_taskTemplatePath) << "\",\n";
    meta << "  \"task_template_version\": \"" << escape(_taskTemplateVersion) << "\",\n";
    meta << "  \"task_source\": \"" << escape(_taskSource) << "\"\n";
    meta << "}\n";
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
