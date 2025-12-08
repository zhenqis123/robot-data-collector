#pragma once

#include <string>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <optional>
#include <vector>

#include "CameraInterface.h"

class Logger;

class DataStorage
{
public:
    DataStorage(const std::string &basePath, Logger &logger);

    void beginRecording(const std::string &captureName,
                        const std::string &subject,
                        const std::string &basePath,
                        const std::vector<CaptureMetadata> &cameraMetas);
    void endRecording();

    std::string basePath() const { return _basePath; }
    void setTaskSelection(const std::string &sceneId,
                          const std::string &taskId,
                          const std::string &templatePath,
                          const std::string &templateVersion,
                          const std::string &source);

    struct StepOverride
    {
        std::string stepId;
        bool done{false};
        int attempts{0};
    };

    struct AnnotationEntry
    {
        std::string source; // script | vml
        std::string sceneId;
        std::string taskId;
        std::string templatePath;
        std::string templateVersion;
        std::string state;
        std::string currentStepId;
        int64_t timestampMs{0};
        std::string triggerType; // key/button etc
        std::vector<StepOverride> stepOverrides;
        std::string notes;
    };

    void logEvent(const std::string &eventName);
    void logAnnotation(const AnnotationEntry &entry);

private:
    std::string _basePath;
    Logger &_logger;
    std::string _captureName;
    std::string _subject;
    std::string _sessionId;
    std::string _sceneId;
    std::string _taskId;
    std::string _taskTemplatePath;
    std::string _taskTemplateVersion;
    std::string _taskSource{"script"};
    std::chrono::system_clock::time_point _sessionStart;
    bool _sessionActive{false};
    std::vector<CaptureMetadata> _cameraMetas;
    std::ofstream _eventStream;
    std::ofstream _annotationStream;

    void writeMetadataFile() const;
    static std::string timestampToIso(const std::chrono::system_clock::time_point &ts);
    bool ensureDirectory(const std::string &path) const;
};
