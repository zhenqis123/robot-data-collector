#pragma once

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

struct VlmConfig
{
    std::string model;
    std::string endpoint;
    std::string promptPath;
    std::string apiKey;
};

struct AudioPromptsConfig
{
    bool enabled{false};
    float volume{1.0f};
    std::string mode{"index_tts"};
    std::string language{"chinese"};
    bool keyframeOnly{false};
    std::unordered_map<std::string, std::string> keybindings;
    struct IndexTts
    {
        std::string endpoint;
        std::vector<std::string> audioPaths;
    } indexTts;
    struct Piper
    {
        std::string voiceOnnx;
        std::string voiceConfig;
        std::string voiceOnnxCn;
        std::string voiceConfigCn;
        std::string espeakData;
        int sampleRate{22050};
    } piper;
    std::unordered_map<std::string, std::string> texts;
};

struct CameraConfig
{
    std::string type;
    std::string serial;
    std::string endpoint;
    int id = 0;
    int width = 0;
    int height = 0;
    int frameRate = 0;
    bool alignDepth{true};
    
    // Generic settings for non-camera devices
    std::unordered_map<std::string, std::string> extraSettings;

    struct StreamConfig
    {
        enum class StreamType { Color, Depth };
        int width = 0;
        int height = 0;
        int frameRate = 0;
        int chunkSize = 0;
        int bitrateKbps = 8000;
        StreamType streamType = StreamType::Color;
    };
    StreamConfig color;
    StreamConfig depth;
};

enum class FiducialType
{
    Aruco,
    AprilTag
};

struct ArucoTarget
{
    FiducialType type{FiducialType::Aruco};
    std::string dictionary;
    std::vector<int> markerIds;
};

class Logger;

class ConfigManager
{
public:
    explicit ConfigManager(Logger &logger);

    bool load(const std::string &path);
    const std::vector<CameraConfig> &getCameraConfigs() const { return _cameraConfigs; }
    std::optional<CameraConfig> getCameraConfigById(int id) const;
    std::string configPath() const { return _path; }
    bool updateCameraConfig(int id, const CameraConfig &config);
    const std::vector<ArucoTarget> &getArucoTargets() const { return _arucoTargets; }
    const std::string &getTasksRootPath() const { return _tasksRoot; }
    const std::string &getCapturesRootPath() const { return _capturesRoot; }
    double getDisplayFpsLimit() const { return _displayFpsLimit; }
    const VlmConfig &getVlmConfig() const { return _vlmConfig; }
    const AudioPromptsConfig &getAudioPromptsConfig() const { return _audioConfig; }

private:
    std::vector<CameraConfig> _cameraConfigs;
    std::vector<ArucoTarget> _arucoTargets;
    std::string _path;
    std::string _tasksRoot;
    std::string _capturesRoot;
    double _displayFpsLimit{0.0};
    VlmConfig _vlmConfig;
    AudioPromptsConfig _audioConfig;
    Logger &_logger;

    static std::pair<int, int> parseResolution(const std::string &value);
    static int defaultDepthChunkSize(int width, int height);
};
