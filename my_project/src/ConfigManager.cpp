#include "ConfigManager.h"

#include <filesystem>

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "Logger.h"

namespace
{
int highestPowerOfTwo(int value)
{
    int result = 1;
    while (result <= value / 2)
        result *= 2;
    return result;
}
} // namespace

ConfigManager::ConfigManager(Logger &logger)
    : _logger(logger)
{
}

bool ConfigManager::load(const std::string &path)
{
    QFile file(QString::fromStdString(path));
    if (!file.open(QIODevice::ReadOnly))
    {
        _logger.error("Failed to open config file: %s", path.c_str());
        return false;
    }

    const auto data = file.readAll();
    const auto doc = QJsonDocument::fromJson(data);
    if (doc.isNull())
    {
        _logger.error("Invalid JSON in config file");
        return false;
    }

    _path = path;
    _cameraConfigs.clear();
    _arucoTargets.clear();
    _vlmConfig = {};
    _audioConfig = {};
    _capturesRoot.clear();
    _displayFpsLimit = 0.0;
    _showDepthPreview = true;

    const auto cameras = doc.object().value("cameras").toArray();
    for (const auto &entry : cameras)
    {
        const auto obj = entry.toObject();
        CameraConfig config;
        config.type = obj.value("type").toString().toStdString();
        config.id = obj.value("id").toInt();
        config.serial = obj.value("serial").toString().toStdString();
        config.endpoint = obj.value("endpoint").toString().toStdString();
        config.alignDepth = obj.value("align_depth").toBool(true);
        const auto resolution = obj.value("resolution").toString().toStdString();
        const auto [width, height] = parseResolution(resolution);
        const int fps = obj.value("frame_rate").toInt();
        auto streamFromObject = [&](const QJsonObject &streamObj,
                                    const std::pair<int, int> &fallbackRes,
                                    int fallbackFps) {
            CameraConfig::StreamConfig stream;
            stream.width = fallbackRes.first;
            stream.height = fallbackRes.second;
            stream.frameRate = fallbackFps;
            if (!streamObj.isEmpty())
            {
                const auto resStr = streamObj.value("resolution").toString().toStdString();
                if (!resStr.empty())
                {
                    const auto pair = parseResolution(resStr);
                    if (pair.first > 0)
                        stream.width = pair.first;
                    if (pair.second > 0)
                        stream.height = pair.second;
                }
                if (streamObj.contains("frame_rate"))
                    stream.frameRate = streamObj.value("frame_rate").toInt(stream.frameRate);
                if (streamObj.contains("chunk_size"))
                    stream.chunkSize = streamObj.value("chunk_size").toInt(0);
                if (streamObj.contains("bitrate_kbps"))
                    stream.bitrateKbps = streamObj.value("bitrate_kbps").toInt(stream.bitrateKbps);
            }
            return stream;
        };

        config.color = streamFromObject(obj.value("color").toObject(), {width, height}, fps);
        config.depth = streamFromObject(obj.value("depth").toObject(), {width, height}, fps);
        if (config.depth.chunkSize <= 0)
            config.depth.chunkSize = defaultDepthChunkSize(config.depth.width, config.depth.height);
        config.width = config.color.width > 0 ? config.color.width : width;
        config.height = config.color.height > 0 ? config.color.height : height;
        config.frameRate = config.color.frameRate > 0 ? config.color.frameRate : fps;

        // Load extra settings for generic devices
        for (auto it = obj.begin(); it != obj.end(); ++it)
        {
            if (it.key() != "type" && it.key() != "id" && it.key() != "serial" && 
                it.key() != "endpoint" && it.key() != "resolution" && 
                it.key() != "frame_rate" && it.key() != "color" && it.key() != "depth")
            {
                config.extraSettings[it.key().toStdString()] = it.value().toVariant().toString().toStdString();
            }
        }

        _cameraConfigs.push_back(config);
    }

    _logger.info("Loaded %zu camera configuration entries", _cameraConfigs.size());

    const auto aru = doc.object().value("aruco_targets").toArray();
    for (const auto &entry : aru)
    {
        const auto obj = entry.toObject();
        ArucoTarget target;
        const auto typeStr = obj.value("type").toString("aruco").toLower();
        if (typeStr == "apriltag")
            target.type = FiducialType::AprilTag;
        else
            target.type = FiducialType::Aruco;

        const auto family = obj.value("family").toString().toStdString();
        target.dictionary = !family.empty() ? family : obj.value("dictionary").toString().toStdString();
        if (target.type == FiducialType::AprilTag && target.dictionary.empty())
            target.dictionary = "tagStandard41h12";
        auto idsArray = obj.value("marker_ids").toArray();
        for (const auto &idVal : idsArray)
            target.markerIds.push_back(idVal.toInt());
        if (!target.dictionary.empty())
            _arucoTargets.push_back(std::move(target));
    }

    _displayFpsLimit = doc.object().value("display_fps_limit").toDouble(0.0);
    _showDepthPreview = doc.object().value("show_depth_preview").toBool(true);

    // tasks_path
    {
        std::filesystem::path configPath(_path);
        auto configDir = configPath.parent_path();
        auto tasksPath = doc.object().value("tasks_path").toString().toStdString();
        if (tasksPath.empty())
            tasksPath = "resources/tasks";
        std::filesystem::path candidate(tasksPath);
        if (candidate.is_relative())
            candidate = configDir / candidate;
        _tasksRoot = candidate.lexically_normal().string();
        _logger.info("Tasks root set to: %s", _tasksRoot.c_str());
    }

    // captures_path
    {
        std::filesystem::path configPath(_path);
        auto configDir = configPath.parent_path();
        auto capturesPath = doc.object().value("captures_path").toString().toStdString();
        if (capturesPath.empty())
            capturesPath = "logs/captures";
        std::filesystem::path candidate(capturesPath);
        if (candidate.is_relative())
            candidate = configDir / candidate;
        _capturesRoot = candidate.lexically_normal().string();
        _logger.info("Captures root set to: %s", _capturesRoot.c_str());
    }

    // vlm config
    {
        std::filesystem::path configPath(_path);
        auto configDir = configPath.parent_path();
        const auto vlmObj = doc.object().value("vlm").toObject();
        _vlmConfig.model = vlmObj.value("model").toString("gemini-2.5-flash").toStdString();
        _vlmConfig.endpoint = vlmObj.value("endpoint")
                                   .toString("https://api.gptoai.top/v1/chat/completions")
                                   .toStdString();
        std::string defaultPrompt = (configDir / "prompts/vlm_task_prompt.txt").string();
        _vlmConfig.promptPath = vlmObj.value("prompt_path")
                                    .toString(QString::fromStdString(defaultPrompt))
                                    .toStdString();
        _vlmConfig.apiKey = vlmObj.value("api_key").toString().toStdString();
    }

    // audio prompts
    {
        const auto audioObj = doc.object().value("audio_prompts").toObject();
        _audioConfig.enabled = audioObj.value("enabled").toBool(false);
        _audioConfig.volume = static_cast<float>(audioObj.value("volume").toDouble(1.0));
        _audioConfig.mode = audioObj.value("mode").toString("index_tts").toLower().toStdString();
        _audioConfig.language = audioObj.value("language").toString("chinese").toLower().toStdString();
        _audioConfig.keyframeOnly = audioObj.value("keyframe_only").toBool(false);
        _audioConfig.indexTts.audioPaths.clear();
        _audioConfig.texts.clear();
        _audioConfig.keybindings.clear();
        const auto indexObj = audioObj.value("index_tts").toObject();
        _audioConfig.indexTts.endpoint = indexObj.value("endpoint").toString().toStdString();
        const auto refs = indexObj.value("audio_paths").toArray();
        for (const auto &r : refs)
            _audioConfig.indexTts.audioPaths.push_back(r.toString().toStdString());
        const auto textsObj = audioObj.value("texts").toObject();
        for (auto it = textsObj.begin(); it != textsObj.end(); ++it)
            _audioConfig.texts[it.key().toStdString()] = it.value().toString().toStdString();
        const auto keysObj = audioObj.value("keybindings").toObject();
        for (auto it = keysObj.begin(); it != keysObj.end(); ++it)
            _audioConfig.keybindings[it.key().toStdString()] = it.value().toString().toStdString();
    }

    return true;
}

std::optional<CameraConfig> ConfigManager::getCameraConfigById(int id) const
{
    for (const auto &cfg : _cameraConfigs)
    {
        if (cfg.id == id)
            return cfg;
    }
    return std::nullopt;
}

std::pair<int, int> ConfigManager::parseResolution(const std::string &value)
{
    auto pos = value.find_first_of("xX");
    if (pos == std::string::npos)
        return {0, 0};
    const auto width = std::stoi(value.substr(0, pos));
    const auto height = std::stoi(value.substr(pos + 1));
    return {width, height};
}

int ConfigManager::defaultDepthChunkSize(int width, int height)
{
    if (width <= 0 || height <= 0)
        return 0;
    constexpr int64_t targetBytes = 32LL * 1024 * 1024;
    const int64_t frameBytes = static_cast<int64_t>(width) * static_cast<int64_t>(height) * 2;
    if (frameBytes <= 0)
        return 0;
    int chunk = static_cast<int>(targetBytes / frameBytes);
    if (chunk < 1)
        chunk = 1;
    chunk = highestPowerOfTwo(chunk);
    if (chunk < 4)
        chunk = 4;
    if (chunk > 128)
        chunk = 128;
    return chunk;
}

bool ConfigManager::updateCameraConfig(int id, const CameraConfig &config)
{
    for (auto &cfg : _cameraConfigs)
    {
        if (cfg.id == id)
        {
            cfg = config;
            _logger.info("Updated camera %d configuration", id);
            return true;
        }
    }
    _logger.warn("Camera %d configuration not found", id);
    return false;
}
