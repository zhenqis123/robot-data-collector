#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <QByteArray>

#include <QObject>

#include "TaskLoader.h"

struct AudioPromptConfig
{
    bool enabled{false};
    float volume{1.0f};
    std::string mode{"index_tts"};
    struct IndexTts
    {
        std::string endpoint;
        std::vector<std::string> audioPaths;
    } indexTts;
    std::unordered_map<std::string, std::string> texts;
};

class Logger;

class AudioPromptPlayer : public QObject
{
    Q_OBJECT
public:
    explicit AudioPromptPlayer(Logger &logger);
    void configure(const AudioPromptConfig &config);
    void setEnabled(bool enabled);
    void setVolume(float volume);

    // Play a prompt for the given key. Optional override text replaces template text.
    void play(const std::string &key,
              const std::string &taskId = "",
              const std::string &stepId = "",
              const std::string &stepDesc = "",
              const std::string &overrideText = "");
    void preloadTexts(const std::vector<std::string> &texts);
    std::string renderText(const std::string &key,
                           const std::string &taskId,
                           const std::string &stepId,
                           const std::string &stepDesc,
                           const std::string &overrideText) const;

private:
    void playFallbackWav();
    Logger &_logger;
    AudioPromptConfig _config;
    QByteArray synthesizeIndexTts(const std::string &text);
    void synthesizeAndPlayIndexTts(const std::string &text);
};
