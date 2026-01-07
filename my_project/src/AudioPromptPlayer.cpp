#include "AudioPromptPlayer.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

#include <QByteArray>
#include <QDateTime>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMediaPlayer>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QThreadPool>
#include <QUrl>
#include <QtConcurrentRun>

#include "Logger.h"

namespace
{
QByteArray floatToInt16Pcm(const QByteArray &floatBytes)
{
    QByteArray pcm;
    const auto *src = reinterpret_cast<const float *>(floatBytes.constData());
    const size_t count = static_cast<size_t>(floatBytes.size() / sizeof(float));
    pcm.resize(static_cast<int>(count * sizeof(int16_t)));
    auto *dst = reinterpret_cast<int16_t *>(pcm.data());
    for (size_t i = 0; i < count; ++i)
    {
        const float clamped = std::max(-1.0f, std::min(1.0f, src[i]));
        dst[i] = static_cast<int16_t>(clamped * 32767.0f);
    }
    return pcm;
}

bool writeWavFile(const QString &path, const QByteArray &pcm, int sampleRate, int channels, int bitsPerSample)
{
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly))
        return false;

    const qint32 byteRate = sampleRate * channels * bitsPerSample / 8;
    const qint16 blockAlign = channels * bitsPerSample / 8;
    const qint32 dataSize = pcm.size();
    const qint32 riffChunkSize = 36 + dataSize;

    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char *>(&riffChunkSize), 4);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    const qint32 subchunk1Size = 16;
    const qint16 audioFormat = 1; // PCM
    f.write(reinterpret_cast<const char *>(&subchunk1Size), 4);
    f.write(reinterpret_cast<const char *>(&audioFormat), 2);
    f.write(reinterpret_cast<const char *>(&channels), 2);
    f.write(reinterpret_cast<const char *>(&sampleRate), 4);
    f.write(reinterpret_cast<const char *>(&byteRate), 4);
    f.write(reinterpret_cast<const char *>(&blockAlign), 2);
    f.write(reinterpret_cast<const char *>(&bitsPerSample), 2);
    f.write("data", 4);
    f.write(reinterpret_cast<const char *>(&dataSize), 4);
    f.write(pcm);
    f.close();
    return true;
}

QString writeTempWavFromFloat(const QByteArray &floatBytes, int sampleRate)
{
    const QByteArray pcm = floatToInt16Pcm(floatBytes);
    const QString path = QDir::temp().filePath(
        QStringLiteral("datacollector_prompt_%1.wav").arg(QDateTime::currentMSecsSinceEpoch()));
    if (!writeWavFile(path, pcm, sampleRate, 1, 16))
        return {};
    return path;
}
} // namespace

AudioPromptPlayer::AudioPromptPlayer(Logger &logger)
    : _logger(logger)
{
}

void AudioPromptPlayer::configure(const AudioPromptConfig &config)
{
    _config = config;
}

void AudioPromptPlayer::setEnabled(bool enabled)
{
    _config.enabled = enabled;
}

void AudioPromptPlayer::setVolume(float volume)
{
    _config.volume = volume;
}

std::string AudioPromptPlayer::renderText(const std::string &key,
                                          const std::string &taskId,
                                          const std::string &stepId,
                                          const std::string &stepDesc,
                                          const std::string &overrideText) const
{
    std::string text = overrideText.empty() ? _config.texts.count(key) ? _config.texts.at(key) : "" : overrideText;
    auto replaceAll = [](std::string src, const std::string &from, const std::string &to) {
        size_t pos = 0;
        while ((pos = src.find(from, pos)) != std::string::npos)
        {
            src.replace(pos, from.size(), to);
            pos += to.size();
        }
        return src;
    };
    text = replaceAll(text, "{task_id}", taskId);
    text = replaceAll(text, "{step_id}", stepId);
    text = replaceAll(text, "{step_desc}", stepDesc);
    return text;
}

void AudioPromptPlayer::play(const std::string &key,
                             const std::string &taskId,
                             const std::string &stepId,
                             const std::string &stepDesc,
                             const std::string &overrideText)
{
    if (!_config.enabled)
        return;
    const auto text = renderText(key, taskId, stepId, stepDesc, overrideText);
    if (text.empty())
        return;
    QtConcurrent::run(QThreadPool::globalInstance(), [this, text]() {
        synthesizeAndPlayIndexTts(text);
    });
}

void AudioPromptPlayer::preloadTexts(const std::vector<std::string> &texts)
{
    (void)texts;
}

QByteArray AudioPromptPlayer::synthesizeIndexTts(const std::string &text)
{
    QNetworkAccessManager mgr;
    QNetworkRequest req(QString::fromStdString(_config.indexTts.endpoint));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    QJsonObject body;
    body["text"] = QString::fromStdString(text);
    QJsonArray arr;
    for (const auto &p : _config.indexTts.audioPaths)
        arr.append(QString::fromStdString(p));
    body["audio_paths"] = arr;
    body["seed"] = 8;
    QNetworkReply *reply = mgr.post(req, QJsonDocument(body).toJson());
    QEventLoop loop;
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();
    QByteArray data;
    if (reply->error() == QNetworkReply::NoError)
        data = reply->readAll();
    reply->deleteLater();
    return data;
}

void AudioPromptPlayer::synthesizeAndPlayIndexTts(const std::string &text)
{
    const auto wavBytes = synthesizeIndexTts(text);
    if (wavBytes.isEmpty())
    {
        playFallbackWav();
        return;
    }
    QString tempPath = QDir::temp().filePath(
        QStringLiteral("datacollector_prompt_%1.wav").arg(QDateTime::currentMSecsSinceEpoch()));
    QFile f(tempPath);
    if (!f.open(QIODevice::WriteOnly))
    {
        playFallbackWav();
        return;
    }
    f.write(wavBytes);
    f.close();

    auto *player = new QMediaPlayer();
    player->setMedia(QUrl::fromLocalFile(tempPath));
    player->setVolume(static_cast<int>(_config.volume * 100));
    QObject::connect(player, &QMediaPlayer::stateChanged, player, &QObject::deleteLater);
    player->play();
}

void AudioPromptPlayer::playFallbackWav()
{
    QString file = QStringLiteral(":/audio/fallback.wav");
    if (!QFile::exists(file))
    {
        _logger.warn("AudioPromptPlayer: fallback wav not found");
        return;
    }
    auto *player = new QMediaPlayer();
    player->setMedia(QUrl::fromLocalFile(file));
    player->setVolume(static_cast<int>(_config.volume * 100));
    QObject::connect(player, &QMediaPlayer::stateChanged, player, &QObject::deleteLater);
    player->play();
}
