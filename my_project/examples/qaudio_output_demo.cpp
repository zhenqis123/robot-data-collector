#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QMediaPlayer>
#include <QUrl>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "piper.h"
}

namespace
{
QByteArray floatToInt16Pcm(const std::vector<float> &samples)
{
    QByteArray pcm;
    pcm.resize(static_cast<int>(samples.size() * sizeof(int16_t)));
    auto *out = reinterpret_cast<int16_t *>(pcm.data());
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const float clamped = std::clamp(samples[i], -1.0f, 1.0f);
        out[i] = static_cast<int16_t>(clamped * 32767.0f);
    }
    return pcm;
}

bool writeWav(const QString &path, const QByteArray &pcm, int sampleRate, int channels, int bitsPerSample)
{
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly))
    {
        qWarning() << "Failed to open wav for write:" << path;
        return false;
    }
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
} // namespace

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);

    const char *voiceOnnx = PIPER_VOICE_ONNX;
    const char *voiceConfig = PIPER_VOICE_CONFIG;
    const char *espeakData = PIPER_ESPEAK_DATA;
    const int sampleRate = PIPER_SAMPLE_RATE;
    const std::string text = "This is a Piper text to speech playback test.";
    const QString wavPath = "/tmp/piper_demo_output.wav";

    piper_synthesizer *synth = piper_create(voiceOnnx, voiceConfig, espeakData);
    if (!synth)
    {
        qWarning() << "Failed to init piper synthesizer.";
        return 1;
    }

    std::vector<float> pcmFloat;
    piper_synthesize_options options = piper_default_synthesize_options(synth);
    const auto startRc = piper_synthesize_start(synth, text.c_str(), &options);
    if (startRc != PIPER_OK)
    {
        qWarning() << "piper_synthesize_start failed:" << startRc;
        piper_free(synth);
        return 1;
    }
    piper_audio_chunk chunk;
    while (piper_synthesize_next(synth, &chunk) != PIPER_DONE)
    {
        pcmFloat.insert(pcmFloat.end(), chunk.samples, chunk.samples + chunk.num_samples);
    }
    piper_free(synth);

    qDebug() << "Generated samples:" << pcmFloat.size();

    QByteArray pcm = floatToInt16Pcm(pcmFloat);
    if (!writeWav(wavPath, pcm, sampleRate, 1, 16))
        return 1;
    qDebug() << "Wrote wav to" << wavPath << "bytes" << pcm.size();

    auto *player = new QMediaPlayer(&app);
    player->setMedia(QUrl::fromLocalFile(wavPath));
    player->setVolume(100);
    QObject::connect(player, &QMediaPlayer::stateChanged, &app, [player](QMediaPlayer::State state) {
        qDebug() << "Player state:" << state << "error:" << player->error();
        if (state == QMediaPlayer::StoppedState)
            QCoreApplication::quit();
    });
    QObject::connect(player, QOverload<QMediaPlayer::Error>::of(&QMediaPlayer::error), &app, [player](QMediaPlayer::Error error) {
        qWarning() << "Player error:" << error << player->errorString();
        QCoreApplication::quit();
    });
    player->play();

    return app.exec();
}
