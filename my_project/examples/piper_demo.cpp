#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>

#include "piper.h"

int main(int argc, char **argv)
{
    // Adjust these paths via compile definitions if needed
#ifdef PIPER_VOICE_ONNX
    std::string voiceOnnx = PIPER_VOICE_ONNX;
#else
    // Default relative path or check environment
    std::string voiceOnnx = "piper1-gpl/libpiper/install/share/piper/en_US-amy-low.onnx";
    if (auto p = std::getenv("PIPER_VOICE_PATH")) voiceOnnx = p;
#endif
#ifdef PIPER_VOICE_CONFIG
    std::string voiceConfig = PIPER_VOICE_CONFIG;
#else
    std::string voiceConfig = voiceOnnx + ".json";
#endif
#ifdef PIPER_ESPEAK_DATA
    std::string espeakData = PIPER_ESPEAK_DATA;
#else
    std::string espeakData = "piper1-gpl/libpiper/install/espeak-ng-data";
    if (auto p = std::getenv("PIPER_ESPEAK_DATA")) espeakData = p;
#endif

    if (argc > 1)
        voiceOnnx = argv[1];
    if (argc > 2)
        voiceConfig = argv[2];
    if (argc > 3)
        espeakData = argv[3];

    if (!std::filesystem::exists(voiceOnnx))
    {
        std::cerr << "Voice ONNX not found: " << voiceOnnx << "\n"
                  << "Please download a Piper voice (\"*.onnx\" and \"*.onnx.json\") into the install/share/piper directory,\n"
                  << "or pass paths as arguments: ./piper_demo <voice.onnx> <voice.onnx.json> <espeak-ng-data>" << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(voiceConfig))
    {
        std::cerr << "Voice config not found: " << voiceConfig << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(espeakData))
    {
        std::cerr << "espeak-ng-data not found: " << espeakData << std::endl;
        return 1;
    }

    piper_synthesizer *synth = piper_create(voiceOnnx.c_str(), voiceConfig.c_str(), espeakData.c_str());
    if (!synth)
    {
        std::cerr << "Failed to create piper synthesizer. Check paths." << std::endl;
        return 1;
    }

    const bool isChinese = voiceOnnx.find("zh_") != std::string::npos;
    const std::string text = "这是一次中文合成测试。请朗读几句自然的中文，检查模型是否能正常输出语音。注意语速、停顿和连贯性。";

    std::ofstream audio_stream("piper_output.raw", std::ios::binary);
    if (!audio_stream.is_open())
    {
        std::cerr << "Failed to open output file" << std::endl;
        piper_free(synth);
        return 1;
    }

    piper_synthesize_options options = piper_default_synthesize_options(synth);
    piper_synthesize_start(synth, text.c_str(), &options);

    piper_audio_chunk chunk;
    while (piper_synthesize_next(synth, &chunk) != PIPER_DONE)
    {
        audio_stream.write(reinterpret_cast<const char *>(chunk.samples),
                           chunk.num_samples * sizeof(float));
    }

    piper_free(synth);
    std::cout << "Wrote piper_output.raw (float32 mono). Play with: aplay -r 22050 -c 1 -f FLOAT_LE -t raw piper_output.raw" << std::endl;
    return 0;
}
