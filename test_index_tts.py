import requests


def main() -> None:
    """
    Simple smoke test for IndexTTS HTTP API.

    Prerequisites:
      1) In another terminal, from repo root:
         cd index-tts-vllm
         python3 api_server.py \
             --model_dir ./checkpoints/Index-TTS-1.5-vLLM \
             --host 127.0.0.1 \
             --port 6006
      2) Ensure 'voice' below exists in assets/speaker.json.
    """

    url = "http://127.0.0.1:6006/audio/speech"

    text = "你好，这是一个 Index TTS 测试语音。"

    # Make sure this matches a key in index-tts-vllm/assets/speaker.json
    # 当前日志中注册的说话人是 "jay_klee"
    voice = "jay_klee"

    payload = {
        "model": "IndexTTS-1.5-vLLM",
        "input": text,
        "voice": voice,
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()

    output_path = "test_tts.wav"
    with open(output_path, "wb") as f:
        f.write(resp.content)

    print(f"TTS 测试完成，已保存到: {output_path}")


if __name__ == "__main__":
    main()
