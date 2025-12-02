#!/usr/bin/env bash
set -euo pipefail

# espeak-ng TTS demo with embedded long text.
# Usage: ./tts_espeak_demo.sh [output.wav]

OUT=${1:-"tts_demo.wav"}
VOICE="en"

# Embedded long test text
read -r -d '' TEXT << 'EOF'
This is a long test passage intended for evaluating text-to-speech systems,
network transmission pipelines, buffer handling, and remote audio playback.
The content includes multiple sentences, varied punctuation, and a natural
flow of phrases to more accurately simulate real-world usage scenarios.
When using TTS engines such as espeak-ng, Flite, or neural systems accessed
through local or remote APIs, it is often helpful to have a sufficiently long
block of text that stresses continuous streaming capabilities as well as
overall voice consistency.

Additionally, such a passage provides an opportunity to observe how well the
system handles intonation, phrasing, pauses, and sentence boundaries. This
text does not contain any special formatting, code snippets, or unusual
Unicode characters, which ensures broad compatibility with most engines and
deployment environments.

In practical testing, this kind of paragraph can also help determine whether
remote transmission—such as PulseAudio forwarding, SSH tunneling, or other
network-based audio solutions—introduces latency, jitter, or audio artifacts.
As you adjust your configuration, experiment with different voice settings,
rates, and pitch levels, and monitor how the playback behaves over time.
EOF

echo "Synthesizing with espeak-ng..."
espeak-ng -v "$VOICE" -w "$OUT" "$TEXT"
echo "Done. Output file: $OUT"
