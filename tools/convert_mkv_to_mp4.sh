#!/usr/bin/env bash
set -euo pipefail

root_dir="${1:-.}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found in PATH" >&2
  exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe not found in PATH" >&2
  exit 1
fi

normalize_bitrate() {
  local bps="$1"
  if [[ -z "$bps" || "$bps" == "N/A" ]]; then
    echo ""
    return
  fi
  echo "$bps"
}

bps_to_k() {
  local bps="$1"
  if [[ -z "$bps" ]]; then
    echo ""
    return
  fi
  local kbps=$((bps / 1000))
  if [[ "$kbps" -le 0 ]]; then
    echo ""
    return
  fi
  echo "${kbps}k"
}

find "$root_dir" -type f -name "*.mkv" -print0 | while IFS= read -r -d '' file; do
  out="${file%.mkv}.mp4"
  if [[ -f "$out" ]]; then
    echo "skip: $out already exists"
    continue
  fi

  vcodec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$file" || true)
  acodec=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$file" || true)

  if [[ "$vcodec" == "h264" && ( -z "$acodec" || "$acodec" == "aac" ) ]]; then
    echo "remux: $file -> $out"
    ffmpeg -hide_banner -y -i "$file" -map 0 -c copy -movflags +faststart "$out"
    continue
  fi

  vbps=$(ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of csv=p=0 "$file" || true)
  vbps=$(normalize_bitrate "$vbps")
  if [[ -z "$vbps" ]]; then
    vbps=$(ffprobe -v error -show_entries format=bit_rate -of csv=p=0 "$file" || true)
    vbps=$(normalize_bitrate "$vbps")
  fi
  vbitrate=$(bps_to_k "$vbps")
  if [[ -z "$vbitrate" ]]; then
    vbitrate="8000k"
  fi

  if [[ -n "$acodec" ]]; then
    abps=$(ffprobe -v error -select_streams a:0 -show_entries stream=bit_rate -of csv=p=0 "$file" || true)
    abps=$(normalize_bitrate "$abps")
    abitrate=$(bps_to_k "$abps")
    if [[ -z "$abitrate" ]]; then
      abitrate="192k"
    fi
    echo "transcode: $file -> $out (v:$vbitrate a:$abitrate)"
    ffmpeg -hide_banner -y -i "$file" \
      -c:v libx264 -preset medium -profile:v high -level 4.2 -pix_fmt yuv420p -b:v "$vbitrate" -maxrate "$vbitrate" -bufsize "$vbitrate" \
      -c:a aac -b:a "$abitrate" \
      -movflags +faststart "$out"
  else
    echo "transcode: $file -> $out (v:$vbitrate, no audio)"
    ffmpeg -hide_banner -y -i "$file" \
      -c:v libx264 -preset medium -profile:v high -level 4.2 -pix_fmt yuv420p -b:v "$vbitrate" -maxrate "$vbitrate" -bufsize "$vbitrate" \
      -an -movflags +faststart "$out"
  fi

done
