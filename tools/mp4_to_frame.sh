#!/usr/bin/env bash
set -euo pipefail

# 检查参数
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <video_path> [output_format: jpg|png] [fps]"
  echo "Example: $0 ./data/my_video.mp4 jpg 30"
  exit 1
fi

VIDEO_PATH="$1"
FORMAT="${2:-jpg}" # 默认为 jpg
FPS="${3:-}"       # 默认为空（即原始帧率）

# 检查输入文件是否存在
if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Error: Video file '$VIDEO_PATH' not found."
  exit 1
fi

# 准备输出目录: /path/to/video.mp4 -> /path/to/video_frames/
DIR_NAME=$(dirname "$VIDEO_PATH")
BASE_NAME=$(basename "$VIDEO_PATH")
FILE_NO_EXT="${BASE_NAME%.*}"
OUT_DIR="${DIR_NAME}/${FILE_NO_EXT}_frames"

if [[ -d "$OUT_DIR" ]]; then
  echo "Warning: Output directory '$OUT_DIR' already exists."
else
  mkdir -p "$OUT_DIR"
fi

echo "Extracting frames from '$BASE_NAME' to '$OUT_DIR'..."

VF_ARGS=()
if [[ -n "$FPS" ]]; then
  echo "Using custom clean FPS: $FPS"
  VF_ARGS+=("-vf" "fps=$FPS")
fi

if [[ "$FORMAT" == "png" ]]; then
  # PNG (无损压缩，文件较大)
  # %06d 表示生成 000001.png, 000002.png 等 6 位数字序号
  ffmpeg -hide_banner -i "$VIDEO_PATH" "${VF_ARGS[@]}" "$OUT_DIR/%06d.png"
else
  # JPG (有损压缩，-q:v 2 为高质量，范围 2-31，2 最好)
  ffmpeg -hide_banner -i "$VIDEO_PATH" "${VF_ARGS[@]}" -q:v 2 "$OUT_DIR/%06d.jpg"
fi

echo "Done. All frames extracted in '$OUT_DIR'."