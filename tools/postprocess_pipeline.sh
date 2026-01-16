#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <captures_root> <tag_map_json>"
  exit 1
fi

ROOT="$1"
TAG_MAP="$2"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" tools/filter_short_sessions.py "$ROOT" --find-meta true
"$PYTHON_BIN" tools/align_depth.py "$ROOT" --delete-original-depth true
"$PYTHON_BIN" tools/sort_timestamps.py "$ROOT" --find-meta true
"$PYTHON_BIN" tools/align_timestamps.py "$ROOT" --find-meta true
"$PYTHON_BIN" tools/encode_videos.py "$ROOT" --find-meta true --output-name color_aligned.mp4
"$PYTHON_BIN" tools/estimate_camera_poses_from_apriltag.py "$ROOT" --tag-map "$TAG_MAP" --pnp-method ippe --no-ransac --output-name camera_poses_apriltag.json --threads 8
"$PYTHON_BIN" tools/postprocess_camera_poses.py "$ROOT" --no-hampel
"$PYTHON_BIN" tools/visualize_session_poses_full.py "$ROOT" --tag-map "$TAG_MAP" --stride 300
"$PYTHON_BIN" tools/run_sam2_apriltag_detect_propagate.py "$ROOT" --cuda-devices 0,1,2,5 --chunk-size 1500 --async-loading-frames --diffueraser-chunk-size 500 --prompt-stride 30
