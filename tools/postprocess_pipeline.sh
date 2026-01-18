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
"$PYTHON_BIN" tools/filter_depth_h5.py "$ROOT" --find-meta true --min-distance 0.2 --max-distance 3.0 \
    --spatial-alpha 0.5 --spatial-delta 20 --spatial-magnitude 2 \
    --temporal-alpha 0.4 --temporal-delta 20
"$PYTHON_BIN" tools/align_depth.py "$ROOT" --delete-original-depth false
"$PYTHON_BIN" tools/depth_h5_viz_video.py "$ROOT" --include-aligned true
"$PYTHON_BIN" tools/sort_timestamps.py "$ROOT" --find-meta true
"$PYTHON_BIN" tools/align_timestamps.py "$ROOT" --find-meta true
"$PYTHON_BIN" tools/encode_videos.py "$ROOT" --find-meta true --output-name color_aligned.mp4
"$PYTHON_BIN" tools/estimate_camera_poses_from_apriltag.py "$ROOT" --tag-map "$TAG_MAP" --pnp-method ippe --no-ransac --output-name camera_poses_apriltag.json --threads 1
"$PYTHON_BIN" tools/postprocess_camera_poses.py "$ROOT" --no-hampel
"$PYTHON_BIN" tools/visualize_session_poses_full.py "$ROOT" --tag-map "$TAG_MAP" --stride 1
"$PYTHON_BIN" tools/run_sam2_apriltag_detect_propagate.py "$ROOT" --cuda-devices 0 --chunk-size 1500 --async-loading-frames --diffueraser-chunk-size 100 --prompt-stride 30
