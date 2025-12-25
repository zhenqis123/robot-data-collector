#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <captures_root> <tag_map_json>"
  exit 1
fi

ROOT="$1"
TAG_MAP="$2"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" tools/align_depth.py "$ROOT" --delete-original-depth true
"$PYTHON_BIN" tools/sort_timestamps.py "$ROOT" --find-meta false

"$PYTHON_BIN" tools/align_timestamps.py "$ROOT" --find-meta false

"$PYTHON_BIN" tools/encode_videos.py "$ROOT" --find-meta false
"$PYTHON_BIN" tools/estimate_camera_poses_from_apriltag.py "$ROOT" --tag-map "$TAG_MAP"
"$PYTHON_BIN" tools/visualize_camera_poses_3d.py "$ROOT" --tag-map "$TAG_MAP"
