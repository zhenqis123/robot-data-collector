#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <captures_root> <tag_map_json> [frame_index] [consistency_thresh]"
  echo "Example: $0 /data/captures /data/tag_map.json 100 0.03"
  exit 1
fi

ROOT="$1"
TAG_MAP="$2"
FRAME_INDEX="${3:-0}"
CONSISTENCY="${4:-0.03}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
# Make TOOLS_DIR an absolute path so we can reliably reference repo-relative resources
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default filter config (repo-relative). Can be overridden by setting FILTER_CFG env var.
FILTER_CFG="${FILTER_CFG:-$TOOLS_DIR/../post_process/resources/point_cloud_filter.json}"
# Default working space JSON (repo-relative). Can be overridden by setting WORKSPACE env var.
WORKSPACE="${WORKSPACE:-$TOOLS_DIR/../post_process/resources/working_space.json}"

# echo "=== Starting 3-View Intersection Reconstruction Pipeline ==="
# echo "Root: $ROOT"
# echo "Tag Map: $TAG_MAP"
# echo "Target Frame: $FRAME_INDEX"
# echo "Consistency Threshold: $CONSISTENCY m"

# # 1. Pre-processing (Filtering & Alignment)
# echo "--- Step 1: Filtering & Alignment ---"
# "$PYTHON_BIN" "$TOOLS_DIR/filter_short_sessions.py" "$ROOT" --find-meta true
# "$PYTHON_BIN" "$TOOLS_DIR/filter_depth_h5.py" "$ROOT" --find-meta true --min-distance 0.2 --max-distance 3.0 \
#     --spatial-alpha 0.5 --spatial-delta 20 --spatial-magnitude 2 \
#     --temporal-alpha 0.4 --temporal-delta 20
# "$PYTHON_BIN" "$TOOLS_DIR/align_depth.py" "$ROOT" --delete-original-depth false
# "$PYTHON_BIN" "$TOOLS_DIR/sort_timestamps.py" "$ROOT" --find-meta true
# "$PYTHON_BIN" "$TOOLS_DIR/align_timestamps.py" "$ROOT" --find-meta true
# "$PYTHON_BIN" "$TOOLS_DIR/encode_videos.py" "$ROOT" --find-meta true --output-name color_aligned.mp4

# # 2. Pose Estimation
# echo "--- Step 2: Pose Estimation ---"
# # Check if poses already exist to save time (optional, tools usually handle this but let's be explicit about intention)
# # We run it to ensure we have the latest.
# "$PYTHON_BIN" "$TOOLS_DIR/estimate_camera_poses_from_apriltag.py" "$ROOT" \
#     --tag-map "$TAG_MAP" \
#     --pnp-method ippe \
#     --no-ransac \
#     --output-name camera_poses_apriltag.json \
#     --threads 1

# "$PYTHON_BIN" "$TOOLS_DIR/postprocess_camera_poses.py" "$ROOT" \
#     --poses-name camera_poses_apriltag.json \
#     --output-name camera_poses_apriltag_post.json \
#     --no-hampel

# 3. Reconstruction
echo "--- Step 3: 3-View Intersection Reconstruction ---"
"$PYTHON_BIN" "$TOOLS_DIR/reconstruct_3view_intersection.py" "$ROOT" \
  --poses "$ROOT/camera_poses_apriltag_post.json" \
  --frame-index "$FRAME_INDEX" \
  --consistency-thresh "$CONSISTENCY" \
  --output-dir "$ROOT/reconstructions_intersection" \
  --filter-config "$FILTER_CFG" \
  --working-space "$WORKSPACE"

echo "=== Pipeline Complete ==="

