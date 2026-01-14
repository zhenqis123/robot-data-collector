#!/usr/bin/env bash
set -euo pipefail

# Usage: ./tools/run_reconstruction_pipeline.sh <capture_root> <tag_map_json> <target_camera_id>

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <captures_root> <tag_map_json> <target_camera_id>"
  echo "Example: $0 /data/sess_01 /data/tag_map.json RealSense_123"
  exit 1
fi

ROOT="$1"
TAG_MAP="$2"
TARGET_CAM="$3"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- Auto-generate Tag Map if missing ---
if [ ! -f "$TAG_MAP" ]; then
  echo ">>> [Info] Tag map not found at $TAG_MAP. Generating from data..."
  
  # 1. Locate meta.json
  META_JSON="$ROOT/meta.json"
  if [ ! -f "$META_JSON" ]; then
     # Fallback: look deeper
     META_JSON=$(find "$ROOT" -maxdepth 2 -name "meta.json" | head -n 1)
  fi
  
  if [ -z "$META_JSON" ] || [ ! -f "$META_JSON" ]; then
     echo "Error: meta.json not found in $ROOT"
     exit 1
  fi
  
  # 2. Locate a RealSense camera directory (prefer one with video/images)
  # We search for directories starting with RealSense
  CAM_DIR=$(find "$ROOT" -maxdepth 1 -type d -name "RealSense_*" | head -n 1)
  
  if [ -z "$CAM_DIR" ]; then
    echo "Error: Could not find any RealSense camera directory in $ROOT to generate tag map."
    exit 1
  fi
  
  echo ">>> Using meta: $META_JSON"
  echo ">>> Using camera data from: $CAM_DIR"
  
  "$PYTHON_BIN" tools/estimate_apriltag_pose.py \
      --images "$CAM_DIR" \
      --meta "$META_JSON" \
      --tag-length 0.034 \
      --output "$TAG_MAP" \
      --no-planar
      
  if [ ! -f "$TAG_MAP" ]; then
    echo "Error: Failed to generate tag map."
    exit 1
  fi
  echo ">>> Tag map generated successfully."
fi

echo ">>> Step 1: Aligning Depth to Color..."
"$PYTHON_BIN" tools/align_depth.py "$ROOT" --delete-original-depth false

echo ">>> Step 2: Sorting and Aligning Timestamps..."
"$PYTHON_BIN" tools/sort_timestamps.py "$ROOT" --find-meta true
"$PYTHON_BIN" tools/align_timestamps.py "$ROOT" --find-meta true

echo ">>> Step 3: Estimating Camera Poses via AprilTag..."
# Generates camera_poses_apriltag.json
"$PYTHON_BIN" tools/estimate_camera_poses_from_apriltag.py "$ROOT" \
    --tag-map "$TAG_MAP" \
    --pnp-method ippe \
    --no-ransac \
    --output-name camera_poses_apriltag.json

echo ">>> Step 4: Post-processing Poses (Smoothing)..."
# Generates session_poses.json
"$PYTHON_BIN" tools/postprocess_camera_poses.py "$ROOT" \
    --poses-name camera_poses_apriltag.json \
    --output-name session_poses.json

echo ">>> Step 5: Reconstructing Scene (Fusion & Cropping)..."
"$PYTHON_BIN" tools/reconstruct_scene.py "$ROOT" \
    --poses "$ROOT/session_poses.json" \
    --target-camera "$TARGET_CAM" \
    --output-dir "$ROOT/reconstructions" \
    --stride 5

echo "Done! Reconstructions saved to $ROOT/reconstructions"
