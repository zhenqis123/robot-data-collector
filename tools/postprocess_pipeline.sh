#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash tools/postprocess_pipeline.sh --root <capture_dir> --tag-length <meters> [options]

Options:
  --root <dir>           Capture root containing meta.json and camera folders
  --tag-length <m>       AprilTag side length in meters (black square)
  --tag-family <name>    AprilTag family (default: tagStandard41h12)
  --ref-camera <id>      Reference camera id for timestamps (default: first in meta.json)
  --tag-camera <id>      Camera id used for tag map detection (default: ref-camera)
  --tag-images <dir>     Override tag images directory (color frames)
  --tag-map <path>       Output tag map path (default: <root>/apriltag_map.json)
  --tag-figure <path>    Output tag map figure path (default: <root>/apriltag_map_3d.png)
  --fps <num>            Video FPS for encoding (default: 30)
  --workers <num>        Worker threads for depth alignment (default: 4)
  --skip-align-depth     Skip depth alignment
  --skip-align-ts        Skip timestamp alignment (frames_aligned.csv)
  --skip-tag-map         Skip tag map estimation
  --skip-camera-poses    Skip per-frame camera pose estimation
  --skip-encode          Skip video encoding + video position annotations
EOF
}

ROOT=""
TAG_LENGTH=""
TAG_FAMILY="tagStandard41h12"
REF_CAMERA=""
TAG_CAMERA=""
TAG_IMAGES=""
TAG_MAP=""
TAG_FIGURE=""
FPS="30"
WORKERS="4"
SKIP_ALIGN_DEPTH="false"
SKIP_ALIGN_TS="false"
SKIP_TAG_MAP="false"
SKIP_CAMERA_POSES="false"
SKIP_ENCODE="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        --tag-length) TAG_LENGTH="$2"; shift 2 ;;
        --tag-family) TAG_FAMILY="$2"; shift 2 ;;
        --ref-camera) REF_CAMERA="$2"; shift 2 ;;
        --tag-camera) TAG_CAMERA="$2"; shift 2 ;;
        --tag-images) TAG_IMAGES="$2"; shift 2 ;;
        --tag-map) TAG_MAP="$2"; shift 2 ;;
        --tag-figure) TAG_FIGURE="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --skip-align-depth) SKIP_ALIGN_DEPTH="true"; shift ;;
        --skip-align-ts) SKIP_ALIGN_TS="true"; shift ;;
        --skip-tag-map) SKIP_TAG_MAP="true"; shift ;;
        --skip-camera-poses) SKIP_CAMERA_POSES="true"; shift ;;
        --skip-encode) SKIP_ENCODE="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$ROOT" ]]; then
    echo "Missing --root"
    usage
    exit 1
fi

ROOT="$(cd "$ROOT" && pwd)"
META_PATH="$ROOT/meta.json"
if [[ ! -f "$META_PATH" ]]; then
    echo "meta.json not found under $ROOT"
    exit 1
fi

if [[ -z "$TAG_MAP" ]]; then
    TAG_MAP="$ROOT/apriltag_map.json"
fi
if [[ -z "$TAG_FIGURE" ]]; then
    TAG_FIGURE="$ROOT/apriltag_map_3d.png"
fi

get_first_camera_id() {
    python3 - <<'PY' "$META_PATH"
import json
import sys
path = sys.argv[1]
meta = json.load(open(path, "r"))
cams = meta.get("cameras", [])
if not cams:
    sys.exit(1)
cid = cams[0].get("id", "")
print(cid)
PY
}

sanitize_id() {
    python3 - <<'PY' "$1"
import sys
value = sys.argv[1]
out = []
for ch in value:
    if ch.isalnum() or ch in "-_":
        out.append(ch)
    else:
        out.append("_")
print("".join(out))
PY
}

if [[ -z "$REF_CAMERA" ]]; then
    REF_CAMERA="$(get_first_camera_id || true)"
fi
if [[ -z "$TAG_CAMERA" ]]; then
    TAG_CAMERA="$REF_CAMERA"
fi

if [[ "$SKIP_TAG_MAP" == "false" ]]; then
    if [[ -z "$TAG_LENGTH" ]]; then
        echo "Missing --tag-length"
        usage
        exit 1
    fi
    if [[ -z "$TAG_IMAGES" ]]; then
        if [[ -z "$TAG_CAMERA" ]]; then
            echo "Failed to determine tag camera id"
            exit 1
        fi
        TAG_IMAGES="$ROOT/$(sanitize_id "$TAG_CAMERA")/color"
    fi
fi

if [[ "$SKIP_ALIGN_DEPTH" == "false" ]]; then
    echo "[pipeline] depth alignment"
    python3 tools/align_depth.py "$ROOT" --workers "$WORKERS" --find-meta true
fi

if [[ "$SKIP_ALIGN_TS" == "false" ]]; then
    echo "[pipeline] timestamp alignment"
    if [[ -n "$REF_CAMERA" ]]; then
        python3 tools/align_timestamps.py "$ROOT" --reference "$REF_CAMERA" --find-meta true
    else
        python3 tools/align_timestamps.py "$ROOT" --find-meta true
    fi
fi

if [[ "$SKIP_TAG_MAP" == "false" ]]; then
    echo "[pipeline] apriltag map"
    python3 tools/estimate_apriltag_pose.py \
        --images "$TAG_IMAGES" \
        --meta "$META_PATH" \
        --tag-length "$TAG_LENGTH" \
        --family "$TAG_FAMILY" \
        --output "$TAG_MAP" \
        --figure "$TAG_FIGURE"
fi

if [[ "$SKIP_CAMERA_POSES" == "false" ]]; then
    echo "[pipeline] camera poses from apriltag"
    python3 tools/estimate_camera_poses_from_apriltag.py "$ROOT" --tag-map "$TAG_MAP"
fi

if [[ "$SKIP_ENCODE" == "false" ]]; then
    echo "[pipeline] encode videos + annotate video positions"
    python3 tools/encode_videos.py "$ROOT" --fps "$FPS"
fi

echo "[pipeline] done"
