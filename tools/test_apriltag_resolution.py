#!/usr/bin/env python3
"""
Quick sanity check: resize frames and verify AprilTag detection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from pupil_apriltags import Detector


def detect_apriltags(image_bgr, family: str) -> list[int]:
    family_norm = family.strip()
    family_lower = family_norm.lower()
    if family_lower in {"41h12", "tag41h12"}:
        family_norm = "tagStandard41h12"
    elif not family_lower.startswith("tag"):
        family_norm = f"tag{family_norm}"

    detector = Detector(families=family_norm)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    return [int(det.tag_id) for det in detections]


def resize_keep_aspect(image_bgr, max_width: int, max_height: int | None):
    height, width = image_bgr.shape[:2]
    if max_width <= 0 and (max_height is None or max_height <= 0):
        return image_bgr

    scale_w = max_width / width if max_width > 0 else 1.0
    scale_h = max_height / height if max_height and max_height > 0 else 1.0
    scale = min(scale_w, scale_h, 1.0)
    if scale == 1.0:
        return image_bgr

    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test AprilTag detection after resizing frames."
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--max-width", type=int, default=640, help="Max width after resize")
    parser.add_argument("--max-height", type=int, default=0, help="Max height after resize (0=ignore)")
    parser.add_argument("--frame-step", type=int, default=10, help="Sample every N frames")
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to test")
    parser.add_argument("--apriltag-family", default="36h11", help="AprilTag family")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"[error] video not found: {video_path}")
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[error] could not open video: {video_path}")
        return 1

    total_tested = 0
    frames_with_tags = 0
    unique_tags: set[int] = set()
    frame_idx = 0
    max_height = args.max_height if args.max_height > 0 else None

    while total_tested < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        resized = resize_keep_aspect(frame, args.max_width, max_height)
        tag_ids = detect_apriltags(resized, args.apriltag_family)
        if tag_ids:
            frames_with_tags += 1
            unique_tags.update(tag_ids)

        total_tested += 1
        frame_idx += 1

    cap.release()

    if total_tested == 0:
        print("[warning] no frames tested")
        return 2

    ratio = frames_with_tags / total_tested
    print(f"Tested frames: {total_tested}")
    print(f"Frames with tags: {frames_with_tags} ({ratio:.2%})")
    print(f"Unique tag IDs: {sorted(unique_tags)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
