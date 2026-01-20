#!/usr/bin/env python3
"""
Run pupil_apriltags detection on a single image.

Usage:
  python -m tools.apriltag_detect_single /path/to/image.png --family tag41h12
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2

try:
    from pupil_apriltags import Detector
except ImportError as exc:
    raise SystemExit("pupil_apriltags is required: pip install pupil_apriltags") from exc


def draw_detections(image_bgr, detections) -> None:
    for det in detections:
        corners = det.corners.astype(int).reshape(-1, 2)
        cv2.polylines(image_bgr, [corners], True, (0, 255, 0), 2)
        center = tuple(det.center.astype(int))
        cv2.circle(image_bgr, center, 3, (0, 0, 255), -1)
        cv2.putText(
            image_bgr,
            f"id={det.tag_id}",
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect AprilTags in a single image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--family", default="tagStandard41h12", help="AprilTag family (default: tag41h12)")
    parser.add_argument("--decimate", type=float, default=1.0, help="Quad decimation (default 1.0)")
    parser.add_argument("--sigma", type=float, default=0.0, help="Quad sigma (default 0.0)")
    parser.add_argument("--threads", type=int, default=0, help="Detector threads (0=auto)")
    parser.add_argument("--out", default=None, help="Optional output image path for visualization")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"[apriltag] image not found: {image_path}")
        return 1

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[apriltag] failed to read image: {image_path}")
        return 1
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    threads = args.threads if args.threads > 0 else max(1, cv2.getNumberOfCPUs())
    detector = Detector(
        families="tagStandard41h12",
        nthreads=threads,
        quad_decimate=args.decimate,
        quad_sigma=args.sigma,
        refine_edges=True,
    )

    detections = detector.detect(image_gray)
    if not detections:
        print("[apriltag] no tags detected")
        return 1

    print(f"[apriltag] detected {len(detections)} tag(s)")
    for det in detections:
        print(f"  id={det.tag_id} decision_margin={det.decision_margin:.2f}")

    if args.out:
        draw_detections(image_bgr, detections)
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image_bgr)
        print(f"[apriltag] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
