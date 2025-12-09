#!/usr/bin/env python3
"""
Batch post-processing for captured sessions:
- Run depth-to-color alignment (using offline_align core)
- Optionally stack two color views vertically into a single video

Usage:
  python postprocess.py /path/to/captures_root --stack color_view1.mp4 color_view2.mp4 --output stacked.mp4

Alignment writes depth_aligned/*.png next to existing color/depth.
Stacking writes a new video at the given output path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2

from offline_align import find_meta_files, process_capture


def stack_videos(top_path: Path, bottom_path: Path, output_path: Path) -> None:
    cap1 = cv2.VideoCapture(str(top_path))
    cap2 = cv2.VideoCapture(str(bottom_path))
    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError("Failed to open input videos")

    fps = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = fps2 if fps2 > 0 else 30.0

    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        raise RuntimeError("Invalid video metadata")

    out_w = max(w1, w2)
    out_h = h1 + h2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 and not ret2:
            break
        if not ret1:
            f1 = np.zeros((h1, w1, 3), dtype=np.uint8)
        if not ret2:
            f2 = np.zeros((h2, w2, 3), dtype=np.uint8)
        if f1.shape[1] != out_w:
            f1 = cv2.resize(f1, (out_w, int(f1.shape[0] * out_w / f1.shape[1])), interpolation=cv2.INTER_LINEAR)
        if f2.shape[1] != out_w:
            f2 = cv2.resize(f2, (out_w, int(f2.shape[0] * out_w / f2.shape[1])), interpolation=cv2.INTER_LINEAR)
        f1 = cv2.resize(f1, (out_w, h1))
        f2 = cv2.resize(f2, (out_w, h2))
        stacked = cv2.vconcat([f1, f2])
        writer.write(stacked)

    cap1.release()
    cap2.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Post-process captures: align depth and optionally stack videos")
    parser.add_argument("root", help="Root directory to search for meta.json and videos")
    parser.add_argument("--stack", nargs=2, metavar=("TOP", "BOTTOM"), help="Two videos to stack vertically")
    parser.add_argument("--output", help="Output stacked video path", default="stacked.mp4")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    metas = find_meta_files(root)
    for m in metas:
        process_capture(m)
    if args.stack:
        top = Path(args.stack[0]).expanduser().resolve()
        bottom = Path(args.stack[1]).expanduser().resolve()
        out = Path(args.output).expanduser().resolve()
        stack_videos(top, bottom, out)


if __name__ == "__main__":
    main()
