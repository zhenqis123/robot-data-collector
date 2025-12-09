#!/usr/bin/env python3
"""
Post-process captures:
 - depth alignment (PNG -> depth_aligned)
 - timestamp alignment (reference camera nearest neighbor)
 - color video encode (H.264 MP4)
 - optional video stacking
 - packing capture to tar.gz

Usage examples:
  python -m tools.postprocess align /path/to/root
  python -m tools.postprocess timestamps /path/to/capture --reference CAM_ID
  python -m tools.postprocess video /path/to/capture --camera CAM_ID --fps 30
  python -m tools.postprocess stack top.mp4 bottom.mp4 --output stacked.mp4
  python -m tools.postprocess pack /path/to/capture
  python -m tools.postprocess all /path/to/root --reference CAM_ID
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local tools package import works when run as script
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR.parent))

from tools import align as align_mod
from tools import timestamps as ts_mod
from tools import videos as vid_mod
from tools import pack as pack_mod


def cmd_align(root: Path) -> None:
    metas = align_mod.find_meta_files(root)
    if not metas:
        print("No meta.json found for alignment")
        return
    for m in metas:
        align_mod.process_capture(m)


def cmd_timestamps(capture: Path, reference: str | None) -> None:
    ts_mod.align_capture(capture, reference)


def cmd_video(capture: Path, camera: str | None, fps: float) -> None:
    # If camera not specified, encode all camera color dirs
    if camera:
        cams = [camera]
    else:
        meta_path = capture / "meta.json"
        import json

        cams = []
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            cams = [c["id"] for c in meta.get("cameras", [])]
    if not cams:
        print(f"[video] no cameras found in {capture}")
        return
    for cid in cams:
        cam_dir = capture / cid
        out = cam_dir / "color.mp4"
        try:
            vid_mod.encode_color_frames(cam_dir, out, fps)
        except Exception as exc:
            print(f"[video] skip {cid}: {exc}")


def cmd_stack(top: Path, bottom: Path, output: Path) -> None:
    vid_mod.stack_videos(top, bottom, output)


def cmd_pack(capture: Path) -> None:
    pack_mod.pack_capture(capture)


def cmd_all(root: Path, reference: str | None, fps: float) -> None:
    # align all captures
    metas = align_mod.find_meta_files(root)
    captures = set()
    for m in metas:
        captures.add(m.parent)
    cmd_align(root)
    # timestamps + video + pack per capture
    for cap in captures:
        ts_mod.align_capture(cap, reference)
        cmd_video(cap, None, fps)
        pack_mod.pack_capture(cap)


def main():
    parser = argparse.ArgumentParser(description="Post-process captures (align, timestamps, video, stack, pack)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_align = sub.add_parser("align", help="Run depth alignment on all captures under root")
    p_align.add_argument("root")

    p_ts = sub.add_parser("timestamps", help="Align timestamps for one capture")
    p_ts.add_argument("capture")
    p_ts.add_argument("--reference", help="Reference camera id", default=None)

    p_vid = sub.add_parser("video", help="Encode color frames to MP4")
    p_vid.add_argument("capture")
    p_vid.add_argument("--camera", help="Camera id (default: all in meta.json)", default=None)
    p_vid.add_argument("--fps", type=float, default=30.0)

    p_stack = sub.add_parser("stack", help="Stack two videos vertically")
    p_stack.add_argument("top")
    p_stack.add_argument("bottom")
    p_stack.add_argument("--output", default="stacked.mp4")

    p_pack = sub.add_parser("pack", help="Pack a capture directory to tar.gz")
    p_pack.add_argument("capture")

    p_all = sub.add_parser("all", help="Run align + timestamps + color video + pack on all captures under root")
    p_all.add_argument("root")
    p_all.add_argument("--reference", default=None)
    p_all.add_argument("--fps", type=float, default=30.0)

    args = parser.parse_args()
    if args.cmd == "align":
        cmd_align(Path(args.root).expanduser().resolve())
    elif args.cmd == "timestamps":
        cmd_timestamps(Path(args.capture).expanduser().resolve(), args.reference)
    elif args.cmd == "video":
        cmd_video(Path(args.capture).expanduser().resolve(), args.camera, args.fps)
    elif args.cmd == "stack":
        cmd_stack(Path(args.top).expanduser().resolve(), Path(args.bottom).expanduser().resolve(), Path(args.output).expanduser().resolve())
    elif args.cmd == "pack":
        cmd_pack(Path(args.capture).expanduser().resolve())
    elif args.cmd == "all":
        cmd_all(Path(args.root).expanduser().resolve(), args.reference, args.fps)


if __name__ == "__main__":
    main()
