#!/usr/bin/env python3
"""
Visualize depth.h5 as a colorized H.264 MP4 using RealSense colorizer.

Usage:
  python -m tools.depth_h5_viz_video /path/to/captures_root
  python -m tools.depth_h5_viz_video /path/to/depth.h5
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


FALLBACK_CODECS = ("H264", "avc1", "X264", "mp4v")


def sanitize_camera_id(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in value)


def find_depth_files(root: Path, include_aligned: bool) -> List[Path]:
    patterns = ["depth.h5"]
    if include_aligned:
        patterns.append("depth_aligned.h5")
    results: List[Path] = []
    for pattern in patterns:
        results.extend(root.rglob(pattern))
    return sorted(set(results))


def load_meta_for_depth(depth_path: Path) -> Dict:
    for parent in depth_path.parents:
        meta = parent / "meta.json"
        if meta.exists():
            try:
                with meta.open("r") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def resolve_depth_scale(meta: Dict, cam_dir_name: str) -> Optional[float]:
    for cam in meta.get("cameras", []):
        cam_id = cam.get("id")
        if cam_id and sanitize_camera_id(cam_id) == cam_dir_name:
            alignment = cam.get("alignment", {})
            scale = alignment.get("depth_scale_m", 0.0)
            try:
                scale_val = float(scale)
            except (TypeError, ValueError):
                return None
            if scale_val > 0:
                return scale_val
    return None


def read_fps_from_timestamps(ts_path: Path) -> Optional[float]:
    if not ts_path.exists():
        return None
    timestamps: List[float] = []
    with ts_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_val = row.get("timestamp_ms", "")
            if not ts_val:
                continue
            try:
                timestamps.append(float(ts_val))
            except ValueError:
                continue
    if len(timestamps) < 2:
        return None
    deltas = np.diff(np.array(timestamps, dtype=np.float64))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None
    median_ms = float(np.median(deltas))
    if median_ms <= 0:
        return None
    return 1000.0 / median_ms


def open_video_writer(output: Path, fps: float, size: Tuple[int, int], codec: str,
                      allow_fallback: bool) -> Tuple[cv2.VideoWriter, str]:
    candidates = [codec] if codec.lower() != "auto" else []
    for candidate in FALLBACK_CODECS:
        if candidate not in candidates:
            candidates.append(candidate)
    tried = []
    for cand in candidates:
        tried.append(cand)
        fourcc = cv2.VideoWriter_fourcc(*cand)
        writer = cv2.VideoWriter(str(output), fourcc, fps, size)
        if writer.isOpened():
            return writer, cand
        writer.release()
    if allow_fallback:
        raise RuntimeError(f"failed to open video writer; tried: {', '.join(tried)}")
    raise RuntimeError(f"failed to open video writer with codec {codec}")


class RealSenseDepthColorizer:
    def __init__(self, width: int, height: int, depth_scale: Optional[float]):
        if rs is None:
            raise RuntimeError("pyrealsense2 not installed")
        self._width = width
        self._height = height
        self._frame_index = 0
        self._device = rs.software_device()
        self._sensor = self._device.add_sensor("Depth")
        if depth_scale is not None:
            try:
                self._sensor.set_option(rs.option.depth_units, float(depth_scale))
            except Exception:
                pass

        stream = rs.video_stream()
        stream.type = rs.stream.depth
        try:
            stream.format = rs.format.z16
        except Exception:
            stream.fmt = rs.format.z16
        stream.index = 0
        stream.uid = 0
        stream.width = width
        stream.height = height
        stream.fps = 30
        intr = rs.intrinsics()
        intr.width = width
        intr.height = height
        intr.ppx = width * 0.5
        intr.ppy = height * 0.5
        intr.fx = float(width)
        intr.fy = float(height)
        intr.model = rs.distortion.none
        intr.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        stream.intrinsics = intr

        profile = self._sensor.add_video_stream(stream)
        if hasattr(profile, "as_video_stream_profile"):
            profile = profile.as_video_stream_profile()
        self._profile = profile

        self._sensor.open(self._profile)
        self._queue = rs.frame_queue(1)
        self._frame_holder: Dict[str, "rs.frame"] = {}
        self._use_queue = True
        try:
            self._sensor.start(self._queue)
        except Exception:
            self._use_queue = False
            if hasattr(self._queue, "enqueue"):
                self._sensor.start(lambda frame: self._queue.enqueue(frame))
                self._use_queue = True
            else:
                self._sensor.start(lambda frame: self._frame_holder.__setitem__("frame", frame))
        self._colorizer = rs.colorizer()

    def close(self) -> None:
        self._sensor.stop()
        self._sensor.close()

    def colorize(self, depth: np.ndarray) -> np.ndarray:
        depth = np.ascontiguousarray(depth.astype(np.uint16))
        frame = rs.software_video_frame()
        frame.profile = self._profile
        data = depth.tobytes()
        if hasattr(frame, "pixels"):
            frame.pixels = data
        elif hasattr(frame, "data"):
            frame.data = data
        else:
            raise RuntimeError("software_video_frame has no data field")
        if hasattr(frame, "stride"):
            frame.stride = self._width * 2
        if hasattr(frame, "bpp"):
            frame.bpp = 2
        if hasattr(frame, "timestamp"):
            frame.timestamp = 0.0
        if hasattr(frame, "domain"):
            frame.domain = rs.timestamp_domain.system_time
        if hasattr(frame, "frame_number"):
            self._frame_index += 1
            frame.frame_number = self._frame_index

        self._sensor.on_video_frame(frame)
        if self._use_queue:
            depth_frame = self._queue.wait_for_frame(5000)
        else:
            depth_frame = None
            for _ in range(50):
                depth_frame = self._frame_holder.get("frame")
                if depth_frame is not None:
                    break
                time.sleep(0.1)
            if depth_frame is None:
                raise RuntimeError("failed to receive depth frame from software device")

        colorized = self._colorizer.colorize(depth_frame)
        color_img = np.asanyarray(colorized.get_data())
        if color_img.ndim == 3 and color_img.shape[2] == 3:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        return color_img


def process_depth_file(depth_path: Path, output_name: Optional[str], fps: float,
                       depth_scale_override: float, codec: str, allow_fallback: bool,
                       overwrite: bool) -> bool:
    if h5py is None:
        print("h5py not installed; cannot read depth.h5 files")
        return False
    if rs is None:
        print("pyrealsense2 not installed; cannot colorize depth")
        return False

    if not depth_path.exists():
        print(f"[depth-viz] missing {depth_path}")
        return False

    out_name = output_name or f"{depth_path.stem}_viz.mp4"
    output = depth_path.parent / out_name
    if output.exists() and not overwrite:
        print(f"[depth-viz] skip {output} (exists, use --overwrite to replace)")
        return True

    meta = load_meta_for_depth(depth_path)
    cam_dir_name = depth_path.parent.name
    depth_scale = depth_scale_override if depth_scale_override > 0 else resolve_depth_scale(meta, cam_dir_name)

    if fps <= 0:
        fps = read_fps_from_timestamps(depth_path.parent / "timestamps.csv") or 30.0

    try:
        with h5py.File(depth_path, "r") as f:
            if "depth" not in f:
                print(f"[depth-viz] missing /depth in {depth_path}")
                return False
            dset = f["depth"]
            if dset.ndim != 3:
                print(f"[depth-viz] unexpected depth shape {dset.shape} in {depth_path}")
                return False
            total, height, width = dset.shape
            if total <= 0:
                print(f"[depth-viz] no frames in {depth_path}")
                return False

            colorizer = RealSenseDepthColorizer(width, height, depth_scale)
            writer: Optional[cv2.VideoWriter] = None
            selected_codec = ""
            count = 0

            try:
                for idx in tqdm(range(total), desc=f"viz {depth_path.name}", unit="frame", leave=False):
                    frame = dset[idx]
                    if frame is None:
                        continue
                    if frame.dtype != np.uint16:
                        frame = frame.astype(np.uint16)
                    color = colorizer.colorize(frame)
                    if writer is None:
                        writer, selected_codec = open_video_writer(
                            output, fps, (color.shape[1], color.shape[0]), codec, allow_fallback
                        )
                        if selected_codec.lower() != codec.lower() and codec.lower() != "auto":
                            print(f"[depth-viz] codec fallback {codec} -> {selected_codec} for {output}")
                    writer.write(color)
                    count += 1
            finally:
                colorizer.close()
                if writer is not None:
                    writer.release()

            print(f"[depth-viz] wrote {output} ({count} frames, {fps:.2f} fps)")
            return True
    except Exception as exc:
        print(f"[depth-viz] failed {depth_path}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Colorize depth.h5 to H.264 MP4 using RealSense colorizer")
    parser.add_argument("input_path", help="Path to depth.h5 or a root directory to scan")
    parser.add_argument("--output-name", default=None, help="Output file name (default: <depth>.stem_viz.mp4)")
    parser.add_argument("--fps", type=float, default=0.0, help="FPS override; 0=auto from timestamps.csv")
    parser.add_argument("--depth-scale", type=float, default=0.0, help="Depth scale override in meters")
    parser.add_argument("--include-aligned", type=str, default="false", choices=["true", "false"],
                        help="Include depth_aligned.h5 when scanning directories")
    parser.add_argument("--codec", default="avc1", help="FourCC codec (default avc1, or 'auto')")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow fallback codecs if requested fails")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    root = Path(args.input_path).expanduser().resolve()
    include_aligned = args.include_aligned.lower() == "true"

    depth_files: List[Path] = []
    if root.is_file() and root.suffix.lower() == ".h5":
        depth_files = [root]
    elif root.is_dir():
        depth_files = find_depth_files(root, include_aligned)
    else:
        print(f"input not found: {root}")
        return 1

    if not depth_files:
        print(f"no depth.h5 files found under {root}")
        return 1

    any_fail = False
    for depth_path in depth_files:
        ok = process_depth_file(
            depth_path,
            args.output_name,
            args.fps,
            args.depth_scale,
            args.codec,
            args.allow_fallback,
            args.overwrite,
        )
        if not ok:
            any_fail = True

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
