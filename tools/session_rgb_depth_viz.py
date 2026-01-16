#!/usr/bin/env python3
"""
Visualize a single RGB frame with its corresponding depth frame in a session.

Uses RealSense colorizer to visualize depth and writes RGB/Depth as separate PNGs.

Usage:
  python -m tools.session_rgb_depth_viz /path/to/session --frame 120
  python -m tools.session_rgb_depth_viz /path/to/session --camera RealSense#123 --frame 120
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def sanitize_camera_id(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in value)


def load_meta(session: Path) -> Dict:
    meta_path = session / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r") as f:
        return json.load(f)


def resolve_cameras(session: Path, meta: Dict, camera_arg: Optional[str]) -> List[Tuple[str, Path]]:
    cameras = meta.get("cameras", [])
    cam_map = {c["id"]: session / sanitize_camera_id(c["id"]) for c in cameras if "id" in c}
    resolved: List[Tuple[str, Path]] = []

    if camera_arg:
        targets = [c.strip() for c in camera_arg.split(",") if c.strip()]
        for target in targets:
            if target in cam_map and cam_map[target].exists():
                resolved.append((target, cam_map[target]))
                continue
            matched = None
            for cam_id, cam_dir in cam_map.items():
                if cam_dir.name == target or cam_dir.name == sanitize_camera_id(target):
                    matched = (cam_id, cam_dir)
                    break
            if matched:
                resolved.append(matched)
                continue
            direct = session / target
            if direct.exists() and direct.is_dir():
                resolved.append((target, direct))
                continue
            raise ValueError(f"camera not found: {target}")
        return resolved

    if cam_map:
        return [(cam_id, cam_dir) for cam_id, cam_dir in cam_map.items() if cam_dir.exists()]

    for child in sorted(session.iterdir()):
        if not child.is_dir():
            continue
        if any((child / name).exists() for name in ("rgb.mkv", "rgb.mp4", "color.mp4", "color.mkv",
                                                    "depth.h5", "depth_aligned.h5")):
            resolved.append((child.name, child))

    if resolved:
        return resolved
    raise ValueError("no cameras found; provide --camera with a camera directory name")


def resolve_depth_scale(meta: Dict, cam_id: str) -> Optional[float]:
    for cam in meta.get("cameras", []):
        if cam.get("id") == cam_id:
            alignment = cam.get("alignment", {})
            scale = alignment.get("depth_scale_m", 0.0)
            try:
                scale_val = float(scale)
            except (TypeError, ValueError):
                return None
            if scale_val > 0:
                return scale_val
    return None


def prefer_mp4_path(path: Path) -> Path:
    if path.suffix.lower() == ".mkv":
        mp4 = path.with_suffix(".mp4")
        if mp4.exists():
            return mp4
    return path


def resolve_rgb_video(cam_dir: Path) -> Optional[Path]:
    ts_path = cam_dir / "timestamps.csv"
    if ts_path.exists():
        with ts_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
        if row and row.get("rgb_path"):
            candidate = cam_dir / row["rgb_path"]
            candidate = prefer_mp4_path(candidate)
            if candidate.exists():
                return candidate
    candidates = [
        cam_dir / "rgb.mkv",
        cam_dir / "rgb.mp4",
        cam_dir / "color.mp4",
        cam_dir / "color.mkv",
    ]
    for candidate in candidates:
        candidate = prefer_mp4_path(candidate)
        if candidate.exists():
            return candidate
    return None


def resolve_depth_source(cam_dir: Path, depth_override: Optional[str]) -> Optional[Path]:
    if depth_override:
        path = Path(depth_override).expanduser()
        if not path.is_absolute():
            path = cam_dir / path
        path = path.resolve()
        if path.exists():
            return path
    candidates = [
        cam_dir / "depth_aligned.h5",
        cam_dir / "depth.h5",
        cam_dir / "depth_aligned",
        cam_dir / "depth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_video_frame(video_path: Path, frame_index: int) -> Optional[np.ndarray]:
    if frame_index <= 0:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def read_depth_from_h5(depth_path: Path, frame_index: int) -> Optional[np.ndarray]:
    if h5py is None:
        raise RuntimeError("h5py not installed; cannot read depth .h5")
    if frame_index <= 0:
        return None
    with h5py.File(depth_path, "r") as f:
        if "depth" not in f:
            raise RuntimeError(f"missing /depth dataset in {depth_path}")
        dset = f["depth"]
        if frame_index > dset.shape[0]:
            return None
        depth = dset[frame_index - 1]
    return depth


def read_depth_from_dir(depth_dir: Path, frame_index: int) -> Optional[np.ndarray]:
    if frame_index <= 0:
        return None
    candidates = [
        depth_dir / f"{frame_index - 1:06d}.png",
        depth_dir / f"{frame_index:06d}.png",
        depth_dir / f"{frame_index - 1}.png",
        depth_dir / f"{frame_index}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            depth = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
            return depth
    pngs = sorted(depth_dir.glob("*.png"))
    if len(pngs) >= frame_index:
        depth = cv2.imread(str(pngs[frame_index - 1]), cv2.IMREAD_UNCHANGED)
        return depth
    return None


def _build_depth_frame(depth: np.ndarray, depth_scale: Optional[float]) -> "rs.frame":
    if rs is None:
        raise RuntimeError("pyrealsense2 not installed; cannot use RealSense colorizer")
    depth = np.ascontiguousarray(depth.astype(np.uint16))
    height, width = depth.shape

    device = rs.software_device()
    sensor = device.add_sensor("Depth")
    if depth_scale is not None:
        try:
            sensor.set_option(rs.option.depth_units, float(depth_scale))
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

    profile = sensor.add_video_stream(stream)
    if hasattr(profile, "as_video_stream_profile"):
        profile = profile.as_video_stream_profile()
    sensor.open(profile)

    queue = rs.frame_queue(1)
    frame_holder: Dict[str, "rs.frame"] = {}
    use_queue = True
    try:
        sensor.start(queue)
    except Exception:
        use_queue = False
        if hasattr(queue, "enqueue"):
            sensor.start(lambda frame: queue.enqueue(frame))
            use_queue = True
        else:
            sensor.start(lambda frame: frame_holder.setdefault("frame", frame))

    frame = rs.software_video_frame()
    frame.profile = profile
    data = depth.tobytes()
    if hasattr(frame, "pixels"):
        frame.pixels = data
    elif hasattr(frame, "data"):
        frame.data = data
    else:
        raise RuntimeError("software_video_frame has no data field")
    if hasattr(frame, "stride"):
        frame.stride = width * 2
    if hasattr(frame, "bpp"):
        frame.bpp = 2
    if hasattr(frame, "width"):
        frame.width = width
    if hasattr(frame, "height"):
        frame.height = height
    if hasattr(frame, "timestamp"):
        frame.timestamp = 0.0
    if hasattr(frame, "domain"):
        frame.domain = rs.timestamp_domain.system_time
    if hasattr(frame, "frame_number"):
        frame.frame_number = 0
    sensor.on_video_frame(frame)

    depth_frame: Optional["rs.frame"]
    if use_queue:
        depth_frame = queue.wait_for_frame(5000)
    else:
        depth_frame = None
        for _ in range(50):
            depth_frame = frame_holder.get("frame")
            if depth_frame is not None:
                break
            time.sleep(0.1)
    sensor.stop()
    sensor.close()
    if depth_frame is None:
        raise RuntimeError("failed to construct depth frame for RealSense colorizer")
    return depth_frame


def colorize_depth_realsense(depth: np.ndarray, depth_scale: Optional[float]) -> np.ndarray:
    if rs is None:
        raise RuntimeError("pyrealsense2 not installed; cannot use RealSense colorizer")
    depth_frame = _build_depth_frame(depth, depth_scale)
    colorizer = rs.colorizer()
    colorized = colorizer.colorize(depth_frame)
    color_img = np.asanyarray(colorized.get_data())
    if color_img.ndim == 3 and color_img.shape[2] == 3:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    return color_img


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize RGB + depth frame for a session")
    parser.add_argument("session", help="Path to session directory (contains meta.json)")
    parser.add_argument("--camera", default=None, help="Camera id or camera folder name")
    parser.add_argument("--frame", type=int, required=True, help="1-based frame index")
    parser.add_argument("--depth", default=None, help="Override depth source path (h5 or folder)")
    parser.add_argument("--depth-scale", type=float, default=0.0, help="Depth scale in meters (override meta.json)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: session)")
    parser.add_argument("--output-rgb", default=None, help="Output RGB PNG path (optional)")
    parser.add_argument("--output-depth", default=None, help="Output depth PNG path (optional)")
    args = parser.parse_args()

    session = Path(args.session).expanduser().resolve()
    if not session.exists():
        print(f"session not found: {session}")
        return 1

    meta = load_meta(session)
    try:
        cameras = resolve_cameras(session, meta, args.camera)
    except ValueError as exc:
        print(exc)
        return 1
    if not cameras:
        print("no cameras found")
        return 1

    if len(cameras) > 1 and (args.output_rgb or args.output_depth):
        print("output-rgb/output-depth only supported with a single camera")
        return 1

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else session
    out_dir.mkdir(parents=True, exist_ok=True)

    any_fail = False
    for cam_id, cam_dir in cameras:
        if not cam_dir.exists():
            print(f"camera directory not found: {cam_dir}")
            any_fail = True
            continue

        rgb_path = resolve_rgb_video(cam_dir)
        if not rgb_path:
            print(f"no rgb video found in {cam_dir}")
            any_fail = True
            continue

        depth_source = resolve_depth_source(cam_dir, args.depth)
        if not depth_source:
            print(f"no depth source found in {cam_dir}")
            any_fail = True
            continue

        rgb = read_video_frame(rgb_path, args.frame)
        if rgb is None:
            print(f"failed to read rgb frame {args.frame} from {rgb_path}")
            any_fail = True
            continue

        try:
            if depth_source.is_dir():
                depth = read_depth_from_dir(depth_source, args.frame)
            else:
                depth = read_depth_from_h5(depth_source, args.frame)
        except RuntimeError as exc:
            print(exc)
            any_fail = True
            continue

        if depth is None:
            print(f"failed to read depth frame {args.frame} from {depth_source}")
            any_fail = True
            continue

        if rs is None:
            print("pyrealsense2 is required for depth colorization")
            return 1

        depth_scale = args.depth_scale if args.depth_scale > 0 else resolve_depth_scale(meta, cam_id)
        try:
            depth_color = colorize_depth_realsense(depth, depth_scale)
        except RuntimeError as exc:
            print(exc)
            any_fail = True
            continue

        rgb_out = Path(args.output_rgb).expanduser().resolve() if args.output_rgb else (
            out_dir / f"{sanitize_camera_id(cam_id)}_frame_{args.frame:06d}_rgb.png"
        )
        depth_out = Path(args.output_depth).expanduser().resolve() if args.output_depth else (
            out_dir / f"{sanitize_camera_id(cam_id)}_frame_{args.frame:06d}_depth.png"
        )
        rgb_out.parent.mkdir(parents=True, exist_ok=True)
        depth_out.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(rgb_out), rgb):
            print(f"failed to write {rgb_out}")
            any_fail = True
            continue
        if not cv2.imwrite(str(depth_out), depth_color):
            print(f"failed to write {depth_out}")
            any_fail = True
            continue

        print(f"wrote {rgb_out}")
        print(f"wrote {depth_out}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
