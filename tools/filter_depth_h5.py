#!/usr/bin/env python3
"""
Apply RealSense depth post-processing filters to depth.h5.

Classic pipeline:
  threshold -> depth->disparity -> spatial -> temporal -> disparity->depth -> hole_filling

Usage:
  python tools/filter_depth_h5.py /path/to/captures_root --find-meta true
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def list_meta_files(root: Path, find_meta: bool, max_depth: int = 2) -> List[Path]:
    if find_meta:
        result: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth > max_depth:
                dirnames[:] = []
                continue
            if "meta.json" in filenames:
                result.append(Path(dirpath) / "meta.json")
        return sorted(result)
    result: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        meta = child / "meta.json"
        if meta.exists():
            result.append(meta)
    return result


def load_meta(meta_path: Path) -> Dict:
    with meta_path.open("r") as f:
        return json.load(f)


def intrinsics_from_json(j: Dict) -> Dict:
    coeffs = list(j.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]))
    if len(coeffs) < 5:
        coeffs = coeffs + [0.0] * (5 - len(coeffs))
    if len(coeffs) > 5:
        coeffs = coeffs[:5]
    return {
        "fx": j["fx"],
        "fy": j["fy"],
        "cx": j["cx"],
        "cy": j["cy"],
        "w": j["width"],
        "h": j["height"],
        "coeffs": coeffs,
    }


def prepare_meta(cam: Dict) -> Dict:
    streams = cam.get("streams", {})
    depth_stream = streams.get("depth", {})
    return {
        "depth_scale": cam["alignment"].get("depth_scale_m", 0.0),
        "Kd": intrinsics_from_json(depth_stream["intrinsics"]),
        "depth_fps": int(depth_stream.get("fps") or 30),
    }


def sanitize_camera_id(cam_id: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in cam_id)


def is_realsense_id(cam_id: str) -> bool:
    return cam_id.startswith("RealSense")


def build_rs_intrinsics(meta_intr: Dict) -> "rs.intrinsics":
    intr = rs.intrinsics()
    intr.width = int(meta_intr["w"])
    intr.height = int(meta_intr["h"])
    intr.ppx = float(meta_intr["cx"])
    intr.ppy = float(meta_intr["cy"])
    intr.fx = float(meta_intr["fx"])
    intr.fy = float(meta_intr["fy"])
    coeffs = list(meta_intr.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]))
    if len(coeffs) < 5:
        coeffs = coeffs + [0.0] * (5 - len(coeffs))
    if len(coeffs) > 5:
        coeffs = coeffs[:5]
    intr.coeffs = coeffs
    intr.model = rs.distortion.none
    return intr


class DepthFilterPipeline:
    def __init__(
        self,
        meta: Dict,
        use_threshold: bool,
        use_disparity: bool,
        use_spatial: bool,
        use_temporal: bool,
        use_hole_filling: bool,
        decimation_magnitude: float,
        min_distance: float,
        max_distance: float,
        spatial_alpha: float,
        spatial_delta: float,
        spatial_magnitude: float,
        temporal_alpha: float,
        temporal_delta: float,
        hole_filling: int,
    ) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 is required for depth filtering")
        self._depth_scale = float(meta.get("depth_scale", 0.0))
        kd = meta["Kd"]
        self._width = int(kd["w"])
        self._height = int(kd["h"])
        self._fps = int(meta.get("depth_fps") or 30)
        self._use_threshold = use_threshold
        self._use_disparity = use_disparity
        self._use_spatial = use_spatial
        self._use_temporal = use_temporal
        self._use_hole_filling = use_hole_filling
        self._use_decimation = decimation_magnitude > 0

        self._device = rs.software_device()
        self._sensor = self._device.add_sensor("Depth")
        if self._depth_scale > 0:
            try:
                self._sensor.set_option(rs.option.depth_units, self._depth_scale)
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
        stream.width = self._width
        stream.height = self._height
        stream.fps = self._fps
        stream.intrinsics = build_rs_intrinsics(kd)
        profile = self._sensor.add_video_stream(stream)
        if hasattr(profile, "as_video_stream_profile"):
            profile = profile.as_video_stream_profile()
        self._profile = profile

        self._queue = rs.frame_queue(1)
        self._sensor.open(self._profile)
        self._sensor.start(self._queue)

        self._decimation = rs.decimation_filter()
        self._threshold = rs.threshold_filter()
        self._depth_to_disp = rs.disparity_transform(True)
        self._spatial = rs.spatial_filter()
        self._temporal = rs.temporal_filter()
        self._disp_to_depth = rs.disparity_transform(False)
        self._hole_filling = rs.hole_filling_filter()

        if self._use_decimation:
            self._decimation.set_option(rs.option.filter_magnitude, float(decimation_magnitude))
        if min_distance >= 0:
            self._threshold.set_option(rs.option.min_distance, float(min_distance))
        if max_distance >= 0:
            self._threshold.set_option(rs.option.max_distance, float(max_distance))
        if spatial_alpha >= 0:
            self._spatial.set_option(rs.option.filter_smooth_alpha, float(spatial_alpha))
        if spatial_delta >= 0:
            self._spatial.set_option(rs.option.filter_smooth_delta, float(spatial_delta))
        if spatial_magnitude >= 0:
            self._spatial.set_option(rs.option.filter_magnitude, float(spatial_magnitude))
        if temporal_alpha >= 0:
            self._temporal.set_option(rs.option.filter_smooth_alpha, float(temporal_alpha))
        if temporal_delta >= 0:
            self._temporal.set_option(rs.option.filter_smooth_delta, float(temporal_delta))
        if hole_filling >= 0:
            try:
                self._hole_filling.set_option(rs.option.holes_fill, int(hole_filling))
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._sensor.stop()
        except Exception:
            pass
        try:
            self._sensor.close()
        except Exception:
            pass

    def _make_frame(self, data: bytes, frame_number: int, timestamp_ms: float) -> "rs.software_video_frame":
        frame = rs.software_video_frame()
        frame.profile = self._profile
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
            frame.timestamp = float(timestamp_ms)
        if hasattr(frame, "domain"):
            frame.domain = rs.timestamp_domain.system_time
        if hasattr(frame, "frame_number"):
            frame.frame_number = int(frame_number)
        if self._depth_scale > 0 and hasattr(frame, "depth_units"):
            frame.depth_units = self._depth_scale
        return frame

    def process(self, depth_raw: np.ndarray, frame_number: int, timestamp_ms: float) -> Optional[np.ndarray]:
        if depth_raw is None or depth_raw.dtype != np.uint16:
            return None
        if depth_raw.shape != (self._height, self._width):
            return None
        depth_raw = np.ascontiguousarray(depth_raw)
        frame = self._make_frame(depth_raw.tobytes(), frame_number, timestamp_ms)
        self._sensor.on_video_frame(frame)
        depth_frame = self._queue.wait_for_frame(5000)
        if depth_frame is None:
            return None

        filtered = depth_frame
        if self._use_decimation:
            filtered = self._decimation.process(filtered)
        if self._use_threshold:
            filtered = self._threshold.process(filtered)
        if self._use_disparity:
            filtered = self._depth_to_disp.process(filtered)
        if self._use_spatial:
            filtered = self._spatial.process(filtered)
        if self._use_temporal:
            filtered = self._temporal.process(filtered)
        if self._use_disparity:
            filtered = self._disp_to_depth.process(filtered)
        if self._use_hole_filling:
            filtered = self._hole_filling.process(filtered)

        out = np.asanyarray(filtered.get_data())
        if out.dtype != np.uint16:
            out = out.astype(np.uint16)
        return out


def should_skip_step(capture_root: Path, step: str) -> bool:
    marker_path = capture_root / "postprocess_markers.json"
    if not marker_path.exists():
        return False
    try:
        payload = json.loads(marker_path.read_text())
    except json.JSONDecodeError:
        return False
    steps = payload.get("steps")
    if not isinstance(steps, dict):
        return False
    return step in steps


def update_marker(capture_root: Path, step: str, info: Dict) -> None:
    marker_path = capture_root / "postprocess_markers.json"
    payload = {}
    if marker_path.exists():
        try:
            payload = json.loads(marker_path.read_text())
        except json.JSONDecodeError:
            payload = {}
    steps = payload.get("steps")
    if not isinstance(steps, dict):
        steps = {}
    done_at = datetime.now(timezone.utc).isoformat()
    entry = dict(info)
    entry["done_at"] = done_at
    steps[step] = entry
    payload["steps"] = steps
    payload["updated_at"] = done_at
    marker_path.write_text(json.dumps(payload, indent=2))
    print(f"[depth-filter] updated marker {marker_path}")


def process_capture(meta_path: Path, args: argparse.Namespace) -> None:
    if h5py is None:
        print("[depth-filter] h5py not installed; cannot filter depth.h5")
        return
    if rs is None:
        print("[depth-filter] pyrealsense2 not installed; cannot filter depth.h5")
        return
    meta = load_meta(meta_path)
    base = meta_path.parent
    cam_entries = {
        c["id"]: c for c in meta.get("cameras", []) if is_realsense_id(str(c.get("id", "")))
    }
    if not cam_entries:
        return
    if args.decimation_magnitude > 0 and not args.allow_decimation:
        print("[depth-filter] decimation changes resolution; pass --allow-decimation to proceed")
        return
    if should_skip_step(base, "filter_depth"):
        print(f"[depth-filter] skip {base}, already filtered")
        return
    processed = []
    for cam_id, cam in cam_entries.items():
        cam_dir = base / sanitize_camera_id(str(cam_id))
        depth_h5 = cam_dir / "depth.h5"
        if not depth_h5.exists():
            continue
        try:
            m = prepare_meta(cam)
        except KeyError as e:
            print(f"[depth-filter] skipping {cam_id} in {base}: missing meta key {e}")
            continue
        out_path = cam_dir / args.output_name
        if out_path.exists() and not args.overwrite:
            print(f"[depth-filter] skip {cam_id}, exists: {out_path}")
            processed.append(str(cam_id))
            continue

        pipeline = None
        try:
            pipeline = DepthFilterPipeline(
                m,
                use_threshold=not args.no_threshold,
                use_disparity=not args.no_disparity,
                use_spatial=not args.no_spatial,
                use_temporal=not args.no_temporal,
                use_hole_filling=not args.no_hole_filling and args.hole_filling >= 0,
                decimation_magnitude=args.decimation_magnitude,
                min_distance=args.min_distance,
                max_distance=args.max_distance,
                spatial_alpha=args.spatial_alpha,
                spatial_delta=args.spatial_delta,
                spatial_magnitude=args.spatial_magnitude,
                temporal_alpha=args.temporal_alpha,
                temporal_delta=args.temporal_delta,
                hole_filling=args.hole_filling,
            )
        except Exception as exc:
            print(f"[depth-filter] skipping {cam_id} in {base}: init failed ({exc})")
            continue

        try:
            with h5py.File(depth_h5, "r") as f:
                if "depth" not in f:
                    print(f"[depth-filter] skipping {cam_id}: missing /depth dataset in {depth_h5}")
                    continue
                dset = f["depth"]
                total = int(dset.shape[0])
                if total <= 0:
                    continue
                timestamps = None
                if "depth_timestamps" in f:
                    timestamps = f["depth_timestamps"][:]
                first_raw = dset[0]
                ts_ms = float(timestamps[0][0]) if timestamps is not None and timestamps.shape[0] > 0 else 0.0
                first_filtered = pipeline.process(first_raw, 1, ts_ms)
                if first_filtered is None:
                    print(f"[depth-filter] skipping {cam_id}: failed to filter first frame")
                    continue
                out_h, out_w = first_filtered.shape[:2]
                if (out_h, out_w) != tuple(dset.shape[1:]):
                    print(f"[depth-filter] skipping {cam_id}: filtered size {out_w}x{out_h} != input {dset.shape[2]}x{dset.shape[1]}")
                    continue
                with h5py.File(out_path, "w") as out_f:
                    out_dset = out_f.create_dataset(
                        "depth",
                        shape=(total, out_h, out_w),
                        dtype=np.uint16,
                        chunks=(1, out_h, out_w),
                    )
                    if timestamps is not None:
                        out_f.create_dataset("depth_timestamps", data=timestamps, dtype=timestamps.dtype)
                    out_dset[0] = first_filtered
                    for idx in tqdm(range(1, total), desc=f"filter {cam_id}", unit="frame", leave=False):
                        depth_raw = dset[idx]
                        ts_ms = float(idx)
                        if timestamps is not None and timestamps.shape[0] > idx:
                            ts_ms = float(timestamps[idx][0])
                        filtered = pipeline.process(depth_raw, idx + 1, ts_ms)
                        if filtered is None:
                            continue
                        out_dset[idx] = filtered
            print(f"[depth-filter] wrote {out_path}")
            processed.append(str(cam_id))
        finally:
            if pipeline is not None:
                pipeline.close()

    if processed:
        update_marker(
            base,
            "filter_depth",
            {
                "cameras": processed,
                "output_file": args.output_name,
                "min_distance": args.min_distance,
                "max_distance": args.max_distance,
                "decimation_magnitude": args.decimation_magnitude,
                "spatial_alpha": args.spatial_alpha,
                "spatial_delta": args.spatial_delta,
                "spatial_magnitude": args.spatial_magnitude,
                "temporal_alpha": args.temporal_alpha,
                "temporal_delta": args.temporal_delta,
                "hole_filling": args.hole_filling,
                "no_threshold": args.no_threshold,
                "no_disparity": args.no_disparity,
                "no_spatial": args.no_spatial,
                "no_temporal": args.no_temporal,
                "no_hole_filling": args.no_hole_filling,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply RealSense depth filters to depth.h5")
    parser.add_argument("root", help="Root directory containing captures/meta.json")
    parser.add_argument("--find-meta", type=str, default="true",
                        choices=["true", "false"],
                        help="Search meta.json recursively (true) or only root/*/meta.json (false)")
    parser.add_argument("--output-name", default="depth_filtered.h5", help="Output H5 name per camera")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    parser.add_argument("--min-distance", type=float, default=-1.0, help="Threshold min distance (m)")
    parser.add_argument("--max-distance", type=float, default=-1.0, help="Threshold max distance (m)")
    parser.add_argument("--decimation-magnitude", type=float, default=-1.0, help="Decimation magnitude (disable with <0)")
    parser.add_argument("--allow-decimation", action="store_true", help="Allow decimation even if it changes resolution")
    parser.add_argument("--spatial-alpha", type=float, default=-1.0, help="Spatial alpha")
    parser.add_argument("--spatial-delta", type=float, default=-1.0, help="Spatial delta")
    parser.add_argument("--spatial-magnitude", type=float, default=-1.0, help="Spatial magnitude")
    parser.add_argument("--temporal-alpha", type=float, default=-1.0, help="Temporal alpha")
    parser.add_argument("--temporal-delta", type=float, default=-1.0, help="Temporal delta")
    parser.add_argument("--hole-filling", type=int, default=-1,
                        help="Hole filling mode (0/1/2). <0 disables")

    parser.add_argument("--no-threshold", action="store_true", help="Disable threshold filter")
    parser.add_argument("--no-disparity", action="store_true", help="Disable disparity transform")
    parser.add_argument("--no-spatial", action="store_true", help="Disable spatial filter")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal filter")
    parser.add_argument("--no-hole-filling", action="store_true", help="Disable hole filling filter")

    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    find_meta = args.find_meta.lower() == "true"
    metas = list_meta_files(root, find_meta, 2)
    if not metas:
        print("No meta.json found")
        return 1
    for meta_path in metas:
        process_capture(meta_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
