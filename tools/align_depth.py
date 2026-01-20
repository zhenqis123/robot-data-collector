#!/usr/bin/env python3
"""
Depth-to-color alignment using saved meta.json and PNG frames or HDF5 depth.

Uses pyrealsense2 (librealsense) align for official alignment behavior.

Usage:
  python -m tools.align /path/to/captures_root --workers 4 --delete-original-depth true

Searches recursively for meta.json files and, for each camera:
- reads intrinsics/extrinsics/depth scale
- aligns depth/*.png or depth.h5 to color resolution
- writes to depth_aligned/*.png
- optionally deletes the original depth files if --delete-original-depth is set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime, timezone

try:
    import h5py
except ImportError:
    h5py = None
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def find_meta_files(root: Path, max_depth: int) -> List[Path]:
    result = []
    
    # 使用 tqdm 包装 os.walk 进度条显示
    for dirpath, dirnames, filenames in tqdm(os.walk(root), desc="Searching for meta.json files", unit="dir"):
        # 计算当前目录的深度
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            continue  # 如果超过最大深度，跳过该目录
        for filename in filenames:
            if filename == "meta.json":
                result.append(Path(dirpath) / filename)
    
    return result


def list_meta_files(root: Path, find_meta: bool, max_depth: int) -> List[Path]:
    if find_meta:
        return find_meta_files(root, max_depth)
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
        "distortion": j.get("distortion") or j.get("distortion_model") or j.get("model"),
    }


def prepare_meta(cam: Dict) -> Dict:
    """Normalize camera metadata into numeric arrays."""
    streams = cam.get("streams", {})
    color_stream = streams.get("color", {})
    depth_stream = streams.get("depth", {})
    return {
        "depth_scale": cam["alignment"].get("depth_scale_m", 0.0),
        "R": np.array(cam["alignment"]["depth_to_color"]["rotation"], dtype=np.float32).reshape(3, 3),
        "t": np.array(cam["alignment"]["depth_to_color"]["translation"], dtype=np.float32).reshape(3, 1),
        "Kc": intrinsics_from_json(color_stream["intrinsics"]),
        "Kd": intrinsics_from_json(depth_stream["intrinsics"]),
        "color_fps": int(color_stream.get("fps") or 30),
        "depth_fps": int(depth_stream.get("fps") or 30),
    }


def distortion_from_meta(value: str) -> int:
    if rs is None:
        return 0
    if not value:
        return rs.distortion.none
    key = str(value).strip().lower()
    mapping = {
        "none": rs.distortion.none,
        "brown_conrady": rs.distortion.brown_conrady,
        "inverse_brown_conrady": rs.distortion.inverse_brown_conrady,
        "modified_brown_conrady": rs.distortion.modified_brown_conrady,
        "ftheta": rs.distortion.ftheta,
        "kannala_brandt4": rs.distortion.kannala_brandt4,
    }
    return mapping.get(key, rs.distortion.none)


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
    intr.model = distortion_from_meta(meta_intr.get("distortion"))
    return intr


class RealSenseAligner:
    def __init__(self, meta: Dict) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 is required for official alignment")
        self._depth_scale = float(meta.get("depth_scale", 0.0))
        self._depth_width = int(meta["Kd"]["w"])
        self._depth_height = int(meta["Kd"]["h"])
        self._color_width = int(meta["Kc"]["w"])
        self._color_height = int(meta["Kc"]["h"])
        self._depth_fps = int(meta.get("depth_fps") or 30)
        self._color_fps = int(meta.get("color_fps") or 30)

        self._device = rs.software_device()
        self._depth_sensor = self._device.add_sensor("Depth")
        self._color_sensor = self._device.add_sensor("Color")
        if self._depth_scale > 0:
            try:
                self._depth_sensor.set_option(rs.option.depth_units, self._depth_scale)
            except Exception:
                pass

        depth_stream = rs.video_stream()
        depth_stream.type = rs.stream.depth
        try:
            depth_stream.format = rs.format.z16
        except Exception:
            depth_stream.fmt = rs.format.z16
        depth_stream.index = 0
        depth_stream.uid = 0
        depth_stream.width = self._depth_width
        depth_stream.height = self._depth_height
        depth_stream.fps = self._depth_fps
        depth_stream.intrinsics = build_rs_intrinsics(meta["Kd"])
        depth_profile = self._depth_sensor.add_video_stream(depth_stream)
        if hasattr(depth_profile, "as_video_stream_profile"):
            depth_profile = depth_profile.as_video_stream_profile()

        color_stream = rs.video_stream()
        color_stream.type = rs.stream.color
        try:
            color_stream.format = rs.format.bgr8
        except Exception:
            color_stream.fmt = rs.format.bgr8
        color_stream.index = 0
        color_stream.uid = 1
        color_stream.width = self._color_width
        color_stream.height = self._color_height
        color_stream.fps = self._color_fps
        color_stream.intrinsics = build_rs_intrinsics(meta["Kc"])
        color_profile = self._color_sensor.add_video_stream(color_stream)
        if hasattr(color_profile, "as_video_stream_profile"):
            color_profile = color_profile.as_video_stream_profile()

        extr = rs.extrinsics()
        extr.rotation = list(np.array(meta["R"], dtype=np.float32).reshape(-1))
        extr.translation = list(np.array(meta["t"], dtype=np.float32).reshape(-1))
        depth_profile.register_extrinsics_to(color_profile, extr)

        self._depth_profile = depth_profile
        self._color_profile = color_profile
        self._syncer = rs.syncer()
        self._depth_sensor.open(self._depth_profile)
        self._color_sensor.open(self._color_profile)
        self._depth_sensor.start(self._syncer)
        self._color_sensor.start(self._syncer)
        self._align = rs.align(rs.stream.color)
        self._frame_index = 0
        blank = np.zeros((self._color_height, self._color_width, 3), dtype=np.uint8)
        self._color_blank = blank.tobytes()

    def _make_frame(self,
                    data: bytes,
                    profile: "rs.video_stream_profile",
                    width: int,
                    height: int,
                    bpp: int,
                    frame_number: int,
                    timestamp_ms: float,
                    depth_units: float = 0.0) -> "rs.software_video_frame":
        frame = rs.software_video_frame()
        frame.profile = profile
        if hasattr(frame, "pixels"):
            frame.pixels = data
        elif hasattr(frame, "data"):
            frame.data = data
        else:
            raise RuntimeError("software_video_frame has no data field")
        if hasattr(frame, "stride"):
            frame.stride = width * bpp
        if hasattr(frame, "bpp"):
            frame.bpp = bpp
        if hasattr(frame, "width"):
            frame.width = width
        if hasattr(frame, "height"):
            frame.height = height
        if hasattr(frame, "timestamp"):
            frame.timestamp = float(timestamp_ms)
        if hasattr(frame, "domain"):
            frame.domain = rs.timestamp_domain.system_time
        if hasattr(frame, "frame_number"):
            frame.frame_number = int(frame_number)
        if depth_units > 0 and hasattr(frame, "depth_units"):
            frame.depth_units = depth_units
        return frame

    def align_depth(self,
                    depth_raw: np.ndarray,
                    frame_number: int,
                    timestamp_ms: float,
                    color_raw: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if depth_raw is None or depth_raw.dtype != np.uint16:
            return None
        if depth_raw.shape != (self._depth_height, self._depth_width):
            return None
        depth_raw = np.ascontiguousarray(depth_raw)
        if color_raw is None:
            color_bytes = self._color_blank
        else:
            color_raw = np.ascontiguousarray(color_raw)
            color_bytes = color_raw.tobytes()
        depth_frame = self._make_frame(
            depth_raw.tobytes(),
            self._depth_profile,
            self._depth_width,
            self._depth_height,
            2,
            frame_number,
            timestamp_ms,
            depth_units=self._depth_scale,
        )
        color_frame = self._make_frame(
            color_bytes,
            self._color_profile,
            self._color_width,
            self._color_height,
            3,
            frame_number,
            timestamp_ms,
        )
        self._depth_sensor.on_video_frame(depth_frame)
        self._color_sensor.on_video_frame(color_frame)
        frameset = self._syncer.wait_for_frames(5000)
        aligned = self._align.process(frameset)
        aligned_depth = aligned.get_depth_frame()
        if not aligned_depth:
            return None
        return np.asanyarray(aligned_depth.get_data())

    def close(self) -> None:
        try:
            self._depth_sensor.stop()
        except Exception:
            pass
        try:
            self._color_sensor.stop()
        except Exception:
            pass
        try:
            self._depth_sensor.close()
        except Exception:
            pass
        try:
            self._color_sensor.close()
        except Exception:
            pass


def sanitize_camera_id(cam_id: str) -> str:
    """Match writer sanitize behavior (non-alnum -> underscore)."""
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in cam_id)

def is_realsense_id(cam_id: str) -> bool:
    return cam_id.startswith("RealSense")


def _parse_frame_index(path: Path, fallback: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return fallback


def _process_camera_task(meta_path_str: str, cam_id: str, delete_original_depth: bool) -> Tuple[str, str, bool, str]:
    if rs is None:
        return ("", cam_id, False, "[align] pyrealsense2 not installed; cannot use official alignment")

    meta_path = Path(meta_path_str)
    meta = load_meta(meta_path)
    base = meta_path.parent
    cam = None
    for entry in meta.get("cameras", []):
        if str(entry.get("id")) == cam_id:
            cam = entry
            break
    if cam is None:
        return (str(base), cam_id, False, f"[align] skip {cam_id}: not found in meta")

    cam_dir = base / sanitize_camera_id(str(cam_id))
    depth_dir = cam_dir / "depth"
    color_dir = cam_dir / "color"
    aligner = None
    try:
        m = prepare_meta(cam)
    except KeyError as e:
        return (str(base), cam_id, False, f"[align] skipping {cam_id} in {base}: missing meta key {e}")
    try:
        aligner = RealSenseAligner(m)
    except Exception as exc:
        return (str(base), cam_id, False, f"[align] skipping {cam_id} in {base}: failed to init aligner ({exc})")
    try:
        depth_files = sorted(depth_dir.glob("*.png")) if depth_dir.exists() else []
        if depth_files and color_dir.exists():
            out_dir = cam_dir / "depth_aligned"
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, path in enumerate(tqdm(depth_files, desc=f"align {cam_id}", unit="frame", leave=False), start=1):
                stem = path.stem
                color_path = color_dir / f"{stem}.png"
                if not color_path.exists():
                    continue
                depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if depth_raw is None or depth_raw.dtype != np.uint16:
                    continue
                frame_index = _parse_frame_index(path, idx)
                aligned = aligner.align_depth(depth_raw, frame_index, float(frame_index))
                if aligned is None:
                    continue
                cv2.imwrite(str(out_dir / f"{stem}.png"), aligned)
                if delete_original_depth:
                    os.remove(path)
            return (str(base), cam_id, True, f"[align] processed PNG depth for camera {cam_id} in {base}")

        depth_h5 = cam_dir / "depth_filtered.h5"
        if not depth_h5.exists():
            depth_h5 = cam_dir / "depth.h5"
        if depth_h5.exists():
            if h5py is None:
                return (str(base), cam_id, False, f"[align] skipping {cam_id}: h5py not installed for {depth_h5}")
            with h5py.File(depth_h5, "r") as f:
                if "depth" not in f:
                    return (str(base), cam_id, False, f"[align] skipping {cam_id}: missing /depth dataset in {depth_h5}")
                dset = f["depth"]
                total = dset.shape[0]
                if total <= 0:
                    return (str(base), cam_id, False, f"[align] skipping {cam_id}: no frames")
                kc = m["Kc"]
                aligned_path = cam_dir / "depth_aligned.h5"
                with h5py.File(aligned_path, "w") as out_f:
                    out_dset = out_f.create_dataset(
                        "depth",
                        shape=(total, int(kc["h"]), int(kc["w"])),
                        dtype=np.uint16,
                        chunks=(1, int(kc["h"]), int(kc["w"])),
                    )
                    for idx in tqdm(range(total), desc=f"align {cam_id}", unit="frame", leave=False):
                        depth_raw = dset[idx]
                        if depth_raw is None or depth_raw.dtype != np.uint16:
                            continue
                        frame_index = idx + 1
                        aligned = aligner.align_depth(depth_raw, frame_index, float(frame_index))
                        if aligned is None:
                            continue
                        out_dset[idx] = aligned
            if delete_original_depth and depth_h5.name == "depth.h5":
                depth_h5.unlink(missing_ok=True)
            return (str(base), cam_id, True, f"[align] processed HDF5 depth for camera {cam_id} in {base}")
        return (str(base), cam_id, False, f"[align] skip {cam_id}: no depth data")
    finally:
        if aligner is not None:
            aligner.close()


def main():
    parser = argparse.ArgumentParser(description="Offline depth-to-color alignment from PNGs and meta.json")
    parser.add_argument("root", help="Root directory containing captures/meta.json")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers per depth.h5")
    parser.add_argument("--find-meta", type=str, default="true",
                        choices=["true", "false"],
                        help="Search meta.json recursively (true) or only root/*/meta.json (false)")
    parser.add_argument("--delete-original-depth", type=str, default="false",
                        choices=["true", "false"],
                        help="Whether to delete original depth files after alignment")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    find_meta = args.find_meta.lower() == "true"
    delete_original_depth = args.delete_original_depth.lower() == "true"
    metas = list_meta_files(root, find_meta, 2)
    if not metas:
        print("No meta.json found")
        return
    tasks: List[Tuple[str, str]] = []
    for m in metas:
        base = m.parent
        if should_skip_step(base, "align_depth"):
            print(f"[align] skip {base}, already aligned depth")
            continue
        meta = load_meta(m)
        cam_entries = {
            c["id"]: c for c in meta.get("cameras", []) if is_realsense_id(str(c.get("id", "")))
        }
        if not cam_entries:
            continue
        for cam_id in cam_entries.keys():
            cam_dir = base / sanitize_camera_id(str(cam_id))
            depth_h5 = cam_dir / "depth_filtered.h5"
            if not depth_h5.exists():
                depth_h5 = cam_dir / "depth.h5"
            depth_dir = cam_dir / "depth"
            color_dir = cam_dir / "color"
            if depth_h5.exists() or (depth_dir.exists() and color_dir.exists()):
                tasks.append((str(m), str(cam_id)))

    if not tasks:
        return

    workers = max(1, args.workers)
    results: Dict[str, List[str]] = {}
    if workers > 1:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            future_map = {
                executor.submit(_process_camera_task, meta_path, cam_id, delete_original_depth): (meta_path, cam_id)
                for meta_path, cam_id in tasks
            }
            for future in as_completed(future_map):
                base, cam_id, ok, msg = future.result()
                if msg:
                    print(msg)
                if ok and base:
                    results.setdefault(base, []).append(cam_id)
    else:
        for meta_path, cam_id in tasks:
            base, cam_id, ok, msg = _process_camera_task(meta_path, cam_id, delete_original_depth)
            if msg:
                print(msg)
            if ok and base:
                results.setdefault(base, []).append(cam_id)

    for base_str, cams in results.items():
        if not cams:
            continue
        update_marker(
            Path(base_str),
            "align_depth",
            {
                "cameras": cams,
                "workers": workers,
                "output_dir": "depth_aligned",
                "output_file": "depth_aligned.h5",
            },
        )


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
    print(f"[align] updated marker {marker_path}")


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


if __name__ == "__main__":
    main()
