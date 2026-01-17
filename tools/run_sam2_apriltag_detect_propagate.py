#!/usr/bin/env python3
"""
Detect AprilTags on every frame and use all detections as SAM2 prompts before
propagating to fill gaps where detection fails, with chunked processing.
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import tempfile
import shutil
import subprocess
import gc
import contextlib
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm

repo_root = Path(__file__).resolve().parent.parent
third_party_root = repo_root / "third_party" / "Grounded-SAM-2"
if third_party_root.exists():
    sys.path.insert(0, str(third_party_root))
diffueraser_root = repo_root / "third_party" / "DiffuEraser"
if diffueraser_root.exists():
    sys.path.insert(0, str(diffueraser_root))

from sam2.build_sam import build_sam2_video_predictor


_APRILTAG_DETECTORS = {}
APRILTAG_FAMILY = "tagStandard41h12"
VIDEO_EXTS = [".mp4", ".mkv", ".avi", ".mov"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM2 tracking using AprilTag detections on all frames."
    )
    parser.add_argument("root", nargs="?", help="Capture root or dataset root for meta.json discovery")
    parser.add_argument("--video", help="Path to input video file")
    parser.add_argument("--diffueraser-chunk-size", type=int, default=500,
                        help="Chunk size for DiffuEraser/ProPainter memory management")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Force device selection (default: auto)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit number of extracted frames (0 = no limit)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Use existing frames in --frames-dir without extracting")
    parser.add_argument("--chunk-size", type=int, default=1500,
                        help="Number of frames per chunk for SAM2 tracking")
    parser.add_argument("--prompt-stride", type=int, default=30,
                        help="Use every Nth detected frame as prompt (default: 30)")
    parser.add_argument("--find-meta", type=str, default="true",
                        choices=["true", "false"],
                        help="Search meta.json recursively (true) or only root/*/meta.json (false)")
    parser.add_argument("--cuda-devices", default="",
                        help="Comma-separated CUDA device indices; one video per device")
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep preloaded frames on CPU to save VRAM (default: True)",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep SAM2 state on CPU to reduce VRAM (default: True)",
    )
    parser.add_argument(
        "--async-loading-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load frames asynchronously (default: False)",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return choice


def enable_cuda_fastpath() -> None:
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def extract_frames(video_path: Path, frames_dir: Path, stride: int, max_frames: int) -> None:
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    print(video_info)
    frame_generator = sv.get_video_frames_generator(
        str(video_path), stride=stride, start=0, end=None
    )
    frames_dir.mkdir(parents=True, exist_ok=True)
    with sv.ImageSink(
        target_dir_path=frames_dir,
        overwrite=True,
        image_name_pattern="{:05d}.jpg",
    ) as sink:
        saved = 0
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break


def list_frame_names(frames_dir: Path) -> List[str]:
    frame_names = [
        p.name
        for p in frames_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(Path(p).stem))
    if not frame_names:
        raise RuntimeError(f"No JPEG frames found in {frames_dir}")
    return frame_names


def sanitize_camera_id(cam_id: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in cam_id)

def is_realsense_id(cam_id: str) -> bool:
    return cam_id.startswith("RealSense")


def find_meta_files(root: Path, max_depth: int) -> List[Path]:
    result: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if "meta.json" in filenames:
            result.append(Path(dirpath) / "meta.json")
    return sorted(result)


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
    print(f"[sam2] updated marker {marker_path}")


def read_encode_output_name(capture_root: Path) -> str:
    marker_path = capture_root / "postprocess_markers.json"
    if not marker_path.exists():
        return ""
    try:
        payload = json.loads(marker_path.read_text())
    except json.JSONDecodeError:
        return ""
    steps = payload.get("steps")
    if not isinstance(steps, dict):
        return ""
    entry = steps.get("encode_videos")
    if not isinstance(entry, dict):
        return ""
    output_name = entry.get("output_name", "")
    return str(output_name) if output_name else ""


def resolve_camera_video(capture_root: Path, cam_id: str) -> Optional[Path]:
    cam_dir = capture_root / sanitize_camera_id(cam_id)
    output_name = read_encode_output_name(capture_root)
    if output_name:
        cand = cam_dir / output_name
        if cand.exists():
            return cand
        print(f"[warning] missing aligned video {cand}")
        return None
    for name in ("color", "rgb"):
        for ext in VIDEO_EXTS:
            cand = cam_dir / f"{name}{ext}"
            if cand.exists():
                return cand
    return None


def build_jobs_from_meta(root: Path, find_meta: bool) -> List[Dict[str, str]]:
    meta_paths: List[Path] = []
    if root.is_file() and root.name == "meta.json":
        meta_paths = [root]
    elif (root / "meta.json").exists():
        meta_paths = [root / "meta.json"]
    else:
        meta_paths = list_meta_files(root, find_meta, 2)
    jobs: List[Dict[str, str]] = []
    for meta_path in meta_paths:
        capture_root = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            print(f"[error] invalid meta.json: {meta_path}")
            continue
        cam_ids = [
            str(c.get("id"))
            for c in meta.get("cameras", [])
            if c.get("id") is not None and is_realsense_id(str(c.get("id")))
        ]
        if not cam_ids:
            print(f"[warning] no cameras in {meta_path}")
            continue
        output_name = read_encode_output_name(capture_root)
        for cam_id in cam_ids:
            video_path = resolve_camera_video(capture_root, cam_id)
            if video_path is None:
                if output_name:
                    print(f"[warning] skipping {cam_id} in {capture_root} (missing aligned video)")
                else:
                    print(f"[warning] no rgb/color video for {cam_id} in {capture_root}")
                continue
            jobs.append(
                {
                    "capture_root": str(capture_root),
                    "cam_id": str(cam_id),
                    "video_path": str(video_path),
                }
            )
    return jobs


def build_outputs(video_path: Path, cam_id: str) -> Dict[str, Path]:
    safe_id = sanitize_camera_id(cam_id)
    base_dir = video_path.parent
    return {
        "frames_dir": base_dir / "frames" / safe_id,
        "results_dir": base_dir / "results" / safe_id,
        "output_video": base_dir / f"{safe_id}_tracking.mp4",
        "output_mask_video": base_dir / f"{safe_id}_masks.mp4",
        "inpaint_dir": base_dir / "inpaint" / safe_id,
        "inpaint_output": base_dir / f"{safe_id}_inpaint.mp4",
    }


def _get_apriltag_detector(family: str):
    family_norm = family.strip()
    family_lower = family_norm.lower()
    if family_lower in {"41h12", "tag41h12"}:
        family_norm = "tagStandard41h12"
    elif not family_lower.startswith("tag"):
        family_norm = f"tag{family_norm}"
    detector = _APRILTAG_DETECTORS.get(family_norm)
    if detector is None:
        try:
            from pupil_apriltags import Detector
        except ModuleNotFoundError as exc:
            raise RuntimeError("pupil_apriltags not available; install pupil-apriltags") from exc
        detector = Detector(families=family_norm)
        _APRILTAG_DETECTORS[family_norm] = detector
    return detector


def detect_apriltag_boxes(image_bgr: np.ndarray, family: str) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    detector = _get_apriltag_detector(family)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    results: List[Tuple[int, Tuple[float, float, float, float]]] = []
    if not detections:
        return results
    for det in detections:
        corners = np.asarray(det.corners, dtype=np.float32)
        xmin = float(corners[:, 0].min())
        ymin = float(corners[:, 1].min())
        xmax = float(corners[:, 0].max())
        ymax = float(corners[:, 1].max())
        results.append((int(det.tag_id), (xmin, ymin, xmax, ymax)))
    return results


def collect_apriltag_detections(
    frames_dir: Path,
    frame_names: List[str],
    family: str
) -> Tuple[Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]], List[int]]:
    detections_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    tag_ids: set[int] = set()
    for frame_idx, frame_name in enumerate(tqdm(frame_names, desc="Detecting AprilTags")):
        frame_path = frames_dir / frame_name
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"[warning] failed to read frame: {frame_path}")
            continue
        detections = detect_apriltag_boxes(image, family)
        if detections:
            detections_by_frame[frame_idx] = detections
            for tag_id, _ in detections:
                tag_ids.add(int(tag_id))
    return detections_by_frame, sorted(tag_ids)


def select_conditioning_frames(
    detections_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]],
    stride: int = 1,
) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    if stride <= 1:
        return dict(detections_by_frame)
    selected: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for idx, frame_idx in enumerate(sorted(detections_by_frame.keys())):
        if idx % stride == 0:
            selected[frame_idx] = detections_by_frame[frame_idx]
    return selected


def copy_chunk_frames(
    frames_dir: Path,
    chunk_frame_names: List[str],
    temp_chunk_dir: Path
) -> None:
    for i, frame_name in enumerate(chunk_frame_names):
        src_path = frames_dir / frame_name
        dst_path = temp_chunk_dir / f"{i:05d}.jpg"
        shutil.copy2(src_path, dst_path)


def normalize_mask_to_frame(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    mask_arr = mask
    if mask_arr.ndim == 3:
        if mask_arr.shape[0] == 1:
            mask_arr = mask_arr.squeeze(0)
        elif mask_arr.shape[2] == 1:
            mask_arr = mask_arr.squeeze(2)
        else:
            mask_arr = mask_arr[0]
    if mask_arr.shape[:2] != (target_h, target_w):
        mask_arr = cv2.resize(mask_arr.astype(np.uint8),
                              (target_w, target_h),
                              interpolation=cv2.INTER_NEAREST)
    return mask_arr.astype(np.uint8)


def sample_uniform_points_in_box(
    box: Tuple[float, float, float, float],
    num_points: int = 10,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = box
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    xmin = float(xmin)
    ymin = float(ymin)
    xmax = float(xmax)
    ymax = float(ymax)
    if num_points <= 0 or xmax == xmin or ymax == ymin:
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        return np.array([[cx, cy]], dtype=np.float32)
    cols = 5
    rows = 2
    xs = np.linspace(xmin, xmax, cols + 2, dtype=np.float32)[1:-1]
    ys = np.linspace(ymin, ymax, rows + 2, dtype=np.float32)[1:-1]
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    if points.shape[0] > num_points:
        points = points[:num_points]
    return points.astype(np.float32)


def expand_box(
    box: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    margin_ratio: float = 1.5,
    min_margin_px: int = 8,
) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = box
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    width = max(1.0, xmax - xmin)
    height = max(1.0, ymax - ymin)
    margin_x = max(min_margin_px, width * margin_ratio)
    margin_y = max(min_margin_px, height * margin_ratio)
    xmin = max(0.0, xmin - margin_x)
    ymin = max(0.0, ymin - margin_y)
    xmax = min(float(frame_w - 1), xmax + margin_x)
    ymax = min(float(frame_h - 1), ymax + margin_y)
    return float(xmin), float(ymin), float(xmax), float(ymax)


def normalize_mask_2d(
    mask: np.ndarray,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None
) -> np.ndarray:
    if mask is None:
        return np.empty((0, 0), dtype=np.uint8)
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        if mask_arr.shape[0] == 1:
            mask_arr = mask_arr.squeeze(0)
        elif mask_arr.shape[2] == 1:
            mask_arr = mask_arr.squeeze(2)
        else:
            mask_arr = mask_arr[0]
    elif mask_arr.ndim == 1 and target_h is not None and target_w is not None:
        if mask_arr.size == target_h * target_w:
            mask_arr = mask_arr.reshape(target_h, target_w)
    if mask_arr.ndim != 2:
        return np.empty((0, 0), dtype=mask_arr.dtype)
    if target_h is not None and target_w is not None:
        if mask_arr.shape[:2] != (target_h, target_w):
            mask_arr = cv2.resize(mask_arr.astype(np.uint8),
                                  (target_w, target_h),
                                  interpolation=cv2.INTER_NEAREST)
    return mask_arr


def masks_from_logits(
    out_mask_logits: torch.Tensor, out_obj_ids: List[int]
) -> Dict[int, np.ndarray]:
    mask_logits = out_mask_logits
    if mask_logits.ndim == 4 and mask_logits.shape[1] == 1:
        mask_logits = mask_logits[:, 0, :, :]
    if mask_logits.ndim == 2:
        if not out_obj_ids:
            return {}
        return {int(out_obj_ids[0]): (mask_logits > 0.0).numpy()}
    if mask_logits.ndim != 3:
        return {}
    num_masks = min(mask_logits.shape[0], len(out_obj_ids))
    return {
        int(out_obj_ids[i]): (mask_logits[i] > 0.0).numpy()
        for i in range(num_masks)
    }


def merge_masks(
    base: Dict[int, np.ndarray],
    update: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    merged = dict(base)
    for obj_id, mask in update.items():
        if obj_id in merged:
            merged[obj_id] = np.logical_or(merged[obj_id], mask)
        else:
            merged[obj_id] = mask
    return merged


def write_video_from_frames_dir(
    frames_dir: Path,
    output_video_path: Path,
    fps: float
) -> None:
    frame_names = [
        p.name for p in frames_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(Path(p).stem.split("_")[-1]))
    if not frame_names:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(str(frames_dir / frame_names[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frames_dir / frame_names[0]}")
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_video_path.with_name(f"{output_video_path.stem}.tmp.mp4")
    writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {temp_output}")

    for frame_name in tqdm(frame_names, desc="Writing video"):
        frame = cv2.imread(str(frames_dir / frame_name))
        if frame is None:
            continue
        writer.write(frame)
    writer.release()
    reencode_to_h264(temp_output, output_video_path)


def write_mask_video(
    frames_dir: Path,
    video_segments: Dict[int, Dict[int, np.ndarray]],
    output_mask_path: Path,
    fps: float
) -> None:
    frame_names = [
        p.name for p in frames_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(Path(p).stem))
    if not frame_names:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(str(frames_dir / frame_names[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frames_dir / frame_names[0]}")
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_mask_path.with_name(f"{output_mask_path.stem}.tmp.mp4")
    writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {temp_output}")

    for frame_idx in tqdm(range(len(frame_names)), desc="Writing mask video"):
        mask_frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame_masks = video_segments.get(frame_idx, {})
        if frame_masks:
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            for mask in frame_masks.values():
                mask_arr = mask.astype(np.uint8)
                if mask_arr.ndim == 3:
                    if mask_arr.shape[0] == 1:
                        mask_arr = mask_arr.squeeze(0)
                    elif mask_arr.shape[2] == 1:
                        mask_arr = mask_arr.squeeze(2)
                    else:
                        mask_arr = mask_arr[0]
                if mask_arr.shape[:2] != (height, width):
                    mask_arr = cv2.resize(mask_arr, (width, height), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, mask_arr)
            combined_mask = np.where(combined_mask > 0, 255, 0).astype(np.uint8)
            mask_frame[:, :, 0] = combined_mask
            mask_frame[:, :, 1] = combined_mask
            mask_frame[:, :, 2] = combined_mask
        writer.write(mask_frame)
    writer.release()
    reencode_to_h264(temp_output, output_mask_path)


def reencode_to_h264(input_path: Path, output_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("[warning] ffmpeg not found; keeping mp4v output")
        return

    def wait_for_stable_file(path: Path, attempts: int = 10, delay_s: float = 0.5) -> bool:
        last_size = -1
        for _ in range(attempts):
            if not path.exists():
                time.sleep(delay_s)
                continue
            size = path.stat().st_size
            if size == last_size and size > 0:
                return True
            last_size = size
            time.sleep(delay_s)
        return path.exists() and path.stat().st_size > 0

    if not wait_for_stable_file(input_path):
        print(f"[warning] input video not stable for reencode: {input_path}")
        return

    temp_output = output_path
    if input_path.resolve() == output_path.resolve():
        temp_output = output_path.with_name(f"{output_path.stem}.h264.tmp.mp4")

    command = [
        ffmpeg_path,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(temp_output),
    ]

    for attempt in range(2):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            if temp_output != output_path:
                temp_output.replace(output_path)
            if input_path != output_path:
                input_path.unlink(missing_ok=True)
            return
        if attempt == 0:
            time.sleep(1.0)
            continue
        print("[warning] ffmpeg reencode failed; keeping mp4v output")
        if result.stderr:
            print(result.stderr.strip())
        if temp_output != output_path and temp_output.exists():
            temp_output.unlink(missing_ok=True)


def split_video_pair(
    input_video: Path,
    input_mask: Path,
    output_dir: Path,
    chunk_size: int,
    mask_dilate_kernel: int = 7,
    mask_dilate_iter: int = 2,
) -> List[Dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap_video = cv2.VideoCapture(str(input_video))
    cap_mask = cv2.VideoCapture(str(input_mask))
    if not cap_video.isOpened() or not cap_mask.isOpened():
        raise RuntimeError("Failed to open input video/mask for chunking")
    fps = cap_video.get(cv2.CAP_PROP_FPS)
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    chunks: List[Dict[str, Path]] = []

    writer_video = None
    writer_mask = None
    chunk_idx = -1
    frame_idx = 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (mask_dilate_kernel, mask_dilate_kernel),
    )
    while True:
        ok_v, frame_v = cap_video.read()
        ok_m, frame_m = cap_mask.read()
        if not ok_v or not ok_m:
            break
        mask_gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
        mask_dilated = cv2.dilate(mask_bin, kernel, iterations=mask_dilate_iter)
        frame_m = cv2.cvtColor(mask_dilated, cv2.COLOR_GRAY2BGR)
        if frame_idx % chunk_size == 0:
            if writer_video is not None:
                writer_video.release()
            if writer_mask is not None:
                writer_mask.release()
            chunk_idx += 1
            video_chunk = output_dir / f"video_{chunk_idx:04d}.mp4"
            mask_chunk = output_dir / f"mask_{chunk_idx:04d}.mp4"
            writer_video = cv2.VideoWriter(str(video_chunk), fourcc, fps, (width, height))
            writer_mask = cv2.VideoWriter(str(mask_chunk), fourcc, fps, (width, height))
            chunks.append({"video": video_chunk, "mask": mask_chunk})
        writer_video.write(frame_v)
        writer_mask.write(frame_m)
        frame_idx += 1

    if writer_video is not None:
        writer_video.release()
    if writer_mask is not None:
        writer_mask.release()
    cap_video.release()
    cap_mask.release()
    return chunks


def concat_videos(video_paths: List[Path], output_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg required to concat inpaint chunks")
    list_file = output_path.with_suffix(".list.txt")
    lines = [f"file '{p.as_posix()}'" for p in video_paths]
    list_file.write_text("\n".join(lines))
    command = [
        ffmpeg_path,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "concat failed")


def run_diffueraser(
    input_video: Path,
    input_mask: Path,
    output_dir: Path,
    chunk_size: int,
    output_path: Path
) -> Path:
    from diffueraser.diffueraser import DiffuEraser
    from propainter.inference import Propainter, get_device

    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    if chunk_size > 0:
        chunk_dir = output_dir / "chunks"
        chunks = split_video_pair(input_video, input_mask, chunk_dir, chunk_size)
        if not chunks:
            raise RuntimeError("No chunks generated for inpainting")
    else:
        chunks = [{"video": input_video, "mask": input_mask}]

    base_model_path = diffueraser_root / "weights" / "stable-diffusion-v1-5"
    vae_path = diffueraser_root / "weights" / "sd-vae-ft-mse"
    diffueraser_path = diffueraser_root / "weights" / "diffuEraser"
    propainter_model_dir = diffueraser_root / "weights" / "propainter"

    os.environ["HF_HUB_OFFLINE"] = "1"
    original_cwd = os.getcwd()
    os.chdir(diffueraser_root)
    try:
        video_inpainting_sd = DiffuEraser(
            device, str(base_model_path), str(vae_path), str(diffueraser_path), ckpt="2-Step"
        )
        if str(device).startswith("cuda") and hasattr(video_inpainting_sd.pipeline, "half"):
            video_inpainting_sd.pipeline = video_inpainting_sd.pipeline.half()
    finally:
        os.chdir(original_cwd)

    os.chdir(diffueraser_root)
    try:
        propainter = Propainter(str(propainter_model_dir), device=device)
    finally:
        os.chdir(original_cwd)

    chunk_outputs: List[Path] = []
    for idx, chunk in enumerate(chunks):
        chunk_video = chunk["video"]
        chunk_mask = chunk["mask"]
        priori_path = output_dir / f"priori_{idx:04d}.mp4"
        chunk_out = output_dir / f"inpaint_{idx:04d}.mp4"
        os.chdir(diffueraser_root)
        try:
            video_info = sv.VideoInfo.from_video_path(str(chunk_video))
            fps = video_info.fps if video_info.fps > 0 else 30.0
            total_frames = video_info.total_frames
            video_length_seconds = max(1.0, total_frames / fps) if total_frames > 0 else 1.0
            with torch.cuda.amp.autocast(enabled=False):
                propainter.forward(
                    str(chunk_video), str(chunk_mask), str(priori_path),
                    video_length=video_length_seconds,
                    ref_stride=min(10, max(1, chunk_size // 2)),
                    neighbor_length=min(10, max(1, chunk_size // 2)),
                    subvideo_length=min(50, max(1, chunk_size)),
                    mask_dilation=8,
                    save_fps=fps
                )
        finally:
            os.chdir(original_cwd)

        os.chdir(diffueraser_root)
        try:
            video_info = sv.VideoInfo.from_video_path(str(chunk_video))
            fps = video_info.fps if video_info.fps > 0 else 30.0
            total_frames = video_info.total_frames
            video_length_seconds = max(1.0, total_frames / fps) if total_frames > 0 else 1.0
            with torch.cuda.amp.autocast(enabled=False):
                video_inpainting_sd.forward(
                    str(chunk_video), str(chunk_mask), str(priori_path), str(chunk_out),
                    max_img_size=960,
                    video_length=video_length_seconds,
                    mask_dilation_iter=8,
                    guidance_scale=None
                )
        finally:
            os.chdir(original_cwd)
        reencode_to_h264(chunk_out, chunk_out)
        chunk_outputs.append(chunk_out)

    if len(chunk_outputs) == 1:
        chunk_outputs[0].replace(output_path)
    else:
        concat_videos(chunk_outputs, output_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    reencode_to_h264(output_path, output_path)
    return output_path


def run_single_video(
    args: argparse.Namespace,
    video_path: Path,
    output_video: Path,
    output_mask_video: Path,
    frames_dir: Path,
    results_dir: Path,
    inpaint_dir: Path,
    inpaint_output: Path,
    device: str,
    capture_root: Optional[Path] = None,
    cam_id: str = "",
) -> int:
    try:
        if not third_party_root.exists():
            print(f"[error] missing Grounded-SAM-2 at {third_party_root}")
            return 1
        if not diffueraser_root.exists():
            print(f"[error] missing DiffuEraser at {diffueraser_root}")
            return 1

        if not video_path.exists():
            print(f"[error] video not found: {video_path}")
            return 1

        if not args.skip_extract:
            extract_frames(video_path, frames_dir, 1, args.max_frames)

        frame_names = list_frame_names(frames_dir)

        os.chdir(third_party_root)
        sys.path.insert(0, str(third_party_root))

        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        if device == "cuda":
            enable_cuda_fastpath()

        video_segments: Dict[int, Dict[int, np.ndarray]] = {}
        tag_ids: set[int] = set()
        prev_chunk_last_masks: Dict[int, np.ndarray] | None = None
        total_frames = len(frame_names)
        num_chunks = math.ceil(total_frames / args.chunk_size)

        video_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "checkpoints/sam2.1_hiera_large.pt",
            device=device,
            dtype=torch_dtype,
        )

        autocast_ctx: contextlib.AbstractContextManager
        if device.startswith("cuda"):
            if hasattr(torch, "autocast"):
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            autocast_ctx = contextlib.nullcontext()

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * args.chunk_size
            end_idx = min((chunk_idx + 1) * args.chunk_size, total_frames)
            chunk_frame_names = frame_names[start_idx:end_idx]
            print(f"[info] processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_frame_names)} frames)")

            with tempfile.TemporaryDirectory() as temp_chunk_dir:
                temp_chunk_dir = Path(temp_chunk_dir)
                copy_chunk_frames(frames_dir, chunk_frame_names, temp_chunk_dir)

                chunk_names = list_frame_names(temp_chunk_dir)
                detections_by_frame, chunk_tag_ids = collect_apriltag_detections(
                    temp_chunk_dir, chunk_names, APRILTAG_FAMILY
                )
                tag_ids.update(chunk_tag_ids)
                conditioning_frames = select_conditioning_frames(
                    detections_by_frame, stride=args.prompt_stride
                )

                first_frame_path = temp_chunk_dir / "00000.jpg"
                first_frame = cv2.imread(str(first_frame_path))
                if first_frame is None:
                    print(f"[warning] failed to read chunk frame: {first_frame_path}")
                    continue
                frame_h, frame_w = first_frame.shape[:2]
                inference_state = None
                try:
                    inference_state = video_predictor.init_state(
                        video_path=str(temp_chunk_dir),
                        offload_video_to_cpu=args.offload_video_to_cpu,
                        offload_state_to_cpu=args.offload_state_to_cpu,
                        async_loading_frames=args.async_loading_frames,
                    )
                    with autocast_ctx:
                        if prev_chunk_last_masks:
                            for obj_id, mask in prev_chunk_last_masks.items():
                                if mask is None or getattr(mask, "size", 0) == 0:
                                    continue
                                mask_arr = normalize_mask_to_frame(mask, frame_h, frame_w)
                                video_predictor.add_new_mask(
                                    inference_state=inference_state,
                                    frame_idx=0,
                                    obj_id=int(obj_id),
                                    mask=mask_arr,
                                )

                        for frame_idx, detections in conditioning_frames.items():
                            for tag_id, box in detections:
                                expanded_box = expand_box(box, frame_w, frame_h)
                                points = sample_uniform_points_in_box(box, num_points=1)
                                labels = np.ones((points.shape[0],), dtype=np.int32)
                                video_predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=frame_idx,
                                    obj_id=int(tag_id),
                                    points=points,
                                    box=np.array(expanded_box, dtype=np.float32),
                                    labels=labels,
                                )

                        if not conditioning_frames and not prev_chunk_last_masks:
                            print("[warning] no AprilTags detected in this chunk and no carry-over masks")
                            continue

                        chunk_segments: Dict[int, Dict[int, np.ndarray]] = {}
                        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                            inference_state
                        ):
                            if device.startswith("cuda"):
                                out_mask_logits = out_mask_logits.to(dtype=torch.float16).cpu()
                            chunk_segments[out_frame_idx] = masks_from_logits(out_mask_logits, out_obj_ids)

                        print("[info] running reverse propagation for chunk")
                        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                            inference_state,
                            reverse=True,
                        ):
                            if device.startswith("cuda"):
                                out_mask_logits = out_mask_logits.to(dtype=torch.float16).cpu()
                            rev_masks = masks_from_logits(out_mask_logits, out_obj_ids)
                            if out_frame_idx in chunk_segments:
                                chunk_segments[out_frame_idx] = merge_masks(
                                    chunk_segments[out_frame_idx], rev_masks
                                )
                            else:
                                chunk_segments[out_frame_idx] = rev_masks

                finally:
                    if inference_state is not None:
                        video_predictor.reset_state(inference_state)
                        del inference_state
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                for out_frame_idx, segments in chunk_segments.items():
                    global_idx = start_idx + out_frame_idx
                    video_segments[global_idx] = segments

                if chunk_segments:
                    last_idx = max(chunk_segments.keys())
                    last_masks = chunk_segments.get(last_idx, {})
                    prev_chunk_last_masks = {
                        int(obj_id): mask.copy()
                        for obj_id, mask in last_masks.items()
                        if mask is not None and hasattr(mask, "size") and mask.size > 0
                    }
                else:
                    prev_chunk_last_masks = None

        del video_predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        tag_labels = {int(tag_id): f"apriltag_{int(tag_id)}" for tag_id in sorted(tag_ids)}
        for frame_idx, frame_name in enumerate(tqdm(frame_names, desc="Annotating frames")):
            frame_path = frames_dir / frame_name
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            segments = video_segments.get(frame_idx, {})
            if segments:
                object_ids = []
                mask_list = []
                for obj_id, mask in segments.items():
                    mask_arr = normalize_mask_2d(mask, image.shape[0], image.shape[1])
                    if mask_arr.size == 0 or mask_arr.ndim != 2:
                        continue
                    mask_arr = mask_arr.astype(bool)
                    if not mask_arr.any():
                        continue
                    object_ids.append(obj_id)
                    mask_list.append(mask_arr)
                if not mask_list:
                    annotated = image
                    cv2.imwrite(str(results_dir / f"annotated_{frame_idx:05d}.jpg"), annotated)
                    continue
                filtered_ids: List[int] = []
                filtered_masks: List[np.ndarray] = []
                for obj_id, mask_arr in zip(object_ids, mask_list):
                    mask_norm = normalize_mask_2d(mask_arr, image.shape[0], image.shape[1])
                    if mask_norm.size == 0 or mask_norm.ndim != 2:
                        continue
                    if not mask_norm.any():
                        continue
                    filtered_ids.append(obj_id)
                    filtered_masks.append(mask_norm.astype(bool))
                if not filtered_masks:
                    annotated = image
                    cv2.imwrite(str(results_dir / f"annotated_{frame_idx:05d}.jpg"), annotated)
                    continue
                masks = np.stack(filtered_masks, axis=0)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0, :, :]
                xyxy = sv.mask_to_xyxy(masks).astype(np.float32)
                detections = sv.Detections(
                    xyxy=xyxy,
                    mask=masks,
                    class_id=np.array(filtered_ids, dtype=np.int32),
                )
                box_annotator = sv.BoxAnnotator()
                label_annotator = sv.LabelAnnotator()
                mask_annotator = sv.MaskAnnotator()
                annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
                annotated = label_annotator.annotate(
                    annotated,
                    detections=detections,
                    labels=[tag_labels.get(i, f"obj_{i}") for i in filtered_ids],
                )
                annotated = mask_annotator.annotate(scene=annotated, detections=detections)
            else:
                annotated = image

            cv2.imwrite(str(results_dir / f"annotated_{frame_idx:05d}.jpg"), annotated)

        video_info = sv.VideoInfo.from_video_path(str(video_path))
        write_video_from_frames_dir(results_dir, output_video, video_info.fps)
        write_mask_video(frames_dir, video_segments, output_mask_video, video_info.fps)

        print("[info] running DiffuEraser inpainting")
        run_diffueraser(
            input_video=video_path,
            input_mask=output_mask_video,
            output_dir=inpaint_dir,
            chunk_size=args.diffueraser_chunk_size,
            output_path=inpaint_output
        )

        if capture_root is not None:
            update_marker(
                capture_root,
                "sam2_apriltag_tracking",
                {
                    "camera_id": cam_id,
                    "input_video": str(video_path),
                    "output_video": str(output_video),
                    "output_mask_video": str(output_mask_video),
                    "inpaint_output": str(inpaint_output),
                },
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0
    finally:
        for path in (frames_dir, results_dir, inpaint_dir):
            if path.exists():
                shutil.rmtree(path)


def _device_worker(device_id: str, jobs: List[Dict[str, str]], args_dict: Dict) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    args = argparse.Namespace(**args_dict)
    device = resolve_device(args.device)
    if device.startswith("cuda"):
        enable_cuda_fastpath()
    exit_code = 0
    for job in jobs:
        video_path = Path(job["video_path"]).expanduser().resolve()
        outputs = build_outputs(video_path, job["cam_id"])
        capture_root = Path(job["capture_root"]).expanduser().resolve()
        if not (capture_root / "meta.json").exists():
            capture_root = None
        code = run_single_video(
            args=args,
            video_path=video_path,
            output_video=outputs["output_video"],
            output_mask_video=outputs["output_mask_video"],
            frames_dir=outputs["frames_dir"],
            results_dir=outputs["results_dir"],
            inpaint_dir=outputs["inpaint_dir"],
            inpaint_output=outputs["inpaint_output"],
            device=device,
            capture_root=capture_root,
            cam_id=job["cam_id"],
        )
        if code != 0:
            exit_code = code
    return exit_code


def _worker_entry(queue: mp.Queue, device_id: str, jobs: List[Dict[str, str]], args_dict: Dict) -> None:
    queue.put(_device_worker(device_id, jobs, args_dict))


def main() -> int:
    args = parse_args()
    if not args.video and not args.root:
        print("[error] provide --video or a root directory")
        return 1
    device = resolve_device(args.device)
    if args.chunk_size < 1:
        print("[error] --chunk-size must be >= 1")
        return 1
    if args.prompt_stride < 1:
        print("[error] --prompt-stride must be >= 1")
        return 1

    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        capture_root = video_path.parent.parent
        if not (capture_root / "meta.json").exists():
            capture_root = None
        outputs = build_outputs(video_path, video_path.parent.name)
        return run_single_video(
            args=args,
            video_path=video_path,
            output_video=outputs["output_video"],
            output_mask_video=outputs["output_mask_video"],
            frames_dir=outputs["frames_dir"],
            results_dir=outputs["results_dir"],
            inpaint_dir=outputs["inpaint_dir"],
            inpaint_output=outputs["inpaint_output"],
            device=device,
            capture_root=capture_root,
            cam_id=video_path.parent.name,
        )

    root = Path(args.root).expanduser().resolve()
    find_meta = args.find_meta.lower() == "true"
    jobs = build_jobs_from_meta(root, find_meta)
    if not jobs:
        print("[error] no videos found via meta.json")
        return 1

    device_list = [d.strip() for d in args.cuda_devices.split(",") if d.strip()]
    if not device_list:
        if device.startswith("cuda"):
            device_list = ["0"]
        else:
            device_list = []

    if not device_list or device == "cpu":
        exit_code = 0
        for job in jobs:
            video_path = Path(job["video_path"]).expanduser().resolve()
            outputs = build_outputs(video_path, job["cam_id"])
            capture_root = Path(job["capture_root"]).expanduser().resolve()
            if not (capture_root / "meta.json").exists():
                capture_root = None
            code = run_single_video(
                args=args,
                video_path=video_path,
                output_video=outputs["output_video"],
                output_mask_video=outputs["output_mask_video"],
                frames_dir=outputs["frames_dir"],
                results_dir=outputs["results_dir"],
                inpaint_dir=outputs["inpaint_dir"],
                inpaint_output=outputs["inpaint_output"],
                device=device,
                capture_root=capture_root,
                cam_id=job["cam_id"],
            )
            if code != 0:
                exit_code = code
        return exit_code

    device_jobs: Dict[str, List[Dict[str, str]]] = {d: [] for d in device_list}
    for idx, job in enumerate(jobs):
        device_jobs[device_list[idx % len(device_list)]].append(job)

    ctx = mp.get_context("spawn")
    workers: List[mp.Process] = []
    results: List[mp.Queue] = []
    args_dict = vars(args)
    for device_id, dev_jobs in device_jobs.items():
        if not dev_jobs:
            continue
        queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_worker_entry, args=(queue, device_id, dev_jobs, args_dict))
        proc.start()
        workers.append(proc)
        results.append(queue)

    exit_code = 0
    for proc, queue in zip(workers, results):
        proc.join()
        if not queue.empty():
            code = queue.get()
            if code != 0:
                exit_code = code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
