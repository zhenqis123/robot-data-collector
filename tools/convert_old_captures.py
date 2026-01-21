#!/usr/bin/env python3
"""
Convert legacy RealSense captures (color/depth PNGs) into rgb.mkv + depth.h5.

Example:
  python3 tools/convert_old_captures.py data_old/captures --fps 30
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import h5py
import numpy as np

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:  # pragma: no cover - optional dependency
    Progress = None


OLD_HEADERS = {
    "timestamp_iso",
    "timestamp_ms",
    "device_timestamp_ms",
    "color_path",
    "depth_path",
}
NEW_HEADERS = [
    "frame_index",
    "timestamp_iso",
    "timestamp_ms",
    "device_timestamp_ms",
    "rgb_path",
    "depth_path",
]


def iter_camera_dirs(root: Path) -> Iterable[Path]:
    matches = []
    for path in root.rglob("RealSense_*"):
        if not path.is_dir():
            continue
        if (path / "color").is_dir() and (path / "depth").is_dir() and (path / "timestamps.csv").exists():
            matches.append(path)
    for path in sorted(matches):
        yield path


def read_header(path: Path) -> List[str]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def is_new_format(cam_dir: Path) -> bool:
    ts_path = cam_dir / "timestamps.csv"
    if not (cam_dir / "rgb.mkv").exists() or not (cam_dir / "depth.h5").exists():
        return False
    if not ts_path.exists():
        return False
    header = read_header(ts_path)
    return "frame_index" in header and "rgb_path" in header and "depth_path" in header


def resolve_path(base: Path, raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return base / candidate


def read_old_timestamps(ts_path: Path) -> Tuple[List[dict], List[str]]:
    with ts_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def build_frame_lists(cam_dir: Path, rows: List[dict]) -> Tuple[List[Path], List[Path]]:
    color_paths: List[Path] = []
    depth_paths: List[Path] = []
    for row in rows:
        color_raw = row.get("color_path") or row.get("rgb_path") or ""
        depth_raw = row.get("depth_path") or ""
        if color_raw and color_raw.lower().endswith(".png"):
            color_paths.append(resolve_path(cam_dir, color_raw))
        if depth_raw and depth_raw.lower().endswith(".png"):
            depth_paths.append(resolve_path(cam_dir, depth_raw))
    if not color_paths:
        color_dir = cam_dir / "color"
        if color_dir.is_dir():
            color_paths = sorted(color_dir.glob("*.png"))
    if not depth_paths:
        depth_dir = cam_dir / "depth"
        if depth_dir.is_dir():
            depth_paths = sorted(depth_dir.glob("*.png"))
    if rows:
        target = len(rows)
        if len(color_paths) > target:
            color_paths = color_paths[:target]
        if len(depth_paths) > target:
            depth_paths = depth_paths[:target]
    return color_paths, depth_paths


def ensure_backup(ts_path: Path) -> None:
    if not ts_path.exists():
        return
    header = read_header(ts_path)
    if "color_path" not in header:
        return
    backup = ts_path.with_name("timestamps_old.csv")
    if backup.exists():
        return
    ts_path.replace(backup)


def detect_image_sequence(frames: List[Path]) -> Optional[Tuple[Path, int, int, str, int]]:
    if not frames:
        return None
    parent = frames[0].parent
    if any(f.parent != parent for f in frames):
        return None
    stems = [f.stem for f in frames]
    if any(not stem.isdigit() for stem in stems):
        return None
    widths = {len(stem) for stem in stems}
    if len(widths) != 1:
        return None
    numbers = sorted(int(stem) for stem in stems)
    for prev, cur in zip(numbers, numbers[1:]):
        if cur != prev + 1:
            return None
    ext = frames[0].suffix
    width = widths.pop()
    return parent, numbers[0], len(numbers), ext, width


def encode_rgb_mkv(frames: List[Path], output: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; please install ffmpeg to encode rgb.mkv")
    seq = detect_image_sequence(frames)
    tmp_output = output.with_suffix(".tmp.mkv")
    try:
        if seq is not None:
            parent, start_number, count, ext, width = seq
            pattern = str(parent / f"%0{width}d{ext}")
            cmd = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                f"{fps}",
                "-start_number",
                str(start_number),
                "-i",
                pattern,
                "-frames:v",
                str(count),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "matroska",
                str(tmp_output),
            ]
        else:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as list_file:
                list_file.write("ffconcat version 1.0\n")
                for frame in frames:
                    list_file.write(f"file '{frame.as_posix()}'\n")
                list_path = list_file.name
            cmd = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-fflags",
                "+genpts",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-r",
                f"{fps}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "matroska",
                str(tmp_output),
            ]
        subprocess.run(cmd, check=True)
        os.replace(tmp_output, output)
    finally:
        if seq is None:
            Path(list_path).unlink(missing_ok=True)
        tmp_output.unlink(missing_ok=True)


def write_depth_h5(
    depth_paths: List[Path],
    output: Path,
    progress: Optional["Progress"] = None,
    task_id: Optional[int] = None,
) -> int:
    sample = None
    for p in depth_paths:
        sample = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if sample is not None:
            break
    if sample is None:
        raise RuntimeError("failed to read any depth frame")
    if sample.ndim != 2:
        raise RuntimeError("depth frames must be single-channel uint16")
    h, w = sample.shape
    depth_count = len(depth_paths)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    try:
        with h5py.File(tmp_output, "w") as f:
            dset = f.create_dataset(
                "depth",
                shape=(depth_count, h, w),
                dtype=sample.dtype,
                chunks=(1, h, w),
            )
            for idx, p in enumerate(depth_paths):
                frame = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if frame is None:
                    frame = np.zeros((h, w), dtype=sample.dtype)
                elif frame.shape != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
                dset[idx] = frame
                if progress is not None and task_id is not None:
                    progress.advance(task_id, 1)
        os.replace(tmp_output, output)
    finally:
        tmp_output.unlink(missing_ok=True)
    return depth_count


def write_new_timestamps(rows: List[dict], output: Path) -> None:
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    try:
        with tmp_output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(NEW_HEADERS)
            for idx, row in enumerate(rows, start=1):
                writer.writerow(
                    [
                        idx,
                        row.get("timestamp_iso", ""),
                        row.get("timestamp_ms", ""),
                        row.get("device_timestamp_ms", ""),
                        "rgb.mkv",
                        "depth.h5",
                    ]
                )
        os.replace(tmp_output, output)
    finally:
        tmp_output.unlink(missing_ok=True)


def convert_camera(
    cam_dir: Path,
    out_dir: Path,
    fps: float,
    overwrite: bool,
    delete_frames: bool,
    dry_run: bool,
    progress: Optional["Progress"] = None,
) -> bool:
    console = progress.console if progress is not None else None
    if out_dir.exists() and is_new_format(out_dir) and not overwrite:
        if console:
            console.print(f"[skip] already converted: {out_dir}")
        else:
            print(f"[skip] already converted: {out_dir}")
        return False
    ts_path = cam_dir / "timestamps.csv"
    if not ts_path.exists():
        if console:
            console.print(f"[skip] missing timestamps.csv: {cam_dir}")
        else:
            print(f"[skip] missing timestamps.csv: {cam_dir}")
        return False
    rows, headers = read_old_timestamps(ts_path)
    if not rows:
        if console:
            console.print(f"[skip] empty timestamps.csv: {cam_dir}")
        else:
            print(f"[skip] empty timestamps.csv: {cam_dir}")
        return False
    if headers and "timestamp_ms" not in headers:
        if console:
            console.print(f"[skip] unexpected timestamps header in {ts_path}")
        else:
            print(f"[skip] unexpected timestamps header in {ts_path}")
        return False
    color_paths, depth_paths = build_frame_lists(cam_dir, rows)
    if not color_paths or not depth_paths:
        if console:
            console.print(f"[skip] missing color/depth paths in {ts_path}")
        else:
            print(f"[skip] missing color/depth paths in {ts_path}")
        return False
    if len(color_paths) < len(rows) or len(depth_paths) < len(rows):
        if console:
            console.print(f"[skip] not enough color/depth frames under {cam_dir}")
        else:
            print(f"[skip] not enough color/depth frames under {cam_dir}")
        return False
    missing = [p for p in color_paths + depth_paths if not p.exists()]
    if missing:
        if console:
            console.print(f"[skip] missing {len(missing)} frames under {cam_dir}")
        else:
            print(f"[skip] missing {len(missing)} frames under {cam_dir}")
        return False
    if dry_run:
        if console:
            console.print(f"[dry-run] {cam_dir} -> {out_dir} ({len(rows)} frames)")
        else:
            print(f"[dry-run] {cam_dir} -> {out_dir} ({len(rows)} frames)")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    if cam_dir == out_dir:
        ensure_backup(out_dir / "timestamps.csv")
    rgb_path = out_dir / "rgb.mkv"
    depth_path = out_dir / "depth.h5"
    ts_out = out_dir / "timestamps.csv"
    if rgb_path.exists() and not overwrite:
        print(f"[skip] rgb.mkv exists: {rgb_path}")
        return False
    if depth_path.exists() and not overwrite:
        print(f"[skip] depth.h5 exists: {depth_path}")
        return False
    if ts_out.exists() and not overwrite and is_new_format(out_dir):
        print(f"[skip] timestamps.csv exists: {ts_out}")
        return False
    if console:
        console.print(f"[convert] {cam_dir} -> {out_dir} ({len(rows)} frames)")
    else:
        print(f"[convert] {cam_dir} -> {out_dir} ({len(rows)} frames)")
    rgb_task = None
    if progress is not None:
        rgb_task = progress.add_task("rgb encode", total=None)
    try:
        encode_rgb_mkv(color_paths, rgb_path, fps)
    finally:
        if progress is not None and rgb_task is not None:
            progress.remove_task(rgb_task)
    depth_task = None
    if progress is not None:
        depth_task = progress.add_task("depth write", total=len(depth_paths))
    try:
        write_depth_h5(depth_paths, depth_path, progress=progress, task_id=depth_task)
    finally:
        if progress is not None and depth_task is not None:
            progress.remove_task(depth_task)
    write_new_timestamps(rows, ts_out)
    if delete_frames and cam_dir == out_dir:
        shutil.rmtree(out_dir / "color", ignore_errors=True)
        shutil.rmtree(out_dir / "depth", ignore_errors=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert legacy color/depth PNG captures to rgb.mkv + depth.h5.")
    parser.add_argument("input_root", type=Path, help="Root directory of old captures")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root; defaults to input root (in-place)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--delete-frames",
        action="store_true",
        help="Delete color/ and depth/ after successful conversion (in-place only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = (args.output_root or input_root).resolve()
    if not input_root.exists():
        print(f"input root not found: {input_root}")
        return 1
    if args.delete_frames and output_root != input_root:
        print("--delete-frames is only supported for in-place conversions")
        return 1

    converted = 0
    camera_dirs = list(iter_camera_dirs(input_root))
    progress = None
    if Progress is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
    if progress is None:
        for cam_dir in camera_dirs:
            try:
                rel = cam_dir.relative_to(input_root)
            except ValueError:
                rel = cam_dir.name
            out_dir = output_root / rel
            try:
                if convert_camera(
                    cam_dir,
                    out_dir,
                    fps=args.fps,
                    overwrite=args.overwrite,
                    delete_frames=args.delete_frames,
                    dry_run=args.dry_run,
                ):
                    converted += 1
            except Exception as exc:
                print(f"[error] {cam_dir}: {exc}")
    else:
        with progress:
            overall = progress.add_task("cameras", total=len(camera_dirs))
            for cam_dir in camera_dirs:
                progress.update(overall, description=str(cam_dir))
                try:
                    rel = cam_dir.relative_to(input_root)
                except ValueError:
                    rel = cam_dir.name
                out_dir = output_root / rel
                try:
                    if convert_camera(
                        cam_dir,
                        out_dir,
                        fps=args.fps,
                        overwrite=args.overwrite,
                        delete_frames=args.delete_frames,
                        dry_run=args.dry_run,
                        progress=progress,
                    ):
                        converted += 1
                except Exception as exc:
                    progress.console.print(f"[error] {cam_dir}: {exc}")
                finally:
                    progress.advance(overall, 1)
    print(f"done: converted {converted} camera folders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
