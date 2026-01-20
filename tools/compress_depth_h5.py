#!/usr/bin/env python3
"""
Compress standard depth.h5 datasets (dataset name: /depth).

Examples:
  python3 tools/compress_depth_h5.py /path/to/depth.h5 --compression gzip --level 4
  python3 tools/compress_depth_h5.py /path/to/captures_root --output-name depth_compressed.h5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

import h5py

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


def iter_depth_h5(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in root.rglob("depth.h5"):
        if path.is_file():
            yield path


def compression_kwargs(compression: str, level: int, shuffle: bool) -> dict:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": level, "shuffle": shuffle}
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": shuffle}
    raise ValueError(f"unsupported compression: {compression}")


def write_compressed(
    input_path: Path,
    output_path: Path,
    compression: str,
    level: int,
    shuffle: bool,
    overwrite: bool,
    progress: Optional["Progress"] = None,
) -> Optional[int]:
    if output_path.exists() and not overwrite:
        print(f"[skip] exists: {output_path}")
        return None

    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    with h5py.File(input_path, "r") as f:
        if "depth" not in f:
            raise RuntimeError(f"missing /depth in {input_path}")
        dset = f["depth"]
        if dset.ndim != 3:
            raise RuntimeError(f"unexpected depth shape {dset.shape} in {input_path}")
        total, height, width = dset.shape
        chunks = dset.chunks or (1, height, width)
        comp_kwargs = compression_kwargs(compression, level, shuffle)
        with h5py.File(tmp_output, "w") as out_f:
            out_dset = out_f.create_dataset(
                "depth",
                shape=(total, height, width),
                dtype=dset.dtype,
                chunks=chunks,
                **comp_kwargs,
            )
            # Copy other datasets/groups as-is.
            for name in f:
                if name == "depth":
                    continue
                f.copy(name, out_f)
            task_id = None
            if progress is not None:
                task_id = progress.add_task(f"frames {input_path.name}", total=total)
            for idx in range(total):
                out_dset[idx] = dset[idx]
                if progress is not None and task_id is not None:
                    progress.advance(task_id, 1)
                elif (idx + 1) % 500 == 0 or (idx + 1) == total:
                    print(f"[write] {input_path.name}: {idx + 1}/{total}")
            if progress is not None and task_id is not None:
                progress.remove_task(task_id)
    os.replace(tmp_output, output_path)
    print(f"[done] {input_path} -> {output_path}")
    return output_path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress depth.h5 files")
    parser.add_argument("input_path", type=Path, help="depth.h5 file or root directory")
    parser.add_argument(
        "--output-name",
        default="depth_compressed.h5",
        help="Output H5 name per camera (ignored for --in-place)",
    )
    parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="Compression codec",
    )
    parser.add_argument("--level", type=int, default=4, help="Gzip level (1-9)")
    parser.add_argument("--shuffle", action="store_true", help="Enable HDF5 shuffle filter")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace depth.h5 in-place using a temporary file",
    )
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        print(f"input not found: {input_path}")
        return 1

    updated = 0
    total_in = 0
    total_out = 0
    depth_files = [p for p in iter_depth_h5(input_path) if p.name == "depth.h5"]
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
        for depth_path in depth_files:
            if args.in_place:
                output_path = depth_path
            else:
                output_path = depth_path.with_name(args.output_name)
            try:
                in_size = depth_path.stat().st_size
                out_size = write_compressed(
                    depth_path,
                    output_path,
                    compression=args.compression,
                    level=args.level,
                    shuffle=args.shuffle,
                    overwrite=args.overwrite or args.in_place,
                )
                if out_size is not None:
                    updated += 1
                    total_in += in_size
                    total_out += out_size
            except Exception as exc:
                print(f"[error] {depth_path}: {exc}")
    else:
        with progress:
            overall = progress.add_task("files", total=len(depth_files))
            for depth_path in depth_files:
                progress.update(overall, description=str(depth_path))
                if args.in_place:
                    output_path = depth_path
                else:
                    output_path = depth_path.with_name(args.output_name)
                try:
                    in_size = depth_path.stat().st_size
                    out_size = write_compressed(
                        depth_path,
                        output_path,
                        compression=args.compression,
                        level=args.level,
                        shuffle=args.shuffle,
                        overwrite=args.overwrite or args.in_place,
                        progress=progress,
                    )
                    if out_size is not None:
                        updated += 1
                        total_in += in_size
                        total_out += out_size
                except Exception as exc:
                    progress.console.print(f"[error] {depth_path}: {exc}")
                finally:
                    progress.advance(overall, 1)
    if total_in > 0:
        ratio = total_out / total_in
        saved = total_in - total_out
        print(
            f"done: compressed {updated} file(s), total {total_in / (1024**2):.2f} MiB -> "
            f"{total_out / (1024**2):.2f} MiB (saved {saved / (1024**2):.2f} MiB, "
            f"{ratio:.2%} of original)"
        )
    else:
        print(f"done: compressed {updated} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
