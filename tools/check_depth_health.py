#!/usr/bin/env python3
"""
Check depth HDF5 files for common issues (missing data, identical frames).

Usage:
  python -m tools.check_depth_health data_temp/data_xizh/captures
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def find_depth_files(root: Path, include_aligned: bool) -> List[Path]:
    patterns = ["depth.h5"]
    if include_aligned:
        patterns.append("depth_aligned.h5")
    results: List[Path] = []
    for pattern in patterns:
        results.extend(root.rglob(pattern))
    return sorted(set(results))


def sample_indices(total: int, samples: int) -> List[int]:
    if total <= 0:
        return []
    if samples <= 1:
        return [0]
    if samples >= total:
        return list(range(total))
    step = (total - 1) / float(samples - 1)
    indices = [int(round(i * step)) for i in range(samples)]
    return sorted(set(indices))


def hash_frame(frame: np.ndarray) -> str:
    return hashlib.md5(frame.tobytes()).hexdigest()


def check_depth_file(path: Path, samples: int) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if h5py is None:
        issues.append("h5py not installed")
        return False, issues
    try:
        with h5py.File(path, "r") as f:
            if "depth" not in f:
                issues.append("missing /depth dataset")
                return False, issues
            dset = f["depth"]
            if dset.ndim != 3:
                issues.append(f"unexpected depth dataset shape: {dset.shape}")
                return False, issues
            total = int(dset.shape[0])
            if total <= 0:
                issues.append("no depth frames")
                return False, issues

            idxs = sample_indices(total, samples)
            hashes = set()
            mins: List[int] = []
            maxs: List[int] = []
            means: List[float] = []

            for idx in idxs:
                frame = dset[idx]
                if frame is None:
                    issues.append(f"failed to read frame {idx + 1}")
                    continue
                if frame.dtype != np.uint16:
                    issues.append(f"unexpected dtype {frame.dtype}")
                hashes.add(hash_frame(frame))
                mins.append(int(frame.min()))
                maxs.append(int(frame.max()))
                means.append(float(frame.mean()))

            if len(hashes) == 1 and total > 1:
                issues.append("sampled frames are identical")
            if mins and maxs and max(maxs) == 0:
                issues.append("sampled frames are all zeros")

            if issues:
                stats = f"stats(min={min(mins)}, max={max(maxs)}, mean={sum(means) / len(means):.2f})"
                issues.append(stats)
    except Exception as exc:
        issues.append(f"failed to read: {exc}")
        return False, issues

    return len(issues) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check depth HDF5 files for integrity issues")
    parser.add_argument("root", help="Root directory containing captures")
    parser.add_argument("--samples", type=int, default=5, help="Number of frames to sample per file")
    parser.add_argument("--include-aligned", type=str, default="true", choices=["true", "false"],
                        help="Include depth_aligned.h5 files")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"root not found: {root}")
        return 1
    if h5py is None:
        print("h5py not installed; cannot read depth.h5 files")
        return 1

    include_aligned = args.include_aligned.lower() == "true"
    depth_files = find_depth_files(root, include_aligned)
    if not depth_files:
        print(f"no depth.h5 files found under {root}")
        return 1

    any_fail = False
    for depth_path in depth_files:
        ok, issues = check_depth_file(depth_path, args.samples)
        rel = depth_path.relative_to(root)
        if ok:
            print(f"[OK] {rel}")
        else:
            any_fail = True
            print(f"[FAIL] {rel}")
            for issue in issues:
                print(f"  - {issue}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
