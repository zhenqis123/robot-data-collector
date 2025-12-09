#!/usr/bin/env python3
"""
Depth visualization utility: colorize depth PNGs.

Supports:
- single or multiple depth files
- folders (recursively find *.png)

Usage examples:
  python -m tools.depth_viz depth1.png depth2.png
  python -m tools.depth_viz /path/to/depth_dir
  python -m tools.depth_viz /path/to/depth_dir --max 4000 --colormap turbo
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np


def collect_depth_paths(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for inp in inputs:
        p = Path(inp).expanduser().resolve()
        if p.is_dir():
            for f in p.rglob("*.png"):
                paths.append(f)
        elif p.is_file():
            paths.append(p)
    return paths


def colorize_depth(depth_path: Path, out_dir: Path, vmax: float, cmap: str) -> None:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        return
    depth_f = depth.astype(np.float32)
    if vmax <= 0:
        vmax = float(np.percentile(depth_f[depth_f > 0], 99.0)) if np.any(depth_f > 0) else 1000.0
    norm = np.clip(depth_f / vmax, 0, 1)
    norm = (norm * 255).astype(np.uint8)
    cmap_map = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS if hasattr(cv2, "COLORMAP_VIRIDIS") else cv2.COLORMAP_JET,
        "plasma": cv2.COLORMAP_PLASMA if hasattr(cv2, "COLORMAP_PLASMA") else cv2.COLORMAP_JET,
    }
    colored = cv2.applyColorMap(norm, cmap_map.get(cmap, cv2.COLORMAP_JET))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (depth_path.stem + "_viz.png")
    cv2.imwrite(str(out_path), colored)


def main():
    parser = argparse.ArgumentParser(description="Colorize depth PNGs")
    parser.add_argument("inputs", nargs="+", help="Depth file(s) or directory(ies)")
    parser.add_argument("--max", type=float, default=0.0, help="Depth max (same units as input). 0=auto (99th percentile).")
    parser.add_argument("--colormap", default="jet", choices=["jet", "turbo", "viridis", "plasma"], help="OpenCV colormap")
    parser.add_argument("--output", default=None, help="Output directory (default: alongside input)")
    args = parser.parse_args()

    paths = collect_depth_paths(args.inputs)
    if not paths:
        print("No depth PNGs found")
        return

    for p in paths:
        out_dir = Path(args.output).expanduser().resolve() if args.output else p.parent
        colorize_depth(p, out_dir, args.max, args.colormap)


if __name__ == "__main__":
    main()
