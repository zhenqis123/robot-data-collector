#!/usr/bin/env python3
"""
Convert 16-bit depth PNG to a colorized visualization (Jet colormap).

Usage:
  python depth_viz.py --input path/to/depth.png --output depth_viz.png --max-depth-mm 8000

Defaults:
  --output: same as input with _viz suffix
  --max-depth-mm: 8000 (matches preview scaling)
"""

import argparse
from pathlib import Path

import cv2


def colorize_depth(depth_path: Path, output_path: Path, max_depth_mm: int) -> None:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
    if depth.dtype != 'uint16':
        print(f"Warning: expected uint16 depth, got {depth.dtype}")

    alpha = 255.0 / float(max_depth_mm)
    depth8 = cv2.convertScaleAbs(depth, alpha=alpha)
    colored = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
    if not cv2.imwrite(str(output_path), colored):
        raise RuntimeError(f"Failed to write {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Colorize 16-bit depth image")
    parser.add_argument("--input", required=True, help="Path to depth PNG (uint16)")
    parser.add_argument("--output", help="Output path (PNG). Default: input with _viz suffix")
    parser.add_argument("--max-depth-mm", type=int, default=8000, help="Max depth in mm for scaling (default 8000)")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")
    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_viz.png")

    colorize_depth(inp, out, args.max_depth_mm)
    print(f"Saved colorized depth to {out}")


if __name__ == "__main__":
    main()
