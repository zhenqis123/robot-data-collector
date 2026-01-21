#!/usr/bin/env python3
"""
Offline depth-to-color alignment using saved meta.json and PNG frames.

Usage:
    python offline_align.py /path/to/captures_root

The script searches the given root recursively for meta.json files, then for
each camera listed in meta.json:
- loads intrinsics/extrinsics/depth scale
- reads depth PNGs from <capture>/<camera>/depth
- aligns them to the color frame size
- writes aligned depth PNGs to <capture>/<camera>/depth_aligned

Requires: numpy, opencv-python
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def find_meta_files(root: Path) -> List[Path]:
    """Find meta.json files recursively."""
    return [p for p in root.rglob("meta.json") if p.is_file()]


def load_meta(meta_path: Path) -> Dict:
    with meta_path.open("r") as f:
        return json.load(f)


def intrinsics_from_json(j: Dict) -> Dict:
    return {
        "fx": j["fx"],
        "fy": j["fy"],
        "cx": j["cx"],
        "cy": j["cy"],
        "w": j["width"],
        "h": j["height"],
    }


def prepare_meta(cam: Dict) -> Dict:
    return {
        "depth_scale": cam["alignment"].get("depth_scale_m", 0.0),
        "R": np.array(cam["alignment"]["depth_to_color"]["rotation"], dtype=np.float32).reshape(3, 3),
        "t": np.array(cam["alignment"]["depth_to_color"]["translation"], dtype=np.float32).reshape(3, 1),
        "Kc": intrinsics_from_json(cam["streams"]["color"]["intrinsics"]),
        "Kd": intrinsics_from_json(cam["streams"]["depth"]["intrinsics"]),
    }


def compute_alignment(depth_raw: np.ndarray, meta: Dict) -> np.ndarray:
    """Align a single depth frame (uint16) to color resolution."""
    scale = float(meta["depth_scale"])
    if scale <= 0:
        raise ValueError("depth_scale_m must be > 0")

    h_d, w_d = depth_raw.shape
    Kd = meta["Kd"]
    Kc = meta["Kc"]
    # Precompute normalized pixel coordinates for depth image
    ys, xs = np.meshgrid(np.arange(h_d, dtype=np.float32), np.arange(w_d, dtype=np.float32), indexing="ij")
    z = depth_raw.astype(np.float32) * scale  # meters
    x = (xs - Kd["cx"]) * z / Kd["fx"]
    y = (ys - Kd["cy"]) * z / Kd["fy"]

    pts = np.stack((x, y, z), axis=-1).reshape(-1, 3).T  # (3, N)
    Pc = meta["R"].dot(pts) + meta["t"]  # (3, N)
    Xc, Yc, Zc = Pc
    eps = 1e-6
    u = (Xc / (Zc + eps)) * Kc["fx"] + Kc["cx"]
    v = (Yc / (Zc + eps)) * Kc["fy"] + Kc["cy"]

    Hc, Wc = int(Kc["h"]), int(Kc["w"])
    aligned = np.zeros((Hc, Wc), dtype=np.uint16)
    valid = (Zc > 0) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
    if not np.any(valid):
        return aligned

    u_i = u[valid].astype(np.int32)
    v_i = v[valid].astype(np.int32)
    z_mm = (Zc[valid] / scale).astype(np.uint16)  # back to depth units (e.g., mm)
    target_idx = v_i * Wc + u_i

    # Keep nearest depth per target pixel
    order = np.argsort(z_mm)
    target_idx = target_idx[order]
    z_mm = z_mm[order]
    seen = np.full(Wc * Hc, False, dtype=bool)
    flat = np.zeros(Wc * Hc, dtype=np.uint16)
    for idx, depth_val in zip(target_idx, z_mm):
        if not seen[idx]:
            flat[idx] = depth_val
            seen[idx] = True
    aligned = flat.reshape(Hc, Wc)
    return aligned


def process_capture(meta_path: Path) -> None:
    meta = load_meta(meta_path)
    cam_entries = {c["id"]: c for c in meta.get("cameras", [])}
    base = meta_path.parent
    for cam_id, cam in cam_entries.items():
        cam_dir = base / cam_id
        depth_dir = cam_dir / "depth"
        color_dir = cam_dir / "color"
        if not depth_dir.exists() or not color_dir.exists():
            continue
        out_dir = cam_dir / "depth_aligned"
        out_dir.mkdir(parents=True, exist_ok=True)
        m = prepare_meta(cam)
        depth_files = sorted(depth_dir.glob("*.png"))
        for depth_path in depth_files:
            stem = depth_path.stem
            color_path = color_dir / f"{stem}.png"
            if not color_path.exists():
                continue
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_raw is None or depth_raw.dtype != np.uint16:
                continue
            aligned = compute_alignment(depth_raw, m)
            cv2.imwrite(str(out_dir / f"{stem}.png"), aligned)
        print(f"[align] processed camera {cam_id} in {base}")


def main():
    parser = argparse.ArgumentParser(description="Offline RealSense depth-to-color alignment from saved PNGs and meta.json")
    parser.add_argument("root", help="Root directory containing captures/meta.json")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")
    metas = find_meta_files(root)
    print(metas)
    if not metas:
        print("No meta.json found")
        return
    for m in metas:
        process_capture(m)


if __name__ == "__main__":
    main()
