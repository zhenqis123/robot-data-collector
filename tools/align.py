#!/usr/bin/env python3
"""
Depth-to-color alignment using saved meta.json and PNG frames.

Usage:
  python -m tools.align /path/to/captures_root --workers 4

Searches recursively for meta.json files and, for each camera:
- reads intrinsics/extrinsics/depth scale
- aligns depth/*.png to color resolution
- writes to depth_aligned/*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
from functools import lru_cache


def find_meta_files(root: Path) -> List[Path]:
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
    """Normalize camera metadata into numeric arrays."""
    return {
        "depth_scale": cam["alignment"].get("depth_scale_m", 0.0),
        "R": np.array(cam["alignment"]["depth_to_color"]["rotation"], dtype=np.float32).reshape(3, 3),
        "t": np.array(cam["alignment"]["depth_to_color"]["translation"], dtype=np.float32).reshape(3, 1),
        "Kc": intrinsics_from_json(cam["streams"]["color"]["intrinsics"]),
        "Kd": intrinsics_from_json(cam["streams"]["depth"]["intrinsics"]),
    }


@lru_cache(maxsize=32)
def precompute_map(depth_shape: tuple, Kd_key: tuple, Kc_key: tuple, R_key: tuple, t_key: tuple):
    h_d, w_d = depth_shape
    fx_d, fy_d, cx_d, cy_d = Kd_key
    fx_c, fy_c, cx_c, cy_c, w_c, h_c = Kc_key
    R = np.array(R_key, dtype=np.float32).reshape(3, 3)
    t = np.array(t_key, dtype=np.float32).reshape(3, 1)

    ys, xs = np.meshgrid(np.arange(h_d, dtype=np.float32), np.arange(w_d, dtype=np.float32), indexing="ij")
    ones = np.ones_like(xs, dtype=np.float32)
    pixels = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T  # (3, N) with z=1 placeholder

    # 3D direction vectors (unit depth)
    X = (pixels[0] - cx_d) / fx_d
    Y = (pixels[1] - cy_d) / fy_d
    Z = np.ones_like(X)
    dirs = np.stack([X, Y, Z], axis=0)  # (3, N)

    dirs_c = R.dot(dirs) + t  # (3,N)
    Xc, Yc, Zc = dirs_c
    u = (Xc / (Zc + 1e-6)) * fx_c + cx_c
    v = (Yc / (Zc + 1e-6)) * fy_c + cy_c
    u = u.reshape(h_d, w_d)
    v = v.reshape(h_d, w_d)

    # Integer target coords and validity mask (independent of depth values)
    u_i = np.round(u).astype(np.int32)
    v_i = np.round(v).astype(np.int32)
    mask = (Zc.reshape(h_d, w_d) > 0) & (u_i >= 0) & (u_i < w_c) & (v_i >= 0) & (v_i < h_c)
    return u_i, v_i, mask


def compute_alignment(depth_raw: np.ndarray, meta: Dict) -> np.ndarray:
    scale = float(meta["depth_scale"])
    if scale <= 0:
        raise ValueError("depth_scale_m must be > 0")

    h_d, w_d = depth_raw.shape
    Kd = meta["Kd"]
    Kc = meta["Kc"]
    u_i, v_i, mask = precompute_map(
        (h_d, w_d),
        (Kd["fx"], Kd["fy"], Kd["cx"], Kd["cy"]),
        (Kc["fx"], Kc["fy"], Kc["cx"], Kc["cy"], Kc["w"], Kc["h"]),
        tuple(meta["R"].flatten()),
        tuple(meta["t"].flatten()),
    )

    depth_valid = depth_raw.astype(np.float32) * scale  # meters
    z = depth_valid[mask]
    if z.size == 0:
        return np.zeros((Kc["h"], Kc["w"]), dtype=np.uint16)

    u_flat = u_i[mask].ravel()
    v_flat = v_i[mask].ravel()
    Hc, Wc = int(Kc["h"]), int(Kc["w"])
    target_idx = v_flat * Wc + u_flat

    # Start with max uint16; will take min per pixel
    out_flat = np.full(Hc * Wc, np.iinfo(np.uint16).max, dtype=np.uint16)
    z_mm = (z / scale).astype(np.uint16)
    np.minimum.at(out_flat, target_idx, z_mm)
    out_flat[out_flat == np.iinfo(np.uint16).max] = 0
    return out_flat.reshape(Hc, Wc)


def sanitize_camera_id(cam_id: str) -> str:
    """Match writer sanitize behavior (non-alnum -> underscore)."""
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in cam_id)


def _process_one(path: Path, color_dir: Path, out_dir: Path, meta: Dict) -> None:
    stem = path.stem
    color_path = color_dir / f"{stem}.png"
    if not color_path.exists():
        return
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None or depth_raw.dtype != np.uint16:
        return
    aligned = compute_alignment(depth_raw, meta)
    cv2.imwrite(str(out_dir / f"{stem}.png"), aligned)


def process_capture(meta_path: Path, workers: int) -> None:
    meta = load_meta(meta_path)
    cam_entries = {c["id"]: c for c in meta.get("cameras", [])}
    base = meta_path.parent
    for cam_id, cam in cam_entries.items():
        cam_dir = base / sanitize_camera_id(str(cam_id))
        depth_dir = cam_dir / "depth"
        color_dir = cam_dir / "color"
        if not depth_dir.exists() or not color_dir.exists():
            continue
        out_dir = cam_dir / "depth_aligned"
        out_dir.mkdir(parents=True, exist_ok=True)
        m = prepare_meta(cam)
        depth_files = sorted(depth_dir.glob("*.png"))
        if not depth_files:
            continue
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in tqdm(
                ex.map(
                    _process_one,
                    depth_files,
                    [color_dir] * len(depth_files),
                    [out_dir] * len(depth_files),
                    [m] * len(depth_files),
                ),
                total=len(depth_files),
                desc=f"align {cam_id}",
                unit="frame",
                leave=False,
            ):
                pass
        print(f"[align] processed camera {cam_id} in {base}")


def main():
    parser = argparse.ArgumentParser(description="Offline depth-to-color alignment from PNGs and meta.json")
    parser.add_argument("root", help="Root directory containing captures/meta.json")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads for alignment")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    metas = find_meta_files(root)
    if not metas:
        print("No meta.json found")
        return
    for m in metas:
        process_capture(m, max(1, args.workers))


if __name__ == "__main__":
    main()
