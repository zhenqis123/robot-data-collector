#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import h5py


def check_camera_dir(cam_dir: Path):
    issues = []
    rgb = cam_dir / "rgb.mkv"
    depth = cam_dir / "depth.h5"
    ts = cam_dir / "timestamps.csv"

    if not rgb.exists() or rgb.stat().st_size == 0:
        issues.append(f"missing or empty rgb.mkv: {rgb}")

    if not depth.exists() or depth.stat().st_size == 0:
        issues.append(f"missing or empty depth.h5: {depth}")

    if not ts.exists() or ts.stat().st_size == 0:
        issues.append(f"missing or empty timestamps.csv: {ts}")

    depth_frames = None
    if depth.exists():
        try:
            with h5py.File(depth, "r") as f:
                if "depth" not in f:
                    issues.append(f"missing /depth dataset in {depth}")
                else:
                    depth_frames = f["depth"].shape[0]
        except Exception as e:
            issues.append(f"failed to read {depth}: {e}")

    ts_rows = []
    if ts.exists():
        try:
            with open(ts, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    ts_rows.append(row)
        except Exception as e:
            issues.append(f"failed to read {ts}: {e}")

    bad_rows = [r for r in ts_rows if len(r) != 6]
    if bad_rows:
        issues.append(f"timestamps.csv has {len(bad_rows)} malformed rows (likely truncated)")

    if depth_frames is not None and ts_rows:
        if depth_frames != len(ts_rows):
            issues.append(f"depth frame count {depth_frames} != timestamps rows {len(ts_rows)}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", help="Path to capture directory")
    args = parser.parse_args()

    cap_dir = Path(args.capture_dir)
    if not cap_dir.exists():
        print(f"capture_dir not found: {cap_dir}")
        return 1

    camera_dirs = [p for p in cap_dir.iterdir() if p.is_dir()]
    if not camera_dirs:
        print(f"no camera subdirs found under: {cap_dir}")
        return 1

    any_issue = False
    for cam in camera_dirs:
        issues = check_camera_dir(cam)
        if issues:
            any_issue = True
            print(f"[FAIL] {cam}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"[OK] {cam}")

    return 1 if any_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())
