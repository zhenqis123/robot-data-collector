#!/usr/bin/env python3
"""
Pack a capture directory into a tar.gz placed alongside it.

Usage:
  python -m tools.pack /path/to/capture
"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def pack_capture(capture_root: Path) -> Path:
    archive_path = capture_root / f"{capture_root.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(capture_root, arcname=capture_root.name)
    print(f"[pack] wrote {archive_path}")
    return archive_path


def main():
    parser = argparse.ArgumentParser(description="Pack a capture directory into tar.gz")
    parser.add_argument("capture", help="Capture root containing meta.json")
    args = parser.parse_args()
    pack_capture(Path(args.capture).expanduser().resolve())


if __name__ == "__main__":
    main()
