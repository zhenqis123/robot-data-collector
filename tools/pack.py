#!/usr/bin/env python3
"""
Pack a capture directory into a tar.gz placed alongside it.

Usage:
  python -m tools.pack /path/to/capture
"""
from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path
from typing import List


def pack_capture(capture_root: Path) -> Path:
    archive_path = capture_root / f"{capture_root.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(capture_root, arcname=capture_root.name)
    print(f"[pack] wrote {archive_path}")
    return archive_path


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


def main():
    parser = argparse.ArgumentParser(description="Pack a capture directory into tar.gz")
    parser.add_argument("root", help="Root directory containing capture folders or meta.json")
    parser.add_argument("--find-meta", type=str, default="true",
                        choices=["true", "false"],
                        help="Search meta.json recursively (true) or only root/*/meta.json (false)")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    find_meta = args.find_meta.lower() == "true"
    metas = list_meta_files(root, find_meta, 2)
    if not metas:
        print("No meta.json found")
        return
    for meta in metas:
        pack_capture(meta.parent)


if __name__ == "__main__":
    main()
