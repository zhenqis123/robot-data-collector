#!/usr/bin/env bash
set -euo pipefail

# Root of the repository
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$ROOT_DIR/my_project"
BUILD_DIR="$PROJECT_DIR/build_release"

echo "Cleaning previous Release build directory: $BUILD_DIR"
rm -rf "$BUILD_DIR"

echo "Configuring CMake (Release)..."
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "Building (Release)..."
cmake --build "$BUILD_DIR" -j

echo
echo "Release build completed."
echo "Binaries are in: $BUILD_DIR"

