#!/usr/bin/env bash
set -euo pipefail
# === Fix for Anaconda/Conda Interference ===
# Temporarily remove Anaconda/Miniconda paths from PATH to prevent CMake 
# from picking up incompatible libraries (HDF5, OpenSSL, etc.)
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "anaconda" | grep -v "miniconda" | tr '\n' ':' | sed 's/:$//')
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
# ===========================================
# Root of the repository
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$ROOT_DIR/my_project"
BUILD_DIR="$PROJECT_DIR/build_release"

echo "Cleaning previous Release build directory: $BUILD_DIR"
rm -rf "$BUILD_DIR"

echo "Configuring CMake (Release)..."
# 使用系统编译器而不是 conda 环境的编译器
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++

echo "Building (Release)..."
cmake --build "$BUILD_DIR" -j --target DataCollectorApp

echo
echo "Release build completed."
echo "Binaries are in: $BUILD_DIR"

