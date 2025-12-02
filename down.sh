#!/usr/bin/env bash
set -euo pipefail

# 目标目录，可通过第一个参数自定义：./download_nvidia_toolkit.sh /path/to/dir
TARGET_DIR="${1:-nvidia-offline}"
BASE_URL="https://mirror.cs.uchicago.edu/nvidia-docker/libnvidia-container/stable/ubuntu20.04/amd64"

FILES=(
  "libnvidia-container1_1.13.5-1_amd64.deb"
  "libnvidia-container-tools_1.13.5-1_amd64.deb"
  "libnvidia-container-dev_1.13.5-1_amd64.deb"
  "nvidia-container-toolkit-base_1.13.5-1_amd64.deb"
  "nvidia-container-toolkit_1.13.5-1_amd64.deb"
  "nvidia-container-runtime_3.13.0-1_all.deb"
  "nvidia-docker2_2.13.0-1_all.deb"
)

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading NVIDIA container toolkit .deb files into: $(pwd)"
for f in "${FILES[@]}"; do
  echo "==> $f"
  wget -c "${BASE_URL}/${f}"
done

echo "All files downloaded."
echo "You can install them with something like:"
echo "  cd $(pwd)"
echo "  sudo dpkg -i libnvidia-container1_1.13.5-1_amd64.deb libnvidia-container-tools_1.13.5-1_amd64.deb"
echo "  sudo dpkg -i libnvidia-container-dev_1.13.5-1_amd64.deb"
echo "  sudo dpkg -i nvidia-container-toolkit-base_1.13.5-1_amd64.deb nvidia-container-toolkit_1.13.5-1_amd64.deb"
echo "  sudo dpkg -i nvidia-container-runtime_3.13.0-1_all.deb nvidia-docker2_2.13.0-1_all.deb"
echo "  sudo apt --fix-broken install -y"
