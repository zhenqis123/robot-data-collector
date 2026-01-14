#!/usr/bin/env zsh
# ======================================================
#  Manus ROS2: Run Data Publisher + Visualization
# ======================================================

echo "🔧 Deactivating conda environment (if any)..."
if type conda &>/dev/null; then
    conda deactivate 2>/dev/null || true
fi

echo "🧩 Resetting library paths..."
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu

echo "🚀 Loading ROS2 Humble environment..."
source /opt/ros/humble/setup.zsh

echo "📦 Loading Manus ROS2 workspace..."
source ~/MANUS_Core_3.0.1_SDK/ros2_ws/install/setup.zsh

# ==============================
# 启动 Manus Data Publisher
# ==============================
echo "🦾 Launching Manus Data Publisher..."
gnome-terminal -- bash -c "
    echo '🦾 Running manus_data_publisher...';
    ros2 run manus_ros2 manus_data_publisher;
    exec bash
"

# ==============================
# 启动 Manus Data Visualization
# ==============================
echo "🎨 Launching Manus Data Visualization..."
gnome-terminal -- bash -c "
    echo '🎨 Running manus_data_viz.py...';
    /usr/bin/python3 ~/MANUS_Core_3.0.1_SDK/ros2_ws/src/manus_ros2/client_scripts/manus_data_viz.py;
    exec bash
"

echo "✅ Both processes started successfully!"

