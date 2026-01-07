#!/bin/bash

# --- 配置区域 ---
PORT="2271"             # 例如 2222
REMOTE_USER="root"           # 远程用户名
REMOTE_HOST="120.92.21.58"        # 远程服务器地址
SOURCE_DIR="./"              # 本地源目录
DEST_DIR="/disk1/xizh/develop/processor"    # 远程目标路径

# 排除列表
EXCLUDES=(
    "index-tts-vllm"
    "data"
    ".git"
    "print_out"
    "outputs"
    "librealsense"
)

# --- 执行逻辑 ---

# 1. 构造排除参数
EXCLUDE_ARGS=""
for item in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$item"
done

echo "🚀 开始远程同步进程..."
echo "📡 目标: ${REMOTE_USER}@${REMOTE_HOST}:${DEST_DIR} (端口: $PORT)"

# 2. 执行 rsync
# -e "ssh -p $PORT": 这是指定端口的关键
# -a: 归档模式
# -v: 详细输出
# -z: 压缩传输（远程传输建议开启，节省带宽）
# --progress: 显示进度
rsync -avz --progress \
    -e "ssh -p $PORT" \
    $EXCLUDE_ARGS \
    "$SOURCE_DIR" \
    "${REMOTE_USER}@${REMOTE_HOST}:${DEST_DIR}"

# 3. 检查结果
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✅ 远程同步成功！"
    echo "------------------------------------------------"
else
    echo "❌ 同步失败，请检查网络、端口或 SSH 密钥配置。"
fi