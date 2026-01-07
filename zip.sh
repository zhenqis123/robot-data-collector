#!/bin/bash

# --- 配置区域 ---
# 备份的文件名（包含时间戳）
BACKUP_NAME="data_collector_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
# 备份存放的路径（建议放在 disk1 下的其他地方，或者当前目录）
BACKUP_DIR="."
# 需要打包的源目录（当前目录）
SOURCE_DIR="."

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

echo "🚀 开始备份进程..."
echo "📂 目标文件: $BACKUP_DIR/$BACKUP_NAME"

# 1. 构造排除参数
EXCLUDE_ARGS=""
for item in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$item"
done

# 2. 检查磁盘空间 (预估需要 50GB)
FREE_SPACE=$(df -m /disk1 | awk 'NR==2 {print $4}')
if [ "$FREE_SPACE" -lt 51200 ]; then
    echo "❌ 错误: /disk1 剩余空间不足 50GB，备份可能失败！"
    exit 1
fi

# 3. 执行打包
# 使用 --totals 可以最后显示总大小
# 如果你的系统有 pigz，可以把 -czvf 换成 --use-compress-program=pigz -cvf 以加速
echo "📦 正在打包 (由于 third_party 较大，请耐心等待)..."
tar $EXCLUDE_ARGS -czvf "$BACKUP_DIR/$BACKUP_NAME" "$SOURCE_DIR" --totals

# 4. 完成后检查
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✅ 备份成功！"
    echo "📊 文件大小: $(du -h $BACKUP_DIR/$BACKUP_NAME | awk '{print $1}')"
    echo "🔗 路径: $BACKUP_DIR/$BACKUP_NAME"
    echo "------------------------------------------------"
else
    echo "❌ 备份过程中出现错误。"
fi