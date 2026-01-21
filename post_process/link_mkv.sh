#!/bin/bash

# 检查是否安装了 jq
if ! command -v jq &> /dev/null; then
    echo "错误: 未找到 'jq' 工具。请先安装: sudo apt install jq"
    exit 1
fi

# 输入参数处理
SEARCH_PATH=$1
OUTPUT_PATH=${2:-"."} # 如果未提供第二个参数，默认为当前目录

# 检查搜索路径是否存在
if [ ! -d "$SEARCH_PATH" ]; then
    echo "错误: 搜索路径 '$SEARCH_PATH' 不存在。"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_PATH"

# 转换输出路径为绝对路径，防止软连接失效
ABS_OUTPUT_PATH=$(realpath "$OUTPUT_PATH")

echo "开始处理路径: $SEARCH_PATH"
echo "输出目录: $ABS_OUTPUT_PATH"
echo "--------------------------------"

# 遍历所有的 sess_ 文件夹
find "$SEARCH_PATH" -type d -name "sess_*" | while read -r sess_dir; do
    
    meta_json="$sess_dir/meta.json"
    mkv_src="$sess_dir/RealSense_151322070562/rgb.mkv"
    sess_name=$(basename "$sess_dir")

    # 1. 检查必要文件是否存在
    if [[ ! -f "$meta_json" ]]; then
        echo "[跳过] 找不到 meta.json: $sess_name"
        continue
    fi

    if [[ ! -f "$mkv_src" ]]; then
        echo "[跳过] 找不到指定的 mkv 文件: $sess_name/RealSense_151322070562/rgb.mkv"
        continue
    fi

    # 2. 从 meta.json 提取信息
    # 提取 subject
    subject=$(jq -r '.subject' "$meta_json")
    
    # 提取 task_template_path 的最后一个词语并去掉 .json 后缀
    task_template_path=$(jq -r '.task_template_path' "$meta_json")
    task_name=$(basename "$task_template_path" .json)

    # 3. 构造新的文件名: subject-task_name-sess_XXXX.mkv
    dst_filename="${subject}-${task_name}-${sess_name}.mkv"
    
    # 4. 创建软连接 (使用绝对路径保证连接有效)
    ln -sf "$(realpath "$mkv_src")" "$ABS_OUTPUT_PATH/$dst_filename"

    echo "[成功] 已链接: $dst_filename"
done

echo "--------------------------------"
echo "任务完成！"