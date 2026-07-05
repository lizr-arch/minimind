#!/usr/bin/env bash
# Download project data from Google Drive
# Usage: bash scripts/download_data.sh
set -euo pipefail

cd "$(dirname "$0")/.."
DATA_DIR="data"
FOLDER_ID="1YiWdaUayxTComVv19jknHiChqDqb24WW"

echo "=== Minimind: downloading data ==="
echo "源: Google Drive (minimind-data)"
echo "目标: $DATA_DIR/"
echo ""

# Method 1: gdown (no auth needed for public folders)
if pip show gdown &>/dev/null; then
    echo "使用 gdown 下载..."
    if [ -d "$DATA_DIR" ] && [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        echo "⚠  $DATA_DIR/ 非空，可能已有数据"
    fi
    mkdir -p "$DATA_DIR"
    # 解压 -f 重新下载已存在的文件
    gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" -O "$DATA_DIR" --remaining-ok
    echo "完成！"
    exit 0
fi

# Method 2: rclone (需要配置 Google Drive remote)
if command -v rclone &>/dev/null; then
    if rclone listremotes 2>/dev/null | grep -q "gdrive:"; then
        echo "使用 rclone 下载..."
        rclone copy --progress gdrive:minimind-data/data/ "$DATA_DIR/"
        echo "完成！"
        exit 0
    fi
fi

echo ""
echo "需要安装 gdown 或配置 rclone:"
echo ""
echo "  方式 A (推荐): pip install gdown"
echo "  方式 B: rclone config (添加 gdrive remote)"
echo ""
echo "然后重新运行此脚本。"
