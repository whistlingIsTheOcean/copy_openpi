#!/usr/bin/env bash
# 部署验证 · 第 1 步：在推理服务端（这台机器）起 pi0 LoRA 服务
#
#   bash serve_flexiv.sh
#
set -euo pipefail
cd "$(dirname "$0")"

# ======== 可按需修改 ========
export CUDA_VISIBLE_DEVICES=1              # GPU 1/2/3 空闲；GPU 0 被别人占，4-7 在训练
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
PORT=8001                                  # 8000 已被占用，必须换端口
CONFIG=pi0_realdata_lora
CKPT=checkpoints/pi0_realdata_lora/my_exp/29999
PROMPT="Pick up the colored cup and move it into the metal cup"
# ===========================

export OPENPI_DATA_HOME=/data/yuxuan/openpi_cache
export HF_LEROBOT_HOME=/data/yuxuan/lerobot_home
# 注意: 绝对不要设置 LEROBOT_HOME（会 ValueError）

if [[ ! -d "$CKPT" ]]; then
  echo "[错误] 找不到 checkpoint 目录: $CKPT" >&2
  echo "可用的是:" >&2
  ls -1 checkpoints/"$CONFIG"/*/ 2>/dev/null >&2 || true
  exit 1
fi

if ! .venv/bin/python -c "import lerobot" 2>/dev/null; then
  echo "[错误] .venv 里 import 不到 lerobot" >&2
  echo "提示: lerobot 是 editable 本地路径安装，不要移动/删除 copy_openpi/lerobot/" >&2
  exit 1
fi

echo "本机可用 IP（客户端要填其中一个）:"
ip -4 -o addr show scope global | awk '{printf "  %-10s %s\n", $2, $4}'
echo
echo "配置: config=$CONFIG  port=$PORT  gpu=$CUDA_VISIBLE_DEVICES"
echo "权重: $CKPT"
echo "健康检查:  curl http://127.0.0.1:$PORT/healthz"
echo
echo "启动中（首次要加载权重 + JIT，可能等 1-2 分钟）..."
echo

exec .venv/bin/python scripts/serve_policy.py \
  --default-prompt="$PROMPT" \
  --port="$PORT" \
  policy:checkpoint \
  --policy.config="$CONFIG" \
  --policy.dir="$CKPT"
