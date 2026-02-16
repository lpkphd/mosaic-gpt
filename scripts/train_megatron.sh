#!/usr/bin/env bash
# MOSAIC-GPT training launcher for Megatron server (2x RTX 3090)
#
# Usage:
#   # MoE model (349M total, 236M active) — recommended first run
#   bash scripts/train_megatron.sh configs/mosaic_small_3090.yaml --grad-accum 32
#
#   # Dense model (122M, GPT-2 scale baseline)
#   bash scripts/train_megatron.sh configs/mosaic_dense_3090.yaml --grad-accum 8
#
#   # With wandb logging
#   bash scripts/train_megatron.sh configs/mosaic_small_3090.yaml --grad-accum 32 --wandb
#
#   # Override max steps
#   bash scripts/train_megatron.sh configs/mosaic_small_3090.yaml --grad-accum 32 --max-steps 10000

set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Environment setup for Megatron server
# ──────────────────────────────────────────────────────────────

# Python environment
VENV="/tmp/spike_v2_env/bin/activate"
if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "ERROR: Virtual environment not found at $VENV"
    echo "Create one with: python -m venv /tmp/spike_v2_env && source $VENV && pip install -e ."
    exit 1
fi

# HuggingFace cache (avoid filling /home)
export HF_HOME=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_cache/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# NCCL settings for 2x 3090 (PCIe, no NVLink)
export NCCL_P2P_DISABLE=1

# PyTorch CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure mosaic-gpt is importable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ──────────────────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────────────────

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [extra args for train.py]"
    echo ""
    echo "Examples:"
    echo "  $0 configs/mosaic_small_3090.yaml --grad-accum 32"
    echo "  $0 configs/mosaic_dense_3090.yaml --grad-accum 8 --wandb"
    exit 1
fi

CONFIG="$1"
shift  # Remaining args passed to train.py

if [ ! -f "$CONFIG" ]; then
    # Try relative to project dir
    CONFIG="$PROJECT_DIR/$CONFIG"
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: Config file not found: $1"
        exit 1
    fi
fi

# Derive run name from config filename
CONFIG_NAME="$(basename "$CONFIG" .yaml)"
RUN_DIR="runs/${CONFIG_NAME}"

# ──────────────────────────────────────────────────────────────
# Launch training with DDP (2 GPUs)
# ──────────────────────────────────────────────────────────────

NUM_GPUS=2
echo "============================================="
echo "  MOSAIC-GPT Training — Megatron Server"
echo "============================================="
echo "Config:    $CONFIG"
echo "Run dir:   $RUN_DIR"
echo "GPUs:      $NUM_GPUS"
echo "Extra args: $*"
echo "PYTHONPATH: $PYTHONPATH"
echo "HF_HOME:    $HF_HOME"
echo "============================================="
echo ""

cd "$PROJECT_DIR"

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29500 \
    experiments/train.py \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR" \
    "$@"
