#!/bin/bash
# Memento M2V Training Script
# Usage: bash run_train.sh
#
# This script trains the LoRA + KeyframeQuery modules on story sequences.
# Requires: 8x NVIDIA GPUs (A100 80GB recommended), ~140GB total VRAM
#
# Key environment variables (override defaults):
#   NGPUS           - Number of GPUs (default: 8)
#   CHECKPOINT_DIR  - Path to Wan2.2-I2V-A14B base model
#   DATA_PATH       - Path to training data JSON
#   OUTPUT_DIR      - Output directory for checkpoints/logs
#   LORA_RANK       - LoRA rank (default: 128)
#   MAX_STEPS       - Max training steps (default: 6000)

set -euo pipefail
cd "$(dirname "$0")"

NGPUS="${NGPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./models/Wan2.2-I2V-A14B}"
DATA_PATH="${DATA_PATH:?Please set DATA_PATH to your training data JSON}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/memento_train}"
LORA_RANK="${LORA_RANK:-128}"
MAX_STEPS="${MAX_STEPS:-6000}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120

torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=$MASTER_PORT \
    train.py \
    --data_path "$DATA_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resolution "480*832" \
    --frame_num 81 \
    --batch_size 1 \
    --lora_rank $LORA_RANK \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --use_gradient_checkpointing_offload \
    --max_memory_frames 8 \
    --skip_prior_shots_prob 0.1 \
    --identity_frame_prob 0.8 \
    --identity_loss_weight 1 \
    --split_identity_attn \
    --split_learnable_query \
    --global_query_num 2 \
    --num_keyframes 12 \
    --selected_local_num 6 \
    --max_steps $MAX_STEPS \
    "$@"
