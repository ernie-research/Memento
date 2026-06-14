#!/bin/bash
# Single story inference script.
# Usage: bash run_inference.sh <story_json> <output_dir> <lora_weight_path>

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1

NGPUS="${NGPUS:-8}"
PORT="${PORT:-8200}"
T2V_MODEL_PATH="${T2V_MODEL_PATH:-./models/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-./models/Wan2.2-I2V-A14B}"
VIDEO_SIZE="${VIDEO_SIZE:-"832*480"}"
LORA_RANK="${LORA_RANK:-128}"

STORY_JSON="${1:?Usage: bash run_inference.sh <story_json> <output_dir> <lora_weight_path>}"
OUTPUT_DIR="${2:?Usage: bash run_inference.sh <story_json> <output_dir> <lora_weight_path>}"
LORA_WEIGHT_PATH="${3:?Usage: bash run_inference.sh <story_json> <output_dir> <lora_weight_path>}"

mkdir -p "$OUTPUT_DIR"

torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=$PORT \
    pipeline_learnable_acce.py \
    --story_script_path "$STORY_JSON" \
    --t2v_model_path "$T2V_MODEL_PATH" \
    --i2v_model_path "$I2V_MODEL_PATH" \
    --lora_weight_path "$LORA_WEIGHT_PATH" \
    --size "$VIDEO_SIZE" \
    --max_memory_size 8 \
    --max_memory_frames 8 \
    --output_dir "$OUTPUT_DIR" \
    --ulysses_size 8 \
    --offload_model \
    --lora_rank $LORA_RANK \
    --mi2v \
    --t2v_first_shot \
    --t5_fsdp \
    --dit_fsdp \
    --split_identity_attn \
    --split_learnable_query \
    --global_query_num 2 \
    --use_subject_recon \
    --use_both_query \
    --compile_dit
