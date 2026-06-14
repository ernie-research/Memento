#!/bin/bash
# Batch inference script with resume support.
# Usage: bash run_inference_batch.sh <story_dir> <output_dir> <lora_weight_path>

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1

NGPUS="${NGPUS:-8}"
BASE_PORT="${BASE_PORT:-8200}"
T2V_MODEL_PATH="${T2V_MODEL_PATH:-./models/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-./models/Wan2.2-I2V-A14B}"
VIDEO_SIZE="${VIDEO_SIZE:-"832*480"}"
LORA_RANK="${LORA_RANK:-128}"

STORY_DIR="${1:?Usage: bash run_inference_batch.sh <story_dir> <output_dir> <lora_weight_path>}"
OUTPUT_BASE_DIR="${2:?Usage: bash run_inference_batch.sh <story_dir> <output_dir> <lora_weight_path>}"
LORA_WEIGHT_PATH="${3:?Usage: bash run_inference_batch.sh <story_dir> <output_dir> <lora_weight_path>}"

STORY_FILES=("$STORY_DIR"/*.json)
TOTAL=${#STORY_FILES[@]}

if [ $TOTAL -eq 0 ]; then
    echo "[ERROR] No json files found in $STORY_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_BASE_DIR"

echo "=========================================="
echo "Batch inference: $TOTAL stories"
echo "Story dir : $STORY_DIR"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "=========================================="

is_story_complete() {
    local output_dir="$1" story_name="$2"
    [ -f "$output_dir/${story_name}.mp4" ]
}

cleanup_gpu() {
    local port="$1"
    sleep 2
    local pids
    pids=$(ps aux 2>/dev/null | grep "[m]aster_port=$port" | awk '{print $2}' || true)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

SUCCESS=0; FAIL=0; SKIP=0; FAIL_LIST=()

for (( i=0; i<TOTAL; i++ )); do
    STORY_FILE="${STORY_FILES[$i]}"
    STORY_NAME=$(basename "$STORY_FILE" .json)
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$STORY_NAME"
    PORT=$((BASE_PORT + (i % 100)))
    LOG_FILE="$OUTPUT_BASE_DIR/${STORY_NAME}.log"
    INDEX=$((i + 1))

    echo ""
    echo "[${INDEX}/${TOTAL}] Story: $STORY_NAME"

    if is_story_complete "$OUTPUT_DIR" "$STORY_NAME"; then
        echo "  SKIP (already complete)"
        SKIP=$((SKIP + 1)); SUCCESS=$((SUCCESS + 1))
        continue
    fi

    # Clean stale files from previous failed run
    if [ -d "$OUTPUT_DIR" ]; then
        find "$OUTPUT_DIR" -mindepth 1 -delete 2>/dev/null || true
    fi
    mkdir -p "$OUTPUT_DIR"

    EXIT_CODE=0
    torchrun \
        --nproc_per_node=$NGPUS \
        --master_port=$PORT \
        pipeline_learnable_acce.py \
        --story_script_path "$STORY_FILE" \
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
        --compile_dit \
        > "$LOG_FILE" 2>&1 || EXIT_CODE=$?

    cleanup_gpu "$PORT"

    if [ $EXIT_CODE -eq 0 ] && is_story_complete "$OUTPUT_DIR" "$STORY_NAME"; then
        echo "  OK"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  FAILED (exit=$EXIT_CODE, log=$LOG_FILE)"
        FAIL=$((FAIL + 1)); FAIL_LIST+=("$STORY_NAME")
    fi
done

echo ""
echo "=========================================="
echo "Done. Success=$SUCCESS (skip=$SKIP), Failed=$FAIL"
if [ ${#FAIL_LIST[@]} -gt 0 ]; then
    printf '  %s\n' "${FAIL_LIST[@]}"
fi
echo "=========================================="
