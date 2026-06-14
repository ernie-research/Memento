#!/bin/bash
# Demo training with fake astronaut data (single-sequence sanity check)
set -euo pipefail
cd "$(dirname "$0")"

export DATA_PATH=./train_data/astronaut.json
export NGPUS="${NGPUS:-8}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/demo_train}"
export MAX_STEPS="${MAX_STEPS:-100}"

bash run_train.sh
