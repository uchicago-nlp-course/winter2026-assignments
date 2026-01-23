#!/usr/bin/env bash
set -euo pipefail

# Runners prefer execution from a3_prompting_dist/
# export CUDA_VISIBLE_DEVICES="1"

BASE_SEED=1
NUM_SELECTIONS=3  # 1 / 3
SHOTS=(
  1
  2
  4
  8
)
SHOTS_CSV=$(IFS=,; echo "${SHOTS[*]}")
LOG_DIR="./logs/gsm"
mkdir -p "$LOG_DIR"

# 0-shot prepass (seeds don't matter; run once per method)
TS=$(date +%Y%m%d_%H%M%S)
python3 prompting_gsm.py \
  --methods direct \
  --shots 0 \
  --base-seed "$BASE_SEED" \
  --num-selections 1 \
  --max-new-tokens 4 \
  --results-dir "./results/gsm/$TS" \
  2>&1 | tee -a "$LOG_DIR/${TS}.log"

TS=$(date +%Y%m%d_%H%M%S)
python3 prompting_gsm.py \
  --methods cot \
  --shots 0 \
  --base-seed "$BASE_SEED" \
  --num-selections 1 \
  --max-new-tokens 512 \
  --results-dir "./results/gsm/$TS" \
  2>&1 | tee -a "$LOG_DIR/${TS}.log"


# Few-shot progressive sweep
TS=$(date +%Y%m%d_%H%M%S)
python3 prompting_gsm.py \
  --methods direct \
  --shots "$SHOTS_CSV" \
  --base-seed "$BASE_SEED" \
  --num-selections "$NUM_SELECTIONS" \
  --max-new-tokens 4 \
  --results-dir "./results/gsm/$TS" \
  2>&1 | tee -a "$LOG_DIR/${TS}.log"

# # few-shot cot is not required; feel free to explore on your own
# TS=$(date +%Y%m%d_%H%M%S)
# python3 prompting_gsm.py \
#   --methods cot \
#   --shots "$SHOTS_CSV" \
#   --base-seed "$BASE_SEED" \
#   --num-selections "$NUM_SELECTIONS" \
#   --max-new-tokens 512 \
#   --results-dir "./results/gsm/$TS" \
#   2>&1 | tee -a "$LOG_DIR/${TS}.log"