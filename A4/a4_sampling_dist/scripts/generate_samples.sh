#!/usr/bin/env bash
# ==============================================================================
# Script for A4 Part 2: Generate and Score Samples
#
# This script runs the generation and scoring experiment for both the base
# Gemma model and the fine-tuned SFT model.
#
# Anticipated format for the SFT model:
# ../a4_sft_solutions/out/[run_name]/checkpoint-[largest_step]
# ==============================================================================
set -euo pipefail

# Handle user-friendly model selection (default to 1b)
MODEL_CHOICE="${1:-1b}"
OUTPUT_DIR="${2:-results}"

mkdir -p "$OUTPUT_DIR"

# 1. Base Model Experiment
echo "Running experiment for BASE model: $MODEL_CHOICE (Output: $OUTPUT_DIR)..."
python generate_samples.py \
    --model-name-or-path "google/gemma-3-${MODEL_CHOICE}-it" \
    --data-path "data/gsm_symbolic_test_100.jsonl" \
    --output-path "${OUTPUT_DIR}/scored_samples_base.jsonl" \
    --num-samples 8 \
    --batch-size 32

# 2. Fine-tuned Model Experiments

case "$MODEL_CHOICE" in
    "1b")
        MODEL_FILTER="*1b-it*"
        ;;
    "4b")
        MODEL_FILTER="*4b-it*"
        ;;
    "all")
        MODEL_FILTER="*"
        ;;
    *)
        # If user provides a custom pattern or full name, use it directly
        MODEL_FILTER="$MODEL_CHOICE"
        ;;
esac

echo "Filtering SFT models using pattern: $MODEL_FILTER"

SFT_OUT_DIR="./sft_out"

if [[ -d "$SFT_OUT_DIR" ]]; then
    shopt -s nullglob
    for run_dir in "$SFT_OUT_DIR"/${MODEL_FILTER}; do
        if [[ -d "$run_dir" ]]; then
            run_name=$(basename "$run_dir")
            # Find the largest checkpoint in this run directory
            SFT_MODEL_PATH=$(find "$run_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
            
            if [[ -n "$SFT_MODEL_PATH" ]]; then
                echo "Found fine-tuned model for run '$run_name' at: $SFT_MODEL_PATH"
                echo "Running experiment for model: $run_name (Output: $OUTPUT_DIR)"
                python generate_samples.py \
                    --model-name-or-path "$SFT_MODEL_PATH" \
                    --data-path "data/gsm_symbolic_test_100.jsonl" \
                    --output-path "${OUTPUT_DIR}/scored_samples_${run_name}.jsonl" \
                    --num-samples 8 \
                    --batch-size 16
            fi
        fi
    done
else
    echo "Warning: SFT output directory $SFT_OUT_DIR not found. Skipping SFT models."
fi
