#!/usr/bin/env bash
# Script to run the reasoning trace analysis for both base and fine-tuned models.
# Usage: bash scripts/analyze_traces.sh [path_to_samples.jsonl]

set -euo pipefail

SAMPLES_PATH="${1:-}"
PLOTS_DIR="plots"
RESULTS_DIR="results/sampling"

mkdir -p "$PLOTS_DIR"

if [ -n "$SAMPLES_PATH" ]; then
    # Run analysis on the provided file
    echo "Running analysis on: $SAMPLES_PATH"
    filename=$(basename "$SAMPLES_PATH" .jsonl)
    suffix=${filename#scored_samples_}
    
    python3 analyze_traces.py \
        --samples-path "$SAMPLES_PATH" \
        --plots-dir "$PLOTS_DIR" \
        --plot-suffix "$suffix"
else
    # Default: run on all scored_samples files in results/sampling
    shopt -s nullglob
    for samples_file in "$RESULTS_DIR"/scored_samples_*.jsonl; do
        filename=$(basename "$samples_file" .jsonl)
        suffix=${filename#scored_samples_}
        
        echo "--------------------------------------------------"
        echo "Analyzing model: $suffix"
        echo "File: $samples_file"
        
        python3 analyze_traces.py \
            --samples-path "$samples_file" \
            --plots-dir "$PLOTS_DIR" \
            --plot-suffix "$suffix" \
            --failure-modes-path "results/failure_modes_$suffix.txt"
    done
fi

echo "Analysis complete. Results in $PLOTS_DIR/"
