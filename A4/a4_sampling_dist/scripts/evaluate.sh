#!/usr/bin/env bash
# Step 3: Evaluate accuracy locally
set -euo pipefail

# Path to the results directory
MODEL_CHOICE="${1:-all}"
RESULTS_DIR="${2:-results/sampling}"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: $RESULTS_DIR directory not found. Did you run get_results.sh?"
    exit 1
fi

# Map friendly names to patterns for finding files
case "$MODEL_CHOICE" in
    "1b")
        pattern="*1b-it*"
        ;;
    "4b")
        pattern="*4b-it*"
        ;;
    "all")
        pattern="*"
        ;;
    *)
        pattern="*$MODEL_CHOICE*"
        ;;
esac

# Iterate over all scored sample files matching the pattern
# We always include 'base' results if they exist, plus the filtered ones.
shopt -s nullglob
files=("$RESULTS_DIR"/scored_samples_base.jsonl "$RESULTS_DIR"/scored_samples_${pattern}.jsonl)

# Keep track of processed files to avoid double-counting (e.g. when pattern is '*')
processed_files=""

# Use a check to avoid "unbound variable" errors with empty arrays in older Bash
for samples_path in ${files[@]+"${files[@]}"}; do
    # Skip if already processed
    if [[ "$processed_files" == *"$samples_path"* ]]; then
        continue
    fi
    processed_files="$processed_files $samples_path"

    if [ -f "$samples_path" ]; then
        # Extract model name from filename for the output directory
        filename=$(basename "$samples_path" .jsonl)
        model_name=${filename#scored_samples_}
        output_dir="$RESULTS_DIR/eval_$model_name"
        
        echo "=========================================="
        echo "Evaluating model: $model_name"
        echo "File: $samples_path"
        echo "Output: $output_dir"
        echo "=========================================="
        
        python3 evaluate.py \
            --samples-path "$samples_path" \
            --output-dir "$output_dir"
        echo ""
    fi
done
