#!/usr/bin/env bash
set -euo pipefail

# Runner assumes execution from a4_sft_dist/
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/huggingface/hub}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-$HOME/.cache/huggingface/datasets}"

export CUDA_VISIBLE_DEVICES="0"

# ==============================================================================
# Parameter arrays (one entry per line for easy commenting)
# ==============================================================================

# Evaluate either:
# - Base model: "google/gemma-3-1b-it"
# - A saved checkpoint dir: "./out/<run_name>/checkpoint-XXXX"
model_name_or_path_list=(
    ## 1. Base models
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"


    ## 2. sft checkpoints: trained on gsm-symbolic
    "./out/gsm_symbolic_train_4500_student-gemma-3-1b-it-maxlen1024-epochs1-lr2e-05-effbsz8-dataseed0/checkpoint-563"
    "./out/gsm_symbolic_train_4500_student-gemma-3-4b-it-maxlen1024-epochs1-lr2e-05-effbsz8-dataseed0/checkpoint-563"
)

eval_file_list=(
    ## 1. gsm-symbolic
    "./data/gsm_symbolic_test_100_student.jsonl"

    ## 2. gsm8k
    "./data/gsm8k_test_100_same_templates.jsonl"
    "./data/gsm8k_test_100_non_overlapping.jsonl"
)

max_new_tokens_list=(
    # 512
    1024
    # 2048
)

seed_list=(
    0
    # 42
)

batch_size_list=(
    # 1
    # 8
    16
)

torch_dtype_list=(
    # bfloat16
    float32
)

# ==============================================================================
# Main evaluation loop
# ==============================================================================
for model_name_or_path in "${model_name_or_path_list[@]}"; do
for eval_file in "${eval_file_list[@]}"; do
for max_new_tokens in "${max_new_tokens_list[@]}"; do
for batch_size in "${batch_size_list[@]}"; do
for torch_dtype in "${torch_dtype_list[@]}"; do
for seed in "${seed_list[@]}"; do

    # Extract model name from path
    # For checkpoints: extract parent directory name (the run_name)
    # For HF models: extract model id after the last "/"
    if [[ "$model_name_or_path" == *"/checkpoint-"* ]]; then
        model_name=$(echo "$model_name_or_path" | awk -F'/' '{print $(NF-1)}')
    else
        model_name=$(echo "$model_name_or_path" | awk -F'/' '{print $NF}')
    fi

    # Extract dataset name from eval_file (filename without extension)
    dataset_name=$(basename "$eval_file" .jsonl)

    # Build descriptive output directory
    TS=$(date +%Y%m%d_%H%M%S)
    output_dir="./eval_results/${dataset_name}/${model_name}/${TS}-max_tokens${max_new_tokens}-bsz${batch_size}-${torch_dtype}-seed${seed}"

    mkdir -p "$output_dir"

    echo "=========================================="
    echo "Starting SFT evaluation"
    echo "  Model: $model_name"
    echo "  Eval file: $dataset_name"
    echo "  Max new tokens: $max_new_tokens"
    echo "  Batch size: $batch_size"
    echo "  Torch dtype: $torch_dtype"
    echo "  Seed: $seed"
    echo "  Output: $output_dir"
    echo "=========================================="

    python3 -m eval \
        --model_name_or_path "$model_name_or_path" \
        --eval_file "$eval_file" \
        --output_dir "$output_dir" \
        --max_new_tokens "$max_new_tokens" \
        --batch_size "$batch_size" \
        --seed "$seed" \
        --torch_dtype "$torch_dtype" \
    2>&1 | tee "$output_dir/sft_eval.log"

    echo "Completed evaluation for: $model_name on $dataset_name"
    echo "----------------------------------------"

done
done
done
done
done
done
