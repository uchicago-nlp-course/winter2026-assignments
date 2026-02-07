#!/usr/bin/env bash
set -euo pipefail

# Runner assumes execution from a4_sft_dist/
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/huggingface/hub}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-$HOME/.cache/huggingface/datasets}"

# ==============================================================================
# Checkpoint cleanup function
# ==============================================================================
cleanup_checkpoints_after_run() {
    local OUTDIR="$1"
    echo ""
    echo "[CLEANUP] Starting checkpoint cleanup in: $OUTDIR"

    if [[ ! -d "$OUTDIR" ]]; then
        echo "[CLEANUP] Skipping: directory does not exist: $OUTDIR"
        return 0
    fi

    # 1) Delete .safetensors directly under OUTDIR
    echo "[CLEANUP] Removing top-level .safetensors files in $OUTDIR (if any)"
    find "$OUTDIR" -maxdepth 1 -type f -name "*.safetensors" -print0 \
        | xargs -0 -r -I{} sh -c 'echo "[CLEANUP] Deleting file: {}"; rm -f "{}" || echo "[CLEANUP] Failed to delete: {}"'

    # 2) Keep only the checkpoint directory with the largest numeric suffix
    echo "[CLEANUP] Pruning older checkpoint-* directories (keeping the largest)"
    mapfile -t __cp_lines < <(
        find "$OUTDIR" -maxdepth 1 -type d -name 'checkpoint-*' -print \
        | while read -r d; do
            bname=$(basename "$d")
            if [[ "$bname" =~ ^checkpoint-([0-9]+)$ ]]; then
                num="${BASH_REMATCH[1]}"
                printf '%d\t%s\n' "$num" "$d"
            fi
        done \
        | sort -n -k1,1
    )

    if (( ${#__cp_lines[@]} == 0 )); then
        echo "[CLEANUP] No numeric checkpoint-* directories found in $OUTDIR"
        return 0
    fi

    local keep_line
    keep_line="${__cp_lines[$((${#__cp_lines[@]}-1))]}"
    local keep_num keep_path
    keep_num=$(awk -F$'\t' '{print $1}' <<<"$keep_line")
    keep_path=$(awk -F$'\t' '{print $2}' <<<"$keep_line")
    echo "[CLEANUP] Will keep: $(basename "$keep_path") (num=$keep_num)"

    for line in "${__cp_lines[@]}"; do
        num=$(awk -F$'\t' '{print $1}' <<<"$line")
        path=$(awk -F$'\t' '{print $2}' <<<"$line")
        if [[ "$path" != "$keep_path" ]]; then
            echo "[CLEANUP] Deleting checkpoint dir: $path"
            rm -rf -- "$path" || echo "[CLEANUP] Failed to delete: $path"
        fi
    done

    # 3) Delete .pt and .pth files in OUTDIR and remaining checkpoint directory
    echo "[CLEANUP] Removing .pt and .pth files in $OUTDIR and checkpoint directories"
    find "$OUTDIR" -maxdepth 1 -type f \( -name "*.pt" -o -name "*.pth" \) -print0 \
        | xargs -0 -r -I{} sh -c 'echo "[CLEANUP] Deleting file: {}"; rm -f "{}" || echo "[CLEANUP] Failed to delete: {}"'

    if [[ -d "$keep_path" ]]; then
        find "$keep_path" -maxdepth 1 -type f \( -name "*.pt" -o -name "*.pth" \) -print0 \
            | xargs -0 -r -I{} sh -c 'echo "[CLEANUP] Deleting file: {}"; rm -f "{}" || echo "[CLEANUP] Failed to delete: {}"'
    fi

    echo "[CLEANUP] Checkpoint cleanup complete for: $OUTDIR"
}

# ==============================================================================
# Parameter arrays (one entry per line for easy commenting)
# ==============================================================================
model_name_or_path_list=(
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
)

train_file_list=(
    "./data/gsm_symbolic_train_4500_student.jsonl"
)

max_seq_length_list=(
    # 512
    1024
    # 2048
)

num_train_epochs_list=(
    1
    # 2
)

learning_rate_list=(
    # "1e-05"
    "2e-05"
    # "5e-05"
)

sample_data_seed_list=(
    0
    # 42
)

effective_bsz_list=(
    8
    # 16
    # 32
)

# ==============================================================================
# Fixed training hyperparameters
# ==============================================================================
# Batch size auto-calculation:
#   effective_bsz = nproc_per_node * per_device_train_batch_size * gradient_accumulation_steps
#   per_device_train_batch_size is determined by model size (extracted from model path):
#     - model_size <= 2B: per_device_train_batch_size = 4
#     - model_size <= 4B: per_device_train_batch_size = 2
#     - model_size >  4B: per_device_train_batch_size = 1
nproc_per_node=1

torch_dtype="bfloat16"
bf16_mixed_prec_training=True
percentage=1.0
wandb_project="a4_sft"

# ==============================================================================
# Base training arguments (parameters NOT iterated over)
# ==============================================================================
base_training_args="\
    --do_train True \
    --use_fast_tokenizer True \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.0 \
    --eval_strategy no \
    --logging_steps 1 \
    --torch_dtype $torch_dtype \
    --bf16 $bf16_mixed_prec_training \
    --tf32 False \
    --fp16 False \
    --overwrite_output_dir True \
    --report_to wandb \
    --optim adamw_torch \
    --percentage $percentage \
    --save_strategy epoch \
    --wandb_project $wandb_project"

# ==============================================================================
# Torchrun rendezvous setup
# ==============================================================================
ID=$RANDOM
PORT=$((29000 + ID % 1000))
header="OMP_NUM_THREADS=8 torchrun --nproc_per_node $nproc_per_node --nnodes 1 \
--rdzv-endpoint=localhost:$PORT \
--rdzv-id=$ID --rdzv_backend c10d \
-m train"

# ==============================================================================
# Main training loop
# ==============================================================================
for model_name_or_path in "${model_name_or_path_list[@]}"; do
for train_file in "${train_file_list[@]}"; do
for max_seq_length in "${max_seq_length_list[@]}"; do
for num_train_epochs in "${num_train_epochs_list[@]}"; do
for learning_rate in "${learning_rate_list[@]}"; do
for sample_data_seed in "${sample_data_seed_list[@]}"; do
for effective_bsz in "${effective_bsz_list[@]}"; do

    # Extract model name from path (everything after the last "/")
    model_name=$(echo "$model_name_or_path" | awk -F'/' '{print $NF}')

    # Extract dataset name from train_file (filename without extension)
    dataset_name=$(basename "$train_file" .jsonl)

    # Extract model size (number before "b-" or "B-") and calculate batch size params
    if [[ "$model_name_or_path" =~ ([0-9]+)[bB]- ]]; then
        model_size="${BASH_REMATCH[1]}"
    else
        model_size=1  # default to small model if pattern not found
    fi

    if (( model_size <= 2 )); then
        per_device_train_batch_size=4
    elif (( model_size <= 4 )); then
        per_device_train_batch_size=2
    else
        per_device_train_batch_size=1
    fi

    gradient_accumulation_steps=$((effective_bsz / (nproc_per_node * per_device_train_batch_size)))

    # Build descriptive run_name (order: maxlen -> epochs -> lr -> effbsz -> dataseed)
    run_name="${dataset_name}-${model_name}-maxlen${max_seq_length}-epochs${num_train_epochs}-lr${learning_rate}-effbsz${effective_bsz}-dataseed${sample_data_seed}"

    # Build output directory
    output_dir="./out/${run_name}"

    mkdir -p "$output_dir"

    echo "=========================================="
    echo "Starting SFT training"
    echo "  Model: $model_name (size: ${model_size}B)"
    echo "  Dataset: $dataset_name"
    echo "  Max seq length: $max_seq_length"
    echo "  Epochs: $num_train_epochs"
    echo "  Learning rate: $learning_rate"
    echo "  Effective batch size: $effective_bsz"
    echo "    nproc_per_node: $nproc_per_node"
    echo "    per_device_train_batch_size: $per_device_train_batch_size"
    echo "    gradient_accumulation_steps: $gradient_accumulation_steps"
    echo "  Data seed: $sample_data_seed"
    echo "  Output: $output_dir"
    echo "=========================================="

    training_args="$base_training_args \
    --model_name_or_path $model_name_or_path \
    --output_dir $output_dir \
    --max_seq_length $max_seq_length \
    --num_train_epochs $num_train_epochs \
    --sample_data_seed $sample_data_seed \
    --learning_rate $learning_rate \
    --train_file $train_file \
    --run_name $run_name \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps"

    eval "$header" "$training_args" 2>&1 | tee "$output_dir/sft_train.log"

    echo "Completed training for: $run_name"

    # Cleanup checkpoints after each run
    cleanup_checkpoints_after_run "$output_dir"

    echo "----------------------------------------"

done
done
done
done
done
done
done
