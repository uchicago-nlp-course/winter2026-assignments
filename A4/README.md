# Assignment 4: Finetuning and Best-of-N Sampling

This assignment explores Supervised Fine-Tuning (SFT) and inference-time strategies like Majority Voting and Best-of-N sampling on mathematical reasoning tasks (GSM-Symbolic).

## Prerequisites

1. **Modal** - create a Modal account as per the handout instructions.
2. **HuggingFace** - accept the [Gemma 3 model license](https://huggingface.co/google/gemma-3-1b-it) on HuggingFace and obtain an [access token](https://huggingface.co/settings/tokens).
3. **Weights & Biases** - create an account at https://wandb.ai and obtain an [API key](https://wandb.ai/authorize).

## Install Dependencies

Requires Python >= 3.11. Install all dependencies locally (needed for the `modal` CLI, test scripts, and optional local execution):

```bash
uv sync
source .venv/bin/activate
```

Or alternatively (even if you prefer pip, we strongly recommend using some kind of virtual environments):

```bash
pip install -r requirements.txt
```

## Part 1: Supervised Fine-Tuning (SFT)

Fine-tune Gemma 3 instruction-tuned models (1B and 4B) on [GSM-Symbolic](https://arxiv.org/abs/2410.05229) math problems, then evaluate on in-distribution (GSM-Symbolic) and out-of-distribution (GSM8K) test sets.

### Directory Structure

```
a4_sft_dist/
├── data/
│   ├── gsm_symbolic_train_4500_student.jsonl   # Training set (4,500 examples)
│   ├── gsm_symbolic_test_100_student.jsonl     # GSM-Symbolic test (100 examples)
│   ├── gsm8k_test_100_same_templates.jsonl     # GSM8K test - same templates as training
│   └── gsm8k_test_100_non_overlapping.jsonl    # GSM8K test - non-overlapping templates
├── scripts/
│   ├── setup_modal.sh          # One-time Modal authentication and secrets setup
│   ├── run_sft_train.sh        # Training launcher with hyperparameter sweeps
│   └── run_sft_eval.sh         # Evaluation launcher
├── data_utils.py               # Dataset loading and encoding (has TODOs)
├── train.py                    # Training entry point (has TODOs)
├── eval.py                     # Evaluation entry point
├── eval_utils.py               # Prompt building and answer extraction utilities
├── training_arguments.py       # Dataclasses for model, data, and training arguments
├── modal_app.py                # Modal cloud GPU orchestration
├── download_modal_volumes.py   # Download checkpoints/results from Modal volumes
└── test_a4_q1.py               # Test script - generates A4-Q1.json for validation
```

### How the Code Works Together

The SFT pipeline has three main stages: data preparation, training, and evaluation.

**Data Preparation** (`data_utils.py`):
- `load_and_prepare_train_dataset()` loads a JSONL file and applies `encode_function()` to each example.
- `encode_function()` converts each (question, answer) pair into tokenized `input_ids`, `labels`, and `attention_mask`. The prompt tokens are masked with `-100` in labels so the model only learns to predict the completion.

**Training** (`train.py`):
- Parses command-line arguments via `training_arguments.py` dataclasses.
- Calls `load_and_prepare_train_dataset()` to get the tokenized dataset.
- Creates a `DataCollatorForSeq2Seq` to handle padding within batches.
- Initializes the HuggingFace `Trainer` with the model, dataset, and collator.
- Runs `trainer.train()` and saves checkpoints to `output_dir`.

**Evaluation** (`eval.py`):
- Loads a trained checkpoint (or base model) and the test JSONL file.
- For each question, builds the prompt using `eval_utils.build_gsm_prompt()`, generates a response, and extracts the predicted numeric answer.
- Compares predictions against gold answers and computes accuracy.

**Orchestration** (`modal_app.py`):
- Defines Modal functions that run `train.py` and `eval.py` on cloud GPUs.
- Mounts Modal volumes for persistent storage of checkpoints and results.
- `launch_all` runs training first, then evaluation on the saved checkpoints.

### Tasks: Complete the TODOs

There are two files with TODO sections to implement:

1. **`data_utils.py`** - implement the `encode_function` loop body and the `load_and_prepare_train_dataset` tokenization and formatting steps.
2. **`train.py`** - create the `data_collator` and `trainer`.

Detailed instructions are provided in the docstrings and inline comments within each file.

After completing all TODOs, run the test script to validate your implementation:

```bash
cd a4_sft_dist
python test_a4_q1.py
```

This generates `A4-Q1.json` in the current directory.

### Running on Modal (Recommended)

#### Modal Setup

Run the setup script once to authenticate and create the required secrets:

```bash
cd a4_sft_dist
bash scripts/setup_modal.sh
```

This will:
1. Verify the `modal` CLI is installed.
2. Authenticate with Modal (runs `modal setup` if needed).
3. Create Modal secrets for W&B (`wandb-secret`) and HuggingFace (`huggingface-secret`).

#### Launch Training and Evaluation

Run both training and evaluation (recommended):

```bash
cd a4_sft_dist
modal run modal_app.py::launch_all
```

> **Tip:** Use the `--detach` (or `-d`) flag (e.g., `modal run --detach modal_app.py::launch_all`) to run the job in the background on Modal servers. This allows you to close your terminal or lose your internet connection without interrupting the execution. You can monitor the progress later via the Modal Dashboard.

You can also run training or evaluation individually:

```bash
modal run modal_app.py::launch_train
modal run modal_app.py::launch_eval
```

> `launch_eval` requires checkpoints from a prior training run to exist on the Modal output volume.

#### Inspecting Results

- **Modal Dashboard** - view real-time logs and app status at https://modal.com/apps.
- **Modal Volumes** - browse saved checkpoints (volume `a4-output`) and evaluation results (volume `a4-eval-results`) directly in the Modal UI. See https://modal.com/docs/guide/volumes.
- **Weights & Biases** - view training loss curves, learning rate schedules, and other metrics at https://wandb.ai under project `a4_sft`.

### Running Locally (Optional)

For students with local GPU access who prefer not to use Modal.

#### Environment Variables

Set the following environment variables before running the scripts:

```bash
export MODEL_CACHE_DIR="$HOME/.cache/huggingface/transformers"
export DATA_CACHE_DIR="$HOME/.cache/huggingface/datasets"
export WANDB_API_KEY="<your-wandb-api-key>"
export HF_TOKEN="<your-hf-token>"
```

#### Training

```bash
cd a4_sft_dist
bash scripts/run_sft_train.sh
```

**Adjusting hyperparameters:** the script uses hyperparameter arrays at the top. Comment or uncomment entries to control the sweep.

#### Evaluation

```bash
cd a4_sft_dist
bash scripts/run_sft_eval.sh
```

Update `model_name_or_path_list` in the script to include your trained checkpoint directories.

#### Downloading Results from Modal

```bash
cd a4_sft_dist

# Download all checkpoints and eval results
python3 download_modal_volumes.py

# Download only eval results
python3 download_modal_volumes.py --volume eval

# Download a specific training run
python3 download_modal_volumes.py --volume output --remote-prefix <run_name>

# List remote files without downloading
python3 download_modal_volumes.py --dry-run
```

## Part 2: Best-of-N Sampling and Majority Voting

In this part, we use the non-finetuned and finetuned `gemma-3-1b-it` to generate 8 samples per question. We then compare two strategies for selecting the final answer:

1. **Majority Voting (Self-Consistency)**: Choosing the most frequent numeric answer among the first n samples.
2. **Best-of-N (BoN)**: Choosing the sample with the highest reward score among the first n samples, using the `Skywork-Reward-V2-Llama-3.1-8B` reward model.

Evaluation is performed for n in {1, 2, 4, 8}.

### Directory Structure

```
a4_sampling_dist/
├── data/                   # Contains the GSM-Symbolic test split (100 examples)
├── scripts/                # Shell scripts to run the experiments
├── utils.py                # Shared utilities for model loading, prompting, and evaluation
├── generate_samples.py     # Main script for generating and scoring samples (has TODOs)
├── evaluate.py             # Script for computing Majority Voting and BoN accuracies (has TODOs)
├── analyze_traces.py       # Script for analyzing failure modes (has TODOs)
├── modal_app.py            # Modal cloud GPU orchestration
├── test_a4_q2.py           # Test script - generates A4-Q2.json for validation
└── results/                # Output directory for predictions and summaries
```

### How the Code Works Together

The sampling pipeline has three main stages: generation/scoring, evaluation, and analysis.

**Sample Generation and Scoring** (`generate_samples.py`):
- Loads the generator model (Gemma) and reward model (Skywork-Reward-V2).
- For each question, `generate_samples_batched()` produces n diverse responses using temperature sampling.
- `score_samples_batched()` scores each response using the reward model.
- Outputs a JSONL file where each line contains the question, gold answer, and n samples with their reward scores.

**Evaluation** (`evaluate.py`):
- Reads the scored samples JSONL file.
- For each n in {1, 2, 4, 8}, computes two metrics:
  - **Majority Voting**: Extracts the numeric answer from each of the first n samples, then picks the most frequent answer.
  - **Best-of-N**: Selects the sample with the highest reward score among the first n, then extracts its numeric answer.
- Compares predictions against gold answers and reports accuracy for each (n, strategy) combination.

**Failure Analysis** (`analyze_traces.py`):
- Identifies questions where the correct answer was generated but the selection strategy failed.
- `check_majority_failure()`: Returns True if majority voting picked wrong despite a correct sample existing.
- `check_bon_failure()`: Returns True if the highest-scored sample was incorrect despite a correct sample existing.
- Generates plots and detailed reports for understanding failure modes.

**Orchestration** (`modal_app.py`):
- Runs `generate_samples.py` on a cloud GPU (generation and reward scoring are compute-intensive).
- Mounts the SFT checkpoint volume so fine-tuned models can be used for generation.
- Stores results in a separate volume for later download and local analysis.

### Tasks: Complete the TODOs

There are three files with TODO sections to implement:

1. **`generate_samples.py`** - implement `generate_samples_batched` (prompt building, tokenization, and generation) and the scoring loop in `score_samples_batched`.
2. **`evaluate.py`** - implement `evaluate_example()` which computes Majority Voting and Best-of-N accuracies for a single question across different $n$ values.
3. **`analyze_traces.py`** - implement `check_majority_failure`, `check_bon_failure`, and `get_unique_preds_count`.

Detailed instructions are provided in the docstrings and inline comments within each file.

### How to Run

**Generate and Score Samples** (Remote):
Run the generation and reward scoring on a Modal GPU.

```bash
cd a4_sampling_dist
modal run modal_app.py::launch_sampling --model 1b
```

**Download Results** (Local):
Pull the scored samples from the Modal cloud volume to your local `results/` folder.

```bash
bash scripts/get_results.sh
```

**Evaluate Accuracy** (Local):
Compute Majority Voting and BoN accuracies locally for both the base and fine-tuned models.

```bash
bash scripts/evaluate.sh
```

This script iterates over the downloaded `.jsonl` files and saves results to `results/sampling/eval_[model_name]/`. Each directory will contain:
`summary.csv`: Accuracies for Majority Voting and Best-of-N at $n \in \{1, 2, 4, 8\}$.

**Analyze Failure Modes** (Local):
Generate summary statistics, plots, and a detailed failure mode report for both the base and fine-tuned models.

```bash
bash scripts/analyze_traces.sh
```

This script iterates over all `.jsonl` files in `results/sampling/` and produces:
- `plots/unique_answers_dist_[model_name].png`: distribution of unique answers per question.
- `results/failure_modes_[model_name].txt`: a comprehensive report containing questions where Majority Voting or Best-of-N failed despite a correct answer being generated (filtered for cases with 3 unique answers).

**Run Test Script** (Local):
After completing all TODOs and the steps above, run the test script to generate `A4-Q2.json` for Gradescope submission.

```bash
python test_a4_q2.py
```

## Submission Instructions

This assignment has two components to submit on Gradescope:

1.  **Written Report**: Submit your final compiled **PDF** report (typeset using the provided LaTeX template) to **Assignment 4 - Written**.
2.  **Coding Implementation**: Submit your code and generated results to **Assignment 4 - Code**.

### Creating the Code Submission Archive

To create the `.zip` file for the coding submission, follow these steps. From the **root** of the `A4` assignment directory (where both `a4_sft_dist` and `a4_sampling_dist` are visible), run the following command:

```bash
zip -r A4.zip a4_sft_dist/ a4_sampling_dist/ -x "**/__pycache__/*" "**/.DS_Store"
```

Upload your A4.zip file to Gradescope.

### Configuration and Path Settings

While the pipeline is designed to be run as is, you may need to adjust paths if you used custom names in Part 1:

1.  **Modal Volumes (`modal_app.py`)**: 
    - `output_vol`: Must point to the same volume name used in Part 1 (default: `a4-output`) to access your fine-tuned checkpoints.
    - `sampling_results_vol`: The volume where Part 2 results are stored (default: `a4-sampling-results`).
    - `huggingface-secret`: Ensure this matches the secret name created in Part 1.
2.  **Syncing Results (`scripts/get_results.sh`)**: 
    - `VOLUME_NAME`: Must match the `sampling_results_vol` name in `modal_app.py`.
3.  **Trace Analysis (`scripts/analyze_traces.sh`)**: 
    - Automatically discovers and analyzes all scored samples in `results/sampling/` if no argument is provided. 
    - Generates model-specific plots and failure mode reports with model-name suffixes to avoid overwriting.

### How Modal Accesses Fine-tuned Models

To perform sampling on your fine-tuned models from Part 1, the following process occurs:

1.  **Mounting**: In `modal_app.py`, the `a4-output` volume (containing your SFT checkpoints) is mounted to the remote path `/root/a4_sampling/sft_out`.
2.  **Discovery**: The `scripts/generate_samples.sh` script looks inside this `sft_out` directory.
3.  **Filtering**: It filters for subdirectories matching your model choice (e.g., directories containing `1b-it` if you chose `--model 1b`).
4.  **Checkpoint Selection**: For each matching run directory, it searches for subdirectories named `checkpoint-*`, sorts them, and picks the one with the largest step number (the most recent checkpoint) to use for generation.