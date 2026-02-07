from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class DataArguments:
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data file (JSONL format)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for preprocessing via datasets.map()."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum total input sequence length after tokenization; longer sequences are truncated."
            )
        },
    )
    sample_data_seed: int = field(
        default=0, metadata={"help": "Seed used for dataset shuffling/sampling."}
    )
    percentage: float = field(
        default=1.0, metadata={"help": "Sampling percentage for the training dataset."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model checkpoint for initialization (HF repo id or local path)."
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use a fast tokenizer (backed by tokenizers) if available."},
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override torch dtype when loading the model. If 'auto', derive dtype from weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name for logging (sets WANDB_PROJECT)."},
    )
