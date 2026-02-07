from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset

from eval_utils import build_gsm_prompt, split_gsm_solution_and_final


ANSWER_PREFIX = "Answer: Let's think step by step."
# Gemma chat template explicitly closes each message with `<end_of_turn>\n`.
END_OF_TURN = "<end_of_turn>\n"


def _build_user_prompt_and_prefix(question: str) -> Tuple[str, str]:
    full_prompt = build_gsm_prompt(question, method="cot")
    if not full_prompt.endswith(ANSWER_PREFIX):
        raise ValueError(
            "A3 GSM CoT prompt did not end with the expected answer prefix. "
            f"Expected suffix: {ANSWER_PREFIX!r}."
        )
    user_prompt = full_prompt[: -len(ANSWER_PREFIX)]
    return user_prompt, ANSWER_PREFIX


def _format_completion(answer_text: str, final_answer: Optional[str]) -> str:
    solution_text, parsed_final = split_gsm_solution_and_final(answer_text)
    final = final_answer if final_answer is not None else parsed_final
    final = str(final).strip()
    solution_text = str(solution_text).rstrip()
    if solution_text:
        return f"{solution_text}\nFinal answer: {final}."
    return f"Final answer: {final}."


@dataclass(frozen=True)
class LMDatasetConfig:
    """Configuration for loading and preprocessing the training dataset.

    Attributes:
        train_file: Path to the JSONL training file containing question/answer pairs.
        max_seq_length: Maximum sequence length for tokenization. Sequences longer
            than this will be truncated. If None, no truncation is applied.
        overwrite_cache: If True, reprocess the dataset even if a cached version
            exists. Passed to Dataset.map(load_from_cache_file=not overwrite_cache).
        preprocessing_num_workers: Number of parallel workers for Dataset.map().
            If None, uses the default (single-process).
        sample_data_seed: Random seed for shuffling the dataset before subsampling.
            Ensures reproducible data ordering across runs.
        percentage: Fraction of the dataset to use (0.0 < percentage <= 1.0).
            After shuffling, only the first (percentage * len(dataset)) examples
            are kept. Useful for quick experiments with smaller data subsets.
    """
    train_file: str
    max_seq_length: Optional[int]
    overwrite_cache: bool
    preprocessing_num_workers: Optional[int]
    sample_data_seed: int
    percentage: float


def parse_torch_dtype(dtype_str: str | None):
    """
    Parse a user-provided dtype string into a torch dtype (or "auto").

    Note: `AutoModelForCausalLM.from_pretrained(..., torch_dtype="auto")` is
    supported by Transformers for auto-derived dtype.
    """

    if dtype_str is None:
        return None
    if dtype_str == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(
            f"torch_dtype must be one of {sorted(mapping.keys()) + ['auto']}, got {dtype_str}."
        )
    return mapping[dtype_str]


def get_data_statistics(lm_datasets):
    """ Get the data statistics of the dataset. """

    def get_length(examples):
        lengths = [len(ids) for ids in examples["input_ids"]]

        completion_lens = []
        for labels in examples["labels"]:
            com_len = sum(1 for label in labels if label > -1)
            completion_lens.append(com_len)
        return {"length": lengths, "c_length": completion_lens}

    if not isinstance(lm_datasets, dict):
        lm_datasets = {"train": lm_datasets}

    for key in lm_datasets:
        dataset = lm_datasets[key]
        data_size = len(dataset)
        dataset = dataset.map(get_length, batched=True)
        lengths = dataset["length"]
        length = sum(lengths) / len(lengths)
        c_lengths = dataset["c_length"]
        c_length = sum(c_lengths) / len(c_lengths)
        print(f"[{key} set] examples: {data_size}; # avg tokens: {length}")
        print(
            f"[{key} set] examples: {data_size}; # avg completion tokens: {c_length}"
        )


def encode_function(
    examples: Dict[str, List],
    tokenizer,
    max_seq_length: Optional[int],
) -> Dict[str, List]:
    """Encode a batch of question-answer pairs into tokenized training examples.

    For each (question, answer) pair, build the prompt and completion texts,
    tokenize them, and create label masks so the model is only trained on the
    completion (assistant) tokens.

    High-level workflow for each example:
    1. Build `prompt_text` using the chat template and the answer prefix.
       - Use `_build_user_prompt_and_prefix(question)` to get `user_prompt`
         and `answer_prefix`.
       - Apply `tokenizer.apply_chat_template(...)` with `tokenize=False`
         and `add_generation_prompt=True` to wrap the user prompt, then
         concatenate the answer prefix.
    2. Build `completion_text` using `_format_completion(answer, final_answer)`,
       then append `END_OF_TURN` and the tokenizer's EOS token.
    3. Concatenate `prompt_text + completion_text` into `full_text`.
    4. Tokenize both `prompt_text` and `full_text` with
       `tokenizer.encode(..., add_special_tokens=False)` (and optional
       truncation if `max_seq_length` is set).
    5. Validate that `prompt_ids` is a prefix of `full_ids`
       (raise `ValueError` if not — this ensures label masking is aligned).
    6. Build output lists:
       - `input_ids`: the full tokenized sequence as a Python list.
       - `labels`: `[-100] * prompt_len + input_ids[prompt_len:]`
         (mask prompt tokens with -100, keep completion token IDs).
       - `attention_mask`: `[1] * len(input_ids)`.

    Example — given question = "What is 2 + 3?", answer = "2 + 3 = 5. #### 5":

        prompt_text (context — NOT trained on):
            "<bos><start_of_turn>user
            Solve the following math problem step by step, and provide the final numeric answer.

            Problem: What is 2 + 3?
            <end_of_turn>
            <start_of_turn>model
            Answer: Let's think step by step."

        completion_text (trained to generate):
            " 2 + 3 = 5.
            Final answer: 5.<end_of_turn>
            <eos>"

    Args:
        examples: A dict with keys "question" (List[str]), "answer" (List[str]),
            and optionally "final_answer" (List[str]).
        tokenizer: A HuggingFace tokenizer with a chat template and EOS token.
        max_seq_length: If set, truncate tokenized sequences to this length.

    Returns:
        A dict with keys "input_ids", "labels", "attention_mask", each a
        List[List[int]]. The labels have -100 for all prompt tokens and real
        token IDs for completion tokens only.
    """
    questions: List[str] = examples["question"]
    answers: List[str] = examples["answer"]
    final_answers: Optional[List[str]] = examples.get("final_answer")

    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        # ===== YOUR CODE HERE =====
        # TODO: implement the loop body for encoding a single (question, answer) pair
        raise NotImplementedError("YOUR CODE HERE")
        # ===== END YOUR CODE =====

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }


def load_and_prepare_train_dataset(
    tokenizer,
    cfg: LMDatasetConfig,
) -> Dataset:
    """Load a JSONL training file and return a tokenized, PyTorch-formatted Dataset.

    The loading, shuffling, and percentage-based subsampling are already done
    below. You need to complete the remaining two steps:

    1. Apply `encode_function` to `raw` using `Dataset.map()` with
       `batched=True`. Check `training_arguments.py` → `DataArguments`
       to determine which attributes of `cfg` correspond to the keyword
       arguments of `Dataset.map()` (e.g., number of parallel workers,
       cache behavior). Also remove all original columns from the raw dataset
       so that only the three output columns remain.

       Note: `Dataset.map()` expects a callable that takes only one argument
       (`examples`), but `encode_function` requires additional arguments
       (`tokenizer`, `max_seq_length`). You will need to adapt
       `encode_function` so it can be passed to `.map()` — consider how
       to bind extra arguments to a function before passing it as a callable.

    2. Set the dataset format to PyTorch tensors for the columns:
       `"input_ids"`, `"labels"`, `"attention_mask"`.

    Args:
        tokenizer: A HuggingFace tokenizer.
        cfg: An LMDatasetConfig with dataset loading parameters.

    Returns:
        A HuggingFace Dataset with PyTorch tensor columns:
        "input_ids", "labels", "attention_mask".
    """
    if not cfg.train_file:
        raise ValueError("No training file provided.")

    raw = load_dataset(
        "json",
        data_files={"train": cfg.train_file},
        cache_dir=os.getenv("DATA_CACHE_DIR"),
    )["train"]

    raw = raw.shuffle(seed=cfg.sample_data_seed)
    if not (0.0 < cfg.percentage <= 1.0):
        raise ValueError(f"percentage must be in (0, 1], got {cfg.percentage}.")
    if cfg.percentage < 1.0:
        num_keep = max(1, int(math.floor(len(raw) * cfg.percentage)))
        raw = raw.select(range(num_keep))

    # ===== YOUR CODE HERE =====
    # TODO: implement dataset tokenization via map() and format setting
    # 1. Use Dataset.map() with encode_function and appropriate arguments
    # 2. Set the dataset format to PyTorch tensors for the required columns with `set_format()`
    raise NotImplementedError("YOUR CODE HERE")
    # ===== END YOUR CODE =====
