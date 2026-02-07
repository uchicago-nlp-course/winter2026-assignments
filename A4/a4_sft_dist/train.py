from __future__ import annotations

import logging
import os
import random
from dataclasses import asdict
from typing import Dict

import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from data_utils import (
    LMDatasetConfig,
    parse_torch_dtype,
    get_data_statistics,
    load_and_prepare_train_dataset,
)
from training_arguments import DataArguments, ModelArguments, TrainingArguments


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    set_seed(training_args.seed)

    if not model_args.model_name_or_path:
        model_args.model_name_or_path = "google/gemma-3-1b-it"
    if model_args.torch_dtype is None:
        # Training default: BF16
        model_args.torch_dtype = "bfloat16"

    cache_dir = os.getenv("MODEL_CACHE_DIR")
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    logger.info(f"tokenizer.chat_template: {tokenizer.chat_template}")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer is missing chat_template; refusing to proceed.")

    logger.info(f"tokenizer.pad_token: {tokenizer.pad_token}")
    if tokenizer.pad_token is None:
        # Do NOT silently add a pad token: the assignment requires failing fast
        # so students notice tokenizer/config issues.
        raise ValueError("Tokenizer is missing pad_token; refusing to proceed.")

    torch_dtype = parse_torch_dtype(model_args.torch_dtype)
    logger.info(
        f"Loading model from {model_args.model_name_or_path} (torch_dtype={model_args.torch_dtype})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
    )

    if not data_args.train_file:
        raise ValueError("--train_file must be provided.")

    train_dataset = load_and_prepare_train_dataset(
        tokenizer,
        cfg=LMDatasetConfig(
            train_file=data_args.train_file,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            preprocessing_num_workers=data_args.preprocessing_num_workers,
            sample_data_seed=data_args.sample_data_seed,
            percentage=data_args.percentage,
        ),
    )

    get_data_statistics(train_dataset)

    for index in random.sample(range(len(train_dataset)), 1):
        sample = train_dataset[index]
        logger.info(f"Sample {index} of the training set: {sample}")

        input_ids = sample["input_ids"]
        logger.info(f"Decoded Input IDs:\n{tokenizer.decode(input_ids)}")

        labels = [label for label in sample["labels"] if label > -1]
        logger.info(f"Decoded Labels:\n{tokenizer.decode(labels)}")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    # ===== YOUR CODE HERE =====
    # TODO: create data_collator and trainer
    # Create a data collator using `DataCollatorForSeq2Seq` and a `Trainer`,
    # both from the transformers package, with appropriate arguments that have
    # been defined in the previous part of this file.
    data_collator = None
    trainer = None
    # ===== END YOUR CODE =====

    logger.info(f"Model args: {asdict(model_args)}")
    logger.info(f"Data args: {asdict(data_args)}")
    logger.info(f"Training args (selected): {_summarize_training_args(training_args)}")

    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info("dist.is_initialized() and dist.get_rank() == 0")
        logger.info(
            f"current master port: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        print(model)
    elif not dist.is_initialized():
        logger.info("not dist.is_initialized()")
        print(model)

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.save_state()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)


def _summarize_training_args(training_args: TrainingArguments) -> Dict[str, object]:
    keys = [
        "output_dir",
        "run_name",
        "do_train",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_train_epochs",
        "max_steps",
        "logging_steps",
        "save_strategy",
        "evaluation_strategy",
        "bf16",
        "fp16",
        "seed",
        "report_to",
    ]
    out: Dict[str, object] = {}
    for k in keys:
        out[k] = getattr(training_args, k, None)
    out["wandb_project"] = training_args.wandb_project
    return out


if __name__ == "__main__":
    main()
