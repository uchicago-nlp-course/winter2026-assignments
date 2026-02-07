from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import ANSWER_PREFIX, parse_torch_dtype
from eval_utils import eq_num, extract_gsm_answer, parse_gold_answer_field, build_gsm_prompt


logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_user_prompt(question: str) -> str:
    full_prompt = build_gsm_prompt(question, method="cot")
    if not full_prompt.endswith(ANSWER_PREFIX):
        raise ValueError(
            "A3 GSM CoT prompt did not end with the expected answer prefix. "
            f"Expected suffix: {ANSWER_PREFIX!r}."
        )
    return full_prompt[: -len(ANSWER_PREFIX)]


@torch.inference_mode()
def _generate_batch(
    model,
    tokenizer,
    device: torch.device,
    questions: List[str],
    max_new_tokens: int,
) -> List[str]:
    prompt_texts = []
    for question in questions:
        user_prompt = _build_user_prompt(question)
        chat_prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Keep evaluation prompts consistent with training: append the CoT
        # answer prefix after `<start_of_turn>model`, not inside the user
        # message.
        prompt_texts.append(chat_prefix + ANSWER_PREFIX)

    inputs = tokenizer(
        prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False,
    ).to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        # Let the model config handle eos_token_id/pad_token_id unless you have a specific override
    )

    # With left-padding, all sequences share the same padded input length.
    # Slice from the end of the (padded) input to extract only new tokens.
    input_len = inputs["input_ids"].shape[1]

    results: List[str] = []
    for seq in outputs:
        new_tokens = seq[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(text.strip())

    return results


def _as_float_maybe(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    p = argparse.ArgumentParser(description="Evaluate a base or SFT-tuned model on GSM-Symbolic.")
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/gemma-3-1b-it",
        help="HF model id or local checkpoint path to evaluate.",
    )
    p.add_argument(
        "--eval_file",
        type=str,
        default="./data/gsm_symbolic_test_100.jsonl",
        help="Eval JSONL file. If missing, falls back to ./data/gsm_symbolic_test_100_student.jsonl.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for eval artifacts. Defaults to ./outputs/<timestamp>/eval.",
    )
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--torch_dtype",
        type=str,
        # Eval default: FP32 for deterministic/portable evaluation (CPU-safe).
        default="float32",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)

    eval_path = Path(args.eval_file)
    if not eval_path.is_file():
        fallback = Path("./data/gsm_symbolic_test_100_student.jsonl")
        if fallback.is_file():
            logger.warning(
                f"Eval file {eval_path} not found; falling back to {fallback}."
            )
            eval_path = fallback
        else:
            raise FileNotFoundError(
                f"Neither {eval_path} nor {fallback} exists; cannot run evaluation."
            )

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("./outputs") / ts / "eval"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cache_dir = os.getenv("MODEL_CACHE_DIR")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer is missing chat_template; refusing to proceed.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=cache_dir,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
    )
    model.to(device)
    model.eval()

    rows = _load_jsonl(eval_path)
    if not rows:
        raise ValueError(f"Eval file {eval_path} is empty.")

    pred_records: List[Dict[str, Any]] = []
    num_correct = 0

    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(rows), batch_size), desc="Evaluating"):
        batch_rows = rows[batch_start : batch_start + batch_size]

        questions: List[str] = []
        golds: List[Optional[float]] = []
        for i, row in enumerate(batch_rows, start=batch_start):
            question = row.get("question")
            if question is None:
                raise ValueError(f"Missing 'question' at row {i}.")
            questions.append(str(question))

            gold = _as_float_maybe(row.get("final_answer"))
            if gold is None:
                gold = parse_gold_answer_field(row.get("answer"))
            golds.append(gold)

        outputs = _generate_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            questions=questions,
            max_new_tokens=args.max_new_tokens,
        )

        for j, (row, output, gold) in enumerate(
            zip(batch_rows, outputs, golds), start=batch_start
        ):
            pred = extract_gsm_answer(output)
            ok = eq_num(pred, gold)
            num_correct += int(ok)

            pred_records.append(
                {
                    "index": j,
                    "question": row.get("question"),
                    "gold": gold,
                    "prediction": pred,
                    "correct": ok,
                    "raw_output": output,
                }
            )

    accuracy = num_correct / len(rows)
    logger.info(f"Accuracy: {accuracy:.4f} ({num_correct}/{len(rows)})")

    # Write outputs
    preds_path = out_dir / "predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for r in pred_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["num_examples", "num_correct", "accuracy", "model_name_or_path", "eval_file"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "num_examples": len(rows),
                "num_correct": num_correct,
                "accuracy": accuracy,
                "model_name_or_path": args.model_name_or_path,
                "eval_file": str(eval_path),
            }
        )

    logger.info(f"Wrote {preds_path}")
    logger.info(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
