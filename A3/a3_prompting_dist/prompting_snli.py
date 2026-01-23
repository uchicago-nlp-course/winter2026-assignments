import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from utils import (
    DEFAULT_MAX_NEW_TOKENS,
    MODEL_NAME,
    SHOT_COUNTS,
    ModelBundle,
    build_snli_prompt,
    compute_accuracy,
    extract_snli_answer,
    load_jsonl,
    make_run_dir,
    model_generate,
    progressive_indices,
    save_jsonl,
    seed_multiples,
    write_meta,
    load_model_and_tokenizer,
)


def run_snli(
    data_dir: Path,
    results_dir: Path,
    methods: List[str],
    shots: List[int],
    base_seed: int,
    num_selections: int,
    max_new_tokens: int,
    model_name: str,
) -> None:
    test = load_jsonl(data_dir / "snli_test_100.jsonl")
    pool = load_jsonl(data_dir / "snli_fewshot_pool_200.jsonl")

    bundle: ModelBundle = load_model_and_tokenizer(model_name=model_name)

    # Use provided results_dir; create it if missing (no extra timestamp)
    base_dir = Path(results_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir # run_dir = base_dir / "snli"
    run_dir.mkdir(parents=True, exist_ok=True)

    # meta
    write_meta(
        run_dir,
        {
            "task": "snli",
            "model_name": bundle.model_name,
            "torch_dtype": str(bundle.torch_dtype).split(".")[-1],
            "base_seed": base_seed,
            "num_selections": num_selections,
            "shots": shots,
            "methods": methods,
            "max_new_tokens": max_new_tokens,
        },
    )

    # Prepare per-run summary CSV
    summary_rows: List[Dict] = []
    preds_records: List[Dict] = []

    # If 0-shot requested, handle separately
    if 0 in shots:
        for method in methods:
            print(f"\n[SNLI] Starting zero-shot | method={method}")
            # Preview prompt with the first test example
            if test:
                preview = build_snli_prompt(
                    test[0]["premise"], test[0]["hypothesis"], method, fewshot_examples=None
                )
                print("[SNLI] Prompt preview (shots=0, first example):\n" + preview + "\n")
            correct_flags = []
            for ex in tqdm(
                test,
                total=len(test),
                desc=f"SNLI | method={method} | shots=0"
            ):
                prompt = build_snli_prompt(
                    ex["premise"], ex["hypothesis"], method, fewshot_examples=None
                )
                out = model_generate(bundle, prompt, max_new_tokens=max_new_tokens)
                pred = extract_snli_answer(out)
                gt = ex["label_text"].lower()
                ok = (pred == gt)
                correct_flags.append(ok)
                preds_records.append(
                    {
                        "method": method,
                        "shots": 0,
                        "seed": None,
                        "premise": ex["premise"],
                        "hypothesis": ex["hypothesis"],
                        "ground_truth": gt,
                        "prediction": pred,
                        "raw_output": out,
                        "correct": ok,
                    }
                )
            acc = compute_accuracy(correct_flags)
            summary_rows.append(
                {
                    "task": "snli",
                    "method": method,
                    "shots": 0,
                    "seed": "-",
                    "num_examples": len(test),
                    "accuracy": acc,
                }
            )

    # Few-shot progressive subsets
    few_shots = [k for k in shots if k > 0]
    if few_shots:
        # Precompute pool tuples
        pool_tuples: List[Tuple[str, str, str]] = [
            (r["premise"], r["hypothesis"], r["label_text"]) for r in pool
        ]
        # Few-shot is restricted to direct-only per assignment
        for method in methods:
            if method != "direct":
                continue
            print(f"\n[SNLI] Starting few-shot | method={method}")
            for seed in seed_multiples(base_seed, num_selections):
                print(f"[SNLI] Seed={seed}")
                rng = np.random.default_rng(seed)
                order = rng.permutation(len(pool_tuples)).tolist()
                idx_by_shot = progressive_indices(order, few_shots)
                for k in sorted(set(few_shots)):
                    indices = idx_by_shot[k]
                    fs_examples = [pool_tuples[i] for i in indices]
                    # Save selection for transparency
                    sel_dir = run_dir / "selections"
                    sel_dir.mkdir(parents=True, exist_ok=True)
                    sel_path = sel_dir / f"selections_{method}_seed{seed}_k{k}.jsonl"
                    save_jsonl(
                        sel_path,
                        (
                            {
                                "pool_index": i,
                                "premise": pool_tuples[i][0],
                                "hypothesis": pool_tuples[i][1],
                                "label": pool_tuples[i][2],
                            }
                            for i in indices
                        ),
                    )
                    print(f"\n\n\n[SNLI] Shots={k} | saved selection -> {sel_path}")
                    # Prompt preview for the first test example
                    if test:
                        preview = build_snli_prompt(
                            test[0]["premise"],
                            test[0]["hypothesis"],
                            method,
                            fewshot_examples=fs_examples,
                        )
                        print(
                            f"[SNLI] Prompt preview (shots={k}, first example):\n" + preview + "\n"
                        )
                    correct_flags = []
                    for ex in tqdm(
                        test,
                        total=len(test),
                        desc=f"SNLI | method={method} | seed={seed} | shots={k}",
                    ):
                        prompt = build_snli_prompt(
                            ex["premise"], ex["hypothesis"], method, fewshot_examples=fs_examples
                        )
                        out = model_generate(
                            bundle, prompt, max_new_tokens=max_new_tokens
                        )
                        pred = extract_snli_answer(out)
                        gt = ex["label_text"].lower()
                        ok = (pred == gt)
                        correct_flags.append(ok)
                        preds_records.append(
                            {
                                "method": method,
                                "shots": k,
                                "seed": seed,
                                "premise": ex["premise"],
                                "hypothesis": ex["hypothesis"],
                                "ground_truth": gt,
                                "prediction": pred,
                                "raw_output": out,
                                "correct": ok,
                            }
                        )
                    acc = compute_accuracy(correct_flags)
                    summary_rows.append(
                        {
                            "task": "snli",
                            "method": method,
                            "shots": k,
                            "seed": seed,
                            "num_examples": len(test),
                            "accuracy": acc,
                        }
                    )

    # Write outputs
    save_jsonl(run_dir / "predictions.jsonl", preds_records)
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "method",
                "shots",
                "seed",
                "num_examples",
                "accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Aggregate max/min/avg over seeds for each (method, shots)
    agg: Dict[Tuple[str, int], List[float]] = {}
    for row in summary_rows:
        key = (row["method"], int(row["shots"]))
        agg.setdefault(key, []).append(float(row["accuracy"]))
    aggregate_json: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (method, k), accs in agg.items():
        stats = {
            "max": max(accs),
            "min": min(accs),
            "avg": sum(accs) / len(accs),
        }
        aggregate_json.setdefault(method, {})[str(k)] = stats
    from utils import save_json

    save_json(run_dir / "aggregate.json", aggregate_json)
    print("\n[SNLI] Aggregate results:")
    print(json.dumps(aggregate_json, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SNLI prompting experiments")
    p.add_argument(
        "--methods",
        type=str,
        default="direct,cot",
        help="Comma-separated methods: direct,cot",
    )
    p.add_argument(
        "--shots",
        type=str,
        default=",".join(str(x) for x in SHOT_COUNTS),
        help="Comma-separated shots. Use 0 for zero-shot.",
    )
    p.add_argument("--base-seed", type=int, default=1)
    p.add_argument("--num-selections", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--results-dir", type=str, default="./results")
    p.add_argument("--model-name", type=str, default=MODEL_NAME)
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent / "data"))
    return p.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    shots = [int(s.strip()) for s in args.shots.split(",") if s.strip()]
    run_snli(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        methods=methods,
        shots=shots,
        base_seed=args.base_seed,
        num_selections=args.num_selections,
        max_new_tokens=args.max_new_tokens,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
