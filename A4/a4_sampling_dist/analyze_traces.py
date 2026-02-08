"""
Script for A4 Part 2: Analyze reasoning traces and sampling behavior.

This script performs the following analyses:
1. Distribution of responses (Unique answers).
2. Majority Vote failures: Correct answer generated but not chosen.
3. Best-of-N (BoN) failures: Correct answer generated but not chosen by RM.

Arguments:
    --samples-path: Path to the JSONL file containing scored samples.
    --plots-dir: Directory to save generated plots.
    --output-path: Path to save the summary statistics JSON.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
from utils import (
    extract_gsm_answer,
    parse_gold_answer_field,
    eq_num,
    load_jsonl,
)

def check_majority_failure(preds: list[float], gold_val: float, has_any_correct: bool) -> bool:
    """Returns True if at least one sample is correct but the majority prediction is not."""
    # TODO: Implement Majority Vote failure check
    # 1. Identify the most common prediction (use Counter)
    # 2. If at least one sample is correct (has_any_correct is True) 
    #    but the majority prediction is NOT correct, return True.
    return False

def check_bon_failure(scores: list[float], preds: list[float], gold_val: float, has_any_correct: bool) -> bool:
    """Returns True if at least one sample is correct but the Best-of-N prediction is not."""
    # TODO: Implement Best-of-N failure check
    # 1. Identify the sample with the highest score (np.argmax)
    # 2. If at least one sample is correct (has_any_correct is True)
    #    but the best-scored prediction is NOT correct, return True.
    return False

def get_unique_preds_count(preds: list[float]) -> int:
    """Returns the number of unique numeric answers in the predictions."""
    # TODO: Implement unique prediction counting
    # 1. Identify unique predictions in 'preds' (ignore None)
    # 2. Return the count of unique predictions
    return 0

def analyze_failure_modes(data: list[dict], output_path: str = "results/failure_modes.txt"):
    """
    Finds and analyzes failure modes for Majority Voting and Best-of-N.
    Specifically looks for cases with 3 unique answers and saves to a .txt file.
    """
    majority_failures = []
    bon_failures = []
    
    for ex in data:
        gold_val = parse_gold_answer_field(ex["gold"])
        samples = ex["samples"]
        preds = [extract_gsm_answer(s["text"]) for s in samples]
        scores = [s["score"] for s in samples]
        valid_preds = [p for p in preds if p is not None]
        
        if not valid_preds: continue
        
        unique_preds = set(valid_preds)
        num_unique = len(unique_preds)
        is_correct = [eq_num(p, gold_val) for p in preds]
        has_any_correct = any(is_correct)
        
        # Majority Vote check
        counts = Counter(valid_preds)
        majority_val = counts.most_common(1)[0][0]
        is_maj_fail = has_any_correct and not eq_num(majority_val, gold_val)
        
        # Best-of-N check
        best_idx = np.argmax(scores)
        best_val = preds[best_idx]
        is_bon_fail = has_any_correct and not eq_num(best_val, gold_val)
        
        info = {
            "question": ex["question"],
            "gold": gold_val,
            "num_unique": num_unique,
            "majority_val": majority_val,
            "best_val": best_val,
            "counts": dict(counts),
            "samples": [s["text"] for s in samples],
            "scores": scores,
            "preds": preds
        }
        
        if num_unique == 3:
            if is_maj_fail: majority_failures.append(info)
            if is_bon_fail: bon_failures.append(info)

    print(f"Saving failure mode analysis to {output_path}...")
    with open(output_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("PART 2: FAILURE MODE ANALYSIS (3 UNIQUE ANSWERS)\n")
        f.write("="*80 + "\n\n")
        
        f.write("INSTRUCTIONS:\n")
        f.write("Select 3 cases from the Majority Voting Failures section and 3 cases from the Best-of-N section.\n")
        f.write("Analyze the failure modes for each selected case.\n\n")

        f.write("1. MAJORITY VOTING FAILURES\n")
        for i, case in enumerate(majority_failures):
            f.write(f"\nCase {i+1}:\n")
            f.write(f"Question: {case['question']}\n")
            f.write(f"Correct Answer (Gold): {case['gold']}\n")
            f.write(f"Majority Answer: {case['majority_val']}\n")
            f.write(f"Answer Distribution: {case['counts']}\n")
            f.write("\nSamples:\n")
            seen_preds = set()
            for s_idx, p in enumerate(case['preds']):
                if p in case['counts'] and p not in seen_preds:
                    f.write(f"--- Sample for prediction {p} ---\n")
                    f.write(f"{case['samples'][s_idx]}\n\n")
                    seen_preds.add(p)
            f.write("-" * 40 + "\n")

        f.write("\n" + "="*80 + "\n")
        f.write("2. BEST-OF-N (BoN) FAILURES\n")
        f.write("="*80 + "\n")
        
        for i, case in enumerate(bon_failures):
            f.write(f"\nCase {i+1}:\n")
            f.write(f"Question: {case['question']}\n")
            f.write(f"Correct Answer (Gold): {case['gold']}\n")
            f.write(f"BoN Selected Answer: {case['best_val']}\n")
            f.write(f"Answer Distribution: {case['counts']}\n")
            f.write("\nSamples:\n")
            
            # Show Gold sample and BoN Selected sample
            found_gold = False
            for s_idx, p in enumerate(case['preds']):
                if eq_num(p, case['gold']) and not found_gold:
                    f.write(f"--- Sample for Gold ({case['gold']}) [Score: {case['scores'][s_idx]}] ---\n")
                    f.write(f"{case['samples'][s_idx]}\n\n")
                    found_gold = True
            best_idx = np.argmax(case['scores'])
            f.write(f"--- Sample for BoN Selected ({case['best_val']}) [Score: {case['scores'][best_idx]}] ---\n")
            f.write(f"{case['samples'][best_idx]}\n\n")
            f.write("-" * 40 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze reasoning traces from scored samples."
    )
    parser.add_argument(
        "--samples-path", 
        type=str, 
        required=True,
        help="Path to the JSONL file with scored samples."
    )
    parser.add_argument(
        "--plots-dir", 
        type=str, 
        default="plots",
        help="Directory to save generated plots."
    )
    parser.add_argument(
        "--failure-modes-path",
        type=str,
        default="results/failure_modes.txt",
        help="Path to save failure mode analysis for Majority Voting and BoN."
    )
    parser.add_argument(
        "--plot-suffix",
        type=str,
        default="",
        help="Suffix to add to the plot filename."
    )
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading samples from {args.samples_path}...")
    data = load_jsonl(args.samples_path)

    # Statistics containers
    stats = {
        "total_questions": len(data),
        "majority_failures": 0,  # Correct exists but not chosen by Majority
        "bon_failures": 0,       # Correct exists but not chosen by BoN
        "any_correct": 0,        # Questions where at least one sample is correct
        "unique_answers_per_q": [],
        "correct_counts_per_q": [],
    }

    # Set font size for plots
    plt.rcParams.update({'font.size': 14})

    print("Analyzing reasoning traces...")
    for i, ex in enumerate(data):
        gold_text = ex["gold"]
        gold_val = parse_gold_answer_field(gold_text)
        
        samples = ex["samples"]
        scores = [s["score"] for s in samples]
        preds = [extract_gsm_answer(s["text"]) for s in samples]
        
        # 1. Answer distribution
        unique_count = get_unique_preds_count(preds)
        stats["unique_answers_per_q"].append(unique_count)
        
        is_correct = [eq_num(p, gold_val) for p in preds]
        correct_count = sum(is_correct)
        stats["correct_counts_per_q"].append(correct_count)
        
        has_any_correct = any(is_correct)
        if has_any_correct:
            stats["any_correct"] += 1
            
        # 2. Majority Vote check
        if check_majority_failure(preds, gold_val, has_any_correct):
            stats["majority_failures"] += 1
                
        # 3. Best-of-N check
        if check_bon_failure(scores, preds, gold_val, has_any_correct):
            stats["bon_failures"] += 1

        # Output partial results to console
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(data)} questions...")

    # Final summary calculations
    summary = {
        "total_questions": stats["total_questions"],
        "any_correct_rate": stats["any_correct"] / stats["total_questions"],
        "majority_failure_rate": (stats["majority_failures"] / stats["any_correct"] 
                                 if stats["any_correct"] > 0 else 0),
        "bon_failure_rate": (stats["bon_failures"] / stats["any_correct"] 
                             if stats["any_correct"] > 0 else 0),
        "avg_unique_answers": np.mean(stats["unique_answers_per_q"]),
    }

    print("\n--- Analysis Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.hist(stats["unique_answers_per_q"], bins=np.arange(0, 10) - 0.5, 
             rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title("Distribution of Unique Answers per Question")
    plt.xlabel("Number of Unique Answers")
    plt.ylabel("Frequency (Questions)")
    plt.xticks(range(0, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_name = "unique_answers_dist"
    if args.plot_suffix:
        plot_name += f"_{args.plot_suffix}"
    
    plt.savefig(plots_dir / f"{plot_name}.png", bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {plots_dir}/")

    # Perform failure mode analysis
    analyze_failure_modes(data, args.failure_modes_path)

if __name__ == "__main__":
    main()
