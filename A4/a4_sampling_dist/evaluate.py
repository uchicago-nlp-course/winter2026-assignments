"""
Script for A4 Part 2: Evaluate Majority Voting and Best-of-N accuracies from scored samples.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from utils import (
    eq_num,
    extract_gsm_answer,
    load_jsonl,
    parse_gold_answer_field,
)

def evaluate_example(ex, n_values):
    """
    Evaluate Majority Voting and Best-of-N for a single example.
    
    Args:
        ex (see in sampling/scored_samples_<model_name>.jsonl): A dictionary containing "gold" (string) and "samples" (list of dicts).
            Each sample dict has "text" (string) and "score" (float).
        n_values: List of integers (n) for which to calculate metrics.
        
    Returns:
        A tuple of (majority_results, bon_results), where each is a 
        dictionary mapping n to a boolean indicating if the prediction 
        for that n was correct.
    """
    # TODO: Implement evaluation for this question
    # 1. Parse the gold answer from ex["gold"] using parse_gold_answer_field.
    # 2. Extract numeric answers from each sample in ex["samples"]
    #    - Each sample is a dict with "text" and "score".
    #    - Use extract_gsm_answer to get the numeric prediction from the text.
    # 3. For each n in n_values, calculate Majority Voting and Best-of-N:
    #    - Majority Voting: Pick the most frequent numeric answer among the first n.
    #    - Best-of-N: Pick the numeric answer associated with the highest reward 
    #      score among the first n.
    # 4. Compare the chosen predictions with the gold answer (use eq_num)
    #    to determine if they are correct.
    
    majority_results = {n: False for n in n_values}
    bon_results = {n: False for n in n_values}
    
    # Your code here
    
    return majority_results, bon_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples_data = load_jsonl(args.samples_path)

    n_values = [1, 2, 4, 8]
    metrics = {n: {"majority": 0, "bon": 0} for n in n_values}

    for ex in samples_data:
        maj_results, bon_results = evaluate_example(ex, n_values)
        for n in n_values:
            if maj_results[n]:
                metrics[n]["majority"] += 1
            if bon_results[n]:
                metrics[n]["bon"] += 1

    # Save summary CSV in a wide format
    summary = []
    for n in n_values:
        summary.append({
            "n": n,
            "majority": metrics[n]["majority"] / len(samples_data),
            "bon": metrics[n]["bon"] / len(samples_data)
        })
    
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "majority", "bon"])
        writer.writeheader()
        writer.writerows(summary)
    
    print(f"\nEvaluation complete. Summary saved to {summary_path}")
    for row in summary:
        print(f"n={row['n']}, majority={row['majority']:.4f}, bon={row['bon']:.4f}")

if __name__ == "__main__":
    main()
