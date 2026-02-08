"""
Data Collection for A4 Q2 (Sampling and Analysis)

Run this script to generate A4-Q2.json for Gradescope submission.
No grading is performed here - only data collection.

Usage:
    python test_a4_q2.py

This will create A4-Q2.json in the same directory.
Submit your coding files (analyze_traces.py, evaluate.py, generate_samples.py) 
and A4-Q2.json to Gradescope.
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
from unittest.mock import MagicMock, mock_open, patch

try:
    import torch
except ImportError:
    torch = MagicMock()

# Import student functions
try:
    from analyze_traces import (
        check_majority_failure,
        check_bon_failure,
        get_unique_preds_count
    )
    from generate_samples import (
        generate_samples_batched,
        score_samples_batched
    )
    # evaluate.py is usually a script, so we'll test its logic by mocking main
    import evaluate 
except ImportError as e:
    print(f"Error importing student modules: {e}")
    # We'll handle this in the collection functions

# Global dictionary to store test results
test_results: Dict[str, Any] = {}

# ==============================================================================
# TEST CASE DEFINITIONS
# ==============================================================================

ANALYZE_TEST_CASES = [
    {
        "name": "basic_majority_success",
        "preds": [5.0, 5.0, 3.0],
        "gold_val": 5.0,
        "has_any_correct": True,
        "expected_unique": 2,
        "expected_maj_fail": False,
        "expected_bon_fail": None, # Depends on scores
    },
    {
        "name": "majority_failure",
        "preds": [3.0, 3.0, 5.0],
        "gold_val": 5.0,
        "has_any_correct": True,
        "expected_unique": 2,
        "expected_maj_fail": True,
    },
    {
        "name": "no_correct_samples",
        "preds": [3.0, 2.0, 1.0],
        "gold_val": 5.0,
        "has_any_correct": False,
        "expected_unique": 3,
        "expected_maj_fail": False,
    },
    {
        "name": "majority_tie",
        "preds": [3.0, 5.0],
        "gold_val": 5.0,
        "has_any_correct": True,
        "expected_unique": 2,
        "expected_maj_fail": True, # Counter.most_common(1) tie-breaks by first appearance
    },
    {
        "name": "all_nones",
        "preds": [None, None],
        "gold_val": 5.0,
        "has_any_correct": False,
        "expected_unique": 0,
        "expected_maj_fail": False,
    }
]

BON_TEST_CASES = [
    {
        "name": "bon_success",
        "scores": [0.1, 0.9, 0.2],
        "preds": [3.0, 5.0, 3.0],
        "gold_val": 5.0,
        "has_any_correct": True,
        "expected_fail": False,
    },
    {
        "name": "bon_failure",
        "scores": [0.9, 0.1, 0.2],
        "preds": [3.0, 5.0, 3.0],
        "gold_val": 5.0,
        "has_any_correct": True,
        "expected_fail": True,
    }
]

EVALUATE_DUMMY_DATA = [
    {
        "gold": "#### 10",
        "samples": [
            {"text": "The answer is 10. #### 10", "score": 0.1},
            {"text": "The answer is 5. #### 5", "score": 0.9},
        ],
        "expected_majority": {1: True, 2: True},
        "expected_bon": {1: True, 2: False}
    },
    {
        "gold": "#### 20",
        "samples": [
            {"text": "The answer is 20. #### 20", "score": 0.8},
            {"text": "The answer is 20. #### 20", "score": 0.2},
        ],
        "expected_majority": {1: True, 2: True},
        "expected_bon": {1: True, 2: True}
    }
]

# ==============================================================================
# DATA COLLECTION FUNCTIONS
# ==============================================================================

def collect_analyze_traces_data():
    """Collect data from analyze_traces.py implementation."""
    print("\nRunning analyze_traces.py tests...")
    result = {"function_tests": [], "error": None}
    
    try:
        # Test get_unique_preds_count
        for tc in ANALYZE_TEST_CASES:
            tc_res = {
                "name": f"unique_{tc['name']}",
                "function": "get_unique_preds_count",
                "input": tc["preds"]
            }
            try:
                tc_res["output"] = get_unique_preds_count(tc["preds"])
                tc_res["reference"] = tc["expected_unique"]
                print(f"  [PASS] get_unique_preds_count: {tc['name']}")
            except Exception as e:
                tc_res["error"] = str(e)
                print(f"  [FAIL] get_unique_preds_count: {tc['name']} - Error: {e}")
            result["function_tests"].append(tc_res)
            
        # Test check_majority_failure
        for tc in ANALYZE_TEST_CASES:
            tc_res = {
                "name": f"maj_{tc['name']}", 
                "function": "check_majority_failure",
                "inputs": {
                    "preds": tc["preds"], 
                    "gold": tc["gold_val"], 
                    "has_any": tc["has_any_correct"]
                }
            }
            try:
                tc_res["output"] = check_majority_failure(
                    tc["preds"], tc["gold_val"], tc["has_any_correct"]
                )
                tc_res["reference"] = tc["expected_maj_fail"]
                print(f"  [PASS] check_majority_failure: {tc['name']}")
            except Exception as e:
                tc_res["error"] = str(e)
                print(f"  [FAIL] check_majority_failure: {tc['name']} - Error: {e}")
            result["function_tests"].append(tc_res)
            
        # Test check_bon_failure
        for tc in BON_TEST_CASES:
            tc_res = {
                "name": f"bon_{tc['name']}",
                "function": "check_bon_failure",
                "inputs": {
                    "scores": tc["scores"],
                    "preds": tc["preds"],
                    "gold": tc["gold_val"],
                    "has_any": tc["has_any_correct"]
                }
            }
            try:
                tc_res["output"] = check_bon_failure(
                    tc["scores"], tc["preds"], tc["gold_val"], tc["has_any_correct"]
                )
                tc_res["reference"] = tc["expected_fail"]
                print(f"  [PASS] check_bon_failure: {tc['name']}")
            except Exception as e:
                tc_res["error"] = str(e)
                print(f"  [FAIL] check_bon_failure: {tc['name']} - Error: {e}")
            result["function_tests"].append(tc_res)
            
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"Critical error in collect_analyze_traces_data: {e}")
        
    test_results["analyze_traces"] = result

def collect_evaluate_data():
    """Collect data from evaluate.py implementation."""
    print("\nRunning evaluate.py tests...")
    result = {"function_tests": [], "error": None}
    
    try:
        # 1. Direct test for evaluate_example
        for i, tc in enumerate(EVALUATE_DUMMY_DATA):
            n_values = [1, 2]
            tc_res = {
                "name": f"evaluate_example_{i}",
                "function": "evaluate_example",
                "inputs": {
                    "gold": tc["gold"],
                    "n_values": n_values
                }
            }
            try:
                from evaluate import evaluate_example
                maj_res, bon_res = evaluate_example(tc, n_values)
                tc_res["output"] = {"majority": maj_res, "bon": bon_res}
                tc_res["reference"] = {
                    "majority": tc["expected_majority"],
                    "bon": tc["expected_bon"]
                }
                print(f"  [PASS] evaluate_example_{i}")
            except Exception as e:
                tc_res["error"] = str(e)
                print(f"  [FAIL] evaluate_example_{i} - Error: {e}")
            result["function_tests"].append(tc_res)

        # 2. Test the main script logic by mocking the components
        from collections import Counter
        from utils import eq_num, extract_gsm_answer, parse_gold_answer_field
        
        n_values = [1, 2]
        metrics = {n: {"majority": 0, "bon": 0} for n in n_values}
        
        # This is the logic the student is supposed to implement
        # We will try to run their code if they refactored it, or we'll look for 
        # a way to trigger it. 
        # For simplicity in this test script, we'll just check if evaluate.py 
        # can be imported and if we can mock its inputs.
        
        with patch("evaluate.load_jsonl", return_value=EVALUATE_DUMMY_DATA), \
             patch("evaluate.Path.mkdir"), \
             patch("evaluate.argparse.ArgumentParser.parse_args",
                   return_value=MagicMock(samples_path="dummy.jsonl", output_dir="dummy_out")), \
             patch("builtins.open", mock_open()), \
             patch("evaluate.csv.DictWriter"):
            
            print("  Testing evaluate.main()...")
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    if 'evaluate' in globals():
                        evaluate.main()
                        result["success"] = True
                    else:
                        raise NameError("Module 'evaluate' was not imported successfully")
                    # Note: We can't print [PASS] here because stdout is redirected
                except Exception as e:
                    result["error"] = str(e)
                    result["traceback"] = traceback.format_exc()
            
            if result.get("success"):
                print("  [PASS] evaluate.main()")
            else:
                print(f"  [FAIL] evaluate.main() - Error: {result.get('error')}")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        
    test_results["evaluate"] = result


def collect_output_validation_data():
    """Validate the format and shape of the generated samples file."""
    print("\nValidating output file format...")
    result = {
        "generate_samples_batched": "fail",
        "score_samples_batched": "fail",
        "error": None
    }
    # Path to the specific file mentioned by the user
    target_file = Path(__file__).parent / "results" / "sampling" / "scored_samples_gsm_symbolic_train_4500_student-gemma-3-1b-it-maxlen1024-epochs1-lr2e-05-effbsz8-dataseed0.jsonl"
    
    if not target_file.exists():
        result["error"] = f"File not found: {target_file}"
        test_results["output_validation"] = result
        print(f"  [FAIL] Output file not found at {target_file}")
        return

    print(f"  Validating: {target_file.name}")
    try:
        shape_correct = True
        scores_correct = True
        valid_entries = 0
        
        with open(target_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                samples = data.get("samples", [])
                
                # 1. Check shape (n=8)
                if len(samples) != 8:
                    shape_correct = False
                    result["error"] = f"Incorrect shape: expected 8 samples, found {len(samples)} at index {data.get('index')}"
                    print(f"  [FAIL] generate_samples_batched: {result['error']}")
                    break
                
                # 2. Check scores (present and type float)
                for i, s in enumerate(samples):
                    score = s.get("score")
                    if score is None or not isinstance(score, (float, int)):
                        scores_correct = False
                        result["error"] = f"Incorrect score format at index {data.get('index')}, sample {i}"
                        print(f"  [FAIL] score_samples_batched: {result['error']}")
                        break
                    
                    if not isinstance(s.get("text"), str):
                        # Still good to check text exists
                        pass 

                if not scores_correct: break
                valid_entries += 1
        
        if shape_correct:
            result["generate_samples_batched"] = "pass"
            print(f"  [PASS] generate_samples_batched (shape correct)")
        
        if scores_correct:
            result["score_samples_batched"] = "pass"
            print(f"  [PASS] score_samples_batched (scores present and float)")

        if not result["error"]:
            print(f"  [PASS] Validated {valid_entries} entries.")
            
    except Exception as e:
        result["error"] = str(e)
        print(f"  [ERROR] Validation failed: {e}")
        
    test_results["output_validation"] = result

def main():
    """Main entry point for data collection."""
    print("Collecting A4 Q2 Data...")
    print("=" * 60)

    # Validating generate_samples_batched and score_samples_batched
    collect_output_validation_data()

    collect_evaluate_data()
    
    collect_analyze_traces_data()
    
    # Save results
    output_path = Path(__file__).parent / "A4-Q2.json"
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_path}")
    print("=" * 60)
    print("\nSubmit analyze_traces.py, evaluate.py, generate_samples.py, and A4-Q2.json to Gradescope")

if __name__ == "__main__":
    main()
