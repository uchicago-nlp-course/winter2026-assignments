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
import torch

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
        "name": "with_nones",
        "preds": [5.0, None, 5.0, 3.0, None],
        "gold_val": 5.0,
        "has_any_correct": True,
        "expected_unique": 2,
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
        ]
    },
    {
        "gold": "#### 20",
        "samples": [
            {"text": "The answer is 20. #### 20", "score": 0.8},
            {"text": "The answer is 20. #### 20", "score": 0.2},
        ]
    }
]

# ==============================================================================
# DATA COLLECTION FUNCTIONS
# ==============================================================================

def collect_analyze_traces_data():
    """Collect data from analyze_traces.py implementation."""
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
            except Exception as e:
                tc_res["error"] = str(e)
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
            except Exception as e:
                tc_res["error"] = str(e)
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
            except Exception as e:
                tc_res["error"] = str(e)
            result["function_tests"].append(tc_res)
            
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        
    test_results["analyze_traces"] = result

def collect_evaluate_data():
    """Collect data from evaluate.py implementation."""
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
            except Exception as e:
                tc_res["error"] = str(e)
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
            
            # If we call evaluate.main(), it should run the TODO code.
            # We want to see what 'metrics' looks like after it runs.
            # However, 'metrics' is local to main(). 
            # So we might need to ask students to put their logic in a function 
            # or we just rely on the analyze_traces tests which cover the core logic.
            
            # Let's try to run it and capture the print outputs as a proxy
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    evaluate.main()
                    output = f.getvalue()
                    result["main_output"] = output
                    result["success"] = True
                except Exception as e:
                    result["error"] = str(e)
                    result["traceback"] = traceback.format_exc()

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        
    test_results["evaluate"] = result

def collect_generate_samples_data():
    """Collect data from generate_samples.py implementation."""
    result = {"function_tests": [], "error": None}
    
    try:
        # Mocking tokenizer and model for generate_samples_batched
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "PRE "
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]), 
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.decode.return_value = " generated text "
        
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        bundle = MagicMock()
        bundle.tokenizer = mock_tokenizer
        bundle.model = mock_model
        bundle.device = "cpu"
        
        questions = ["Q1"]
        n = 1
        max_tokens = 10
        
        # Test generate_samples_batched
        tc_gen = {
            "name": "generate_samples_batched_basic",
            "function": "generate_samples_batched"
        }
        try:
            output = generate_samples_batched(bundle, questions, n, max_tokens)
            tc_gen["output_structure"] = str(type(output))
            tc_gen["output_len"] = len(output)
            tc_gen["inner_len"] = len(output[0]) if output else 0
            
            # Check if chat template was called correctly
            call_args = mock_tokenizer.apply_chat_template.call_args
            if call_args:
                tc_gen["template_called"] = True
                # Check for ANSWER_PREFIX in prompt_texts (internal to student code)
                # We can't easily check internal variables, so we check if 
                # tokenizer was called with something that looks like a prompt.
        except Exception as e:
            tc_gen["error"] = str(e)
        result["function_tests"].append(tc_gen)
        
        # Test score_samples_batched
        rm_tokenizer = MagicMock()
        rm_tokenizer.bos_token = "<s>"
        rm_tokenizer.apply_chat_template.return_value = "<s>CONV"
        rm_tokenizer.return_value = {"input_ids": torch.tensor([[1]])}
        
        rm_model = MagicMock()
        rm_model.return_value = MagicMock(logits=torch.tensor([[5.0]]))
        
        rm_bundle = MagicMock()
        rm_bundle.tokenizer = rm_tokenizer
        rm_bundle.model = rm_model
        rm_bundle.device = "cpu"
        
        tc_score = {
            "name": "score_samples_batched_basic",
            "function": "score_samples_batched"
        }
        try:
            scores = score_samples_batched(rm_bundle, ["Q1"], [["R1"]])
            tc_score["output"] = scores
            tc_score["output_type"] = str(type(scores))
        except Exception as e:
            tc_score["error"] = str(e)
        result["function_tests"].append(tc_score)
        
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        
    test_results["generate_samples"] = result

def main():
    """Main entry point for data collection."""
    print("Collecting A4 Q2 Data...")
    print("=" * 60)
    
    collect_analyze_traces_data()
    analyze_err = test_results["analyze_traces"].get("error")
    print(f"analyze_traces.py: {'FAIL' if analyze_err else 'pass'}")
    if analyze_err: print(f"  Error: {analyze_err}")
    
    collect_evaluate_data()
    eval_err = test_results["evaluate"].get("error")
    print(f"evaluate.py: {'FAIL' if eval_err else 'pass'}")
    if eval_err: print(f"  Error: {eval_err}")
    
    collect_generate_samples_data()
    gen_err = test_results["generate_samples"].get("error")
    print(f"generate_samples.py: {'FAIL' if gen_err else 'pass'}")
    if gen_err: print(f"  Error: {gen_err}")
    
    # Save results
    output_path = Path(__file__).parent / "A4-Q2.json"
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)
        
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print("=" * 60)
    print("\nSubmit analyze_traces.py, evaluate.py, generate_samples.py, and A4-Q2.json to Gradescope")

if __name__ == "__main__":
    main()
