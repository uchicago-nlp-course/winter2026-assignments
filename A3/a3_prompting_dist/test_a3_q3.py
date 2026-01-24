"""
Data Collection for A3 Q3 (Prompting)

Run this script to generate A3-Q3.json for Gradescope submission.
No grading is performed here - only data collection.

This version stores both student prompts and reference prompts for Jaccard similarity comparison.

Usage:
    python test_a3_q3.py

This will create A3-Q3.json in the same directory.
Submit both utils.py and A3-Q3.json to Gradescope.
"""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock
import traceback

from utils import apply_chat_template, build_snli_prompt, build_gsm_prompt


# Global dictionary to store test results
test_results: Dict[str, Any] = {}


# ==============================================================================
# REFERENCE PROMPTS (hardcoded for grading comparison)
# ==============================================================================

REFERENCE_PROMPTS = {
    "snli_direct_zero_shot": """You are given a premise and a hypothesis. Classify their relationship as one of: entailment, neutral, or contradiction. Provide only the final label.

Premise: A man is walking in the park.
Hypothesis: A person is outside.
Answer:""",

    "snli_cot_zero_shot": """You are given a premise and a hypothesis. Classify their relationship as one of: entailment, neutral, or contradiction, after thinking step by step.

Premise: A man is walking in the park.
Hypothesis: A person is outside.
Answer: Let's think step by step.""",

    "snli_direct_few_shot": """You are given a premise and a hypothesis. Classify their relationship as one of: entailment, neutral, or contradiction. Provide only the final label.

Premise: A dog is running.
Hypothesis: An animal is moving.
Answer: entailment

Premise: The sky is blue.
Hypothesis: It is raining.
Answer: contradiction

Premise: A man is walking in the park.
Hypothesis: A person is outside.
Answer:""",

    "gsm_direct_zero_shot": """Solve the following math problem. Provide only the final numeric answer.

Problem: John has 5 apples. He gives 2 to Mary. How many apples does John have now?
Answer:""",

    "gsm_cot_zero_shot": """Solve the following math problem step by step, and provide the final numeric answer.

Problem: John has 5 apples. He gives 2 to Mary. How many apples does John have now?
Answer: Let's think step by step.""",

    "gsm_direct_few_shot": """Solve the following math problem. Provide only the final numeric answer.

Problem: What is 2 + 2?
Answer: 4

Problem: What is 10 - 3?
Answer: 7

Problem: John has 5 apples. He gives 2 to Mary. How many apples does John have now?
Answer:""",
}


def collect_apply_chat_template_data():
    """Collect data about apply_chat_template implementation."""
    result = {
        "function_name": "apply_chat_template",
        "test_cases": [],
        "error": None,
        "traceback": None
    }
    
    try:
        # Create a mock tokenizer with chat template support
        mock_tokenizer = Mock()
        
        # Test case 1: Simple prompt
        test_prompt = "What is the capital of France?"
        expected_output = "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n"
        mock_tokenizer.apply_chat_template.return_value = expected_output
        
        output = apply_chat_template(mock_tokenizer, test_prompt)
        
        # Collect call information
        call_args = mock_tokenizer.apply_chat_template.call_args
        messages = call_args[0][0] if call_args else None
        
        result["test_cases"].append({
            "name": "simple_prompt",
            "messages_count": len(messages) if messages else 0,
            "has_user_role": messages[0]["role"] == "user" if messages else False,
            "content_matches": messages[0]["content"] == test_prompt if messages else False,
            "tokenize_false": call_args[1].get("tokenize") == False if call_args else False,
            "add_generation_prompt": call_args[1].get("add_generation_prompt") == True if call_args else False,
        })

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    test_results["apply_chat_template"] = result


def collect_build_snli_prompt_data():
    """Collect data about build_snli_prompt implementation."""
    result = {
        "function_name": "build_snli_prompt",
        "test_cases": [],
        "error": None,
        "traceback": None
    }
    
    try:
        premise = "A man is walking in the park."
        hypothesis = "A person is outside."
        
        # Test case 1: Direct zero-shot
        student_prompt = build_snli_prompt(
            premise=premise,
            hypothesis=hypothesis,
            method="direct",
            fewshot_examples=None
        )
        reference_prompt = REFERENCE_PROMPTS["snli_direct_zero_shot"]

        result["test_cases"].append({
            "name": "direct_zero_shot",
            "student_prompt": student_prompt,
            "reference_prompt": reference_prompt,
            "prompt_length": len(student_prompt),
        })
        
        # Test case 2: CoT zero-shot
        student_prompt = build_snli_prompt(
            premise=premise,
            hypothesis=hypothesis,
            method="cot",
            fewshot_examples=None
        )
        reference_prompt = REFERENCE_PROMPTS["snli_cot_zero_shot"]

        result["test_cases"].append({
            "name": "cot_zero_shot",
            "student_prompt": student_prompt,
            "reference_prompt": reference_prompt,
            "prompt_length": len(student_prompt),
        })
        
        # Test case 3: Direct few-shot
        fewshot_examples = [
            ("A dog is running.", "An animal is moving.", "entailment"),
            ("The sky is blue.", "It is raining.", "contradiction"),
        ]
        
        student_prompt = build_snli_prompt(
            premise=premise,
            hypothesis=hypothesis,
            method="direct",
            fewshot_examples=fewshot_examples
        )
        reference_prompt = REFERENCE_PROMPTS["snli_direct_few_shot"]

        result["test_cases"].append({
            "name": "direct_few_shot",
            "student_prompt": student_prompt,
            "reference_prompt": reference_prompt,
            "prompt_length": len(student_prompt),
        })

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    test_results["build_snli_prompt"] = result


def collect_build_gsm_prompt_data():
    """Collect data about build_gsm_prompt implementation."""
    result = {
        "function_name": "build_gsm_prompt",
        "test_cases": [],
        "error": None,
        "traceback": None
    }
    
    try:
        question = "John has 5 apples. He gives 2 to Mary. How many apples does John have now?"
        
        # Test case 1: Direct zero-shot
        student_prompt = build_gsm_prompt(
            question=question,
            method="direct",
            fewshot_examples=None
        )
        reference_prompt = REFERENCE_PROMPTS["gsm_direct_zero_shot"]

        result["test_cases"].append({
            "name": "direct_zero_shot",
            "student_prompt": student_prompt,
            "reference_prompt": reference_prompt,
            "prompt_length": len(student_prompt),
        })

        # Test case 2: CoT zero-shot
        student_prompt = build_gsm_prompt(
            question=question,
            method="cot",
            fewshot_examples=None
        )
        reference_prompt = REFERENCE_PROMPTS["gsm_cot_zero_shot"]

        result["test_cases"].append({
            "name": "cot_zero_shot",
            "student_prompt": student_prompt,
            "reference_prompt": reference_prompt,
            "prompt_length": len(student_prompt),
        })
        
        # Test case 3: Direct few-shot
        fewshot_examples_direct = [
            ("What is 2 + 2?", "4"),
            ("What is 10 - 3?", "7"),
        ]
        
        student_prompt = build_gsm_prompt(
            question=question,
            method="direct",
            fewshot_examples=fewshot_examples_direct
        )
        reference_prompt = REFERENCE_PROMPTS["gsm_direct_few_shot"]

        result["test_cases"].append({
            "name": "direct_few_shot",
            "student_prompt": student_prompt,
            "reference_prompt": reference_prompt,
            "prompt_length": len(student_prompt),
        })

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    test_results["build_gsm_prompt"] = result


def main():
    """Main entry point for data collection."""
    print("Collecting A3 Q3 Data...")
    print("="*60)
    
    collect_apply_chat_template_data()
    status = "✓" if test_results['apply_chat_template']['error'] is None else "✗"
    print(f"apply_chat_template: {status}")
    
    collect_build_snli_prompt_data()
    status = "✓" if test_results['build_snli_prompt']['error'] is None else "✗"
    print(f"build_snli_prompt: {status}")
    
    collect_build_gsm_prompt_data()
    status = "✓" if test_results['build_gsm_prompt']['error'] is None else "✗"
    print(f"build_gsm_prompt: {status}")
    
    # Save results
    output_path = Path(__file__).parent / "A3-Q3.json"
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("="*60)
    print(f"Results saved to: {output_path}")
    print("="*60)
    print("\nSubmit both utils.py and A3-Q3.json to Gradescope")


if __name__ == "__main__":
    main()
