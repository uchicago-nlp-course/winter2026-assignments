from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple


"""
Utilities copied from `../../A3/a3_prompting_solutions/utils.py`.

We intentionally duplicate these here so A4 can be self-contained and keep
training/evaluation prompt + grading conventions consistent with A3.
"""


# Prompt builders
def build_gsm_prompt(
    question: str,
    method: str,
    fewshot_examples: Optional[List[Tuple]] = None,
) -> str:
    if method == "cot":
        header = (
            "Solve the following math problem step by step, and provide the final numeric answer."
        )
    else:
        header = "Solve the following math problem. Provide only the final numeric answer."

    demo = ""
    if fewshot_examples:
        parts: List[str] = []
        if method == "cot":
            # Expect tuples of (question, solution_text, final_answer_str)
            for tup in fewshot_examples:
                if len(tup) != 3:
                    raise ValueError(
                        "For GSM CoT few-shot, examples must be (question, solution, final_answer)."
                    )
                q, solution, final_ans = tup  # type: ignore[misc]
                parts.append(
                    f"Problem: {q}\nAnswer: Let's think step by step. {solution}. Final answer: {final_ans}."
                )
        else:
            # Direct: expect tuples of (question, final_answer_str)
            for tup in fewshot_examples:
                if len(tup) != 2:
                    raise ValueError(
                        "For GSM direct few-shot, examples must be (question, final_answer)."
                    )
                q, final_ans = tup  # type: ignore[misc]
                parts.append(f"Problem: {q}\nAnswer: {final_ans}")
        demo = "\n\n".join(parts) + "\n\n"

    if method == "cot":
        query = f"Problem: {question}\nAnswer: Let's think step by step."
    else:
        query = f"Problem: {question}\nAnswer:"

    return f"{header}\n\n{demo}{query}"


_NUM_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)")


def split_gsm_solution_and_final(answer_text: str) -> Tuple[str, str]:
    """
    Split a GSM answer field into (solution_text, final_answer_str).

    Uses the '####' delimiter as in the GSM files.
    """
    if answer_text is None:
        return "", ""
    s = str(answer_text).strip()
    marker = "####"
    idx = s.rfind(marker)
    if idx != -1:
        sol = s[:idx].strip()
        tail = s[idx + len(marker) :].strip()
        nums = _NUM_RE.findall(tail)
        final_str = nums[-1] if nums else tail
        return sol, final_str
    raise ValueError("No '####' marker found in GSM answer text for splitting.")


# Extraction/Grading
def parse_gold_answer_field(answer_field: str) -> Optional[float]:
    if answer_field is None:
        return None
    s = str(answer_field).replace(",", "")
    nums = _NUM_RE.findall(s)
    if not nums:
        raise ValueError("No numeric answer found in answer field.")
    try:
        return float(nums[-1])
    except ValueError as e:
        raise ValueError(f"Failed to parse numeric answer from answer field: {e}") from e


def extract_gsm_answer(model_resp: str) -> Optional[float]:
    s = (model_resp or "").replace(",", "")
    # Prefer after cues if present
    for cue in ["final answer:", "answer:"]:
        idx = s.lower().rfind(cue)
        if idx != -1:
            sub = s[idx:]
            nums = _NUM_RE.findall(sub)
            if nums:
                try:
                    return float(nums[-1])
                except ValueError:
                    pass
    # Fallback: last number in whole string
    nums = _NUM_RE.findall(s)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def eq_num(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def compute_accuracy(bools: Iterable[bool]) -> float:
    lst = list(bools)
    if not lst:
        return 0.0
    return sum(1 for b in lst if b) / len(lst)

