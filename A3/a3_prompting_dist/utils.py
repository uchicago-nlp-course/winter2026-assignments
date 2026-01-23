import json
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Constants and config
MODEL_NAME = "google/gemma-3-1b-it"
SHOT_COUNTS = [1, 2, 4, 8]
RESULTS_DIR = "./results"
DEFAULT_MAX_NEW_TOKENS = 512


@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    device: torch.device
    model_name: str
    torch_dtype: torch.dtype


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.to(device)

    # pad_token_id fallback
    if getattr(model.config, "pad_token_id", None) is None:
        eos_id = getattr(model.config, "eos_token_id", None)
        if eos_id is None and hasattr(tokenizer, "eos_token_id"):
            eos_id = tokenizer.eos_token_id
        model.config.pad_token_id = eos_id

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_name=model_name,
        torch_dtype=torch_dtype,
    )


def make_run_dir(base: str = RESULTS_DIR) -> Path:
    import datetime as _dt

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path(base) / ts
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_chat_template(tokenizer, prompt: str) -> str:
    """
    Apply the chat template to a prompt string.

    Args:
        tokenizer: HuggingFace tokenizer with chat template support
        prompt: The raw prompt string to wrap

    Returns:
        The formatted prompt string with chat template applied
    """
    # TODO: Implement this function
    raise NotImplementedError("TODO: Implement this function")


def model_generate(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    chat = apply_chat_template(bundle.tokenizer, prompt)
    inputs = bundle.tokenizer(chat, return_tensors="pt").to(bundle.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = bundle.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            # Let the model config handle eos_token_id/pad_token_id unless you have a specific override
        )
    new_tokens = out[0][input_len:] # may have problem when batching/padding is involved
    text = bundle.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


# Prompt builders
def build_snli_prompt(
    premise: str,
    hypothesis: str,
    method: str,
    fewshot_examples: Optional[List[Tuple[str, str, str]]] = None,
) -> str:
    """
    Build a prompt for the SNLI classification task.

    Args:
        premise: The premise sentence
        hypothesis: The hypothesis sentence
        method: Either "direct" or "cot" (chain-of-thought)
        fewshot_examples: Optional list of (premise, hypothesis, label) tuples for few-shot

    Returns:
        The constructed prompt string
    """
    # TODO: Implement this function
    # Refer to the prompt templates in the handout
    raise NotImplementedError("TODO: Implement this function")


def build_gsm_prompt(
    question: str,
    method: str,
    fewshot_examples: Optional[List[Tuple]] = None,
) -> str:
    """
    Build a prompt for the GSM-Symbolic math problem task.

    Args:
        question: The math problem question
        method: Either "direct" or "cot" (chain-of-thought)
        fewshot_examples: Optional list of example tuples for few-shot
            - For direct: (question, final_answer) tuples
            - For cot: (question, solution, final_answer) tuples

    Returns:
        The constructed prompt string
    """
    # TODO: Implement this function
    # Refer to the prompt templates in the handout
    raise NotImplementedError("TODO: Implement this function")


def split_gsm_solution_and_final(answer_text: str) -> Tuple[str, str]:
    """
    Split a GSM answer field into (solution_text, final_answer_str).

    Primary split uses the '####' delimiter as in the student files. If absent,
    fall back to the last numeric token in the string and remove it (and any
    trailing cues like 'Final answer:' or 'Answer:') from the solution text.
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
    else:
        raise ValueError("No '####' marker found in GSM answer text for splitting.")


# Extraction/Grading
_SNLI_RE = re.compile(r"\b(entailment|neutral|contradiction)\b", re.IGNORECASE)


def extract_snli_answer(text: str) -> Optional[str]:
    matches = list(_SNLI_RE.finditer(text))
    if not matches:
        return None
    # Use last match per spec
    label = matches[-1].group(1).lower()
    return label


def compute_accuracy(bools: Iterable[bool]) -> float:
    lst = list(bools)
    if not lst:
        return 0.0
    return sum(1 for b in lst if b) / len(lst)


_NUM_RE = re.compile(r"-?\d+\.?\d*")


def parse_gold_answer_field(answer_field: str) -> Optional[float]:
    # Expect patterns like '#### 123'. Fallback: last number.
    if answer_field is None:
        return None
    s = str(answer_field).replace(",", "")
    nums = _NUM_RE.findall(s)
    if not nums:
        raise ValueError("No numeric answer found in answer field.")
    try:
        return float(nums[-1])
    except ValueError as e:
        raise ValueError(f"Failed to parse numeric answer from answer field: {e}")


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


def progressive_indices(order: List[int], shots: List[int]) -> Dict[int, List[int]]:
    # Given a full permutation order, return prefix indices for each K in shots
    out: Dict[int, List[int]] = {}
    for k in sorted(set(shots)):
        if k < 0:
            raise ValueError("shots must be >= 0")
        out[k] = order[:k]
    return out


def seed_multiples(base_seed: int, num: int) -> List[int]:
    return [base_seed * i for i in range(1, num + 1)]


def write_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    save_json(run_dir / "meta.json", meta)
