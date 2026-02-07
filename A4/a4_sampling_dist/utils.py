"""
Utility functions for A4: Best-of-N Sampling on GSM Data.
Inherits conventions from A3 prompting solutions.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Constants
GEMMA_MODEL = "google/gemma-3-1b-it"
REWARD_MODEL = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
DEFAULT_MAX_NEW_TOKENS = 1024
ANSWER_PREFIX = "Answer: Let's think step by step."

@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    device: Any
    model_name: str
    torch_dtype: Any

def load_model_and_tokenizer(
    model_name: str,
    is_reward_model: bool = False,
) -> ModelBundle:
    """Loads model and tokenizer for inference."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding side for batching:
    # Generator needs left-padding for generate()
    # Reward model (classification) typically uses right-padding
    tokenizer.padding_side = "right" if is_reward_model else "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use bfloat16 for efficiency if on GPU
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if is_reward_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device if device.type == "cuda" else None,
            num_labels=1,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device if device.type == "cuda" else None,
        )
    
    if device.type != "cuda":
        model.to(device)

    # pad_token_id fallback
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_name=model_name,
        torch_dtype=torch_dtype,
    )

def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def save_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def append_jsonl(path: Union[str, Path], row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_gsm_prompt(question: str) -> str:
    """Builds the GSM prompt for CoT (without the final answer prefix)."""
    header = (
        "Solve the following math problem step by step, "
        "and provide the final numeric answer."
    )
    query = f"Problem: {question}"
    return f"{header}\n\n{query}"

def extract_gsm_answer(model_resp: str) -> Optional[float]:
    """Extracts the final numeric answer from the model's response."""
    s = (model_resp or "").replace(",", "")
    num_re = re.compile(r"-?\d+\.?\d*")
    
    for cue in ["final answer:", "answer:"]:
        idx = s.lower().rfind(cue)
        if idx != -1:
            sub = s[idx:]
            nums = num_re.findall(sub)
            if nums:
                try:
                    return float(nums[-1])
                except ValueError:
                    pass
    
    nums = num_re.findall(s)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None

def parse_gold_answer_field(answer_field: str) -> Optional[float]:
    """Parses the ground truth answer from the dataset."""
    if answer_field is None:
        return None
    s = str(answer_field).strip()
    marker = "####"
    idx = s.rfind(marker)
    num_re = re.compile(r"-?\d+\.?\d*")
    if idx != -1:
        tail = s[idx + len(marker) :].strip()
        nums = num_re.findall(tail)
        if nums:
            return float(nums[-1].replace(",", ""))
    
    # Fallback to last number in entire string
    s = s.replace(",", "")
    nums = num_re.findall(s)
    if not nums:
        return None
    return float(nums[-1])

def eq_num(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def apply_chat_template(tokenizer, messages: Union[List[dict], str], add_generation_prompt: bool = True) -> str:
    """Wraps messages in the tokenizer's chat template."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    
    # Remove duplicate BOS token if present (common requirement for some models)
    if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
        # Only strip if the tokenizer adds it and we want to control it manually
        # This is explicitly recommended for Skywork-Reward-V2
        pass 
        
    return formatted
