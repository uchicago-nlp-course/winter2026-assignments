"""
Script for A4 Part 2: Generate N=8 samples per question and score them using Skywork-Reward-V2.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import torch

from utils import (
    GEMMA_MODEL,
    REWARD_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    ANSWER_PREFIX,
    build_gsm_prompt,
    load_jsonl,
    load_model_and_tokenizer,
    save_jsonl,
    append_jsonl,
    set_seed,
    apply_chat_template,
)

def generate_samples_batched(
    bundle,
    questions: list[str],
    n: int,
    max_new_tokens: int
) -> list[list[str]]:
    """Generate n sampled responses for each question using the language model.

    For each question, this function:
    1. Builds a GSM-style prompt and applies the chat template
    2. Appends the chain-of-thought answer prefix
    3. Generates n diverse responses using temperature sampling

    Args:
        bundle (see in utils.py): A model bundle containing:
            - model: The HuggingFace causal language model
            - tokenizer: The corresponding tokenizer
            - device: The torch device (e.g., "cuda" or "cpu")
        questions: A list of math word problems (raw question strings).
        n: Number of samples to generate per question.
        max_new_tokens: Maximum number of new tokens to generate per sample.

    Returns:
        A list of lists, where the outer list has length len(questions) and
        each inner list contains n generated response strings (completions only,
        excluding the prompt).
    """
    # TODO: Implement batched generation
    # 1. Build prompt texts for each question:
    #    - Use build_gsm_prompt(q)
    #    - Apply chat template: bundle.tokenizer.apply_chat_template(...)
    #      with tokenize=False and add_generation_prompt=True.
    #    - Append ANSWER_PREFIX to the result.
    # 2. Tokenize the prompt texts:
    #    - Use bundle.tokenizer(...) with padding=True and add_special_tokens=False.
    #    - Move tensors to bundle.device.
    # 3. Configure generation kwargs:
    #    - Set max_new_tokens, num_return_sequences=n, do_sample=True,
    #      temperature=0.7, and top_p=0.95.
    # 4. Generate sequences using bundle.model.generate(**inputs, **gen_kwargs).
    
    # Placeholder for student implementation: replace these with your actual logic
    inputs = None   # Tokenized inputs dictionary
    input_len = 0  # Number of tokens in the prompt
    outputs = None  # Model generation outputs

    with torch.no_grad():
        ### YOUR CODE HERE ###
        
        ### END OF YOUR CODE ###
        
        # outputs shape: (batch_size * n, total_len)
        all_texts = []
        for i in range(len(questions)):
            prompt_texts_batch = []
            for j in range(n):
                idx = i * n + j
                text = bundle.tokenizer.decode(
                    outputs[idx][input_len:], 
                    skip_special_tokens=True
                ).strip()
                prompt_texts_batch.append(text)
            all_texts.append(prompt_texts_batch)
            
    return all_texts

def score_samples_batched(
    rm_bundle,
    questions: list[str],
    responses_batch: list[list[str]]
) -> list[list[float]]:
    """Score all generated samples using a reward model.

    For each (question, response) pair, this function:
    1. Formats the conversation using the reward model's chat template
    2. Strips redundant BOS tokens (as recommended by Skywork-Reward-V2)
    3. Computes a scalar reward score

    Args:
        rm_bundle (see in utils.py): A reward model bundle containing:
            - model: The HuggingFace reward model (e.g., Skywork-Reward-V2)
            - tokenizer: The corresponding tokenizer
            - device: The torch device (e.g., "cuda" or "cpu")
        questions: A list of math word problems (raw question strings).
        responses_batch: A list of lists, where responses_batch[i] contains
            the n generated responses for questions[i].

    Returns:
        A list of lists with the same shape as responses_batch, where
        each element is the reward score (float) for the corresponding response.
        Higher scores indicate better quality responses according to the
        reward model.
    """
    flat_questions = []
    flat_responses = []
    
    for q, resps in zip(questions, responses_batch):
        flat_questions.extend([q] * len(resps))
        flat_responses.extend(resps)
        
    formatted_list = []
    for q, resp in zip(flat_questions, flat_responses):
        # Skywork-Reward-V2 expects a specific conversation format
        # and recommends stripping the redundant BOS token added by templates.
        conv = [{"role": "user", "content": q}, {"role": "assistant", "content": resp}]
        formatted = apply_chat_template(rm_bundle.tokenizer, conv, add_generation_prompt=False)
        
        if rm_bundle.tokenizer.bos_token and formatted.startswith(rm_bundle.tokenizer.bos_token):
            formatted = formatted[len(rm_bundle.tokenizer.bos_token):]
        formatted_list.append(formatted)
            
    # Process in reward model batches to avoid OOM if n is very large
    rm_bsz = 32
    all_scores = []
    
    for i in range(0, len(formatted_list), rm_bsz):
        batch = formatted_list[i : i + rm_bsz]
        # TODO: Perform reward model inference
        # 1. Tokenize the batch using rm_bundle.tokenizer 
        #    (set return_tensors="pt", padding=True, add_special_tokens=False)
        # 2. Pass inputs to rm_bundle.model
        # 3. Extract scores from the logits (shape: [batch_size, 1]) using logits[:, 0]
        scores = []
        all_scores.extend(scores)
        
    # Reshape scores back to (batch_size, n)
    reshaped_scores = []
    idx = 0
    for resps in responses_batch:
        n = len(resps)
        reshaped_scores.append(all_scores[idx : idx + n])
        idx += n
        
    return reshaped_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=GEMMA_MODEL,
                        help="HF model id or local checkpoint path.")
    parser.add_argument("--data-path", type=str, 
                        default="data/gsm_symbolic_test_100.jsonl")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output JSONL path. If None, derived from model name.")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of questions to process in parallel.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    model_id = args.model_name_or_path
    model_name_clean = model_id.split("/")[-1]
    if "checkpoint-" in model_id:
        parent_name = Path(model_id).parent.name
        model_name_clean = f"{parent_name}_{model_name_clean}"
    
    output_path = args.output_path
    if output_path is None:
        output_path = f"results/scored_samples_{model_name_clean}.jsonl"

    print(f"Loading generator: {model_id}")
    gen_bundle = load_model_and_tokenizer(model_id)
    
    print(f"Loading reward model: {REWARD_MODEL}")
    rm_bundle = load_model_and_tokenizer(REWARD_MODEL, is_reward_model=True)
    
    test_data = load_jsonl(args.data_path)
    
    # Clear or initialize output file
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Gen & Score"):
        batch = test_data[i : i + args.batch_size]
        questions = [ex["question"] for ex in batch]
        
        samples_batch = generate_samples_batched(
            gen_bundle, questions, args.num_samples, args.max_new_tokens
        )
        scores_batch = score_samples_batched(rm_bundle, questions, samples_batch)
        
        for j, ex in enumerate(batch):
            res_item = {
                "index": i + j,
                "question": ex["question"],
                "gold": ex["answer"],
                "samples": [
                    {"text": s, "score": sc} 
                    for s, sc in zip(samples_batch[j], scores_batch[j])
                ]
            }
            append_jsonl(output_path, res_item)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
