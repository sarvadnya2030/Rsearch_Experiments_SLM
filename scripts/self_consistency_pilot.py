"""Stage 2.5 pilot: self-consistency / majority-vote baseline on a tiny
subsample (first 10 test examples, matching Stage 2's original smoke-test
convention). Samples k_max completions per question at temperature>0, then
evaluates majority-vote accuracy at k=1,2,4,8 by slicing the same samples
(no need to regenerate per k).

Usage:
    python scripts/self_consistency_pilot.py --out results/exp02_5_self_consistency/pilot_n10.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.gsm8k import load_gsm8k_test, to_examples, build_completion_prompt  # noqa: E402
from src.evaluation.answer_extraction import extract_final_answer, is_correct  # noqa: E402


@torch.no_grad()
def sample_k(model, tokenizer, device, prompt, k, temperature, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    responses = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in outputs]
    return responses


def majority_vote(answers: list[str | None]) -> str | None:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16).to(device)
    model.eval()

    examples = to_examples(load_gsm8k_test())[: args.n_examples]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    per_example = []
    with open(args.out, "w") as out_f:
        for ex in examples:
            prompt = build_completion_prompt(ex.question)
            responses = sample_k(model, tokenizer, device, prompt, args.k_max, args.temperature, args.max_new_tokens)
            extracted = []
            for r in responses:
                res = extract_final_answer(r, hit_max_new_tokens=False)
                extracted.append(res["extracted_answer"])
            record = {
                "example_id": ex.example_id,
                "reference_answer": ex.reference_answer,
                "extracted_answers": extracted,
                "responses": responses,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            per_example.append(record)
            print(f"ex={ex.example_id} ref={ex.reference_answer} samples={extracted}")

    # Evaluate majority-vote accuracy at k = 1, 2, 4, 8 (and greedy-single-sample as k=1 proxy)
    for k in [1, 2, 4, min(8, args.k_max)]:
        if k > args.k_max:
            continue
        correct = 0
        for rec in per_example:
            vote = majority_vote(rec["extracted_answers"][:k])
            if is_correct(vote, rec["reference_answer"]):
                correct += 1
        print(f"k={k}: majority-vote accuracy = {correct}/{len(per_example)} = {correct/len(per_example):.2%}")


if __name__ == "__main__":
    main()
