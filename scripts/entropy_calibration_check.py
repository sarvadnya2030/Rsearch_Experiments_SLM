"""Stage 2 calibration check: does the model's own per-token entropy at the
point it commits to a final answer predict whether that answer is right or
wrong? Teacher-forces each saved generation through the model once (no new
sampling) and reads off entropy at the answer-token span located via the
same phrase-pattern extraction used for scoring.

Usage:
    python scripts/entropy_calibration_check.py --run results/exp00_baseline/run_20260905_194735 \
        --out results/exp00_baseline/entropy_calibration.jsonl
"""
import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.gsm8k import build_completion_prompt  # noqa: E402
from src.evaluation.answer_extraction import _ANSWER_PHRASE_PATTERNS, _truncate_at_continuation  # noqa: E402


def find_answer_char_span(response: str):
    """Locate the character span of the matched answer number within the
    ORIGINAL (untruncated) response string, using the same phrase patterns
    and truncation as scoring, so entropy is measured at the exact span
    extraction relied on."""
    truncated = _truncate_at_continuation(response)
    for pattern in _ANSWER_PHRASE_PATTERNS:
        matches = list(re.finditer(pattern, truncated, flags=re.IGNORECASE))
        if matches:
            m = matches[-1]
            return m.start(1), m.end(1)
    return None


@torch.no_grad()
def compute_answer_entropy(model, tokenizer, device, question: str, response: str, hit_max_new_tokens: bool):
    span = find_answer_char_span(response)
    if span is None:
        return None

    prompt = build_completion_prompt(question)
    full_text = prompt + response
    enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=2048)
    offsets = enc["offset_mapping"][0].tolist()
    input_ids = enc["input_ids"].to(device)

    prompt_char_len = len(prompt)
    answer_start = prompt_char_len + span[0]
    answer_end = prompt_char_len + span[1]

    token_idx_span = [
        i for i, (s, e) in enumerate(offsets)
        if e > answer_start and s < answer_end
    ]
    if not token_idx_span:
        return None

    logits = model(input_ids).logits[0]  # [seq_len, vocab]
    entropies = []
    for tok_idx in token_idx_span:
        if tok_idx == 0:
            continue
        dist = F.softmax(logits[tok_idx - 1].float(), dim=-1)
        ent = -(dist * torch.log(dist + 1e-12)).sum().item()
        entropies.append(ent)
    if not entropies:
        return None
    return sum(entropies) / len(entropies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16).to(device)
    model.eval()

    with open(os.path.join(args.run, "generations.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_ok, n_skip = 0, 0
    with open(args.out, "w") as out_f:
        for i, row in enumerate(rows):
            entropy = compute_answer_entropy(
                model, tokenizer, device,
                row["question"], row["model_response"], row["hit_max_new_tokens"],
            )
            if entropy is None:
                n_skip += 1
                continue
            n_ok += 1
            result = {
                "example_id": row["example_id"],
                "is_correct": row["is_correct"],
                "extraction_method": row["extraction_method"],
                "termination_status": row["termination_status"],
                "answer_entropy": entropy,
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(rows)}] ok={n_ok} skip={n_skip}")

    print(f"done: {n_ok} scored, {n_skip} skipped (no extractable answer span)")


if __name__ == "__main__":
    main()
