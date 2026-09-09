# Stage 2 loose end — rerun the 47 `capped_no_answer` examples from the full
# 1319-example Experiment 0 baseline run (run_20260905_194735) under a much
# higher max_new_tokens budget, to confirm (rather than assume) that raising
# the cap wouldn't have helped them (docs/research_log.md Finding 3 already
# checked this on a 100-example sample and found no evidence it would — this
# extends that check to every actual capped_no_answer case from the full run).
#
# FULLY SELF-CONTAINED: installs every dependency, downloads the model and
# dataset fresh, reruns exactly these 47 example_ids, and writes both a JSONL
# results file and a Markdown summary. Paste into one empty Colab/Kaggle cell
# with a GPU runtime and run.
#
# Uses the EXACT same answer-extraction/scoring logic and completion prompt
# as the local repo (copied inline) so results are directly comparable.

import os
import subprocess
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch", "transformers", "datasets", "accelerate"],
    check=True,
)

import json
import re
import time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MAX_NEW_TOKENS = 2048  # was 512 in the original run
TEMPERATURE = None  # greedy, matching the original baseline run (do_sample=False)

# The exact 47 example_ids (GSM8K test split, 0-indexed) whose
# termination_status was "capped_no_answer" in run_20260905_194735.
CAPPED_NO_ANSWER_IDS = [
    7, 8, 15, 39, 43, 150, 162, 189, 209, 309, 330, 331, 357, 427, 499, 532,
    593, 628, 637, 672, 711, 724, 726, 780, 815, 855, 885, 916, 943, 984,
    1003, 1031, 1070, 1087, 1089, 1096, 1110, 1117, 1122, 1129, 1159, 1166,
    1183, 1215, 1216, 1273, 1303,
]

OUT_JSONL = "rerun_capped_no_answer_maxtok2048.jsonl"
OUT_MD = "rerun_capped_no_answer_report.md"

# ---------------------------------------------------------------------------
# Answer extraction / scoring — copied verbatim from
# src/evaluation/answer_extraction.py so results match the local repo exactly.
# ---------------------------------------------------------------------------

_ANSWER_PHRASE_PATTERNS = [
    r"####\s*([\-\$]?[\d,]*\.?\d+%?)",
    r"final answer is[:\s]*([\-\$]?[\d,]*\.?\d+%?)",
    r"the answer is[:\s]*([\-\$]?[\d,]*\.?\d+%?)",
    r"answer:\s*([\-\$]?[\d,]*\.?\d+%?)",
]
_ANY_NUMBER_PATTERN = r"[\-\$]?[\d,]*\.?\d+%?"
_CONTINUATION_MARKERS = [
    r"\n\s*\[Question\]",
    r"\n\s*Question\s*:",
    r"\n\s*Q\s*:",
]
_CONTINUATION_PATTERN = re.compile("|".join(_CONTINUATION_MARKERS), flags=re.IGNORECASE)


def _truncate_at_continuation(response: str) -> str:
    match = _CONTINUATION_PATTERN.search(response)
    return response[: match.start()] if match else response


def normalize_number(raw):
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1].strip()
    s = s.replace("$", "").replace(",", "").strip()
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    if is_percent:
        value = value / 100.0
    if value == int(value):
        return str(int(value))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def extract_predicted_answer_with_method(response: str):
    if not response:
        return None, "none"
    truncated = _truncate_at_continuation(response)
    for pattern in _ANSWER_PHRASE_PATTERNS:
        matches = re.findall(pattern, truncated, flags=re.IGNORECASE)
        if matches:
            normalized = normalize_number(matches[-1])
            if normalized is not None:
                return normalized, "phrase"
    fallback_matches = re.findall(_ANY_NUMBER_PATTERN, truncated)
    fallback_matches = [m for m in fallback_matches if re.search(r"\d", m)]
    if fallback_matches:
        normalized = normalize_number(fallback_matches[-1])
        if normalized is not None:
            return normalized, "fallback"
    return None, "none"


def classify_termination(response: str, hit_cap: bool):
    raw_answer, method = extract_predicted_answer_with_method(response)
    has_answer = method == "phrase" or (method == "fallback" and not hit_cap)
    if hit_cap:
        return (raw_answer if method == "phrase" else None), ("capped_with_answer" if method == "phrase" else "capped_no_answer")
    return raw_answer, ("stopped_with_answer" if has_answer else "stopped_no_answer")


def is_correct(predicted, reference):
    if predicted is None or reference is None:
        return False
    pred_norm = normalize_number(predicted)
    ref_norm = normalize_number(reference)
    return pred_norm is not None and pred_norm == ref_norm


def build_completion_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


# ---------------------------------------------------------------------------
# Load model + dataset
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print("WARNING: no GPU detected — this will be very slow at max_new_tokens=2048.")

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
model.eval()

print("Loading GSM8K test split")
dataset = load_dataset("openai/gsm8k", "main", split="test")


def reference_answer_of(row) -> str:
    tail = row["answer"].split("####")[-1].strip().replace(",", "")
    return tail


# ---------------------------------------------------------------------------
# Rerun
# ---------------------------------------------------------------------------

results = []
now_answered = 0
now_correct = 0

for i, example_id in enumerate(CAPPED_NO_ANSWER_IDS):
    row = dataset[example_id]
    question = row["question"]
    reference_answer = reference_answer_of(row)
    prompt = build_completion_prompt(question)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - start

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    hit_cap = len(new_tokens) >= MAX_NEW_TOKENS

    extracted_answer, termination_status = classify_termination(response_text, hit_cap)
    correct = is_correct(extracted_answer, reference_answer)

    if extracted_answer is not None:
        now_answered += 1
    if correct:
        now_correct += 1

    record = {
        "example_id": example_id,
        "question": question,
        "reference_answer": reference_answer,
        "rerun_max_new_tokens": MAX_NEW_TOKENS,
        "rerun_response_text": response_text,
        "rerun_response_tokens": int(len(new_tokens)),
        "rerun_extracted_answer": extracted_answer,
        "rerun_termination_status": termination_status,
        "rerun_is_correct": correct,
        "rerun_generation_time_sec": round(gen_time, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    results.append(record)
    print(
        f"[{i+1}/{len(CAPPED_NO_ANSWER_IDS)}] id={example_id} "
        f"capped_no_answer -> {termination_status} "
        f"({len(new_tokens)} tok, {gen_time:.1f}s, correct={correct})"
    )

with open(OUT_JSONL, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

now_still_no_answer = sum(1 for r in results if r["rerun_termination_status"] in ("capped_no_answer",))

summary = f"""# Rerun report: capped_no_answer @ max_new_tokens={MAX_NEW_TOKENS}

Source: 47 `capped_no_answer` examples from `run_20260905_194735`
(full 1319-example baseline, original `max_new_tokens=512`).

- Now produced SOME coherent answer: {now_answered}/{len(CAPPED_NO_ANSWER_IDS)}
  (were 0/{len(CAPPED_NO_ANSWER_IDS)} under the original cap, by definition)
- Now correct: {now_correct}/{len(CAPPED_NO_ANSWER_IDS)}
- Still capped with no answer even at {MAX_NEW_TOKENS} tokens: {now_still_no_answer}/{len(CAPPED_NO_ANSWER_IDS)}

Interpretation: if `now_still_no_answer` stays high, this confirms Finding 3's
conclusion (these are genuinely non-convergent/degenerate-loop generations,
not reasoning cut off too early) at full scale rather than just the 100-example
sample. If several flip to `stopped_with_answer` or `capped_with_answer`,
that's new evidence Finding 3's 100-example sample missed and the baseline's
`max_new_tokens=512` choice should be revisited.
"""
with open(OUT_MD, "w") as f:
    f.write(summary)

print("\n" + summary)
print(f"Wrote {OUT_JSONL} and {OUT_MD}")
print("Download both files from the Colab file browser (or `files.download(...)`) to bring them back into the local repo.")
