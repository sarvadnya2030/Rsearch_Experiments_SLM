# Stage 2.5 — Self-consistency / majority-vote baseline for Qwen3-0.6B-Base on GSM8K.
# FULLY SELF-CONTAINED, start to end: installs every dependency, downloads the
# model and dataset fresh, runs the batched sampling experiment, and writes
# both a JSONL results file and a Markdown report. Assumes NOTHING is already
# in the notebook — paste this into one empty cell of a Colab or Kaggle
# notebook with a GPU runtime (T4/P100 free tier is plenty) and run it.
#
# What it does: for each question, samples k completions at temperature>0,
# then evaluates majority-vote accuracy at k=1,2,4,8,16 (extending past the
# k=8 we could run locally, since Colab/Kaggle's ~16GB VRAM allows a much
# bigger batch than the RTX 2070's 8GB).
#
# Uses the EXACT same answer-extraction/scoring logic as the local repo
# (src/evaluation/answer_extraction.py), copied inline so results are
# directly comparable to the local n=100 run (k=1:40%, k=2:40%, k=4:47%, k=8:60%).

import os
import subprocess
import sys

# Must be set before any CUDA context is created (i.e. before the first
# tensor .to("cuda") call below) to actually take effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch", "transformers", "datasets", "accelerate"],
    check=True,
)

import json
import platform
import re
import time
from collections import Counter
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Answer extraction / scoring — copied verbatim from
# src/evaluation/answer_extraction.py so results match the local runs exactly.
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


def extract_final_answer(response: str):
    # Sampling runs (do_sample=True) don't track a token cap here the way the
    # greedy baseline did, so treat every response as "stopped" for extraction
    # purposes (matches the not-hit_max_new_tokens branch of the local logic).
    raw_answer, method = extract_predicted_answer_with_method(response)
    return raw_answer if (method == "phrase" or raw_answer is not None) else None


def is_correct(predicted, reference):
    if predicted is None or reference is None:
        return False
    pred_norm = normalize_number(predicted)
    ref_norm = normalize_number(reference)
    return pred_norm is not None and pred_norm == ref_norm


def build_completion_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def majority_vote(answers):
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Config — adjust these
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
N_EXAMPLES = 100          # matches the local run for direct comparison
QUESTION_BATCH_SIZE = 8   # just a starting guess — generate_batch() halves automatically on OOM,
                          # so this doesn't need hand-tuning per GPU (T4/P100/A100/local 2070/...)
K_MAX = 16                # push past the local run's k=8 ceiling; k is never auto-reduced,
                          # since it's the question COUNT per batch that drives memory use
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512
SEED = 42

# Auto-pick a writable output dir: Kaggle's /kaggle/working, else Colab's
# /content, else the current directory (covers any other Jupyter host too).
import os
if os.path.isdir("/kaggle/working"):
    OUT_DIR = "/kaggle/working"
elif os.path.isdir("/content"):
    OUT_DIR = "/content"
else:
    OUT_DIR = "."
OUT_JSONL = f"{OUT_DIR}/n100_self_consistency.jsonl"
OUT_MD = f"{OUT_DIR}/n100_self_consistency_report.md"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    print("WARNING: no CUDA GPU detected — check the notebook's runtime type "
          "(Colab: Runtime > Change runtime type > GPU; Kaggle: Settings > Accelerator > GPU). "
          "This will run on CPU and be extremely slow.")
gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
print("device:", device, "| GPU:", gpu_name)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # required for correct batched generation on a causal LM

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
model.eval()

ds = load_dataset("openai/gsm8k", "main", split="test")
examples = []
for idx, row in enumerate(ds.select(range(N_EXAMPLES))):
    ref_answer = row["answer"].split("####")[-1].strip().replace(",", "")
    examples.append({"example_id": idx, "question": row["question"], "reference_answer": ref_answer})


@torch.no_grad()
def _generate_batch_raw(prompts, k):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    padded_prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=1.0,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id,
    )
    texts = tokenizer.batch_decode(outputs[:, padded_prompt_len:], skip_special_tokens=True)
    return [texts[i * k : (i + 1) * k] for i in range(len(prompts))]


def generate_batch(prompts, k, _depth=0):
    """Self-adapting batched generation: tries the whole sub-batch at once;
    on OOM, halves the question sub-batch and retries each half
    recursively. This means the script finds a working batch size on
    whatever GPU it's given (T4, P100, A100, ...) instead of needing the
    QUESTION_BATCH_SIZE constant hand-tuned per machine. k is never
    reduced — generating k samples for a single question is always cheap;
    it's the number of DIFFERENT questions batched together that eats
    memory, so that's what gets split.
    """
    try:
        return _generate_batch_raw(prompts, k)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise  # can't split a single question any further
        mid = len(prompts) // 2
        if _depth == 0:
            print(f"  OOM at batch of {len(prompts)} questions (k={k}) — splitting and retrying "
                  f"(this GPU's safe batch size will settle automatically)")
        left = generate_batch(prompts[:mid], k, _depth + 1)
        right = generate_batch(prompts[mid:], k, _depth + 1)
        return left + right


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
per_example = []
batch_timings = []
t_start = time.time()
run_started_at = datetime.now(timezone.utc).isoformat()

with open(OUT_JSONL, "w") as out_f:
    for batch_start in range(0, len(examples), QUESTION_BATCH_SIZE):
        batch = examples[batch_start : batch_start + QUESTION_BATCH_SIZE]
        prompts = [build_completion_prompt(ex["question"]) for ex in batch]

        t0 = time.time()
        grouped_texts = generate_batch(prompts, K_MAX)  # self-adapts on OOM, see above
        dt = time.time() - t0
        batch_timings.append(dt / len(batch))

        for ex, responses in zip(batch, grouped_texts):
            extracted = [extract_final_answer(r) for r in responses]
            record = {
                "example_id": ex["example_id"],
                "reference_answer": ex["reference_answer"],
                "extracted_answers": extracted,
                "responses": responses,
            }
            out_f.write(json.dumps(record) + "\n")
            per_example.append(record)

        out_f.flush()
        elapsed = time.time() - t_start
        print(f"batch [{batch_start}:{batch_start+len(batch)}] took {dt:.1f}s ({dt/len(batch):.1f}s/question) | elapsed {elapsed:.1f}s")

total_time = time.time() - t_start
avg_per_question = total_time / len(per_example)
print(f"\ndone: {len(per_example)} examples in {total_time:.1f}s ({avg_per_question:.2f}s/example)")

# ---------------------------------------------------------------------------
# Accuracy-vs-k curve
# ---------------------------------------------------------------------------
k_results = {}
for k in [1, 2, 4, 8, 16]:
    if k > K_MAX:
        continue
    correct = sum(
        1 for rec in per_example
        if is_correct(majority_vote(rec["extracted_answers"][:k]), rec["reference_answer"])
    )
    acc = correct / len(per_example)
    k_results[k] = (correct, acc)
    print(f"k={k}: majority-vote accuracy = {correct}/{len(per_example)} = {acc:.2%}")

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
gpu_mem_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None

md_lines = [
    "# Stage 2.5 — Self-Consistency Baseline (Colab/Kaggle run)",
    "",
    f"- **Run started (UTC):** {run_started_at}",
    f"- **Model:** {MODEL_NAME}",
    f"- **Device:** {device} ({gpu_name})",
    f"- **Python:** {platform.python_version()} | torch: {torch.__version__}",
    f"- **n_examples:** {N_EXAMPLES} | **question_batch_size:** {QUESTION_BATCH_SIZE} | **k_max:** {K_MAX}",
    f"- **temperature:** {TEMPERATURE} | **max_new_tokens:** {MAX_NEW_TOKENS} | **seed:** {SEED}",
    f"- **Total time:** {total_time:.1f}s ({avg_per_question:.2f}s/example, {sum(batch_timings)/len(batch_timings):.2f}s/example avg per-batch)",
]
if gpu_mem_gb is not None:
    md_lines.append(f"- **Peak GPU memory:** {gpu_mem_gb} GB")
md_lines += [
    "",
    "## Accuracy vs. k (majority vote)",
    "",
    "| k | correct | accuracy |",
    "|---|---|---|",
]
for k, (correct, acc) in sorted(k_results.items()):
    md_lines.append(f"| {k} | {correct}/{len(per_example)} | {acc:.2%} |")

md_lines += [
    "",
    "## Notes",
    "",
    "- Scoring logic (answer extraction, number normalization) is copied verbatim from",
    "  `src/evaluation/answer_extraction.py` in the local repo, so these numbers are",
    "  directly comparable to local runs.",
    "- k=1 here uses temperature-sampled decoding, NOT greedy — expect it to score below",
    "  the original greedy baseline (52.16% on the full 1319-example set) even before any",
    "  voting benefit kicks in at higher k.",
    "",
    f"Raw per-example generations (including full reasoning text) saved to `{OUT_JSONL}`.",
]

with open(OUT_MD, "w") as md_f:
    md_f.write("\n".join(md_lines) + "\n")

print(f"\nJSONL results: {OUT_JSONL}")
print(f"Markdown report: {OUT_MD}")
print("\nDownload both files and drop them into results/exp02_5_self_consistency/")
print("in the local repo (rename the .jsonl to avoid clobbering the local n100_batched.jsonl).")
