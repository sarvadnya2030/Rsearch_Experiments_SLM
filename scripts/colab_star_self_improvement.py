# Stage 4, Arm 2 — STaR (Self-Taught Reasoner) self-improvement for Qwen3-0.6B-Base.
# FULLY SELF-CONTAINED, start to end: installs every dependency, downloads the
# model and dataset fresh, runs the batched STaR generation pass, and writes
# both a JSONL results file and a Markdown report. Assumes NOTHING is already
# in the notebook — paste into one empty cell of a Colab or Kaggle notebook
# with a GPU runtime and run it.
#
# What STaR does (no teacher model, no RL): for each training question,
# sample k own-attempt completions; if any reach the correct answer, keep
# the first correct one as a training example. If none do, fall back to
# few-shot HINT-based backward rationalization — show the model the correct
# answer and ask it to derive reasoning toward it — and keep that only if
# it genuinely reaches the answer (verified, not assumed). Zero-shot hints
# were tested locally and failed badly on this base model (either ignored
# entirely, or the model faked arithmetic to force-match the number without
# real derivation) — few-shot demonstrations fixed this, confirmed on a
# local pilot (80% -> 90% kept, with the hint path now producing genuine
# derivations). That fix is already baked into this script.
#
# Uses the EXACT same answer-extraction/scoring logic and completion-prompt
# format as the local repo, copied inline so results are directly comparable.

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
import platform
import re
import time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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


def extract_final_answer(response: str):
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


# Few-shot demonstrations of genuine backward rationalization. Needed because
# Qwen3-0.6B-Base has no instruction tuning and will not reliably follow a
# zero-shot embedded hint instruction — piloted locally, it either ignored
# the hint entirely or faked arithmetic to force-match the given number
# without real derivation. These are ORIGINAL toy problems, NOT from GSM8K,
# to avoid any contamination — they only demonstrate the completion pattern.
_HINT_FEWSHOT_PREFIX = """Question: A farmer has 12 sheep. He buys 5 more sheep. How many sheep does he have now?
(The final answer to this question is 17. Show the step-by-step reasoning that leads to this answer.)
Answer: The farmer starts with 12 sheep. He buys 5 more sheep, so he now has 12 + 5 = 17 sheep. The answer is 17.

Question: A bakery makes 8 trays of muffins with 6 muffins on each tray. They sell 20 muffins. How many muffins are left?
(The final answer to this question is 28. Show the step-by-step reasoning that leads to this answer.)
Answer: The bakery makes 8 trays with 6 muffins each, so they have 8 x 6 = 48 muffins in total. They sell 20 muffins, so they have 48 - 20 = 28 muffins left. The answer is 28.

Question: A store had 90 apples. They sold some apples and have 34 left. How many apples did they sell?
(The final answer to this question is 56. Show the step-by-step reasoning that leads to this answer.)
Answer: The store started with 90 apples and now has 34 left. The number sold is the difference: 90 - 34 = 56. The answer is 56.

"""


def build_hint_prompt(question: str, answer: str) -> str:
    return (
        _HINT_FEWSHOT_PREFIX +
        f"Question: {question}\n"
        f"(The final answer to this question is {answer}. Show the step-by-step reasoning "
        f"that leads to this answer.)\nAnswer:"
    )


# ---------------------------------------------------------------------------
# Config — adjust these
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
N_EXAMPLES = 2000
QUESTION_BATCH_SIZE = 8   # starting guess — generate_adaptive() halves automatically on OOM
K_OWN_ATTEMPT = 4         # samples per question in phase 1 (own-attempt)
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512
SEED = 42
PAUSE_ABOVE_C = 80        # harmless on well-cooled datacenter GPUs, kept as a safety net
RESUME_BELOW_C = 65

# Prefer Google Drive if we're on Colab and it mounts successfully — every
# record is already written+flushed per batch below, so writing straight to
# a mounted Drive path is enough on its own to survive a session disconnect
# (no extra background-copy thread needed). Falls back to Kaggle's working
# dir, then /content, then cwd, exactly as before, if Drive isn't available
# (e.g. on Kaggle, or if you decline the mount prompt).
OUT_DIR = None
if os.path.isdir("/content"):
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT_DIR = "/content/drive/MyDrive/slm-reasoning-research/exp04_star"
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"Google Drive mounted — writing output to {OUT_DIR} (survives a session disconnect)")
    except Exception as e:
        print(f"Could not mount Google Drive ({e}) — falling back to ephemeral /content storage. "
              "If this session disconnects, in-progress output will be lost.")
if OUT_DIR is None:
    if os.path.isdir("/kaggle/working"):
        OUT_DIR = "/kaggle/working"
    elif os.path.isdir("/content"):
        OUT_DIR = "/content"
    else:
        OUT_DIR = "."
OUT_JSONL = f"{OUT_DIR}/star_n{N_EXAMPLES}.jsonl"
OUT_MD = f"{OUT_DIR}/star_n{N_EXAMPLES}_report.md"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    print("WARNING: no CUDA GPU detected — check the notebook's runtime type.")
gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
print("device:", device, "| GPU:", gpu_name)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
model.eval()

ds = load_dataset("openai/gsm8k", "main", split="train")
examples = []
for idx, row in enumerate(ds.select(range(N_EXAMPLES))):
    ref_answer = row["answer"].split("####")[-1].strip().replace(",", "")
    examples.append({"example_id": idx, "question": row["question"], "reference_answer": ref_answer})


def gpu_temp_c():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip())
    except Exception:
        return None


def wait_for_cooldown():
    temp = gpu_temp_c()
    if temp is None or temp < PAUSE_ABOVE_C:
        return
    print(f"GPU at {temp}C >= {PAUSE_ABOVE_C}C, pausing until below {RESUME_BELOW_C}C...")
    while True:
        time.sleep(15)
        temp = gpu_temp_c()
        if temp is None or temp < RESUME_BELOW_C:
            print(f"resuming (temp={temp}C)")
            return


@torch.no_grad()
def _generate_raw(prompts, k, temperature):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    padded_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        top_p=1.0,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id,
    )
    texts = tokenizer.batch_decode(outputs[:, padded_len:], skip_special_tokens=True)
    return [texts[i * k : (i + 1) * k] for i in range(len(prompts))]


def generate_adaptive(prompts, k, temperature, _depth=0):
    """Self-adapting batched generation: on OOM, halves the question
    sub-batch and retries each half recursively. k is never reduced — it's
    the number of DIFFERENT questions batched together that drives memory."""
    try:
        return _generate_raw(prompts, k, temperature)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise
        mid = len(prompts) // 2
        if _depth == 0:
            print(f"  OOM at batch of {len(prompts)} questions (k={k}) — splitting and retrying")
        left = generate_adaptive(prompts[:mid], k, temperature, _depth + 1)
        right = generate_adaptive(prompts[mid:], k, temperature, _depth + 1)
        return left + right


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
n_own, n_hint_ok, n_hint_failed, n_total = 0, 0, 0, 0
t_start = time.time()
run_started_at = datetime.now(timezone.utc).isoformat()

# Resume support: if OUT_JSONL already has content (e.g. re-running this
# cell after a session disconnect and reconnect), skip example_ids already
# done and APPEND rather than overwrite — otherwise reconnecting would wipe
# out everything Drive just saved for you, defeating the whole point.
done_ids = set()
if os.path.exists(OUT_JSONL) and os.path.getsize(OUT_JSONL) > 0:
    with open(OUT_JSONL) as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    if done_ids:
        examples = [ex for ex in examples if ex["example_id"] not in done_ids]
        print(f"found existing output with {len(done_ids)} examples already done — "
              f"resuming, {len(examples)} remaining")

file_mode = "a" if done_ids else "w"
with open(OUT_JSONL, file_mode) as out_f:
    for batch_start in range(0, len(examples), QUESTION_BATCH_SIZE):
        wait_for_cooldown()
        batch = examples[batch_start : batch_start + QUESTION_BATCH_SIZE]

        # Phase 1: own-attempt sampling
        own_prompts = [build_completion_prompt(ex["question"]) for ex in batch]
        own_grouped = generate_adaptive(own_prompts, K_OWN_ATTEMPT, TEMPERATURE)

        needs_hint = []
        for ex, responses in zip(batch, own_grouped):
            kept = None
            for r in responses:
                extracted = extract_final_answer(r)
                if is_correct(extracted, ex["reference_answer"]):
                    kept = r
                    break
            if kept is not None:
                # Truncate at the first sign of a hallucinated continuation
                # (base model rambling into a new, fabricated question after
                # already answering — confirmed present in 88% of own_attempt
                # traces from the n=2000 run before this fix). Scoring already
                # accounted for this via extract_final_answer, but the SAVED
                # trace must be the clean version too, or the student model
                # would be trained to imitate the rambling.
                record = {
                    "example_id": ex["example_id"],
                    "question": ex["question"],
                    "reference_answer": ex["reference_answer"],
                    "star_trace": _truncate_at_continuation(kept).rstrip(),
                    "source": "own_attempt",
                }
                out_f.write(json.dumps(record) + "\n")
                n_own += 1
            else:
                needs_hint.append(ex)

        # Phase 2: few-shot hint-based backward rationalization for the rest
        if needs_hint:
            hint_prompts = [build_hint_prompt(ex["question"], ex["reference_answer"]) for ex in needs_hint]
            hint_grouped = generate_adaptive(hint_prompts, 1, TEMPERATURE)
            for ex, responses in zip(needs_hint, hint_grouped):
                r = responses[0]
                extracted = extract_final_answer(r)
                if is_correct(extracted, ex["reference_answer"]):
                    record = {
                        "example_id": ex["example_id"],
                        "question": ex["question"],
                        "reference_answer": ex["reference_answer"],
                        "star_trace": _truncate_at_continuation(r).rstrip(),
                        "source": "hint_rationalized",
                    }
                    out_f.write(json.dumps(record) + "\n")
                    n_hint_ok += 1
                else:
                    n_hint_failed += 1

        out_f.flush()
        n_total += len(batch)
        elapsed = time.time() - t_start
        print(f"[{n_total}/{len(examples)}] own={n_own} hint_ok={n_hint_ok} hint_failed={n_hint_failed} "
              f"| elapsed {elapsed:.1f}s")

total_time = time.time() - t_start
# n_total/n_own/n_hint_ok/n_hint_failed only cover THIS session's work; add
# back what was already on disk from a prior session so a resumed run's
# summary reflects the true grand total, not just what happened just now.
grand_total_processed = n_total + len(done_ids)
kept_total_this_session = n_own + n_hint_ok
kept_total = kept_total_this_session + len(done_ids)  # previously-done rows were all "kept" by definition
kept_rate = kept_total / grand_total_processed if grand_total_processed else 0.0
print(f"\ndone: {kept_total}/{grand_total_processed} kept overall ({kept_rate:.1%}) — "
      f"this session added {kept_total_this_session}/{n_total} in {total_time:.1f}s "
      f"({n_own} own attempts, {n_hint_ok} hint-rationalized, {n_hint_failed} failed even with hint)")

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
gpu_mem_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None

md_lines = [
    "# Stage 4 Arm 2 — STaR Self-Improvement (Colab/Kaggle run)",
    "",
] + ([f"- **Resumed run:** {len(done_ids)} examples were already done from a prior session; "
      f"stats below marked 'this session' cover only the newly-added work."] if done_ids else []) + [
    f"- **Run started (UTC):** {run_started_at}",
    f"- **Model:** {MODEL_NAME}",
    f"- **Device:** {device} ({gpu_name})",
    f"- **Python:** {platform.python_version()} | torch: {torch.__version__}",
    f"- **n_examples:** {N_EXAMPLES} | **question_batch_size (start):** {QUESTION_BATCH_SIZE} | **k_own_attempt:** {K_OWN_ATTEMPT}",
    f"- **temperature:** {TEMPERATURE} | **max_new_tokens:** {MAX_NEW_TOKENS} | **seed:** {SEED}",
    f"- **Total time:** {total_time:.1f}s ({total_time/n_total:.2f}s/example)" if n_total else "- **Total time:** n/a",
]
if gpu_mem_gb is not None:
    md_lines.append(f"- **Peak GPU memory:** {gpu_mem_gb} GB")

md_lines += [
    "",
    "## Yield breakdown",
    "",
    "| Source | Count | % of total |",
    "|---|---|---|",
    f"| own_attempt (model already got it right, sampled {K_OWN_ATTEMPT}x) | {n_own} | {n_own/n_total:.1%} |" if n_total else "",
    f"| hint_rationalized (few-shot hint recovered it) | {n_hint_ok} | {n_hint_ok/n_total:.1%} |" if n_total else "",
    f"| discarded (failed even with hint) | {n_hint_failed} | {n_hint_failed/n_total:.1%} |" if n_total else "",
    f"| **Total kept for training** | **{kept_total}** | **{kept_rate:.1%}** |",
    "",
    "## Notes",
    "",
    "- Scoring logic and prompt format are copied verbatim from the local repo",
    "  (`src/evaluation/answer_extraction.py`, `src/data/gsm8k.py`), so results are",
    "  directly comparable to local runs.",
    "- The hint-rationalization prompt uses FEW-SHOT demonstrations (3 original, non-GSM8K",
    "  toy problems) rather than a zero-shot instruction. This was a deliberate fix after a",
    "  local pilot found zero-shot hints either got ignored entirely, or caused the base model",
    "  to fake arithmetic to force-match the given number rather than deriving it genuinely",
    "  (e.g. '...however the answer is 624, which is 7488 - 624 = 6864 - 624 = 624').",
    "  Spot-check a sample of `hint_rationalized` traces before trusting them at scale — this",
    "  fix was validated on a small local pilot (n=10), not exhaustively.",
    "",
    f"Raw per-example traces saved to `{OUT_JSONL}`.",
]

with open(OUT_MD, "w") as md_f:
    md_f.write("\n".join(l for l in md_lines if l is not None) + "\n")

print(f"\nJSONL results: {OUT_JSONL}")
print(f"Markdown report: {OUT_MD}")
print("\nDownload both and drop them into results/exp04_star/ in the local repo")
print("(rename the .jsonl to avoid clobbering the local star_n2000.jsonl if one exists).")
