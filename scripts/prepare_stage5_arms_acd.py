"""
Stage 5, Arms A/C/D — derived from Stage 3's GPT-OSS-20B teacher traces
(the arms that need reflection/verification content GSM8K's own reference
solutions don't have — see docs/research_log.md Finding 7 and
research-plan.md Stage 3/4 scope-narrowing decision).

- Arm A (full reasoning): the teacher trace, unmodified.
- Arm C (verification stripped): drop any section headed by a bold
  markdown heading containing "verif" (e.g. "**Verification moment**").
- Arm D (reflection stripped): drop any section headed by a bold
  markdown heading containing "reflect" (e.g. "**Reflection moment**").

Verified structurally first (see the ad-hoc check that produced these
numbers): across a sample of 1000 correct teacher traces, 99.9% contain a
"verification" heading and 99.8% a "reflection" heading, both as markdown
bold section headers — a reliable, consistent split point, not something
this script assumes without checking.

The final "The answer is N" sentence is always extracted and preserved
separately before section-stripping, specifically so that dropping the
verification/reflection section (which sometimes trails right before it)
can never accidentally remove the actual answer statement.

Every derived Arm C/D trace is re-verified against ground truth after
stripping (never trust a transformation blindly) — a trace is dropped
from that arm's output if stripping somehow breaks its extractable answer.

Usage:
    python scripts/prepare_stage5_arms_acd.py
    python scripts/prepare_stage5_arms_acd.py --teacher-file results/exp03_teacher/traces_n2000.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.answer_extraction import extract_final_answer, is_correct
from src.utils.logging import get_logger

logger = get_logger("prepare_stage5_arms_acd")

_HEADER_RE = re.compile(r"^\*\*([^*\n]{1,60})\*\*\.?:?\s*(.*)$", re.DOTALL)
_ANSWER_SENTENCE_RE = re.compile(r"[^.\n]*\banswer is\b[^.\n]*\.?", re.IGNORECASE)


def split_sections(trace: str) -> list[tuple[str | None, list[str]]]:
    """Group a trace's paragraphs under the nearest preceding bold header."""
    paragraphs = re.split(r"\n\s*\n", trace.strip())
    sections: list[list] = []
    for para in paragraphs:
        m = _HEADER_RE.match(para.strip())
        if m:
            header_text, rest = m.group(1).strip(), m.group(2).strip()
            sections.append([header_text, [rest] if rest else []])
        elif sections:
            sections[-1][1].append(para)
        else:
            sections.append([None, [para]])
    return [(h, paras) for h, paras in sections]


def strip_by_header_keyword(trace: str, keyword: str) -> str:
    """Remove sections whose header contains `keyword`, always preserving
    the final 'the answer is N' sentence regardless of which section it
    trails."""
    answer_match = list(_ANSWER_SENTENCE_RE.finditer(trace))
    answer_sentence = answer_match[-1].group(0).strip() if answer_match else None
    body = trace
    if answer_match:
        body = trace[: answer_match[-1].start()].rstrip()

    sections = split_sections(body)
    kept_chunks = []
    for header, paras in sections:
        if header is not None and keyword in header.lower():
            continue
        chunk = ("**" + header + "**\n\n" if header else "") + "\n\n".join(p for p in paras if p)
        if chunk.strip():
            kept_chunks.append(chunk.strip())

    result = "\n\n".join(kept_chunks)
    if answer_sentence:
        result = (result + "\n\n" + answer_sentence).strip()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-file", type=str, default="results/exp03_teacher/traces_n2000.jsonl")
    parser.add_argument("--out-dir", type=str, default="results/exp05_arms_acd")
    args = parser.parse_args()

    with open(args.teacher_file) as f:
        all_rows = [json.loads(line) for line in f]
    correct_rows = [r for r in all_rows if r.get("is_correct")]
    logger.info(f"Loaded {len(all_rows)} teacher rows, {len(correct_rows)} correct (is_correct=True) — using only those")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_a, arm_c, arm_d = [], [], []
    c_verify_failed = c_no_header = 0
    d_verify_failed = d_no_header = 0

    for r in correct_rows:
        trace = r["teacher_trace"]
        reference_answer = r["reference_answer"]

        arm_a.append(
            {
                "example_id": r["example_id"],
                "question": r["question"],
                "reference_answer": reference_answer,
                "reasoning_trace": trace,
                "arm": "A_full",
                "source": "teacher_gpt_oss_20b",
            }
        )

        if "verif" not in trace.lower():
            c_no_header += 1
        else:
            stripped = strip_by_header_keyword(trace, "verif")
            extracted = extract_final_answer(stripped, hit_max_new_tokens=False)["extracted_answer"]
            if is_correct(extracted, reference_answer):
                arm_c.append(
                    {
                        "example_id": r["example_id"],
                        "question": r["question"],
                        "reference_answer": reference_answer,
                        "reasoning_trace": stripped,
                        "arm": "C_no_verification",
                        "source": "teacher_gpt_oss_20b_stripped",
                    }
                )
            else:
                c_verify_failed += 1

        if "reflect" not in trace.lower():
            d_no_header += 1
        else:
            stripped = strip_by_header_keyword(trace, "reflect")
            extracted = extract_final_answer(stripped, hit_max_new_tokens=False)["extracted_answer"]
            if is_correct(extracted, reference_answer):
                arm_d.append(
                    {
                        "example_id": r["example_id"],
                        "question": r["question"],
                        "reference_answer": reference_answer,
                        "reasoning_trace": stripped,
                        "arm": "D_no_reflection",
                        "source": "teacher_gpt_oss_20b_stripped",
                    }
                )
            else:
                d_verify_failed += 1

    n = len(correct_rows)
    paths = {
        "A": out_dir / f"arm_a_full_n{n}.jsonl",
        "C": out_dir / f"arm_c_no_verification_n{n}.jsonl",
        "D": out_dir / f"arm_d_no_reflection_n{n}.jsonl",
    }
    for key, records in (("A", arm_a), ("C", arm_c), ("D", arm_d)):
        with open(paths[key], "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    logger.info(f"Arm A (full): {len(arm_a)}/{n} -> {paths['A']}")
    logger.info(
        f"Arm C (no verification): {len(arm_c)}/{n} kept, "
        f"{c_no_header} had no verification header, {c_verify_failed} verify-failed after stripping -> {paths['C']}"
    )
    logger.info(
        f"Arm D (no reflection): {len(arm_d)}/{n} kept, "
        f"{d_no_header} had no reflection header, {d_verify_failed} verify-failed after stripping -> {paths['D']}"
    )

    print("\n--- Sample Arm C (no verification) ---")
    print(json.dumps(arm_c[0], indent=2)[:1500])
    print("\n--- Sample Arm D (no reflection) ---")
    print(json.dumps(arm_d[0], indent=2)[:1500])


if __name__ == "__main__":
    main()
