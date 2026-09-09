"""Post-process a raw STaR output file: truncate each star_trace at the
first sign of a hallucinated continuation (the model rambling into a new,
unrelated question after already answering), the same way scoring already
does — but scoring only used the truncated view internally and never
overwrote what got SAVED. 88.1% of own_attempt traces and 11.6% of
hint_rationalized traces in the n=2000 Colab run contained this garbage in
the saved field, which would have taught the student model to imitate
rambling into a fabricated new problem after answering.

Re-verifies each cleaned trace still reaches the correct answer after
truncation (it always should, since truncation only removes content AFTER
the answer was already stated) and reports before/after stats.

Usage:
    python scripts/clean_star_traces.py --in "/path/to/star_n2000 (2).jsonl" \
        --out results/exp04_star/star_n2000_cleaned.jsonl
"""
import argparse
import json
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.answer_extraction import extract_final_answer, is_correct  # noqa: E402

_CONTINUATION_MARKERS = [
    r"\n\s*\[Question\]",
    r"\n\s*Question\s*:",
    r"\n\s*Q\s*:",
]
_CONTINUATION_PATTERN = re.compile("|".join(_CONTINUATION_MARKERS), flags=re.IGNORECASE)


def truncate_at_continuation(response: str) -> str:
    match = _CONTINUATION_PATTERN.search(response)
    return response[: match.start()].rstrip() if match else response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path)]
    n_affected = 0
    n_still_correct = 0
    n_broke = 0
    cleaned = []

    for r in rows:
        original = r["star_trace"]
        truncated = truncate_at_continuation(original)
        was_affected = truncated != original
        n_affected += was_affected

        res = extract_final_answer(truncated, hit_max_new_tokens=False)
        still_correct = is_correct(res["extracted_answer"], r["reference_answer"])
        if was_affected:
            if still_correct:
                n_still_correct += 1
            else:
                n_broke += 1
                print(f"WARNING: ex={r['example_id']} no longer correct after truncation — dropping", file=sys.stderr)
                continue  # don't keep a record that fails re-verification

        r_clean = dict(r)
        r_clean["star_trace"] = truncated
        r_clean["was_truncated"] = was_affected
        cleaned.append(r_clean)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in cleaned:
            f.write(json.dumps(r) + "\n")

    print(f"\ninput: {len(rows)} rows")
    print(f"affected by hallucinated continuation: {n_affected} ({n_affected/len(rows):.1%})")
    print(f"  still correct after truncation: {n_still_correct}")
    print(f"  broke (no longer correct, dropped): {n_broke}")
    print(f"output: {len(cleaned)} rows written to {args.out}")


if __name__ == "__main__":
    main()
