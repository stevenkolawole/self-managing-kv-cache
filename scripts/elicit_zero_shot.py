#!/usr/bin/env python3
"""
E1: Zero-shot elicitation of inline management tokens.
Runs one prompt variant on MATH-500 and measures emission rate,
structural validity, and accuracy.

Usage:
    python scripts/elicit_zero_shot.py \
        --model 7b --variant 1 \
        --problems data/math500_annotated.jsonl \
        --output data/e1/v1_7b.jsonl
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_model, load_traces, save_trace, extract_boxed, answers_match
from src.tokens import parse_management_tokens

# ── Prompt variants ────────────────────────────────────────────────────────────

VARIANTS = {
    1: ("minimal",
        "When you realize you've gone down a wrong reasoning path, "
        "mark it with <FORGET seg_N> (where N is the segment number) before moving on."),

    2: ("typed",
        "You have memory management tokens:\n"
        "- <FORGET seg_N>: marks segment N as a dead-end you are abandoning\n"
        "- <SUMMARY>brief note</SUMMARY>: summarizes what you learned before forgetting\n"
        "- <BOOKMARK seg_N>: pins an important segment to keep\n"
        "Use them naturally as you reason through the problem."),

    3: ("procedural",
        "As you solve problems, manage your reasoning explicitly:\n"
        "1. Number your reasoning segments mentally (seg_0, seg_1, ...).\n"
        "2. When you catch a mistake and backtrack, emit <FORGET seg_N> for the abandoned segment.\n"
        "3. Optionally precede it with <SUMMARY>brief lesson</SUMMARY>.\n"
        "4. Use <BOOKMARK seg_N> to flag segments you will refer back to.\n"
        "Emit these tokens inline without breaking your reasoning flow."),

    4: ("segment_aware",
        "You are a reasoning model with explicit KV cache management.\n"
        "Segments are bounded by self-correction markers (\"Wait\", \"Actually\", etc.).\n"
        "seg_0 is text before your first correction; seg_1 is the next, and so on.\n"
        "At each self-correction, emit exactly one of:\n"
        "- <FORGET seg_N> if the preceding segment was a dead-end\n"
        "- <SUMMARY>key insight</SUMMARY><FORGET seg_N> if it had useful partial results\n"
        "- <BOOKMARK seg_N> if it is critical and you must refer back to it"),

    5: ("few_shot",
        "When you realize you've made a mistake, emit a management token before correcting.\n\n"
        "Example:\n"
        "...I set x = 3, so x² = 9. Wait, actually I misread — x = 2. <FORGET seg_0>\n"
        "Starting over with x = 2: x² = 4...\n\n"
        "Use <FORGET seg_N> to mark abandoned reasoning. "
        "Optionally precede with <SUMMARY>lesson</SUMMARY>. "
        "Use <BOOKMARK seg_N> to pin segments you will refer back to."),
}

# ── Metrics ────────────────────────────────────────────────────────────────────

_VALID_FORGET   = re.compile(r"<FORGET\s+seg_\d+>")
_VALID_BOOKMARK = re.compile(r"<BOOKMARK\s+seg_\d+>")
_VALID_SUMMARY  = re.compile(r"<SUMMARY>[^<]*</SUMMARY>")


def emission_stats(text):
    tokens     = parse_management_tokens(text)
    n_forget   = sum(1 for t in tokens if t["type"] == "forget")
    n_bookmark = sum(1 for t in tokens if t["type"] == "bookmark")
    n_summary  = len(_VALID_SUMMARY.findall(text))
    n_total    = n_forget + n_bookmark + n_summary
    n_valid    = (len(_VALID_FORGET.findall(text)) +
                  len(_VALID_BOOKMARK.findall(text)) + n_summary)
    return {"n_forget": n_forget, "n_bookmark": n_bookmark, "n_summary": n_summary,
            "n_total": n_total, "n_valid": n_valid, "any_emitted": n_total > 0}


# ── Inference ──────────────────────────────────────────────────────────────────

def generate(model, tok, system, problem, max_new_tokens):
    import torch
    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": problem}]
    prompt  = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tok.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="7b")
    p.add_argument("--variant",        type=int, required=True, choices=list(VARIANTS))
    p.add_argument("--problems",       default="data/math500_annotated.jsonl")
    p.add_argument("--output",         required=True)
    p.add_argument("--n",              type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=16384)
    args = p.parse_args()

    name, system = VARIANTS[args.variant]
    print(f"Variant {args.variant} ({name}) | model: {args.model}")

    model, tok = load_model(args.model)
    traces     = load_traces(args.problems)
    if args.n:
        traces = traces[:args.n]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    n_correct = n_emitted = n_valid_any = 0

    for i, trace in enumerate(traces):
        problem = trace.get("problem", "")
        gt      = trace.get("ground_truth", "")
        if not problem or not gt:
            continue

        gen_text = generate(model, tok, system, problem, args.max_new_tokens)
        correct  = answers_match(extract_boxed(gen_text), gt)
        stats    = emission_stats(gen_text)

        if correct:              n_correct  += 1
        if stats["any_emitted"]: n_emitted  += 1
        if stats["n_valid"] > 0: n_valid_any += 1

        save_trace({"problem": problem, "ground_truth": gt,
                    "variant": args.variant, "variant_name": name,
                    "generated_text": gen_text, "correct": correct, **stats}, out)

        if (i + 1) % 10 == 0 or i == 0:
            d = i + 1
            print(f"[{d}/{len(traces)}]  acc={n_correct/d:.1%}  "
                  f"emission={n_emitted/d:.1%}  valid={n_valid_any/d:.1%}")

    n = len(traces)
    print(f"\n── E1 v{args.variant} ({name}) · {args.model} ──────────────────────")
    print(f"  Accuracy:       {n_correct/n:.1%}")
    print(f"  Emission rate:  {n_emitted/n:.1%}  (any management token)")
    print(f"  Valid emission: {n_valid_any/n:.1%}  (correctly formatted)")
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()
