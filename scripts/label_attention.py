#!/usr/bin/env python3
"""
Post-hoc eager attention pass. Computes per-segment attention mass from
post-correction tokens back to each segment, then populates dead_end_flags.

Uses forward hooks so only one layer's attention is in memory at a time.

Usage:
    python scripts/label_attention.py \
        --input  data/math500_annotated.jsonl \
        --output data/math500_labeled.jsonl \
        --model 7b [--threshold 0.05] [--max_tokens 4096]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_traces, save_trace, load_model
from src.segments import extract_think_span


def seg_token_indices(gen_text, think_start, segs, tok, prompt_len):
    """Map segment char spans (relative to think_text) → absolute token indices."""
    enc = tok(gen_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    result = []
    for seg in segs:
        cs = think_start + seg["char_start"]
        ce = think_start + seg["char_end"]
        toks = [prompt_len + i for i, (a, b) in enumerate(offsets) if a < ce and b > cs]
        result.append(toks)
    return result


def compute_masses(token_ids, seg_idx_lists, model):
    """
    One forward pass with output_attentions=True.
    Hook intercepts each layer's attention weights, accumulates per-segment
    post-correction attention mass, then replaces the tensor with None so
    it isn't stored in the output tuple (memory efficiency).
    Returns list of per-segment mean attention mass, averaged over layers.
    """
    seq_len = len(token_ids)
    mass_accum = [0.0] * len(seg_idx_lists)
    n_layers = [0]

    def hook(module, input, output):
        if not isinstance(output, tuple) or len(output) < 2:
            return
        attn_w = output[1]
        if not isinstance(attn_w, torch.Tensor):
            return
        n_layers[0] += 1
        w = attn_w[0].mean(0).cpu().float()  # [seq, seq], averaged over heads
        for i, toks in enumerate(seg_idx_lists):
            if not toks:
                continue
            seg_end = max(toks) + 1
            post = list(range(seg_end, seq_len))
            if post:
                mass_accum[i] += float(w[post, :][:, toks].mean())
        # Replace attn weights with None — frees GPU tensor; parent collects None instead
        return (output[0], None) + output[2:]

    hooks = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if all(hasattr(m, a) for a in ("q_proj", "k_proj", "v_proj", "o_proj"))
    ]
    input_ids = torch.tensor([token_ids]).to(model.device)
    try:
        with torch.no_grad():
            model(input_ids, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    n = n_layers[0] or 1
    return [m / n for m in mass_accum]


def label_trace(trace, model, tok, threshold, max_tokens):
    ann = trace.get("segment_annotations", {})
    if ann.get("skipped"):
        return False

    segs       = ann.get("segment_spans", [])
    token_ids  = trace.get("token_ids", [])
    prompt_len = trace.get("prompt_len", 0)
    gen_text   = trace.get("generated_text", "")

    if not segs or not token_ids or not gen_text:
        return False
    if len(token_ids) > max_tokens:
        ann["skipped_reason"] = f"too_long ({len(token_ids)} > {max_tokens})"
        return False

    _, think_start, _ = extract_think_span(gen_text)
    seg_idxs = seg_token_indices(gen_text, think_start, segs, tok, prompt_len)
    masses   = compute_masses(token_ids, seg_idxs, model)
    flags    = [m < threshold for m in masses]

    dead_chars  = sum(s["n_chars"] for s, f in zip(segs, flags) if f)
    total_chars = sum(s["n_chars"] for s in segs) or 1

    ann["segment_attention_mass"] = masses
    ann["dead_end_flags"]         = flags
    ann["dead_end_fraction"]      = dead_chars / total_chars
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--model",      default="7b")
    p.add_argument("--threshold",  type=float, default=0.05)
    p.add_argument("--max_tokens", type=int,   default=16384)
    p.add_argument("--max",        type=int,   default=None)
    args = p.parse_args()

    model, tok = load_model(args.model, eager=True)
    traces = load_traces(args.input)
    if args.max:
        traces = traces[:args.max]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    ok = skipped = 0
    for i, trace in enumerate(traces):
        try:
            labeled = label_trace(trace, model, tok, args.threshold, args.max_tokens)
        except Exception as e:
            print(f"[{i}] error: {e}")
            labeled = False

        if labeled:
            ok += 1
        else:
            skipped += 1
        save_trace(trace, out)
        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(traces)}] ok={ok} skipped={skipped}")

    print(f"Done. {ok} labeled, {skipped} skipped → {out}")


if __name__ == "__main__":
    main()
