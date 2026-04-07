#!/usr/bin/env python3
"""
E0: Empirical characterization of dead-end tokens.
Expects annotated traces from collect_traces.py.
Produces stats.json + figures/.

Usage:
    python scripts/characterize.py \\
        --input data/math500_annotated.jsonl \\
        --output_dir data/e0_characterization
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_traces


def trace_stats(trace):
    text = trace.get("generated_text", "")
    if not text:
        return None
    ann = trace.get("segment_annotations", {})
    if ann.get("skipped"):
        return None

    # Segment lengths
    spans = ann.get("segment_spans", [])
    dead_flags = ann.get("dead_end_flags", [])  # may be empty until label_attention runs
    seg_lens = [s["n_chars"] for s in spans]
    dead_lens = [s["n_chars"] for s, d in zip(spans, dead_flags) if d]
    live_lens = [s["n_chars"] for s, d in zip(spans, dead_flags) if not d]
    dead_chars = sum(dead_lens)
    think_chars = sum(seg_lens) or 1

    # Bigram repetition rate
    words = text.split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    counts = Counter(bigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    rep_rate = repeated / max(len(bigrams), 1)

    # Marker positions normalised to [0, 1]
    marker_pos_norm = [p / max(think_chars, 1) for p in ann.get("marker_positions", [])]

    return {
        "n_gen_tokens":      len(trace.get("token_ids", [])) - trace.get("prompt_len", 0),
        "n_markers":         ann["n_markers"],
        "n_segments":        ann["n_segments"],
        "seg_lens":          seg_lens,
        "dead_lens":         dead_lens,
        "live_lens":         live_lens,
        "dead_fraction":     dead_chars / think_chars,
        "rep_rate":          rep_rate,
        "marker_pos_norm":   marker_pos_norm,
        "correct":           ann.get("correct", trace.get("correct", False)),
    }


def stats_summary(rows, key, label):
    vals = [r[key] for r in rows if not np.isnan(r.get(key, float("nan")))]
    if not vals:
        return {}
    a = np.array(vals)
    return {f"{label}_mean": float(a.mean()), f"{label}_median": float(np.median(a)),
            f"{label}_p95": float(np.percentile(a, 95)), f"{label}_std": float(a.std())}


def make_figures(rows, out_dir):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        print("matplotlib/seaborn not installed — skipping figures")
        return

    figs = out_dir / "figures"
    figs.mkdir(exist_ok=True)

    plots = [
        ([r["n_gen_tokens"] for r in rows],        "Generated tokens",           "kv_growth.png"),
        ([r["dead_fraction"]*100 for r in rows],   "Dead-end % of think block",  "dead_end_dist.png"),
        ([r["rep_rate"]*100 for r in rows],        "Repeated bigrams %",         "repetition.png"),
        ([r["n_markers"] for r in rows],           "Self-correction markers",    "marker_count.png"),
    ]
    for vals, xlabel, fname in plots:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(vals, bins=30, edgecolor="white", alpha=0.85)
        ax.axvline(np.median(vals), color="firebrick", linestyle="--",
                   label=f"Median: {np.median(vals):.1f}")
        ax.set_xlabel(xlabel); ax.legend(); fig.tight_layout()
        fig.savefig(figs / fname, dpi=150); plt.close(fig)

    # Marker position distribution
    pos = [x for r in rows for x in r["marker_pos_norm"]]
    if pos:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(pos, bins=20, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Marker position in think block (0=start, 1=end)")
        fig.tight_layout(); fig.savefig(figs / "marker_pos.png", dpi=150); plt.close(fig)

    print(f"Figures → {figs}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--output_dir", default="data/e0_characterization")
    p.add_argument("--no_figures", action="store_true")
    args = p.parse_args()

    traces = load_traces(args.input)
    rows = [s for t in traces if (s := trace_stats(t)) is not None]
    print(f"{len(rows)} valid traces ({len(traces) - len(rows)} skipped)")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"n_traces": len(rows)}
    for key, label in [
        ("n_gen_tokens", "gen_tokens"), ("n_markers", "markers"),
        ("dead_fraction", "dead_frac"), ("rep_rate", "rep_rate"),
    ]:
        summary.update(stats_summary(rows, key, label))

    # Flatten segment lengths
    all_seg = [n for r in rows for n in r["seg_lens"]]
    if all_seg:
        summary["seg_len_median"] = float(np.median(all_seg))
        summary["seg_len_p95"]    = float(np.percentile(all_seg, 95))

    with open(out / "stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── E0 Summary ──────────────────────────────────")
    print(f"  Traces:              {summary['n_traces']}")
    print(f"  Gen tokens (median): {summary.get('gen_tokens_median', 'n/a'):.0f}")
    print(f"  Markers / trace:     {summary.get('markers_mean', 'n/a'):.1f} mean")
    print(f"  Dead-end frac (mean):{summary.get('dead_frac_mean', 'n/a'):.1%}")
    print(f"  Bigram repetition:   {summary.get('rep_rate_mean', 'n/a'):.1%} mean")
    print(f"  Segment len (median):{summary.get('seg_len_median', 'n/a'):.0f} chars")
    print(f"  Stats → {out}/stats.json")

    if not args.no_figures:
        make_figures(rows, out)


if __name__ == "__main__":
    main()
