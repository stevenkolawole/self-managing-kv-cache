#!/usr/bin/env python3
"""
Annotate existing kvcache traces with segment boundaries.
Adds segment_annotations to each trace dict and re-emits as JSONL.

Usage:
    python scripts/collect_traces.py \\
        --input /home/skolawol/workspace/kvcache/data/math500_traces.jsonl \\
        --output data/math500_annotated.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_traces, save_trace
from src.segments import segment_trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--marker_tier", default="strong")  # unused for now; kept for CLI compat
    args = p.parse_args()

    traces = load_traces(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    ok, skipped = 0, 0
    for i, trace in enumerate(traces):
        ann = segment_trace(trace)
        if ann is None:
            skipped += 1
            trace["segment_annotations"] = {"skipped": True}
        else:
            ok += 1
            trace["segment_annotations"] = ann
        save_trace(trace, out)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1}/{len(traces)}] ok={ok} skipped={skipped}")

    print(f"Done. {ok} annotated, {skipped} skipped → {out}")


if __name__ == "__main__":
    main()
