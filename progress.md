# Progress Log

---

## 2026-04-07

### E0 — Empirical Characterization (MATH-500, 100 traces)

**Status:** Core pipeline done. Scale-up (1K traces, AIME) pending.

| Metric | Value |
|--------|-------|
| Dead-end fraction | **79% mean** |
| Self-correction markers | **51.3 / trace mean** |
| Bigram repetition | **45.4%** |
| Segment length | **253 chars median** |
| Gen tokens | **2074 median** |
| Attention-mass threshold | **0.05** |

Best run: job 6998624, RTX PRO 6000 Blackwell 95GB, 79/100 labeled.
L40S nodes (44GB) OOM on ~25 longest traces — always request Blackwell for label_attention.py.

Dead-end fraction is inflated by selection bias (long traces fail → short/easy traces dominate),
but 79% is a conservative floor. True value likely higher on hard problems.

**Scripts built:**
- `scripts/collect_traces.py` — annotates kvcache traces with segment boundaries
- `scripts/label_attention.py` — post-hoc eager attention pass, populates dead_end_flags
- `scripts/characterize.py` — E0 stats + figures

**What E0 tells us:**
The dead-end token problem is severe (≥79% of thinking tokens wasted) and the
self-correction signal is dense (51 markers/trace). The labeling pipeline works.

---

### Infrastructure

| Item | Status |
|------|--------|
| P4 — hindsight labeling script | done (`label_attention.py`) |
| Slurm job for E0 pipeline | done (`slurm/run_e0_label_characterize.sh`) |

---

### Infrastructure note
All slurm scripts now carry `--constraint=A100_80GB|H100|H200|RTX_PRO_6000` to avoid L40S (44GB)
OOM failures on long traces.

---

## E1 — Zero-shot Elicitation (2026-04-08)

### 32B Results (V1–V5, 100 problems each)

| V | Name | Accuracy | Any emitted |
|---|------|----------|-------------|
| 1 | minimal | **64%** | 0% |
| 2 | typed | 46% | 0% |
| 3 | procedural | 49% | **1%** (1 real `<SUMMARY>`) |
| 4 | segment_aware | 26% | 0% |
| 5 | few_shot | 56% | 0% |

True emission rate: **0.2%** (1 real token / 500 traces). Models reason *about* tokens rather than emitting them. Accuracy degrades monotonically with prompt complexity. Full analysis in `analysis.md`.

### V6–V9: Improved Variants (reasoning-internal framing)

V1–V5 framed tokens as output formatting — the wrong mental model. V6–V9 use a research-backed approach: inline examples at self-correction moments, identity priming, output priming (prefix injection), and recency suffix.

| Variant | Style | Key design |
|---------|-------|------------|
| V1 | minimal | baseline — `<FORGET>` only, single sentence |
| V6 | rich_few_shot | 3 inline examples at correction moments + rules |
| V7 | identity | persona framing ("native to how you think") + same 3 examples |
| V8 | output_primed | minimal system + prefix injected into assistant turn |
| V9 | recency_suffix | minimal system + reminder appended to user message |

**V2–V5 dropped** from 7B rerun (already 32B baseline; output-framing approach abandoned).

### Full E1 Results (2026-04-09, all runs complete)

**7B:**

| V | Acc | FORGET | BOOKMARK | Any emitted |
|---|-----|--------|----------|-------------|
| V1 | **71%** | 0 | 0 | 0% |
| V6 | 63% | 0 | 1 | 1% |
| V7 | 64% | 0 | 0 | 0% |
| V8 | 70% | 0 | 0 | 0% |
| V9 | **71%** | 0 | 0 | 0% |

**32B (all 9 variants):**

| V | Acc | FORGET | BOOKMARK | Any emitted |
|---|-----|--------|----------|-------------|
| V1 | 64% | 0 | 0 | 0% |
| V2 | 46% | 0 | 0 | 0% |
| V3 | 49% | 0 | 0 | 1% (SUMMARY) |
| V4 | 26% | 0 | 0 | 0% |
| V5 | 56% | 0 | 0 | 0% |
| V6 | 38% | 0 | 0 | 0% |
| V7 | 43% | 0 | 0 | 0% |
| **V8** | **73%** | **9** | **0** | **9%** |
| V9 | 54% | 1 | 2 | 3% |

**Winner: V8 (output_primed)** — 9% FORGET emission on 32B, highest accuracy of all variants (+9pp above V1 baseline). 7B produces 0 emissions regardless of variant — needs SFT.

**Critical caveat on V8 emission quality:** All 9 emitting traces were read and classified. Only 1 of 9 (trace_082: `3^{2x}+19=10^x`) shows genuine target behavior — FORGET at a real approach switch, mid-reasoning, before the answer is found. The other 8 show wrong-intent patterns:
- Post-hoc bulk erasure (trace_040, trace_094): all segments wiped after solving correctly
- Pathological verification loop (trace_075): re-derives same correct answer 11 times, forgets each
- Self-doubt on correct reasoning (trace_050, trace_086): FORGET emitted when no mistake was made
- False triggers (trace_025, trace_046, trace_096): FORGET at end, recanted, or on correct steps

**SFT implication:** V8 emitting traces cannot be used naively as positive training examples. Doing so would teach bulk-erasure and anxiety-loop patterns. Only FORGET-at-approach-switch traces are valid positives. E2 hindsight labeling is the primary data source.

**Script:** `scripts/elicit_zero_shot.py`  
**Output:** `data/e1/v{N}_{7b,32b}.jsonl` + `_meta.jsonl` sidecar + `generated/` individual txt files

---

## Current

**Status:** E1 complete.  
**Next:** E1.4 precision/recall of V8 32B emissions vs. E0 hindsight labels → E2 trace collection (10K–20K)
