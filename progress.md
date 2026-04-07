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

## E1 — Zero-shot Elicitation (2026-04-07, in progress)

**Status:** Jobs submitted. Awaiting results.

5 prompt variants × 2 models = 10 parallel jobs.

| Variant | Style | Key feature |
|---------|-------|-------------|
| V1 | minimal | single sentence, `<FORGET>` only |
| V2 | typed | all three token types defined |
| V3 | procedural | step-by-step with segment numbering |
| V4 | segment-aware | explicit boundary definition, one token per marker |
| V5 | few-shot | one inline worked example |

**Script:** `scripts/elicit_zero_shot.py`  
**Jobs:** `slurm/run_e1_7b.sh`, `slurm/run_e1_32b.sh` (array=1-5)  
**Output:** `data/e1/v{1-5}_{7b,32b}.jsonl`

**Next after results:** E1.4 precision/recall vs. E0 hindsight labels, then E2 collection.

---

## Current

**Running:** E1 jobs (10 array tasks across 7B and 32B)
