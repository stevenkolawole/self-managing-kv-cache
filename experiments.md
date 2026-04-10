# Self-Managing KV Cache — Experiment Tracker

**Project:** Fine-tuning reasoning models to emit inline special tokens (`<FORGET id>`, summary tokens, `<BOOKMARK id>`) to manage their own KV cache during generation.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` skipped

**Models:** DeepSeek-R1-Distill-Qwen-7B (dev), DeepSeek-R1-Distill-Qwen-32B (scale), Sky-T1-32B (transfer)  
**Datasets:** MATH-500, AIME 2024, GSM8K, LiveCodeBench  
**Baselines:** ThinKV, FreeKV, LouisKV, ShadowKV, RaaS, Crystal-KV, R-KV, H2O, SnapKV, FullKV

---

## Engineering Prerequisites
> Must be built before E4/E5. Can develop in parallel with E0–E3.

| # | Task | Status | Notes |
|---|------|--------|-------|
| P1 | Segment ID scheme — define how segments are delimited and referenced by ID in inline tokens | `[ ]` | Options: sequential numbering, marker-indexed, fixed-size windows |
| P2 | Serving hook — vLLM/SGLang modification that intercepts `<FORGET id>` / summary tokens mid-decode and executes PagedAttention block eviction | `[ ]` | Reference: ThinKV CT kernel, SideQuest SGLang cursor eviction |
| P3 | KV cache simulator for training — lightweight simulator tracking which blocks are evicted during GRPO rollouts so model experiences its own decisions | `[ ]` | Closes training/inference mismatch; needed before E4 |
| P4 | Hindsight labeling script — automated pipeline for steps E2.1–E2.5 | `[x]` | `scripts/label_attention.py` — eager forward pass with hooks, populates dead_end_flags |
| P5 | Eval harness — unified runner for all benchmarks + metrics (peak KV MB, TPOT, accuracy, useful retention rate) | `[ ]` | Needed before E5 |

---

## E0 — Empirical Characterization
> **Goal:** Quantify the dead-end token problem before designing anything. Generates motivation figures; calibrates E2 labeling thresholds.  
> **Dependency:** None — run first.  
> **Models:** DeepSeek-R1-Distill-Qwen-7B  
> **Datasets:** MATH-500, AIME 2024

| # | Task | Status | Notes |
|---|------|--------|-------|
| E0.1 | Collect ~1K completed reasoning traces from 7B on MATH-500 + AIME | `[~]` | 100 traces collected (from kvcache project, MATH-500 only); AIME + scale-up pending |
| E0.2 | Curate self-correction marker vocabulary ("Wait", "Actually", "Let me reconsider", "Hmm", "That's wrong", etc.) and measure coverage | `[x]` | Vocabulary in `src/segments.py`; 51.3 markers/trace mean confirms strong coverage |
| E0.3 | Measure distribution of dead-end segment lengths (tokens preceding each self-correction marker): mean, median, p95, max | `[x]` | Segment len median 253 chars; figures in `data/e0_characterization/figures/` |
| E0.4 | Compute attention mass from all post-marker tokens to each pre-marker segment (avg across layers/heads); plot distribution | `[x]` | `scripts/label_attention.py` done; threshold 0.05 confirmed; 79/100 labeled (21 OOM on longest traces, need Blackwell node) |
| E0.5 | Measure KV cache growth curve over generation length; identify inflection points at phase transitions | `[ ]` | |
| E0.6 | Measure token repetition rates per trace (reproduce R-KV's 8–14× trace inflation claim on Qwen-7B distill) | `[x]` | 45.4% bigram repetition — corroborates R-KV; gen tokens median 2074 |
| E0.7 | Correlate self-correction markers with ThinKV's Transition-type attention sparsity pattern | `[ ]` | |
| E0.8 | Summarize findings into motivation figures for paper | `[~]` | Histograms generated; need paper-quality pass |

**Key output (2026-04-07, MATH-500 100 traces):**
- Dead-end fraction: **79% mean** (79/100 traces labeled; best run on RTX PRO 6000 Blackwell 95GB)
- Self-correction markers: **51.3/trace mean** — dense labeling signal
- Bigram repetition: **45.4%** — corroborates R-KV trace inflation
- Segment length: **253 chars median**
- Gen tokens: **2074 median**
- Attention-mass threshold for E2: **0.05**

---

## E1 — Zero-shot Elicitation Study
> **Goal:** Establish the zero-shot baseline; scope the gap that fine-tuning must close.  
> **Dependency:** None — runs in parallel with E0.  
> **Models:** 7B and 32B  
> **Datasets:** MATH-500, GSM8K

| # | Task | Status | Notes |
|---|------|--------|-------|
| E1.1 | Design prompt variants ranging from minimal to elaborate instruction | `[x]` | V1–V5 baseline/ablation (output-framing, largely ineffective); V6–V9 improved (reasoning-internal framing: rich_few_shot, identity, output_primed, recency_suffix) — in `scripts/elicit_zero_shot.py` |
| E1.2 | Run prompts on 7B — measure emission rate, structural validity, accuracy delta | `[x]` | Done: V1,6,7,8,9 — all 0% emission; V8 70% acc, V1/V9 71% |
| E1.3 | Run prompts on 32B — same metrics; check if scale provides zero-shot capability | `[x]` | Done: V8 best — 9% FORGET emission, 73% acc (above V1 baseline). V6/V7 hurt accuracy. |
| E1.4 | Score precision and recall of emitted tokens vs. E0 hindsight labels | `[ ]` | Requires E1.2/E1.3 output; focus on V8 32B (only variant with real emission) |
| E1.5 | Qualitative analysis: examples of good/bad/hallucinated token placement | `[x]` | Done: 5 behavioral patterns identified across 9 V8 32B emitting traces — see `analysis.md`. Only 1/9 traces shows target behavior (approach-switch FORGET); 8/9 show wrong-intent patterns (post-hoc bulk erasure, verification loops, self-doubt). SFT data must filter to approach-switch only. |
| E1.6 | Write up zero-shot baseline numbers for paper | `[ ]` | |

**Key output (2026-04-09, all variants complete):**
- V8 (output_primed) 32B: **9% FORGET emission, 73% acc** — only variant with meaningful emission AND accuracy gain.
- V8 qualitatively correct: proper segment tracking, FORGET at genuine dead-ends (see `analysis.md` trace_082).
- V6/V7 hurt 32B accuracy: −26pp and −21pp vs. V1. Long prompts disrupt chain-of-thought.
- 7B: 0 emissions across all variants. Fine-tuning required.
- Gap to close: 79% hindsight dead-end rate vs. 9% zero-shot emission → ~70pp.

**Next:** E1.4 precision/recall of V8 32B emissions vs. E0 hindsight labels.

---

## E2 — Hindsight Labeling Pipeline
> **Goal:** Produce the training dataset for E3 and E4.  
> **Dependency:** E0 (for attention-mass threshold).  
> **Models:** 7B (traces); optionally 32B or stronger model as annotator for summary tokens

| # | Task | Status | Notes |
|---|------|--------|-------|
| E2.1 | Collect ~10K–20K reasoning traces from 7B on MATH-500 + GSM8K train splits | `[ ]` | |
| E2.2 | Run self-correction marker detection on all traces | `[ ]` | Using vocabulary from E0.2 |
| E2.3 | Define segment boundaries (marker-to-marker); apply attention-mass threshold from E0.4 to label dead-end segments | `[ ]` | Key design decision: which layers/heads to use. E1 qualitative analysis confirms FORGET must fire mid-reasoning at approach switches — not post-hoc, not on self-doubt. Hindsight labels provide ground truth for filtering. |
| E2.4 | Generate 2–4 summary tokens per dead-end segment (using 7B or stronger annotator) | `[ ]` | Ablated in E6d; investigate using stronger annotator vs. self-annotation |
| E2.5 | Construct preferred traces (management tokens inserted) vs. rejected traces (raw) | `[ ]` | Preferred: `<FORGET id>` + summary tokens at each dead-end; rejected: unmodified |
| E2.6 | Compute dataset statistics: how many management tokens per trace on average, segment length distribution, summary token quality (spot check) | `[ ]` | |
| E2.7 | Run labeling threshold sensitivity check (vary threshold ±50%; does label set change dramatically?) | `[ ]` | Early version of E6e |

**Key output:** Training dataset of labeled trace pairs. Consider releasing as a dataset contribution.

---

## E3 — Cold-start SFT
> **Goal:** Give the model a stable initialization before RL. Acts as go/no-go gate before committing GRPO compute.  
> **Dependency:** E2  
> **Models:** 7B (primary); verify on 32B if 7B succeeds

| # | Task | Status | Notes |
|---|------|--------|-------|
| E3.1 | SFT on preferred traces from E2 — `<FORGET>` only variant (no summary tokens) | `[ ]` | Baseline for E3.3 |
| E3.2 | SFT on preferred traces from E2 — `<FORGET>` + summary tokens variant | `[ ]` | Main cold-start checkpoint |
| E3.3 | Measure management token precision and recall vs. E0 hindsight labels | `[ ]` | |
| E3.4 | Measure task accuracy on MATH-500 + AIME vs. unmodified base model | `[ ]` | Does SFT on modified traces hurt reasoning? |
| E3.5 | Measure token placement quality: are tokens appearing after self-correction markers or randomly? | `[ ]` | |
| E3.6 | Measure format validity rate: what fraction of emitted tokens are structurally valid? | `[ ]` | |
| E3.7 | **Go/no-go decision:** if precision < threshold or accuracy drops >3%, revisit E2 labeling before proceeding to E4 | `[ ]` | |

**Key output:** Initialized checkpoint for GRPO; first viability signal.

---

## E4 — GRPO Training
> **Goal:** Main training result. Teach the model to emit management tokens reliably via RL.  
> **Dependency:** E3 (cold-start checkpoint), P3 (KV cache simulator)  
> **Models:** 7B (iterate); 32B (after recipe validated)

### Reward Design
```
r = accuracy_reward + λ · memory_efficiency_reward
accuracy_reward      = 1 if final answer correct, 0 otherwise
memory_efficiency_reward = 1 − (KV bytes at answer token / FullKV baseline bytes)
```
GRPO hyperparameters (starting point): G=64 completions/question, β=0.04 KL coefficient (added directly to loss), ε=0.2 clipping.

| # | Task | Status | Notes |
|---|------|--------|-------|
| E4.1 | Implement GRPO training loop with outcome supervision (E4a) | `[ ]` | Simpler baseline; all tokens in a trace share group-normalized reward |
| E4.2 | Implement process supervision variant (E4b): per-step reward for each management token = memory saved × (1 − future attention mass to evicted segment) | `[ ]` | PAV-style; more compute but better credit assignment |
| E4.3 | Add format reward: small bonus for structurally valid management tokens | `[ ]` | Mirrors DeepSeek-R1's format reward |
| E4.4 | λ sweep: train E4a at λ ∈ {0.1, 0.3, 0.5, 0.7}; pick λ* that keeps accuracy within 1% of FullKV | `[ ]` | Reported as tradeoff curve |
| E4.5 | Train 7B with E4a (outcome supervision) at λ* | `[ ]` | Main 7B checkpoint |
| E4.6 | Train 7B with E4b (process supervision) at λ* | `[ ]` | Upper-bound variant |
| E4.7 | Train 32B at λ* after 7B recipe validated | `[ ]` | |
| E4.8 | Monitor for degenerate behaviors: `<FORGET>` spam, never-emit, accuracy collapse | `[ ]` | Tune β if needed |
| E4.9 | Measure accuracy + peak KV cache on MATH-500 + AIME after training | `[ ]` | |

**Key output:** Fine-tuned 7B + 32B checkpoints; λ–accuracy tradeoff curve.

---

## E5 — Main Results Table
> **Goal:** Head-to-head comparison against all baselines across all benchmarks.  
> **Dependency:** E4, P2 (serving hook), P5 (eval harness)  
> **Models:** 7B, 32B, Sky-T1-32B  
> **Datasets:** MATH-500, AIME 2024, GSM8K, LiveCodeBench

| # | Task | Status | Notes |
|---|------|--------|-------|
| E5.1 | Run FullKV baseline on all models × datasets (accuracy upper bound) | `[ ]` | |
| E5.2 | Run H2O and SnapKV baselines | `[ ]` | Classical attention heuristics |
| E5.3 | Run RaaS baseline | `[ ]` | |
| E5.4 | Run Crystal-KV baseline | `[ ]` | Most directly analogous heuristic |
| E5.5 | Run R-KV baseline | `[ ]` | |
| E5.6 | Run ShadowKV baseline | `[ ]` | |
| E5.7 | Run FreeKV baseline | `[ ]` | Note: two papers share this name; confirm which is the intended citation |
| E5.8 | Run LouisKV baseline | `[ ]` | |
| E5.9 | Run ThinKV baseline | `[ ]` | Most competitive; most important comparison |
| E5.10 | Run our system (zero-shot, from E1) | `[ ]` | Lower bound of our approach |
| E5.11 | Run our system (SFT only, from E3) | `[ ]` | |
| E5.12 | Run our system (GRPO outcome, from E4.5) | `[ ]` | Main result |
| E5.13 | Run our system (GRPO process, from E4.6) | `[ ]` | Upper bound variant |
| E5.14 | Compile full results table: peak KV (MB), TPOT (ms/token), accuracy (%), useful retention rate (%) | `[ ]` | |

---

## E6 — Ablations
> **Dependency:** Checkpoints from E3 and E4. Runs concurrently with E5.

### E6a — Token Type Ablation
> Which token types drive the gains?

| # | Task | Status | Notes |
|---|------|--------|-------|
| E6a.1 | `<FORGET>` only | `[ ]` | Already from E3.1 |
| E6a.2 | `<FORGET>` + summary tokens | `[ ]` | Already from E3.2 / main E4 checkpoint |
| E6a.3 | `<FORGET>` + summary tokens + `<BOOKMARK>` | `[ ]` | Full system |
| E6a.4 | Compile accuracy + KV delta across configurations | `[ ]` | |

### E6b — Training Algorithm Ablation
> How much does each training stage contribute?

| # | Task | Status | Notes |
|---|------|--------|-------|
| E6b.1 | Zero-shot (E1 result) | `[ ]` | Already done |
| E6b.2 | SFT only (E3 result) | `[ ]` | Already done |
| E6b.3 | GRPO outcome supervision (E4a) | `[ ]` | Already done |
| E6b.4 | GRPO process supervision (E4b) | `[ ]` | Already done |
| E6b.5 | GRPO + final DPO alignment pass | `[ ]` | Optional; tests whether DPO alignment adds anything after RL |
| E6b.6 | Compile accuracy + KV delta across training stages | `[ ]` | |

### E6c — Reward Weight λ Sweep
> Already partially run in E4.4 — present as a full tradeoff curve here.

| # | Task | Status | Notes |
|---|------|--------|-------|
| E6c.1 | Plot accuracy vs. memory efficiency across λ ∈ {0.1, 0.3, 0.5, 0.7} | `[ ]` | Reuses E4.4 checkpoints |

### E6d — Summary Token Count
> How many summary tokens per evicted segment are needed?

| # | Task | Status | Notes |
|---|------|--------|-------|
| E6d.1 | Train/evaluate with 1 summary token per segment | `[ ]` | |
| E6d.2 | Train/evaluate with 2 summary tokens | `[ ]` | |
| E6d.3 | Train/evaluate with 4 summary tokens | `[ ]` | Default |
| E6d.4 | Train/evaluate with 8 summary tokens | `[ ]` | |
| E6d.5 | Plot accuracy retention vs. compression ratio across counts | `[ ]` | |

### E6e — Hindsight Labeling Threshold Sensitivity
> Is the labeling pipeline robust?

| # | Task | Status | Notes |
|---|------|--------|-------|
| E6e.1 | Relabel E2 dataset with threshold × 0.5 (stricter: fewer dead-ends) | `[ ]` | |
| E6e.2 | Relabel E2 dataset with threshold × 1.5 (looser: more dead-ends) | `[ ]` | |
| E6e.3 | SFT + light GRPO on both; compare downstream accuracy and KV efficiency | `[ ]` | |

---

## E7 — Analysis
> **Dependency:** E5 results.

| # | Task | Status | Notes |
|---|------|--------|-------|
| E7.1 | False eviction analysis: what fraction of `<FORGET>` decisions evict segments that later receive non-trivial attention? Compare to hindsight-optimal labeling from E0 | `[ ]` | Main failure mode quantified |
| E7.2 | Generalization to Sky-T1-32B: use fine-tuned 7B's traces as demonstrations to prompt Sky-T1-32B; measure whether management token behavior transfers without fine-tuning | `[ ]` | |
| E7.3 | Trace length and structure: does GRPO training change overall trace length? Are traces shorter with equivalent accuracy? | `[ ]` | |
| E7.4 | Phase structure visualization: plot attention-sparsity heatmaps before/after training to show management tokens align with phase transitions | `[ ]` | |
| E7.5 | Qualitative trace examples: 3–5 full examples showing management tokens in context, what segment was evicted, what summary captured | `[ ]` | Most persuasive figure in the paper |

---

## Notes & Decisions Log
> Running record of design choices, surprises, and pivots.

| Date | Note |
|------|------|
| | |
