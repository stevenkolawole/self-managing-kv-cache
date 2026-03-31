# Self-Managing KV Cache via Inline Management Tokens — Technical Research Overview

*Last updated: March 2026*

---

## 1. Problem Statement and Motivation

### The Core Problem

Transformer-based reasoning models — DeepSeek-R1, QwQ, NVIDIA AceReason, Sky-T1, and their distilled variants — generate extremely long chains of thought during inference. A single AIME problem may require 20,000–60,000 tokens of internal reasoning before producing a final answer. The KV cache grows linearly with sequence length:

```
Memory ≈ 2 × n_layers × n_heads × d_head × seq_len × bytes_per_element
```

For DeepSeek-R1-Distill-Qwen-32B at 60k tokens (BF16): approximately 200GB — far beyond a single H100's 80GB. At 7B parameters, a 30k-token reasoning trace consumes ~20GB, dominating the available memory budget and drastically reducing viable batch sizes.

The standard response is **KV cache eviction**: retain only a budget of K token entries and discard the rest. But this raises the central question: *which tokens are worth keeping?*

### The Structural Failure of All Existing Methods

Every existing KV cache management method — H2O, SnapKV, ThinKV, RaaS, Crystal-KV, R-KV, FreeKV, ShadowKV, LouisKV — makes eviction decisions based on **past attention patterns**. The assumption is that tokens which have been heavily attended to in the past are likely to be important in the future. This assumption is systematically wrong for one specific and common event in reasoning traces: **dead-end branches**.

When a reasoning model self-corrects — generating tokens like "Wait, that approach is wrong. Let me try a different method..." — the preceding hundreds or thousands of tokens of exploratory computation have a precisely zero probability of being referenced again. They are semantically inert from the moment the self-correction begins. But from the perspective of any backward-looking system:

- Their **cumulative attention scores** are high (they were actively referenced during the exploration)
- Their **recent attention scores** are high (they were referenced moments ago)
- Their **key-vector norms** are high (they participated in dense computation)

All existing signals therefore *retain* these tokens, exactly when they should be *evicted immediately*. The methods are not merely imprecise — they are structurally incapable of detecting dead-end branches at the moment the branches are abandoned. Detection happens only gradually, as the post-correction trace accumulates attention to different tokens and the dead-end tokens' relative scores decay. By then, the tokens have occupied cache for thousands of additional steps.

This failure mode is not a marginal inefficiency. R-KV (2025) documents that reasoning traces are 8–14× longer than reference solutions due to the accumulation of exploratory and self-correction content. Crystal-KV (2026) finds that removing dead-end tokens can *improve* final-answer accuracy (105% of full-cache performance at 16% KV budget on AIME 2024), not merely preserve it — the dead-end tokens are not just wasteful, they are actively misleading subsequent attention.

### The Core Hypothesis

The only agent that can detect a dead-end at the moment of abandonment is the model generating the trace. When the model generates "Wait, that approach doesn't work," it has — implicitly — evaluated the preceding branch and decided it is worthless. This semantic judgment exists in the model's internal state before the self-correction tokens are even generated. No external system can access it; no backward-looking signal can substitute for it.

**Our hypothesis:** A reasoning model can be fine-tuned to externalize this judgment by emitting special inline management tokens within its reasoning trace. The model learns to prospectively signal — at the moment of self-correction — which prior segments should be evicted from the KV cache, and to emit compact summary tokens capturing any residual information worth preserving from those segments before they are evicted.

This is a strictly forward-looking approach. The model is the only observer that knows which segments are dead before they become dead, and fine-tuning teaches it to act on that knowledge in real time.

### What This Approach Is Not

It is important to distinguish from two adjacent ideas:

**Not a backward-looking heuristic**: H2O, ThinKV, Crystal-KV, and RaaS all operate on past attention patterns. Our model emits management tokens *while* generating the self-correction, before the dead-end tokens' attention signatures have degraded.

**Not an auxiliary thread**: SideQuest (2026) uses a separate parallel generation thread that evaluates which *tool outputs* in the agentic context are stale. Our management tokens are inline within the primary reasoning trace. There is no auxiliary thread, no separate context, and no restriction to tool outputs — the model manages its own self-generated reasoning tokens.

### The Dead-End Token Lifecycle

```
Step 1 (t=0):     Model begins exploring approach A
                   KV cache accumulating: [Q][approach_A_tokens...]
                   All methods: retain these tokens (actively attended)

Step 2 (t=5000):  Model generates: "Wait, approach A is wrong."
                   All backward methods: still retain (high cumulative attention)
                   Our method: <FORGET seg_A> emitted → evict immediately

Step 3 (t=5001):  Model begins approach B
                   All backward methods: seg_A still in cache, consuming memory
                   Our method: seg_A gone; cache has space for approach B
```

The gap between Step 2 and the eventual eviction under backward methods is the entire remaining reasoning trace — potentially thousands of steps. At 60k tokens with 30k dead-end tokens, this represents up to 50% wasted cache capacity.

---

## 2. Proposed Approach

### Token Vocabulary

Three special tokens are added to the model's vocabulary and the model is trained to emit them inline within the `<think>...</think>` block:

**`<FORGET seg_id>`** — Signals that segment `seg_id` (a contiguous span of previously generated reasoning tokens, identified by a segment ID scheme) is semantically closed and should be evicted from the KV cache immediately. The serving layer intercepts this token during generation and executes the eviction.

**`<SUMMARY>...(tokens)...</SUMMARY>`** — Emitted immediately before a `<FORGET>`, this wraps 2–8 dense summary tokens that capture any residual useful information from the segment before eviction. The summary tokens remain in cache after the segment is evicted, functioning as a compressed proxy for the segment's content. This replaces the original `<COMPRESS>` idea, which required an online quantization kernel — summary tokens are learned behavior, no kernel required.

**`<BOOKMARK seg_id>`** — Pins segment `seg_id` and exempts it from any future eviction pressure. Used for segments containing key intermediate results, definitions, or constraints that the model anticipates needing throughout the remainder of the trace.

### The Summary Token Design

The decision to replace `<COMPRESS id bits>` with summary tokens deserves explicit motivation. `<COMPRESS>` would require:
1. The model to specify a target bit-width
2. A serving-layer kernel to quantize existing KV entries mid-sequence
3. Online integration of mixed-precision attention

Summary tokens instead require only that the model learn to write dense, semantically rich representations of dead-end content — a capability directly in-distribution with the model's existing pre-training. CCoT (Cheng & Van Durme, 2024) demonstrates that models can be trained to compress reasoning chains into a small number of dense "contemplation tokens" with minimal accuracy loss. ShadowKV (2024) independently validates that per-chunk mean vectors ("landmarks") are effective approximate attention proxies. Both confirm that the summary-token approach is achievable and sufficient.

### Segment Identification

The segment ID scheme must be defined before training. The primary candidate is **marker-indexed segmentation**: a segment begins at either the start of the trace or the immediately preceding self-correction marker, and ends at the current self-correction marker. This is natural, requires no additional tokens, and aligns with the semantic unit that hindsight labeling targets.

Alternative: **fixed-size windowed segmentation** (e.g., every 512 tokens gets a segment ID), which is simpler to implement but loses semantic alignment.

The choice is an empirical question; see E2 design decisions.

### Training Pipeline

The model is trained in three stages:

**Stage 0 — Zero-shot elicitation (baseline, not a training stage):** Evaluate whether instruction prompting alone can elicit useful management tokens. This bounds the fine-tuning gap.

**Stage 1 — Cold-start SFT:** Supervised fine-tuning on hindsight-labeled preferred traces. Hindsight labeling: given a completed correct trace, identify dead-end segments (spans preceding self-correction markers with near-zero future attention mass), insert optimal `<FORGET>` + summary tokens into the preferred trace, construct (preferred, rejected) pairs. SFT on the preferred traces gives the model a stable initialization before RL.

**Stage 2 — GRPO with joint reward:** Group Relative Policy Optimization with:
```
r = accuracy_reward + λ · memory_efficiency_reward
accuracy_reward = 1 if final answer correct, 0 otherwise
memory_efficiency_reward = 1 − (KV bytes at answer token / FullKV baseline)
```
Process supervision (step-level rewards for individual management token decisions) preferred over outcome-only supervision for credit assignment; see Section 5 on training design.

**Optional Stage 3 — DPO alignment:** A final direct preference optimization pass for preference alignment after RL convergence.

---

## 3. Related Work — Detailed Comparison

### 3.1 H2O — Heavy-Hitter Oracle (Zhang et al., NeurIPS 2024)

**Core mechanism:** Tracks cumulative attention received by each token across all decode steps. Tokens accumulating the largest share of total attention ("heavy hitters") are retained; others are evicted. Budget is split evenly between heavy hitters and the most recent tokens.

**Novelty:** First principled, theoretically grounded (dynamic submodular maximization) approach to eviction-based KV cache management. Demonstrates that 5% of tokens receive most attention mass, validating the viability of aggressive eviction.

**What it misses:** In reasoning traces, cumulative attention is precisely the wrong signal for dead-end detection. Dead-end tokens accumulate high cumulative scores during the exploration phase — they are retained exactly when they should be evicted. The 50/50 heavy-hitter/recency split is fixed and untuneable. Eviction is irreversible: once a token is evicted, it cannot be recovered for non-monotonic tasks. The bias toward older tokens (which have had more time to accumulate scores) penalizes recent but important computation.

**Relevance:** Primary baseline for attention-based eviction. Our method outperforms H2O by detecting dead-ends at abandonment time rather than waiting for cumulative scores to decay. The submodular framing provides a theoretical template if guarantees are needed.

---

### 3.2 SnapKV (Li et al., NeurIPS 2024)

**Core mechanism:** Prefill-time eviction only. Uses the attention pattern over the last α prompt tokens to identify which context tokens each attention head will need during generation. Retains top-K per head. Selection is done once before generation begins; zero per-step overhead.

**Novelty:** Observation-window-based: the model's own question-processing attention fingerprint identifies relevant context. Stable per-head attention profiles validate once-at-prefill selection.

**What it misses:** Entirely inapplicable to long generation traces. SnapKV cannot know at prefill time which tokens the model will generate 15,000 decode steps later. It solves the long-input problem, not the long-output problem. The assumption of stable attention fingerprints also breaks down for self-generated tokens, which evolve during generation.

**Relevance:** Upstream complement. SnapKV handles the prompt KV cache; our inline tokens handle the growing generation KV cache. The two are designed to compose, not compete.

---

### 3.3 RaaS — Reasoning-Aware Attention Sparsity (Hu et al., arXiv 2502)

**Core mechanism:** Characterizes two token categories empirically:
- **Milestone tokens**: heavily attended during an intermediate reasoning step, permanently irrelevant once that step concludes. Premature eviction forces repetition loops.
- **Phoenix tokens**: low attention for long stretches, then surge back at synthesis. Nearly always in the prefill (user query), not the generated trace.

Algorithm: LRU eviction with timestamps refreshed whenever a token ranks in the top-50% of attention at any step. All prefill tokens preserved unconditionally. Achieves O(L) memory and O(L) time simultaneously.

**Novelty:** First empirical characterization of milestone vs. phoenix as distinct categories. LRU timestamps handle temporal importance patterns that cumulative attention cannot. Demonstrates that H2O-style eviction causes a 24.2% "failure rate" (attention map anomalies forcing repetition loops) on reasoning traces — establishing the severity of the problem.

**What it misses:** Still backward-looking: milestone tokens are detected only as their LRU timestamp ages out, not at the moment of abandonment. The predictive gap remains. Long prefill preservation may exhaust the budget if the user prompt is large. No evaluation outside math reasoning benchmarks.

**Relevance:** The milestone/phoenix characterization directly grounds our work — it provides empirical evidence that reasoning traces have structurally distinct token utility patterns. Our `<BOOKMARK>` token is the model-driven equivalent of "unconditional prefill preservation": the model explicitly marks its own load-bearing intermediate results rather than relying on the heuristic that all prefill tokens are important.

---

### 3.4 ThinKV (Ramachandran et al., ICLR 2026 Oral)

**Core mechanism:** Classifies reasoning trace tokens into three thought types via KDE on attention sparsity from 4 representative layers, refreshed every 128 decode steps:
- **Reasoning (R):** systematic deduction; moderate sparsity; highest importance → FP8 (8-bit)
- **Execution (E):** calculation/code; dense attention; moderate importance → NVFP4 (4-bit)
- **Transition (T):** backtracking/uncertainty; highest sparsity; lowest importance → Ternary (2-bit)

Combines differential quantization (TBQ) with progressive eviction triggered at Transition detection (TBE, retention schedule {64, 32, 16, 8, 4} tokens per aging segment). Custom PagedAttention kernel extension (CT) for in-place eviction without gather compaction. Near-lossless at <5% KV retention; 5.8× throughput improvement over SOTA.

**Novelty:** First reasoning-aware compression at thought-segment granularity. Counterfactual KL-divergence analysis establishing importance ordering R >> E >> T. CT kernel eliminating compaction overhead. Best SOTA on reasoning-model KV compression.

**What it misses:** Classification is backward-looking — thought type is assigned after the segment is generated, using aggregate statistics over the most recent 128 tokens. There is no prospective detection: ThinKV identifies that a segment was Transition-type *after* it has already been generated and accumulated in cache. The 4-token minimum retention floor prevents complete segment eviction. Offline calibration of sparsity thresholds per model family. Classification is attention-sparsity-based, not semantic: it detects the statistical *properties* of attention, not the *meaning* of the segment.

**Relevance:** The primary system-level baseline. Our approach most directly extends ThinKV's insight that thought types should drive compression decisions, by having the model declare its own thought-type boundaries via inline tokens rather than requiring the serving system to infer them. ThinKV's TBE retention schedule ({64, 32, 16, 8, 4}) is a concrete policy to compare against when the model itself specifies eviction scope. The CT kernel is the implementation reference for serving-layer eviction infrastructure.

---

### 3.5 Crystal-KV (arXiv 2601)

**Core mechanism:** Distinguishes two empirical categories in reasoning KV entries:
- **CrystalKV:** Intermittent but durable attention patterns; these tokens support final answer correctness and must be retained.
- **SlipKV:** Strong local attention that fades as reasoning progresses; these tokens maintain flow locally but become misleading context.

Scores tokens using a Combined Recency and Frequency (CRF) metric with exponential decay: `CRF(t) = λ^(t−tᵢ) · CRF(tᵢ) + 1{hit at t}`. Tokens whose CRF drops below threshold are classified SlipKV and evicted. Layer-wise adaptive budget allocation based on utilization. 90.89% memory savings; 7.57× throughput; up to 12.24× on 16K reasoning traces. On AIME, *removing* SlipKV achieves 105% of full-cache accuracy at 16% KV budget.

**Novelty:** CrystalKV/SlipKV dichotomy is the most granular empirical characterization of reasoning-trace KV utility patterns. The finding that selective eviction improves accuracy is a strong empirical argument for the value of dead-end detection. Direct application to DeepSeek-R1 distilled models.

**What it misses:** Still backward-looking: CRF decays reflect past access patterns, not prospective needs. The ground-truth CrystalKV classification requires knowing which think-phase tokens influenced the final answer — unavailable during generation, approximated by the CRF proxy. Hyperparameter sensitivity: λ and top-p require per-task tuning. Narrow evaluation (only DeepSeek-R1 distills; only math and code).

**Relevance:** Crystal-KV's CrystalKV/SlipKV dichotomy is precisely the distinction our system trains models to make themselves via inline tokens. `<BOOKMARK>` ≈ "this is CrystalKV"; `<FORGET>` ≈ "this has become SlipKV". The 105% accuracy finding is one of the strongest empirical arguments for our training objective: the model should learn to call these transitions *in advance*, which Crystal-KV cannot do. The LRFU score provides a computable signal for hindsight labeling in E2.

---

### 3.6 R-KV — Redundancy-Aware KV Cache Compression (arXiv 2505)

**Core mechanism:** Addresses a failure mode of all attention-based importance scoring on reasoning models: reasoning traces contain heavy token repetition (8–14× trace inflation over reference solutions), and repeated tokens accumulate unfairly high attention scores. Adds an explicit redundancy penalty on top of importance scoring: `Z = λ·I − (1−λ)·R`, where I is attention-based importance and R is cosine similarity to other retained tokens. Optimal at λ≈0.1 (heavily weights redundancy removal). Achieves O(N) memory with constant footprint via segment-based buffer management.

**Key results:** Full performance at 34% KV budget on MATH-500; 105% of full-cache on AIME-2024 at 16% budget (removing redundancy improves accuracy). 6.6× throughput improvement.

**What it misses:** Evaluated only on DeepSeek-R1 distilled models, only math benchmarks. O(B²) cosine similarity matrix computation. Training-free ceiling: cannot teach the model to generate less redundant traces, only to compress post-hoc.

**Relevance:** The 8–14× trace inflation statistic quantifies the opportunity our system targets. More importantly, R-KV's training-free ceiling is exactly what our approach overcomes: if the model emits `<FORGET>` + summary tokens immediately when it starts repeating itself, the redundant tokens never accumulate in cache. Our summary tokens replace the repetitive recapitulation at the source, which is strictly better than post-hoc deduplication.

---

### 3.7 SideQuest (Kariyappa & Suh, NVIDIA, arXiv 2602)

**Core mechanism:** Long-horizon agentic tasks accumulate tool responses (retrieved documents) in the KV cache. Every K=4 turns, a parallel auxiliary generation thread is triggered via "Memory management mode" in the context, causing the fine-tuned model to output structured JSON deletion commands identifying which cursor-indexed tool outputs are stale. Stale cursors are evicted from the KV cache. The auxiliary thread's tokens are then discarded. Fine-tuned on 215 traces with hindsight-annotated last-use indices plus logit distillation (λ=500) to prevent forgetting. 56–65% token reduction; +83.9% throughput; ≤2% accuracy loss.

**Novelty:** First demonstration that a reasoning model can be trained to actively manage its own KV cache via structured generated output. Proves that hindsight supervision (using a stronger model to annotate last-use indices) is sufficient training signal. Proves that 215 training examples produces meaningful generalization. The parallel-thread design prevents management output from contaminating the primary reasoning context.

**What it misses:** Scope restricted to tool responses (discrete cursor-identified documents). The model's own CoT reasoning tokens — the dominant memory consumer for single-turn reasoning — are completely untouched. For tasks without tool calls, SideQuest provides zero compression. SideQuest's own authors explicitly flag "thought pruning" as future work.

**Relevance:** This is the most directly analogous prior work. SideQuest proves the paradigm; our paper extends it in three concrete ways: (1) inline rather than auxiliary thread — management tokens appear in the primary reasoning stream, (2) token-level rather than document-level granularity, (3) targeting self-generated CoT tokens rather than external tool outputs. The hindsight annotation approach and the distillation loss coefficient (λ=500) are directly applicable. The parallel-thread design is a concrete architectural alternative we should ablate against (Experiment E6b).

---

### 3.8 FreeKV (Liu et al., ICLR 2026 / arXiv 2505)

**Core mechanism:** KV retrieval method (all tokens kept in CPU memory; serving system selects which to bring to GPU per decode step). Key observation: adjacent decode steps have >0.84 query vector cosine similarity across most attention heads, meaning the important KV set barely changes step-to-step. Speculatively reuses the prior step's retrieved KV set (shifted off the critical path), with fine-grained per-head correction when similarity drops below threshold τ. Hardware-aware dual memory layout (NHD on GPU, HND on CPU) + double-buffered streamed recall hides PCIe latency. 13× speedup over prior retrieval methods; near-lossless accuracy.

**What it misses:** Retrieval, not eviction: all tokens still reside in CPU memory; no reduction in total memory footprint. Selection is reasoning-agnostic (query similarity, not semantic importance). The >0.84 similarity assumption may break during reasoning phase transitions where the model's focus shifts sharply — precisely the moments when our management tokens would fire.

**Relevance:** The >0.84 inter-step similarity finding is directly actionable for training: our model should emit management tokens *sparsely*, only at genuine phase transitions when similarity drops. If the model can sense when its own query intent is shifting (a metacognitive capability DeepSeek-R1-Zero spontaneously develops), it can learn to emit tokens at the right moments. FreeKV's infrastructure (NHD/HND dual layout, double-buffered recall) is relevant for the serving layer when managing the warm tier of non-evicted tokens.

---

### 3.9 ShadowKV (Sun et al., arXiv 2410)

**Core mechanism:** Pre-RoPE keys are extremely low-rank (sharp singular value decay). SVD compresses pre-RoPE keys to rank r=160 during prefill; chunk landmarks (per-chunk key means) stored as retrieval indices. Values offloaded to CPU. Per-step: compute approximate attention over landmarks, fetch top-k value chunks from CPU via PCIe (overlapped with computation via CUDA streams). Adjacent steps have >60% overlapping chunk access patterns; misses-only fetching reduces PCIe traffic significantly.

**Key results:** 3.04× throughput improvement; 6× larger batch sizes; matches full attention at 1.56% sparse KV budget across 6 models on RULER, LongBench, NIAH at up to 1M tokens.

**Relevance:** Two distinct relevances: (1) **Landmark chunks** — ShadowKV's per-chunk mean vectors serving as compressed attention targets are structurally equivalent to our summary tokens. The finding that a mean vector is a sufficient proxy for chunk-level attention validates the summary token design independently. (2) **Infrastructure** — ShadowKV's dual-precision (compressed keys, full values) and CPU offloading architecture is the reference design for a warm tier in any hybrid eviction/retrieval system that builds on our approach.

---

### 3.10 LouisKV (Wu et al., arXiv 2510)

**Core mechanism:** Observes temporal locality in KV access patterns: critical entries matter in bursts, then become less relevant. Also observes that input-prompt KV and output-generated KV have structurally different distribution patterns requiring different strategies. Semantic-aware retrieval triggers fire only at semantic boundaries (not every decode step). Decoupled fine-grained management for input vs. output segments. Custom Triton/CUDA kernels for KV clustering. 4.7× speedup over SOTA retrieval methods; near-lossless accuracy on long-input-long-output scenarios.

**Relevance:** The temporal locality observation directly motivates where our model should emit management tokens: at semantic boundaries it naturally produces (end of reasoning step, self-correction marker). LouisKV's decoupled input/output strategy suggests our system should treat prompt-derived KV (which has phoenix-like patterns per RaaS) differently from generation-derived KV (which is what our tokens target). Sparse trigger design aligns with FreeKV's high inter-step similarity finding: emit tokens rarely, at genuine transitions.

---

### 3.11 DPO — Direct Preference Optimization (Rafailov et al., NeurIPS 2023)

**Core mechanism:** Eliminates the explicit reward model and RL loop from RLHF. Derives a closed-form equivalence between the RLHF objective and a binary cross-entropy loss over preference pairs (x, y_w, y_l):

```
L_DPO = −E[log σ(β log[π_θ(y_w|x)/π_ref(y_w|x)] − β log[π_θ(y_l|x)/π_ref(y_l|x)])]
```

The policy π_θ is the only parameter; π_ref is frozen. No reward model, no PPO, no rollouts.

**Key limitation for our use case:** DPO is outcome-level. It rewards or penalizes entire traces, not individual token decisions. A `<FORGET>` token at position 500 that was excellent cannot receive credit if the trace-level outcome is incorrect for an unrelated reason. Process-level rewards — needed to credit individual management token decisions — are not natively supported.

**Relevance:** Viable for a final preference-alignment stage after GRPO convergence (Stage 3 in training pipeline). Simpler infrastructure than GRPO. β controls KL from the reference policy, preventing degenerate management token spam. Not the right primary training algorithm for learning *when* to emit management tokens.

---

### 3.12 DeepSeek-R1 (DeepSeek-AI, arXiv 2501) and GRPO

**Core mechanism:** Multi-stage training pipeline: (1) cold-start SFT on a small set of long CoT examples to stabilize format and avoid RL instability, (2) GRPO RL on verifiable rewards (math correctness, code test cases, format compliance), (3) rejection-sampling SFT on RL-generated traces mixed with general data, (4) final GRPO on mixed reasoning + alignment rewards.

**GRPO specifics:** For each question, sample G=64 completions. Estimate advantage via group-normalized rewards: `Â = (r − mean(r)) / std(r)`. No critic/value network required. KL penalty added directly to loss (β=0.04, not as reward penalty). PPO clipping (ε=0.2) retained. Two modes: outcome supervision (all tokens share trace-level reward) and process supervision (each step gets step-level reward; token advantages accumulate forward).

**Emergent behaviors under pure RL (R1-Zero):** Self-reflection (model revisits and corrects earlier reasoning), extended chains on harder problems, spontaneous backtracking with self-correction markers — the precise behaviors our system targets.

**Key limitations:** RL instability at small scales (direct RL on 7B produces poor reasoning; cold-start SFT first is necessary). Prompt sensitivity. Verifiability gap: RL works well for math/code where reward is binary and automatic; harder for open-ended tasks.

**Relevance:** GRPO is the training algorithm for Stage 2. The cold-start SFT → GRPO sequence directly mirrors DeepSeek-R1's pipeline. The emergent self-reflection finding is the most important: if models spontaneously develop self-correction behavior under RL pressure, they can plausibly learn to emit management tokens at those corrections under an additional memory efficiency incentive. The distillation pathway (large model learns via RL → SFT distillation into smaller model) gives us a deployment story: train on 32B, distill inline-token behavior into 7B.

---

### 3.13 Compressed Chain of Thought — CCoT (Cheng & Van Durme, arXiv 2412)

**Core mechanism:** Inserts special "contemplation tokens" into the generation stream — dense continuous vectors representing entire intermediate reasoning steps in compact form. The model is trained to emit a variable number of contemplation tokens instead of verbose step-by-step reasoning, then decode from them back to text (making them interpretable). Token count is controllable at inference time for a performance-latency tradeoff.

**Relevance:** Direct precedent for the summary token design. CCoT proves that models can be trained to compress reasoning chains into a small number of special dense tokens with minimal accuracy loss, and that these tokens are decodable to text (semantically grounded). For our system, the summary tokens before a `<FORGET>` are analogous to contemplation tokens: they capture the useful content of a dead-end branch before the branch is evicted. The variable-count design (2–8 tokens per segment in our ablation E6d) mirrors CCoT's controllable compression ratio.

---

### 3.14 Recurrent Memory Transformer — RMT (Bulatov et al., NeurIPS 2022)

**Core mechanism:** Augments a standard Transformer with memory tokens prepended and appended to each input segment. Output memory tokens become input memory tokens for the next segment, implementing a trained recurrence. Memory tokens have no fixed semantics; they learn to carry whatever information future segments need. Handles sequences far beyond training context length.

**Relevance:** Architectural proof-of-concept for the inline-token approach. RMT shows that a model can learn to write summary information into special tokens at segment boundaries and read it back later — effectively learning a KV management protocol through training. Our `<SUMMARY>...<FORGET>` pattern is the within-sequence generalization of RMT's cross-segment memory: the model learns to write a memory capsule at abandonment time rather than at a fixed segment boundary.

---

### 3.15 Vision Transformers Need Registers (Darcet et al., ICLR 2024)

**Core mechanism:** Background image tokens in ViTs develop anomalously high norms because the model co-opts them as scratch space for computations that don't fit into patch tokens. Adding designated "register tokens" provides explicit scratch space, freeing patch tokens to represent clean features. Eliminates high-norm artifact tokens; improves dense prediction.

**Relevance:** Provides empirical evidence that transformer models spontaneously develop a need for scratch-space tokens and will hijack other tokens if none are designated. The inline management tokens in our system are a controlled, explicit version of this: rather than hoping the model discovers scratch-space tokens (and hijacks semantically important positions), training designates management tokens for this purpose. ViT Registers also shows that the spatial attention maps become cleaner and more interpretable after register tokens are added — an analogous "cleaning" effect may appear in reasoning traces after the model learns to offload dead-end content into `<FORGET>` + summary patterns.

---

### 3.16 Process Reward Models — PRM (Lightman et al., ICLR 2024) and PAV (Setlur et al., arXiv 2410)

**PRM core mechanism:** Human annotators label each step in a multi-step math solution as correct or incorrect. A process reward model predicts step-level correctness. At inference time, best-of-N search uses the PRM to score solutions step-by-step rather than only at the end.

**PAV core mechanism:** Instead of predicting step correctness, the PAV model measures *progress* — the change in probability of reaching a correct final answer after taking a step. Computed by rolling a weaker prover forward from each step: `progress(step j) = P(correct | step j done) − P(correct | step j not done)`. Weaker provers suffice for useful progress signals. >8% accuracy improvement over ORM at 1.5–5× better compute efficiency.

**Relevance:** Both papers define the training paradigm for step-level rewards, directly applicable to our process supervision variant (E4b). For each `<FORGET>` emission, the per-step reward is:
```
step_reward = memory_saved(seg_id) × (1 − future_attention_mass_to_evicted_segment)
```
The second factor requires rolling a prover forward from the post-eviction state — exactly PAV's approach. This is expensive but provides superior credit assignment: the model is rewarded not for evicting tokens, but for evicting tokens that it genuinely won't need. PAV's finding that weaker provers suffice means we can use the 7B model itself as the prover for 32B training.

---

### 3.17 Quest — Query-Aware Sparse KV Retrieval (Tang et al., ICML 2024)

**Core mechanism:** Segments KV cache into fixed-size pages; stores per-page min/max key statistics. At each decode step, uses the current query vector to estimate each page's criticality via element-wise products with min/max statistics. Loads only top-K pages for full attention. 7.03× overall latency reduction; handles up to 1M token contexts.

**Relevance:** Quest and our approach are synergistic, not competitive. Our inline tokens aggressively evict provably unneeded KV entries; Quest efficiently retrieves the remaining entries per decode step. Combined: (1) inline tokens eliminate dead-end segments permanently, (2) Quest efficiently serves the surviving segments. The composability is architectural: our serving hook runs at eviction time; Quest runs at attention time.

---

### 3.18 StreamingLLM / Attention Sinks (Xiao et al., ICLR 2024)

**Core mechanism:** First formal characterization of attention sinks: initial tokens (positions 0–4) consistently receive disproportionate attention regardless of content, because softmax requires sum-to-1 and these tokens serve as a "dump." Fix: retain sink tokens + sliding recent window. Enables stable generation up to 4M tokens.

**Relevance:** Our `<BOOKMARK>` token is an active, semantically meaningful attention sink. StreamingLLM's finding that adding a pre-designated sink token during pre-training makes the role more stable is relevant: if management tokens are added to the vocabulary, initializing them as attention sinks may improve training stability. The attention sink observation also informs what should *not* be `<FORGET>`-ed: initial tokens, like sinks, tend to be important across long contexts.

---

## 4. Gaps Closed by This Work

### Gap 1: Prospective Dead-End Detection

**Status of all prior work:** ThinKV (post-hoc Transition classification), RaaS (LRU timestamp decay), Crystal-KV (CRF decay), R-KV (redundancy accumulation) — all detect dead-end tokens *after* they have ceased to be useful, relying on the accumulation of disuse evidence. The detection lag is the entire remaining reasoning trace after abandonment.

**What we close:** The model detects and signals dead-end branches at the moment of abandonment — when "Wait, that's wrong" is generated, not after 5,000 additional tokens of alternative reasoning have diluted the dead-end tokens' relative attention scores. This is a structural improvement, not a marginal one. No attention heuristic can replicate it because attention heuristics are inherently backward-looking.

### Gap 2: Token-Granular Self-Management of CoT Tokens

**Status:** SideQuest manages tool responses at document/cursor granularity. No method manages the model's own CoT reasoning tokens via model-generated signals.

**What we close:** The model emits management signals at token-span granularity within its own CoT. This is the direct extension SideQuest identifies as future work ("thought pruning"). The inline token design avoids SideQuest's auxiliary thread complexity.

### Gap 3: Lossless Compression via Summary Tokens

**Status:** All eviction methods (ThinKV, H2O, Crystal-KV) permanently destroy evicted token information. Retrieval methods (FreeKV, ShadowKV) avoid loss but preserve all tokens in CPU memory. No method allows the model to selectively compress a segment's content into a denser representation before eviction.

**What we close:** `<SUMMARY>...<FORGET>` creates a compact representation of a dead-end segment before eviction. Information from the dead-end that is genuinely useful (e.g., the reason an approach failed) is preserved in summary tokens and remains accessible to subsequent attention. This is strictly superior to pure eviction (which destroys all information) and more efficient than full retrieval (which preserves all information at full cost).

### Gap 4: FlashAttention Compatibility (Distinguishing from the Epiphany-Aware KVC project)

**Note:** The parallel hidden-state-variance project (see `/home/skolawol/workspace/kvcache/`) makes FA2 compatibility a primary argument for avoiding attention-based signals. Our approach has a different but compatible FA2 story: the serving-layer KV eviction triggered by inline tokens operates on the PagedAttention block table (not the attention weight matrix), and does not require materialising the attention matrix. The model's own generation of management tokens happens in the normal autoregressive forward pass — FA2-compatible. The only FA2-incompatible operation is the hindsight labeling pipeline (which needs full attention matrices), but that is offline.

---

## 5. Training Design — Key Decisions and Tradeoffs

### 5.1 DPO vs. GRPO

DPO is operationally simpler (no rollouts, no group sampling, no reward infrastructure). But its fundamental constraint for this project is outcome-level credit assignment: the DPO loss rewards or penalizes entire traces based on which was preferred, with no mechanism to credit the `<FORGET>` token at position 500 specifically.

GRPO with process supervision is more complex but provides token-level credit:

```python
# For each management token emission at step t:
# memory_saved = (evicted_segment_size / total_cache_size)
# future_attention = mean attention mass from tokens t+1..T to evicted segment
# step_reward(t) = memory_saved × (1 − future_attention)
```

The advantage formula accumulates forward: `Â_{i,t} = Σ_{j≥t} γ^(j-t) · r̃_j`, giving the management token credit for all future memory savings it enables.

**Decision:** GRPO with process supervision for Stage 2. DPO optionally for Stage 3 (alignment, not learning *when* to emit tokens). This mirrors the DeepSeek-R1 pipeline and is supported by the PAV finding that progress-based step rewards outperform outcome-only by >8% at better compute efficiency.

### 5.2 Training/Inference Mismatch

During GRPO rollouts, the model emits `<FORGET seg_id>` tokens. If the KV cache simulator does not actually evict the corresponding tokens during the rollout, the model experiences different context at positions t+1..T than it will at inference time. The evicted tokens remain available for attention during training but not during deployment, creating a systematic generalization failure.

**Mitigation:** The KV cache simulator (Engineering Prerequisite P3) must execute evictions during GRPO rollouts in real time. This is the most technically demanding prerequisite: the simulator must be fast enough not to bottleneck the training loop, and must handle the PagedAttention block table correctly.

**Alternative (lower bound):** Train with full context but add a penalty for any `<FORGET>` target that is attended to after the `<FORGET>` token is emitted. This does not close the mismatch but penalizes the model for emitting tokens it will violate at inference time.

### 5.3 Format and Structural Rewards

A small format reward prevents degenerate emissions (malformed segment IDs, summary tokens outside `<SUMMARY>...</SUMMARY>` brackets, `<BOOKMARK>` applied to nonexistent segments). This mirrors DeepSeek-R1's format reward for `<think>...</think>` structure and should fire per-emission.

### 5.4 Cold-Start Data Quality

SideQuest's key finding: 215 training examples suffice for meaningful generalization. This is encouraging, but SideQuest's training examples have an advantage: tool outputs have explicit cursor identifiers, making hindsight "last-use" annotation unambiguous. CoT segment boundaries are defined by our labeling pipeline (E2), which introduces approximation at two levels:

1. Attention-mass threshold for dead-end classification (calibrated from E0; ablated in E6e)
2. Summary token quality (written by model or annotator; ablated implicitly by E6d)

If SFT on these approximate labels produces poor precision/recall (E3 go/no-go), the likely fix is (a) stricter attention-mass threshold or (b) using a stronger annotator (32B or larger) for summary generation, matching SideQuest's use of gpt-oss-120b.

### 5.5 Vocabulary Extension vs. Token Repurposing

Adding `<FORGET>`, `<SUMMARY>`, and `<BOOKMARK>` to the model vocabulary requires:
- Extending the embedding matrix (randomly initialized new rows)
- Extending the LM head (new output logit columns)
- SFT on cold-start data to assign meaningful representations to these tokens before GRPO

Alternative: repurpose existing rare tokens (e.g., unused Unicode characters) as management tokens without vocabulary extension. This avoids embedding matrix surgery but may inherit confounding representations from pre-training.

**Decision:** Vocabulary extension, following standard practice (e.g., DeepSeek-R1's format tokens). Randomly initialized embeddings allow the model to develop clean management-token representations from scratch during SFT.

---

## 6. Open Questions

The following are unresolved at the time of writing. Answers will emerge from experiments.

1. **Metacognitive accuracy:** Can a fine-tuned 7B model reliably predict which of its own reasoning segments are dead-ends before they fully play out? The zero-shot baseline (E1) provides the first signal.

2. **Summary token informativeness:** Are 2–4 summary tokens sufficient to preserve the useful content of a dead-end branch? The ablation E6d tests counts; qualitative analysis E7.5 evaluates content.

3. **Cold-start data scale:** SideQuest used 215 examples. Will CoT management require more, given the more complex and variable nature of CoT boundaries vs. cursor-indexed tool outputs?

4. **Process supervision compute budget:** E4b's step-level rewards require prover rollouts from each management token. At G=64 completions × average 5 management tokens per trace, this is 320 prover rollouts per training question. Feasible on A100 clusters; may require approximation (e.g., shorter rollouts, sampling fewer management steps to evaluate).

5. **Generalization across domains:** The model is trained on math (MATH-500, AIME) and tested on code (LiveCodeBench). Do management token patterns generalize, or is the behavior domain-specific? E5 tests this directly.

6. **Interaction with generation length:** Does the model learn to compress traces (emit management tokens to stay within a target cache budget) or does it maintain trace length and simply manage the cache more efficiently? The ideal outcome is the former. E7.3 analyzes this.

7. **FreeKV citation disambiguation:** The proposal's reference list cites FreeKV as OpenReview ID wXAn7orB1H ("Boosting KV Cache Retrieval"). The publicly accessible FreeKV paper is arXiv:2505.13109 (same title, same core mechanism). Confirm whether these are the same work at different stages of submission before finalizing related work section.

---

## 7. Positioning Statement (for Paper Introduction)

We propose the first KV cache management method that:
1. **Detects dead-end branches at the moment of abandonment**, not post-hoc as backward-looking attention heuristics eventually degrade
2. **Operates by model self-report via inline generation**, requiring no auxiliary thread, no external classifier, and no serving-system inference about token importance
3. **Preserves useful information before eviction** via summary tokens, combining the memory efficiency of eviction with partial information recovery previously available only in full-retrieval methods
4. **Targets self-generated CoT reasoning tokens** — the dominant memory consumer in single-turn reasoning, which SideQuest (the closest prior model-driven work) explicitly excludes

The approach is complementary to all existing serving-system heuristics: inline tokens handle the prospective dead-end detection problem; methods like ThinKV, Crystal-KV, and Quest handle residual compression of the surviving KV cache.

---

## 8. Key Figures for the Paper

1. **Motivation figure:** KV cache growth curve over reasoning trace generation; dead-end segment fraction; annotation showing the detection lag of backward-looking methods vs. inline token emission time
2. **Architecture diagram:** Inline token positions in a sample reasoning trace; serving-layer interception; PagedAttention block table state before/after `<FORGET>`
3. **Training pipeline diagram:** E0 characterization → E2 labeling → E3 cold-start SFT → E4 GRPO, with reward decomposition shown
4. **Main results table:** Peak KV cache, TPOT, accuracy, useful retention rate across all baselines and our variants
5. **λ–accuracy tradeoff curve:** Memory efficiency vs. accuracy as a function of GRPO reward weight λ
6. **Ablation heatmap:** Token type × training stage matrix of accuracy and memory efficiency
7. **Qualitative trace example:** Annotated reasoning trace showing management tokens in context, evicted segment highlighted, summary token content decoded

---

*See [experiments.md](experiments.md) for the full experiment tracking document.*
