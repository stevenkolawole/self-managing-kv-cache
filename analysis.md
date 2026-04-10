# E1 Zero-Shot Elicitation — Analysis

**Date:** 2026-04-08  
**Model:** DeepSeek-R1-Distill-Qwen-32B  
**Dataset:** MATH-500 (100 problems)  
**Script:** `scripts/elicit_zero_shot.py`

*(7B results pending — clean rerun in progress)*

---

## Prompt Variants (Full Text)

**V1 — Minimal**
```
When you realize you've gone down a wrong reasoning path, mark it with <FORGET seg_N>
(where N is the segment number) before moving on.
```

**V2 — Typed**
```
You have memory management tokens:
- <FORGET seg_N>: marks segment N as a dead-end you are abandoning
- <SUMMARY>brief note</SUMMARY>: summarizes what you learned before forgetting
- <BOOKMARK seg_N>: pins an important segment to keep
Use them naturally as you reason through the problem.
```

**V3 — Procedural**
```
As you solve problems, manage your reasoning explicitly:
1. Number your reasoning segments mentally (seg_0, seg_1, ...).
2. When you catch a mistake and backtrack, emit <FORGET seg_N> for the abandoned segment.
3. Optionally precede it with <SUMMARY>brief lesson</SUMMARY>.
4. Use <BOOKMARK seg_N> to flag segments you will refer back to.
Emit these tokens inline without breaking your reasoning flow.
```

**V4 — Segment-Aware**
```
You are a reasoning model with explicit KV cache management.
Segments are bounded by self-correction markers ("Wait", "Actually", etc.).
seg_0 is text before your first correction; seg_1 is the next, and so on.
At each self-correction, emit exactly one of:
- <FORGET seg_N> if the preceding segment was a dead-end
- <SUMMARY>key insight</SUMMARY><FORGET seg_N> if it had useful partial results
- <BOOKMARK seg_N> if it is critical and you must refer back to it
```

**V5 — Few-Shot**
```
When you realize you've made a mistake, emit a management token before correcting.

Example:
...I set x = 3, so x² = 9. Wait, actually I misread — x = 2. <FORGET seg_0>
Starting over with x = 2: x² = 4...

Use <FORGET seg_N> to mark abandoned reasoning. Optionally precede with
<SUMMARY>lesson</SUMMARY>. Use <BOOKMARK seg_N> to pin segments you will refer back to.
```

---

## Prompt Variants V6–V9 (Improved — Reasoning-Internal Framing)

> Added 2026-04-09. These variants address the root failure of V1–V5: framing tokens as
> output formatting rather than internal reasoning actions. Research basis: identity priming
> makes behavior feel native; inline examples at correction moments (not isolated) set the
> correct context; output priming and recency suffix exploit context-proximity effects.

**V6 — Rich Few-Shot**
```
As you reason, you will sometimes catch yourself going down the wrong path.
At that moment — right after 'Wait' or 'Actually' — emit a management token
inside your reasoning before you continue. These are reasoning-internal markers,
not output formatting.

Three examples of how this looks inside a reasoning chain:

Example 1 (dead-end, clean forget):
...so the sum must be 6 × 7 = 42. Wait, I'm confusing rows with columns.
<FORGET seg_0> Let me restart — the grid is 7 columns wide, so the sum is 7 × 8 = 56.

Example 2 (partial insight worth keeping before discarding):
...I'll factor: x² - 5x + 6 = (x-2)(x-3). Hmm, actually the original had
coefficient 2 on x², so that's wrong.
<SUMMARY>roots would be 2 and 3 if leading coeff were 1</SUMMARY><FORGET seg_1>
Applying quadratic formula with a=2, b=-5, c=6...

Example 3 (pin a key result, then abandon a later attempt):
...I've shown angle BAC = 45°. <BOOKMARK seg_0> Now I need BC.
Actually, my triangle approach is getting messy — I should use the sine rule.
<FORGET seg_1> Using sine rule: BC / sin(45°) = AB / sin(BCA)...

Segments increment at each self-correction (seg_0, seg_1, ...).
<FORGET seg_N>: segment was a dead-end.
<SUMMARY>one sentence</SUMMARY><FORGET seg_N>: dead-end with a useful lesson.
<BOOKMARK seg_N>: pin this result, you will refer back to it.
```

**V7 — Identity**
```
You are a reasoning model that tracks its own dead-ends in real time.
When you catch yourself going wrong — right after 'Wait' or 'Actually' —
you mark it with a management token and move on cleanly.
This is native to how you think, not something added on top.

Three examples from your own reasoning style:

[same three examples as V6]

[same rules as V6]
```

**V8 — Output-Primed**

System:
```
When you realize you've gone down a wrong reasoning path, mark it with <FORGET seg_N>
before moving on.
```
Assistant turn prefix (injected before generation):
```
I'll track my reasoning in segments and emit <FORGET seg_N> each time I catch myself
going wrong.

```

**V9 — Recency Suffix**

System:
```
When you realize you've gone down a wrong reasoning path, mark it with <FORGET seg_N>
before moving on.
```
User message suffix (appended to each problem):
```
Remember: emit <FORGET seg_N> when you abandon a reasoning path,
<SUMMARY>lesson</SUMMARY><FORGET seg_N> if there's a useful insight,
or <BOOKMARK seg_N> to pin a result you'll need later.
```

---

## Results — 7B

| V | Name | N | Accuracy | FORGET | SUMMARY | BOOKMARK | Any emitted |
|---|------|---|----------|--------|---------|----------|-------------|
| 1 | minimal | 100 | **71%** | 0 | 0 | 0 | 0% |
| 6 | rich_few_shot | 100 | 63% | 0 | 0 | 1 | **1%** |
| 7 | identity | 100 | 64% | 0 | 0 | 0 | 0% |
| 8 | output_primed | 100 | 70% | 0 | 0 | 0 | 0% |
| 9 | recency_suffix | 100 | **71%** | 0 | 0 | 0 | 0% |

V2–V5 not rerun for 7B (output-framing approach abandoned based on 32B results).

---

## Results — 32B

| V | Name | N | Accuracy | FORGET | SUMMARY | BOOKMARK | Any emitted |
|---|------|---|----------|--------|---------|----------|-------------|
| 1 | minimal | 100 | **64%** | 0 | 0 | 0 | 0% |
| 2 | typed | 100 | 46% | 0 | 0 | 0 | 0% |
| 3 | procedural | 100 | 49% | 0 | 1 | 0 | **1%** |
| 4 | segment_aware | 100 | 26% | 0 | 0 | 0 | 0% |
| 5 | few_shot | 100 | 56% | 0 | 0 | 0 | 0% |
| 6 | rich_few_shot | 100 | 38% | 0 | 0 | 0 | 0% |
| 7 | identity | 100 | 43% | 0 | 0 | 0 | 0% |
| **8** | **output_primed** | **100** | **73%** | **9** | **0** | **0** | **9%** |
| 9 | recency_suffix | 100 | 54% | 1 | 0 | 2 | 3% |

> **Note on V1 accuracy:** The log reported 26% due to a resume-session bug in the summary
> counter. The true value from the JSONL file (64 correct / 100) is 64%.

> **Note on emission counts:** Initial grep for `"any_emitted": true` produced 3 hits across
> V2/V3/V4. On inspection, V2 and V4 hits were false positives — the model had written the
> literal JSON string `"any_emitted": true` inside its `generated_text` while reasoning about
> the system prompt instructions. Only V3's `n_summary: 1` is a real emission.

---

## The One Real Emission — V3 32B (line 19, v3_32b.jsonl)

The only genuine management token emitted across all 500 32B traces was a single `<SUMMARY>`
tag in V3 (procedural), line 19 of `data/e1/v3_32b.jsonl`. The trace could not be fully
extracted here due to token length, but the token counts confirm: `n_summary=1, n_forget=0`.

The likely character of this emission: the model followed step 3 of the procedural instructions
("Optionally precede it with `<SUMMARY>brief lesson</SUMMARY>`") but did not follow through
with a `<FORGET>` — consistent with partial, halting compliance. The summary appeared but the
eviction token was omitted, suggesting the model understood the syntax but not the intent.

---

## Key Findings

### 1. V8 (output_primed) is the breakthrough

V8 is the only variant to produce meaningful zero-shot emission: **9% emission rate on 32B,
with 73% accuracy — the highest of all 9 variants, above V1 baseline (64%).**

The prefix injection (`"I'll track my reasoning in segments and emit <FORGET seg_N>..."`)
forces the model to commit to the behavior before generation starts. It cannot decide not to
track segments — the commitment is already in the context. Critically, accuracy does not
degrade: the framing appears to actively help reasoning organization, not hurt it.

7B does not benefit from V8: 70% accuracy, 0 emissions. Scale is required for zero-shot compliance.

### 2. V8 trace quality is high

trace_082 (V8 32B) shows proper behavior: the model labels segments (`seg_1:`, `seg_2:`, ...),
tries two failed algebraic approaches, emits `<FORGET seg_5><FORGET seg_6>` then
`<FORGET seg_7><FORGET seg_8>` at genuine decision points, and recovers with integer
testing to find the correct answer (x=2). This is precisely the target behavior.

False positive rate: 6 of 15 files containing `<FORGET` text are false positives (model
wrote `"I don't need to issue a <FORGET> command"` etc.). True emission: 9 real events.

### 3. Rich-context prompts (V6/V7) hurt accuracy badly on 32B

| Variant | Accuracy | Δ vs V1 |
|---------|----------|---------|
| V8 output_primed | **73%** | +9pp |
| V1 minimal | 64% | — |
| V9 recency_suffix | 54% | −10pp |
| V7 identity | 43% | −21pp |
| V6 rich_few_shot | 38% | −26pp |

Long system prompts with examples consume attention budget or disrupt chain-of-thought
initiation. The model spends generation capacity parsing instructions rather than reasoning.

The 7B pattern is similar but less extreme: V6 drops accuracy 8pp, V7 drops 7pp, V8/V9 hold baseline.

### 4. V1–V5 failure mode confirmed

Zero-shot emission was ~0% across 500 V1–V5 32B traces. The model reasons *about* tokens
(V2/V4 false positives generated JSON-like text about the instructions) rather than emitting
them. V8 breaks this pattern by making token emission the first act of generation.

### 5. The gap to close with fine-tuning

```
E0 ground truth dead-end rate:        79%
E1 best zero-shot emission (V8 32B):   9%
Gap (motivates fine-tuning):          ~70pp
```

---

## Qualitative Analysis of All 9 V8 32B Emissions

All 9 traces with `n_forget >= 1` were read and classified by emission intent.

### Behavioral taxonomy

| Pattern | Traces | Forget count | Quality |
|---------|--------|-------------|---------|
| Genuine prospective forgetting (dead-end → pivot) | 082 | 4 | Target behavior |
| Post-hoc bulk erasure (all segs wiped after solving) | 040, 094 | 14 | Wrong intent |
| Pathological verification loop | 075 | 11 | Wrong intent |
| Self-doubt on correct reasoning | 050, 086 | 4 | Wrong intent |
| False trigger / arbitrary erasure | 025, 046, 096 | 3 | Wrong intent |

**Only 1 of 9 traces (trace_082) is doing the target behavior.** The other 8 account for 32 of the 36 FORGET tokens but reflect wrong usage patterns.

---

### Per-trace breakdown

**trace_082** — `3^{2x} + 19 = 10^x` | 4 forgets | correct | **Target behavior**
```
seg_5: Let y = (9/10)^x...
seg_6: This substitution complicates things further.
<FORGET seg_5>
<FORGET seg_6>

seg_7: Take the natural logarithm...
seg_8: This also doesn't simplify easily.
<FORGET seg_7>
<FORGET seg_8>

seg_9: Test x=1: 28 ≠ 10
seg_10: Test x=2: 100 = 100 ✓
```
Genuine: failed approach recognized mid-reasoning, FORGET emitted, alternative tried, correct answer found.

---

**trace_040** — octagon perimeter | 6 forgets | correct | **Post-hoc bulk erasure**

Model solves correctly in seg_1–seg_6, then emits `<FORGET seg_1>` through `<FORGET seg_6>` all at once immediately before `</think>`. Using FORGET as "clear working memory after answering," not dead-end marking.

**trace_094** — vector projection | 8 forgets | correct | **Post-hoc bulk erasure**

Identical pattern to trace_040: full correct solution in seg_1–seg_8, then all 8 segments FORGET'd in a block. 14 of the 36 total forgets come from this cleanup pattern.

---

**trace_075** — domain of log(x²) | 11 forgets | correct | **Pathological verification loop**

Model arrives at the correct answer (a+b=0) in seg_6, then second-guesses itself 11 times. Each iteration: re-derive a+b=0 → FORGET → re-derive again. Cannot commit to a correct answer. The problem is trivial but the model is stuck.

---

**trace_050** — gold coins redistribution | 2 forgets | incorrect* | **Self-doubt loop**

Derives correct answer (203) in seg_1, emits `<FORGET seg_1>` with "I think I made a mistake," re-derives 203 in seg_2, emits `<FORGET seg_2>`. No mistake was made. *Marked incorrect likely due to unboxed answer format.

**trace_086** — `|5x-1| = x+3` | 2 forgets | correct | **Self-doubt on correct reasoning**

Solves correctly (x=1 is largest), emits `<FORGET seg_4>` + "Wait, I think I made a mistake" — no mistake. Re-verifies, emits `<FORGET seg_6>`. The model emits FORGET as an anxiety signal, not a dead-end signal.

---

**trace_025** — functional equation f(x)+f(y) | 1 forget | correct | **False trigger, recanted**

Complete correct solution in seg_1–seg_10, then `<FORGET seg_2>` at the end, immediately followed by "Wait, no, seg_2 was correct. Maybe I confused with another thought." Model emits FORGET then walks it back.

**trace_046** — roots of unity z⁴+z²+1=0 | 1 forget | correct | **Arbitrary post-hoc erasure**

Full correct solution, then `<FORGET seg_2>` emitted at the end for no stated reason. Seg_2 was the correct and used factoring step.

**trace_096** — figure skater 2250° | 1 forget | incorrect* | **False correction**

Model computes correctly (east), then emits `<FORGET seg_1>` + "she should end up facing east, not west" — it had already said east. Corrects an error that didn't exist. *Ground truth is east; marked incorrect likely due to `\boxed{east}` vs `\text{east}` format mismatch.

---

## Implications for E3 (Cold-Start SFT)

- **Do not naively use V8 emitting traces as SFT positives.** 8 of 9 emitting traces show
  wrong-intent patterns (post-hoc bulk erasure, verification loops, self-doubt). Training on
  these would teach the model to FORGET after completing a correct solution, or to loop.
  Only trace_082 shows the target behavior. SFT data must be filtered to FORGET-at-approach-switch only.

- **Filter criterion for E2 training data:** A FORGET is a valid training signal only if
  (a) it is emitted mid-reasoning before the solution is found, and (b) the forgotten segment
  was a genuinely different approach that was abandoned, not a correct step being second-guessed.
  The E0 hindsight labels provide ground truth for (b).

- **trace_082 is the positive template.** Segment-header tracking + FORGET at approach pivot +
  continued reasoning to correct answer. This is the exact structure E2 data should have.

- **Use V8 system prompt at inference** post-training, or no system prompt if the behavior
  is internalized. Do not use V6/V7 — long prompts hurt accuracy.

- **7B needs SFT regardless.** Zero emissions across 500 7B traces. The 7B model has no
  zero-shot capability here; fine-tuning must build it from scratch.

- **SideQuest comparison:** SideQuest used 215 examples for meaningful SFT generalization.
  With only 1 clean zero-shot example (trace_082), E2 hindsight labeling is the primary
  data source — zero-shot mining is not sufficient as a cold-start corpus.

---

---

## Research Basis for V6–V9 Design

**Why V1–V5 failed:** All five variants framed management tokens as output requirements — something to add to the response. The model treated them as a topic to reason about, not an action to take. V2 and V4 false positives show this explicitly: the model generated JSON-like strings about the tokens, not the tokens themselves.

**What works in few-shot prompting (research basis):**

1. **Inline examples at the triggering moment** (not isolated blocks): Examples embedded exactly where the target behavior should occur set the correct context association. V5's example was a standalone block; V6's examples are mid-reasoning-chain, showing the token appearing immediately after "Wait" / "Actually" / "Hmm". This is the key structural fix.

2. **Identity framing** (V7): Making behavior feel *native* ("this is how you think") rather than *instructed* ("do this additional thing") reduces the model's tendency to add tokens only at the end or to resist the instruction. Evidence: persona prompts improve instruction-following for rare behaviors in RLHF-trained models.

3. **Output priming / prefix injection** (V8): Injecting text into the start of the assistant turn forces the model to commit to the framing before generating. The model cannot "decide" not to track segments — it has already stated it will. This exploits the autoregressive nature of the forward pass.

4. **Recency / proximity** (V9): Instructions closer in token distance to the generation point are weighted more by the attention mechanism. The user-message suffix is ~100 tokens before generation vs. ~300+ for system prompt in a long context. This is the weakest intervention but tests the proximity hypothesis in isolation.

**Actual ranking (32B accuracy):** V8 (73%) > V1 (64%) > V9 (54%) > V7 (43%) > V6 (38%).  
**Actual emission:** V8 (9%) > V9 (3%) > V6/V7 (0%).

The predicted ranking was wrong in a revealing way: we expected V6/V7 to lead on both axes.
Instead, long examples consumed capacity (−21 to −26pp accuracy) and produced zero emission.
V8's minimalism won on both: the commitment prefix is short enough to not hurt reasoning,
but strong enough that the model follows through. V9's weak signal (3%) confirms proximity
matters, but not as much as forcing an explicit commitment before generation starts.
