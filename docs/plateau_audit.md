# CORAL v4.2 Plateau Audit — Why Is Token Accuracy Stuck at ~56%?

**Date:** 2026-03-31
**Status at audit:** task loss ≈ 0.9, total loss ≈ 14.5, token accuracy ≈ 56%, exact accuracy = 0%
**Reference:** TRM achieves 87.4% exact accuracy on the same benchmark.

---

## Summary: Top 3 Causes (Ranked)

| Rank | Hypothesis | Confidence | Proposed Experiment |
|------|-----------|------------|---------------------|
| 1 | **Gradient depth gap (32×)**: detach between segments limits gradient to 21 backbone applications per update vs TRM's 672 | **HIGH** | `configs/phase1_deep_gradient.yaml`: K_max=1, inner_steps=336 |
| 2 | **Embedding weight decay at 1.0**: token/position embeddings shrink to ~49% natural scale by end of training | **MEDIUM** | Add `"embedding"` check to no-decay param group in optimizer |
| 3 | **Halting loss noise (~28% of total)**: BCE at 0% exact accuracy drives q_halt → −∞ and injects noise gradient into backbone | **LOW-MEDIUM** | Disable halting loss for first 5000 steps via warmup coefficient |

---

## 1. Gradient Depth Analysis

### Finding

With `detach()` between segments, each segment is an **independently optimized 21-step program**.

- Each parameter update flows through exactly **21 backbone applications** (one segment's inner loop).
- Between segments, `z_states = [z.detach() for z in z_states]` severs the computation graph.
- Deep supervision provides signal at every segment, but each segment's loss only updates parameters based on that segment's 21 steps — not across segment boundaries.

**TRM comparison:**
- TRM: 6 inner steps × 112 outer steps = 672 total backbone applications, all in **one computation graph**.
- Gradient flows through all 672 applications simultaneously.
- TRM's own paper identified full-backprop as the critical change: 56.5% → 87.4% exact accuracy.
- CORAL's effective gradient depth: 21 (one segment) — a **32× gap**.

**What 56% token accuracy actually means:**
Sudoku-extreme puzzles have ~17–30 given cells and ~51–64 empty cells per puzzle.
At 56% token accuracy (assuming the model correctly copies all given cells):
- With 26 given cells: ~35% of empty cells are solved correctly.
- With 22 given cells: ~39% of empty cells are solved correctly.
- Exact accuracy = 0%: even getting one cell wrong fails the whole puzzle.

The model has learned good local constraint propagation but cannot globally solve puzzles. This is exactly the symptom of insufficient gradient depth — local patterns are learnable in 21 steps, global consistency requires deeper backpropagation.

### Why Deep Supervision Doesn't Close the Gap

Each segment is optimized to take its (detached) input state and improve it in 21 steps. This is analogous to **unrolled optimization with supervision at each step**. The backbone learns "given THIS state, improve it." It cannot learn to chain improvements across segments because the gradient chain is severed.

TRM's full backprop teaches the backbone "given the INITIAL encoding, iteratively refine over 672 steps to reach the solution." The backbone learns the complete transformation, not just local improvement.

### The Critical Experiment

**`configs/phase1_deep_gradient.yaml`**: K_max=1, inner_steps_override=336.

- Total backbone applications: 336 (same as current 21×16).
- Effective gradient depth: 336 (vs current 21).
- All 336 steps in one computation graph, no detach.
- Equivalent to TRM's 2-layer × 336 = 672 transformer layer applications.

Memory note: 336 in-graph steps at B=64 requires ~7–9 GB for activations (bfloat16).
Config uses batch_size=32 to stay within 24GB VRAM budget.

---

## 2. Loss Function Audit

### Task Loss — All 81 Cells

**Finding:** ALL 81 cells (given + empty) contribute to the task loss. No masking of given cells.

**Code path:**
`build_sudoku_dataset.py` line 119: `return arr + 1` — both inputs and labels are +1 shifted. Labels are the full solution (digits 1–9 → tokens 2–10) for ALL cells.
`metadata.ignore_label_id = 0` — but labels are 2–10 after +1 shift, so value=0 never appears → no masking occurs.
`losses.py` line 162: `mask = labels != IGNORE_LABEL_ID` → mask is all-True for all 81 cells.

**Assessment:** Training on given cells is HRM/TRM standard practice, not a bug. For given cells, the model learns to copy (digit in input → same digit in output), which is easy. For empty cells, it must reason. The copy task dilutes ~27–37% of gradient signal toward trivially-learnable targets. This contributes to the plateau but is not the primary cause.

### Halting Loss

**Finding:** L_halt ≈ 0.5 × BCE(q_halt, seq_is_correct) is computed at every segment. At 0% exact accuracy, `seq_is_correct = 0` for all puzzles, making the halting target always 0.

**Quantitative impact:**
- L_halt per segment at init (logit≈0): 0.5 × log(2) ≈ **0.347**
- Weighted sum over 16 segments (linear weighting): **≈ 5.55**
- Observed total loss: ≈ 14.5
- Halting loss fraction: **≈ 28% of total loss**

This is significant. The halting network reads `z_states` and produces a scalar; its gradient flows back into the backbone. At 0% exact accuracy, this gradient consistently says "your representations should predict no-halt." This creates **constant downward pressure on halt confidence** that flows into backbone representations — potentially pulling representations away from their task-optimal configuration.

The halting loss is adding ~28% gradient noise during the period when the model is trying to learn basic Sudoku reasoning. TRM disables similar losses early in training for this reason.

### Stablemax vs Softmax

No issues found. Stablemax is numerically stable, computed in float64. The cast `logits.to(torch.float64)` in `_log_stablemax` is correct.

### Linear Weighting Gradient Conflict

**Finding:** With 16 segments and linear weighting:
- Segment 0: weight = 2/17 ≈ 0.12 (refined by 0 prior steps)
- Segment 15: weight = 32/17 ≈ 1.88 (refined by 315 prior no-grad steps)

Early-segment gradients come from a coarsely-refined state; late-segment gradients from a well-refined state. These could point in opposing directions in parameter space, causing the optimizer to receive conflicting signals. However, the linear weighting already down-weights early segments significantly, partially mitigating this.

---

## 3. Learning Rate and Schedule

### Finding

The cosine schedule with `warmup=500, total_steps=20000, min_lr=0.1×lr_max`:

| Step | LR ratio | Effective LR |
|------|----------|-------------|
| 500  | 1.000    | 7.00e-05    |
| 1000 | 0.998    | 6.99e-05    |
| 2000 | 0.985    | 6.90e-05    |
| 5000 | 0.874    | 6.12e-05    |
| 10000| 0.520    | 3.64e-05    |
| 15000| 0.154    | 1.08e-05    |
| 20000| 0.100    | 7.00e-06    |

**Assessment:** LR is NOT the cause of the plateau. At step 2000 (where the plateau begins), LR is still at 98.5% of its maximum. The schedule decays slowly — it only reaches 50% at step 10000. The plateau at step 2000 is a representation learning problem, not an optimization rate problem.

The minimum LR of 7e-6 (at step 20000) is reasonable and not causing underfitting.

---

## 4. Input Encoding

### Vocabulary and Token Space

**Encoding:**
- Token 0: PAD (never appears in inputs after `+1` shift)
- Token 1: empty cell marker (inputs only; appears where digit is unknown)
- Tokens 2–10: digits 1–9

**Vocab_size = 11** is correct. The decoder has logits for all 11 tokens. The model never needs to predict 0 or 1 (they never appear in labels), so 2/11 of decoder capacity is unused. This is minor and matches HRM/TRM practice.

**Empty cell marker (token 1):** This embedding carries the critical "I need to be solved" signal. It must survive through 336 backbone applications (if we implement hypothesis #1). With input_injection at every step, the empty marker embedding is re-added every backbone call — this is correct and ensures the signal doesn't fade.

### Input Injection Correctness (Verified)

After commit `3896194`, `evaluator.py` correctly uses `core.forward()` which includes `input_signal = z1_init` re-injected at every inner step. The encoder output (including the empty cell token embedding and positional embeddings) is added to every backbone input. ✓

### Potential Issue: Input Injection Scale

`input_signal = z1_init` is the raw encoder output, which has been `input_norm`'d (LayerNorm). This is at scale ~1.0. The backbone state `z` grows with each iteration (pre-norm accumulates residuals). After T steps:

```
z ≈ z_0 + T × backbone_output_scale
```

With T=336 and `backbone_output_scale ≈ 0.5` (typical pre-norm residual), `||z|| ≈ 168`. Meanwhile `||z1_init|| ≈ 1.0`. The input injection becomes negligibly small relative to the accumulated residual stream by step ~50. **This may be causing the model to ignore the task constraints late in the recursion.**

A possible fix: normalize `z1_init` adaptively or inject at a learned scale. Not implementing now — flagging for investigation.

---

## 5. Weight Decay

### Finding: Embeddings Are Decayed at WD=1.0

`optimizer.py` lines 58–64:
```python
if param.ndim <= 1 or name.endswith(".bias"):
    no_decay_params.append(param)  # WD = 0
else:
    decay_params.append(param)  # WD = 1.0
```

All `nn.Embedding` weights have `ndim=2` → they fall into `decay_params` (WD=1.0).

**Affected parameters:**
- `adapter.token_emb.weight` — shape [11, 512] — DECAYED
- `adapter.row_emb.weight` — shape [9, 512] — DECAYED
- `adapter.col_emb.weight` — shape [9, 512] — DECAYED
- `core.level_emb.embeddings.weight` — shape [1, 512] — DECAYED

**Quantitative impact over 20,000 steps:**
Using the actual cosine schedule (compound decay):

```
final_scale = Π(1 - lr_t × wd) for t=1..20000 ≈ 0.487
```

Embeddings are pushed to **~49% of their unconstrained scale** by the end of training. At WD=1.0 and early LR (7e-5), each step decays embeddings by 0.007%. Over 20,000 steps this accumulates to 51% shrinkage.

**Why this matters:**
1. `token_emb.weight[1]` — the empty cell embedding — is being regularized toward zero, reducing its ability to signal "solve me" through 336 backbone iterations.
2. `row_emb` and `col_emb` — the positional embeddings — encode Sudoku grid structure. Decaying them toward zero removes structural priors.
3. Standard practice (GPT-2, LLaMA, T5) excludes all embeddings from weight decay, because decaying them conflates magnitude with representation quality.

**Correctly excluded from decay (ndim=1):**
- `cond_gate` — shape [1] ✓
- `row_bias`, `col_bias`, `box_bias` — shape [1] ✓

**Proposed fix:** Add `"emb"` or `"embedding"` check to the no-decay condition in `optimizer.py`. No architectural changes required.

---

## 6. Deep Supervision Weighting

### Finding: Possibly Hurting, Not Helping

With `deep_supervision_weighting="linear"`, the total gradient from all 16 segments is a weighted mixture where early segments (coarse state) contribute less but still participate.

**Critical question:** Is deep supervision better or worse than single-segment training?

Evidence from the audit prompt:
- Diagnostic with "last segment only" training: 38% at step 100, ~55% extrapolated
- Full trainer with all 16 segments: 56% at step 2000

This is **marginal improvement** from deep supervision despite 16× more compute per step. If single-segment-only training reaches 55% extrapolated, and 16-segment reaches 56%, deep supervision is not providing a meaningful improvement per gradient step.

Possible explanation: With 16 independent gradient updates (one per segment), the backbone is being optimized for 16 different initial states simultaneously. This creates a more general solution but also more gradient variance. The benefit of "more training signal" (16 loss terms) is offset by "more conflicting gradient" (16 different starting states pulling in different directions).

**The K=1 experiment directly tests this.** If K=1 with 336 inner steps dramatically outperforms K=16 with 21 inner steps, deep supervision is indeed hurting. If they're comparable, the gradient depth hypothesis is confirmed and deep supervision is neutral.

---

## 7. Architecture Comparison to TRM

### Verified Matches

| Feature | TRM | CORAL | Status |
|---------|-----|-------|--------|
| Pre-norm | ✓ | ✓ | `backbone.py` lines 190–191: `x = x + attn(norm1(x))` |
| Residual-only updates | ✓ | ✓ | `z = z_new + gate * (cond - z)` at gate=0.01 ≈ pure residual |
| Input re-injection every step | ✓ | ✓ | `backbone_in += input_injection` at every inner step |
| Single timescale (N=1) | ✓ | ✓ | n_levels=1 |
| Shared backbone weights | ✓ | ✓ | one `CoralBackbone` for all steps |

### Key Difference: Gradient Strategy

| Strategy | TRM | CORAL (current) | CORAL (proposed) |
|----------|-----|-----------------|------------------|
| Gradient depth | 672 apps (all in one graph) | 21 apps (one segment) | 336 apps (all in one graph) |
| Outer steps | 112 (no_grad outer, grad for last) OR all-grad | 16 (detach between) | 1 (no detach) |
| Deep supervision | No (single output) | Yes (16 outputs) | No (single output) |

**Note on TRM's gradient strategy:** The audit prompt presents two conflicting claims about TRM:
1. "TRM effective gradient depth: 672 backbone applications in one computation graph"
2. "TRM gradient strategy: no_grad for all-but-last outer step, grad for last step only"

If (2) is correct, TRM's actual gradient depth is only 6 steps (last inner loop), but those 6 steps start from a state refined by 111 no-grad outer iterations. This would mean CORAL's 21 steps per segment already exceeds TRM's gradient depth, and the issue is **state quality** (not gradient depth): TRM's final 6 gradient steps start from a near-optimal state, while CORAL's 21 gradient steps start from a much earlier state.

Either way, **the K=1 experiment is the right test**:
- If the bottleneck is gradient depth → K=1 with 336 steps dramatically improves.
- If the bottleneck is state quality → K=1 with 336 steps also improves (because the single segment runs all 336 steps in one graph, building state quality and gradient signal simultaneously).

### Latent Bug: `_log_precision_metrics` in `train.py`

`train.py` line 279 still uses a manual `_run_level` call without `input_injection` or `attention_bias`:
```python
z_states[i] = trainer.core._run_level(z_states[i], i, cfg.timescale_base, None)
```
This is only executed when `use_predictive_coding=True` (not in baseline mode) and only for logging, not for training. **Does not affect baseline training.** Should be fixed before running PC experiments.

---

## Experiment Proposals

### Experiment 1 (Hypothesis #1): Full Gradient Depth — `phase1_deep_gradient.yaml`

**Change:** K_max=1, inner_steps_override=336, batch_size=32
**Expected outcome:** If gradient depth is the bottleneck, this should break through the 56% plateau quickly.
**Decision point:** If token accuracy > 70% at step 2000, gradient depth confirmed as primary cause.
**Config:** See `configs/phase1_deep_gradient.yaml`

### Experiment 2 (Hypothesis #2): Fix Embedding Weight Decay

**Change:** In `optimizer.py`, add embedding parameter names to the no-decay group:
```python
if param.ndim <= 1 or name.endswith(".bias") or "emb" in name:
    no_decay_params.append(param)
```
**Expected outcome:** Embeddings maintain ~100% scale instead of decaying to ~49%. Richer representations for empty cell marker and positional structure.
**Can be run alongside Experiment 1** — orthogonal change.

### Experiment 3 (Hypothesis #3): Halting Loss Warmup

**Change:** Add `halting_loss_warmup_steps: int = 5000` to TrainingConfig. Scale L_halt by 0 during warmup and linearly ramp to 1.0.
**Rationale:** At 0% exact accuracy, halting targets are all 0 → pure noise gradient into backbone. Let the backbone first learn basic reasoning before adding halting pressure.
**Risk:** Low — halting is not on the critical path for accuracy.

### Experiment 4 (Input Injection Scale): Normalize Input Signal

**Hypothesis:** After T=100+ inner steps, the residual stream norm `||z||` exceeds `||z1_init||` by 50–100×, making input injection negligible.
**Test:** Log `z.norm()` and `z1_init.norm()` at steps t=0, 50, 100, 200, 336 during a forward pass.
**Fix if confirmed:** Add a learnable scale `input_injection_scale = nn.Parameter(1.0)` and inject `input_injection_scale * z1_init`.

---

## Files Created

- `docs/plateau_audit.md` — this document
- `configs/phase1_deep_gradient.yaml` — Experiment 1 config (K=1, T=336, full gradient depth)
