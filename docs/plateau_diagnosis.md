# CORAL v4 — Plateau Diagnosis: Why Baseline Stalls at ~60% Token Accuracy

**Date:** 2026-04-08  
**Status:** Full diagnostic audit  
**Symptom:** Baseline eval token accuracy plateaus at ~60% (~45% on empty cells, ~100%
on given cells) across 7+ configurations regardless of depth, data, or hyperparameters.  
**Architecture is functional:** Model can overfit a single puzzle to 100% in 20 steps.

---

## Orientation: What does 60% actually mean?

For Sudoku-Extreme hard bucket (50–59 empty cells, ~68% of cells empty):

```
overall_token_acc = empty_acc × 0.68 + given_acc × 0.32
0.60 ≈ 0.45 × 0.68 + 1.00 × 0.32
```

The model gets nearly 100% on given cells (just copies the input) and ~45% on empty
cells. Since 1/9 ≈ 11% is random chance, 45% is substantially above random — the model
IS learning something. The question is why it learns THIS MUCH but no more.

**Hypothesis consistent with 45% empty-cell accuracy:**  
The model learns first-order constraint propagation (which digits are ruled out by the
same row/col/box) but not multi-step constraint propagation (chaining constraints
transitively). First-order resolution solves roughly 40–50% of empty cells in extreme
Sudoku ("naked singles" and "hidden singles"). The remaining cells need deductive chains
that require multi-step reasoning, and the model never learns those.

This hypothesis is testable and motivates a specific target for experiments: move from
45% → 65% empty accuracy (solve hidden doubles, box-line reductions) as a waypoint.

---

## Section 1 — Attention Pattern Analysis

**Files read:** `coral/model/backbone.py` — `RotaryAttention`, `_apply_rope`

### 1.1 RoPE implementation

CORAL uses the complex-exponential form (`torch.polar` → `view_as_complex`), pairing
adjacent dimensions: `(x[0],x[1]), (x[2],x[3]), ...`

TRM uses the `rotate_half` form (`LLaMA-style`): `q*cos + rotate_half(q)*sin`, pairing
`(x[i], x[i+D/2])`.

**These are mathematically equivalent implementations of the same rotation.** Both
correctly encode relative position. No bug here.

RoPE is correctly applied to **q and k but not v** (standard). SDPA is called with
correct `[B, H, L, dk]` shape after transpose. No off-by-one for L=81: `freqs_cis[:81]`
takes exactly the right prefix of a precomputed length-1024 buffer.

### 1.2 Attention mask

The structural bias (`row_bias * same_row + col_bias * same_col + box_bias * same_box`)
is cast to query dtype and passed as a float additive bias to SDPA. This is correct —
SDPA accepts float additive masks. The three learnable scalars initialise to **0.0**, so
the structural bias is a no-op at init and must be learned from gradient signal. This is
intentional.

### 1.3 1D RoPE on a 2D grid

RoPE treats the 81 cells as a 1D sequence. Cell `(r,c)` at flat index `r*9+c` is given
position `r*9+c`. Cells in the same row have consecutive positions (small relative
distance → high attention affinity in RoPE). Cells in the same column have relative
distance 9. Cells in the same box have relative distances up to 18.

The adapter's 2D learned position embeddings (`row_emb + col_emb`) provide the correct
2D structure in the input signal. RoPE provides additional 1D position signal in the
attention layers. These are not in conflict — they provide different information —
but neither alone encodes the full Sudoku constraint structure. The local attention
bias (if used) provides the third signal. Together they are sufficient in principle.

**No bug found in attention. No contribution to plateau.**

---

## Section 2 — Position Encoding: Confirmed embed_scale Bug

**File:** `coral/adapters/grid.py`, `encode()`, lines 102–104

```python
emb = tok + pos
if self.embed_scale:
    emb = emb * math.sqrt(self.d_model)   # ← scale applied BEFORE norm
return self.input_norm(emb)               # ← LayerNorm applied AFTER
```

**This is a confirmed bug.** LayerNorm (and RMSNorm) is scale-invariant:

```
input_norm(c * x) = (c * x) / rms(c * x) = (c * x) / (c * rms(x)) = x / rms(x)
```

Multiplying by `sqrt(512) ≈ 22.6` before LayerNorm has **zero effect**. The embed_scale
setting makes no difference. `z1_init` always exits `encode()` with RMS = 1 per
dimension, giving L2 norm ≈ sqrt(512) ≈ 22.6, regardless of the `embed_scale` flag.

**TRM's correct application:**

```python
# TRM: embed_scale applied AFTER any normalisations
return self.embed_scale * embedding   # No LayerNorm after this
```

TRM's `embed_tokens` is initialised with `std = 1/sqrt(hidden_size)`, so token
embeddings have RMS ≈ 1/sqrt(d) per dimension. After `embed_scale * embedding`:
RMS ≈ `sqrt(d) * (1/sqrt(d))` = 1 per dimension, but L2 norm ≈ `sqrt(L * d)` where
L includes puzzle positions. The scale ensures the **input signal has magnitude
proportional to** `sqrt(d * L)`, not just `sqrt(d)` as in CORAL.

**Why this matters:**

In `_run_level`, the backbone sees:

```
backbone_in = project_up(z) + input_injection + level_emb + ts_emb
```

`input_injection = z1_init` has norm ≈ 22.6.  
`z` (evolved state) has norm ≈ 22.6 after the first step, growing slowly afterward.

Without embed_scale working, z and input_injection have the **same order of magnitude**.
At every inner step the backbone receives a 50/50 blend of "current reasoning state" and
"raw input encoding." If the intended embed_scale had worked, input_injection would be
~22× larger, strongly anchoring the backbone to the input constraints at every step.

The practical effect: the backbone can partially "ignore" the input injection because z
and z1_init contribute equally. This weakens constraint enforcement for given digits and
may allow the reasoning state to drift away from puzzle constraints, particularly in
later segments.

**Fix:** Move embed_scale to after input_norm:

```python
emb = tok + pos
emb = self.input_norm(emb)
if self.embed_scale:
    emb = emb * math.sqrt(self.d_model)
return emb
```

**Experiment to verify:** Run baseline with fixed embed_scale (applied post-norm) vs
current (cancelled by pre-norm). If this is causal, expect:
- Faster convergence on given cells (already near 100%, so the signal would be in
  faster early convergence rather than ceiling)
- Higher empty-cell accuracy at plateau (stronger constraint anchoring)

---

## Section 3 — Recurrence Dynamics: Two Structural Issues

**File:** `coral/model/coral_core.py`, `_run_level()`, `forward()`

### 3.1 Initial state is the input embedding (no learned init)

CORAL initialises `z_states[0] = z1_init` (the encoder output, which varies per puzzle).
`z1_init` is also used as `input_signal` (the injection at every step). This means:

**At step 0 of segment 0:**
```
backbone_in = project_up(z1_init)      # = z1_init (identity for level 0)
            + z1_init                   # input_injection
            + level_emb + ts_emb        # small (std=0.02 → norm ≈ 0.45)
            ≈ 2 * z1_init               # the state IS the input, doubled
```

The model's initial reasoning state is the input itself. There is no input-agnostic
"reasoning seed" that the model can specialise through training.

**TRM's approach (both variants):**

```python
self.L_init = nn.Buffer(
    trunc_normal_init_(torch.empty(hidden_size), std=1), persistent=True
)
# At the start of each ACT step, reset to L_init if halted:
z_L = torch.where(carry.halted, self.L_init, carry.z_L)
```

`L_init` is a learned *constant* vector — the same starting point for every puzzle.
Over training, `L_init` specialises to be a good initialisation for the iterative
refinement dynamics. The input enters only via injection, not as the starting state.

**Why this matters for the plateau:** With CORAL's setup, the recurrence is:

```
z_{t+1} = backbone(z_t + z1_init + emb)
```

For given cells, z1_init already encodes the correct digit → the model quickly learns
to "copy" z1_init into the answer. For empty cells, z1_init encodes token 1 (empty) +
position. The model must move z away from this encoding toward the actual answer through
iteration. But the continuous re-injection of z1_init creates a "gravitational pull"
back toward the input representation at every step.

The fixed point of this recurrence satisfies:
```
z* = backbone(z* + z1_init + emb)
```

This fixed point is a function of z1_init, not of a learned reasoning starting point.
The model is essentially learning "what does the answer look like as a function of the
input's fixed point?" rather than "how do I iteratively refine a state toward an answer?"

### 3.2 No consolidation phase: injection at every single step

**TRM single-z forward structure (trm_singlez.py):**

```python
# H_cycles-1 outer loops WITHOUT gradient:
for _H_step in range(self.config.H_cycles - 1):
    for _L_step in range(self.config.L_cycles):
        z_L = self.L_level(z_L + input_embeddings, **seq_info)   # injection
    z_L = self.L_level(z_L, **seq_info)                          # ← consolidation, NO injection
# Final loop WITH gradient:
for _L_step in range(self.config.L_cycles):
    z_L = self.L_level(z_L + input_embeddings, **seq_info)       # injection
z_L = self.L_level(z_L, **seq_info)                              # ← consolidation, NO injection
```

After each injection cycle, TRM runs **one transformer pass with NO input injection**.
This "consolidation step" allows z to integrate the injected input information through
self-attention and FFN **without the input pulling it in a particular direction**. It is
a pure recurrent transformation of the current state.

**CORAL has no equivalent.** Every inner step injects z1_init:

```python
for t in range(n_steps):
    backbone_in = project_up(z) + level_emb + ts_emb + input_injection
    z = project_down(backbone(backbone_in))
```

There is no step where the backbone can process z without the input present.
The model never has a "think for yourself" step. Complex deductive chains require
maintaining a hypothesis state that is updated by constraint checking, not by the
raw input. Without a consolidation step, z is always a function of the raw input
at every step, limiting the depth of reasoning the recurrence can perform.

**Fix:** Add one consolidation step at the end of each inner loop:

```python
# Last step: no injection — consolidation
backbone_in = project_up(z) + level_emb + ts_emb   # no input_injection
z = project_down(backbone(backbone_in))
```

Or (matching TRM more closely): group every `L_cycles` injection steps with one
consolidation step, alternating.

**Experiment to verify:** Add 1 consolidation step at the end of each inner loop
(total steps increase from 21 to 22). If causal, expect better performance on harder
difficulty buckets (60+ empty cells) where multi-step reasoning is required.

---

## Section 4 — Deep Supervision: Gradient Tension (Minor)

**File:** `coral/training/trainer.py`, `train_step()`

With K_max=16 segments and linear weighting:
- Segment 0 loss weight ≈ 0.12 (very small)
- Segment 15 loss weight ≈ 1.88

The segment-0 loss asks "produce a good answer after 21 inner steps." This gradient
pushes the backbone toward fast answering, which is in mild tension with "careful
multi-step refinement" needed at segment 15.

However, TRM applies the same principle (one loss per ACT step). The linear weighting
(introduced in Session 10) already down-weights early segments. This is unlikely to be a
primary cause of the plateau, but it does mean the backbone receives conflicting gradient
pressure from early vs late segments.

**Assessment:** Minor issue. The linear weighting already partially addresses this.
Not a primary root cause. No immediate fix needed.

---

## Section 5 — Halting Network: Mean Pool Contaminates Cell Representations

**File:** `coral/model/halting.py`, `HaltingNetwork.forward()`

```python
pooled = pooled + self.level_projs[i](z.mean(dim=1))   # mean over ALL 81 cells
```

CORAL's halting loss gradients flow back through `z.mean(dim=1)` into ALL 81 cell
representations. The halting loss says "predict whether the current answer is globally
correct" — a scalar signal distributed back to every cell state uniformly.

**TRM's approach:**

```python
q_logits = self.q_head(z_out[:, 0])   # first puzzle_emb position ONLY
```

TRM uses a **dedicated first token** (the puzzle embedding position) for Q-value
computation. Halting loss gradients flow only to this one position, not to the 81
solution cells. This is a clean separation: the 81 cells learn to encode per-cell
answers; the dedicated token learns to estimate global correctness.

**Why this matters:** In CORAL, the halting loss adds a small but consistent gradient
signal to every cell state that says "your aggregate representation should predict
whether the whole puzzle is solved." This is a different objective from "your
representation should enable decoding the correct digit for this cell." The interference
is small per step but persistent across all 16 segments × 21 steps = 336 backbone
applications per forward pass.

**Fix:** Add a dedicated `[CLS]`-style token at position 0 of the sequence, projected
from a mean pool or prepended to the 81-cell sequence. Route halting gradients through
this token only by detaching z before passing to the halting network, then adding a
residual gradient path through the CLS token.

Alternatively (simpler): `self.halting(z_states)` → `self.halting([z.detach() for z in
z_states])`. This fully isolates halting from cell representation gradients, at the cost
of halting having no gradient path to the backbone. The halting network still trains
(its own weights receive gradients from the halting loss), but it learns from a
frozen/detached z.

---

## Section 6 — Decoder Bottleneck

**File:** `coral/adapters/grid.py`, `decode()` and `_init_weights()`

```python
self.decoder = nn.Linear(self.d_model, self.vocab_size, bias=True)
nn.init.normal_(self.decoder.weight, std=(self.d_model ** -0.5))  # std ≈ 0.044
```

The 512→11 linear decoder with bias=True is more than sufficient capacity for 9 digits.
TRM uses the same dimension and same init std (LeCun normal), but without bias. The
presence of a bias term in CORAL's decoder is a minor advantage (not a disadvantage).

**Assessment:** Not a root cause. No action needed.

---

## Section 7 — State Norm Growth: Pre-Norm vs TRM's Post-Norm

**File:** `coral/model/backbone.py`, `TransformerLayer.forward()`

CORAL uses pre-norm (norm applied BEFORE attention/FFN, residual stream not normalised):
```python
x = x + self.attn(self.norm1(x))   # pre-norm: residual can grow
x = x + self.ffn(self.norm2(x))
```

TRM uses post-norm (norm applied AFTER residual addition):
```python
hidden_states = rms_norm(hidden_states + self.self_attn(...))   # bounded output
hidden_states = rms_norm(hidden_states + self.mlp(...))         # always unit RMS
```

**Key consequence:** TRM's z_L always has RMS = 1 per dimension (norm ≈ sqrt(512) ≈ 22.6)
after every transformer block. This is a fixed invariant regardless of how many
recurrence steps are applied. With post-norm:

- z at step t has unit RMS
- backbone_in = z + input_injection has norm ≈ sqrt(2) * 22.6 ≈ 32 (well-conditioned)
- backbone_out returns to unit RMS (post-norm normalises it)

CORAL's z norm after multiple steps:
- At init: z norm ≈ 22.6 (from input_norm)
- After each backbone step: the pre-norm residual additions can grow the norm
- Level/timescale embeddings add tiny signal (std=0.02 → norm ≈ 0.45, negligible)
- Main growth driver: `x + attn(norm(x))` where attn output can have similar magnitude to x

In practice, with weight decay=1.0 and gradient clipping=1.0, weights stay small and
norm growth per step is modest. Over 21 inner steps, z might grow by ~1.5–2×
(from ≈22.6 to ≈34–45). This is not catastrophic but makes the system less predictable
than TRM's always-unit-norm dynamics.

**The CLAUDE.md reasoning for pre-norm** ("allows the residual stream to grow freely
across recurrence steps") is debatable: TRM uses post-norm with 168+ backbone
applications and achieves 87.4% exact accuracy. Post-norm's bounded dynamics are
empirically sufficient and arguably better for stable multi-step recurrence.

**Assessment:** Moderate concern. Not the primary cause of the plateau, but switching
to post-norm would bring CORAL's dynamics closer to TRM's and improve consistency.
Pre-norm is an explicit architectural decision that may be worth revisiting.

---

## Section 8 — TRM vs CORAL: Full Line-by-Line Comparison

**Files:** `models/recursive_reasoning/trm_singlez.py` (single-level TRM, most comparable
to CORAL N=1 baseline), `models/recursive_reasoning/trm.py` (two-level, most comparable
to CORAL N=2 baseline).

### 8.1 Complete difference table

| Aspect | TRM (singlez) | CORAL N=1 baseline |
|--------|--------------|-------------------|
| Norm type | **Post-norm** (always unit RMS) | Pre-norm (can grow) |
| Initial state | **Learned constant** `L_init` (input-agnostic) | `z1_init` = encoder output |
| Input injection | `z_L + input_embeddings` (added to z before layers) | `project_up(z) + input_injection + embeddings` |
| Consolidation | **1 step without injection** after each L_cycle | None — injection every step |
| embed_scale | Applied **after** embeddings (effective) | Applied before LayerNorm (**cancelled**) |
| Decode source | `z_L` (same state that reasons) | `z_states[0]` (same) ✓ |
| Q/halt head | **First token only** (dedicated position) | Mean pool all 81 cells |
| Level/step embeddings | None added per step | `level_emb + ts_emb` added at every step |
| Token embedding init | std = 1/sqrt(d) (LeCun normal) | std = 0.02 (smaller — different scale) |
| Backbone bias | None | `bias=True` on decoder only |
| RMSNorm | No learnable scale | No learnable scale (same) ✓ |

### 8.2 The forward path side-by-side

**TRM single-z (one outer H_cycle with gradient):**
```python
# (H_cycles-1) outer loops, NO gradient:
for _H_step in range(H_cycles - 1):
    for _L_step in range(L_cycles):
        z_L = L_level(z_L + input_embeddings)   # injection
    z_L = L_level(z_L)                          # consolidation

# Final 1 outer loop, WITH gradient:
for _L_step in range(L_cycles):
    z_L = L_level(z_L + input_embeddings)       # injection
z_L = L_level(z_L)                              # consolidation

output = lm_head(z_L)                           # decode
q_logits = q_head(z_L[:, 0])                   # halt from first token
new_carry = InnerCarry(z_L=z_L.detach())        # carry is detached
```

**CORAL baseline (_run_level, per segment):**
```python
for t in range(n_steps):                       # n_steps injection steps, NO consolidation
    backbone_in = project_up(z) + level_emb(t) + ts_emb(t) + input_injection
    z = project_down(backbone(backbone_in))

# After loop:
logits = decode_fn(z)                          # decode from z
h, q_halt, q_cont = halting(z.mean(dim=1))    # halt from mean pool of all cells
```

### 8.3 Additional diagnostic bug: wrong empty-cell mask in repr_diagnostics

**File:** `coral/training/trainer.py`, `compute_repr_diagnostics()`, line 308:

```python
empty_mask = (inputs == 0).cpu()   # ← BUG: empty cells are token 1, not 0
```

Empty cells in Sudoku are encoded as token value **1** (`inputs == 1`), not 0. Token 0
is pad (never present in Sudoku puzzles since all 81 cells have a valid token). This
means `compute_repr_diagnostics` collects states for **no cells** (all-False mask), and
the diagnostics `repr/same_digit_similarity`, `repr/effective_rank` etc. are either NaN
or computed on an empty set. The diagnostic function returns incorrect/empty results.

This does not affect training (it's only used for representation analysis), but any
W&B metrics from `repr/*` are meaningless.

**Fix:** Change `(inputs == 0)` to `(inputs == 1)` in line 308 of `trainer.py`.

---

## Summary of Root Causes

### Root Cause 1 (Confirmed Bug): embed_scale cancelled by LayerNorm

**Location:** `coral/adapters/grid.py`, `encode()`, line 102–104  
**Severity:** HIGH — affects all experiments  
**Effect:** Input injection `z1_init` has the same magnitude as the evolved state `z`,
weakening constraint anchoring. The backbone can partially ignore the given-digit
signal since it doesn't dominate. The setting `embed_scale: true/false` has zero effect.  
**Fix:** Apply `embed_scale` after `input_norm`, not before.  
**Verifying experiment:** Run two configs identical except `embed_scale` placement;
compare empty-cell accuracy curves. Also compare to disabling embed_scale entirely (which
is currently the same as enabling it).

---

### Root Cause 2 (Structural): No consolidation phase

**Location:** `coral/model/coral_core.py`, `_run_level()`  
**Severity:** HIGH — likely primary driver of multi-step reasoning ceiling  
**Effect:** z is always a function of z1_init at every step. The backbone never
processes z without input contamination. Complex deductive chains (e.g., hidden triples,
swordfish patterns) require maintaining a hypothesis state that evolves by reasoning, not
just by re-reading the input. Without a consolidation step, the recurrent attractor is
determined by the input, limiting the depth of deduction to first-order constraints.  
**Fix:** Add 1 consolidation step (no input_injection) at the end of each inner loop
in `_run_level`. Optionally group `L_cycles` injection steps + 1 consolidation step
as a repeating unit (matching TRM's structure exactly).  
**Verifying experiment:** Compare N=1 baseline with and without 1 consolidation step
at the end of the 21-step inner loop. Expect improvement specifically on the 60+ empty
bucket (extreme puzzles needing multi-step deduction).

---

### Root Cause 3 (Structural): Initial state = input embedding

**Location:** `coral/model/coral_core.py`, `forward()`, line 302  
**Severity:** MEDIUM — interacts with Root Cause 2  
**Effect:** `z_states = [z1_init]` starts the reasoning state as the encoder output.
For empty cells, z1_init encodes "this cell is empty" + 2D position, which is identical
for all empty cells sharing the same position (and similar across positions since the
position embedding has small std=0.02). The model must learn to move z away from this
"empty" representation through recurrence. TRM uses a learned constant `L_init`
initialised with `trunc_normal(std=1)`, allowing the starting point to specialise
through training as a good "reasoning seed."  
**Fix:** Add `self.z1_init = nn.Parameter(torch.empty(config.backbone_dim))` initialised
with `trunc_normal(std=1)`. Use this as z1 starting state and keep z1_init as injection only:

```python
# In forward():
z1_learned_init = self.z1_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
z_states = [z1_learned_init]   # learned constant, not input-dependent
input_signal = z1_init          # injection remains input-dependent
```

**Verifying experiment:** Run with learned init vs input init, same config. If learned
init helps, the improvement should be visible in all difficulty buckets. If input init
is actually load-bearing (the model uses the initial z to identify given cells), the
experiment will show no improvement or regression.

---

## Prioritised Experiment Order

| Priority | Experiment | Expected signal | Config change |
|----------|-----------|-----------------|---------------|
| 1 | Fix embed_scale (move after input_norm) | Faster early training, higher plateau | `grid.py` 3-line change |
| 2 | Add consolidation step (1 no-injection step at end of inner loop) | Better 50+ empty accuracy | `coral_core.py` `_run_level` |
| 3 | Fix repr_diagnostics empty mask (`== 0` → `== 1`) | Correct W&B repr/* metrics | `trainer.py` 1-line change |
| 4 | Switch to learned z_init (separate from input_injection) | Cleaner reasoning dynamics | `coral_core.py` forward |
| 5 | Detach z_states before halting network | Eliminates gradient contamination | `coral_core.py` 1-line change |
| 6 | Switch to post-norm (matches TRM dynamics) | Bounded, more predictable state | `backbone.py` 4-line change |
| 7 | Q-head on dedicated first token instead of mean pool | Cleaner halt signal | `halting.py` + `coral_core.py` |

**Experiments 1, 2, 3** are the highest-confidence changes. Do them together in one
commit, then run a full Phase 1 training run to compare against the baseline.

---

## Diagnostic Predictions

If the analysis is correct:

- **embed_scale fix alone:** Expect empty-cell accuracy to rise from ~45% to ~50–55%,
  with faster convergence. Given-cell accuracy remains near 100%.

- **Consolidation step alone:** Expect improvement specifically on 60+ empty bucket.
  30–49 and 50–59 buckets (solvable by first-order propagation) may not change much.
  Consolidation step should enable the model to learn hidden singles and pairs.

- **Both together:** Should move plateau from ~45% to ~60–65% empty accuracy,
  approaching the regime where exact accuracy starts to become meaningful.

- **If neither helps:** The plateau is caused by something else — the most likely
  remaining hypothesis is that a 512-dim single-level state is insufficient to represent
  the constraint state for 55+ empty Sudoku cells simultaneously. In that case:
  switch to N=2 hierarchy with full PC, which increases effective reasoning capacity.

---

## Appendix: Confirmed Bugs Summary

| # | File | Line | Bug | Impact |
|---|------|------|-----|--------|
| 1 | `coral/adapters/grid.py` | 102–104 | `embed_scale` applied before `input_norm` — has zero effect | Training (all runs) |
| 2 | `coral/training/trainer.py` | 308 | `(inputs == 0)` should be `(inputs == 1)` for empty cells | Diagnostics only |
