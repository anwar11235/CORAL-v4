# TRM vs CORAL-v4 — Training Audit

**Date:** 2026-03-31  
**Purpose:** Identify why TRM achieves 87.4% exact accuracy on Sudoku-Extreme while CORAL
plateaus at ~45% empty-cell token accuracy.

**TRM source:** `C:\Users\mauha\dev-projects\TinyRecursiveModels`  
**CORAL source:** `C:\Users\mauha\dev-projects\CORAL-v4`

---

## Critical Up-Front Corrections

The preliminary analysis (from an automated audit pass) claimed "TRM masks given cells and
trains on empty cells only." **This is incorrect.** Both TRM and CORAL:
- Store labels as the full 81-cell solution (digits 1–9 → token indices 2–10)
- Set `ignore_label_id=0` in metadata, but no label ever equals 0 (solutions are complete)
- Compute loss on **all 81 cells** — no cells are masked in either codebase

The real differences are structural. See below.

---

## Preliminary: What TRM Actually Achieves 87.4%

The 87.4% model is **not** the attention variant of TRM. From the README (lines 64–78):

```bash
# 87.4% model:
python pretrain.py arch=trm ...
    arch.mlp_t=True arch.pos_encodings=none   # ← MLP over L, NOT attention

# Attention variant:
python pretrain.py arch=trm ...
    # mlp_t not set                           # ← transformer attention
# Expected: Around 75% exact-accuracy
```

**CORAL uses attention (transformer backbone), which is comparable to TRM's 75% variant.**
The 87.4% vs ~0% exact accuracy gap involves a fundamentally different architecture. The
fair comparison is TRM-attention (~75%) vs CORAL-attention (unknown exact %).

---

## 1. Data Encoding — IDENTICAL

| Property | TRM | CORAL |
|----------|-----|-------|
| Empty cell token | 1 | 1 |
| Digits 1–9 tokens | 2–10 | 2–10 |
| Vocab size | 11 | 11 |
| Given cells in labels | full solution (2–10) | full solution (2–10) |
| Given cells masked? | **No** | **No** |

Both dataset builders are near-identical (CORAL forked from TRM). `ignore_label_id=0`
never fires because no solution cell has token value 0.

---

## 2. Architecture

| Property | TRM (att) | TRM (mlp_t) | CORAL Phase 1 |
|----------|-----------|-------------|---------------|
| Inner module | Transformer (self-attn) | MLP over L dimension | Transformer (self-attn) |
| d_model | 512 | 512 | 512 |
| n_heads | 8 | — (no attn) | 8 |
| FFN expansion | 4 | 4 | 4 |
| Layers per backbone call | 2 | 2 | 2 |
| Positional encoding | **RoPE** | **none** | Learned 2D (row+col) |
| Hierarchy | 2-level (H + L) | 2-level (H + L) | 1-level |
| H_cycles | 3 | 3 | — |
| L_cycles per H | 6 | 6 | — |
| inner_steps per segment | 18 (3×6) | 18 (3×6) | 21 (override) |
| Max outer segments (K_max) | 16 | 16 | 16 |
| Per-puzzle embeddings | **yes (512-dim, 16 tokens)** | **yes (512-dim, 16 tokens)** | **no** |
| Structural attn bias | no | no | row/col/box (disabled in phase1) |

### 2a. TRM's Two-Level Hierarchy

TRM has two shared states — `z_H` (high, slow) and `z_L` (low, fast):

```python
# Per outer segment:
# H_cycles-1 = 2 passes with NO GRADIENT:
with torch.no_grad():
    for _H_step in range(H_cycles - 1):       # 2 no-grad H-cycles
        for _L_step in range(L_cycles):        # 6 L-cycles each
            z_L = L_level(z_L, z_H + input_embeddings)   # L ← H+input
        z_H = L_level(z_H, z_L)                           # H ← L
# 1 pass WITH GRADIENT:
for _L_step in range(L_cycles):               # 6 L-cycles with grad
    z_L = L_level(z_L, z_H + input_embeddings)
z_H = L_level(z_H, z_L)
```

- Input embeddings injected at **every L-level call** (before transformer layers)
- `z_H` updated only once per H-cycle from `z_L` (slow global context)
- `z_L` updated 6× per H-cycle from `z_H + input` (fast local refinement)
- Gradient flows through only 12 backbone applications (1 H-cycle × 6 L × 2 layers)

CORAL (phase 1 baseline): single z state, input injected at every inner step, full gradient
through all 21 steps.

### 2b. Per-Puzzle Embeddings (TRM only)

TRM has `puzzle_emb_ndim=512` — a `CastedSparseEmbedding` table of shape
`[num_puzzle_identifiers, 512]`, trained with `CastedSparseEmbeddingSignSGD` at lr=1e-2
(separate from the model's AdamATan2 optimizer).

These 16 "prefix tokens" are prepended to the 81-cell sequence. The LM head strips them:
```python
output = self.lm_head(z_H)[:, self.puzzle_emb_len:]   # remove 16 prefix tokens
```

**Effect:** Each unique puzzle gets a learnable embedding that the model can exploit to
store puzzle-specific information. With 1000 base puzzles × 1000 augmentations = 1M
effective examples, the puzzle embedding can learn a compressed representation of the
solution space for each original puzzle (augmented versions share the same puzzle_id only
if they originate from the same base puzzle — needs verification).

CORAL has no equivalent mechanism.

---

## 3. Training Hyperparameters

| Parameter | TRM (Sudoku) | CORAL Phase 1 |
|-----------|--------------|---------------|
| **Optimizer** | **AdamATan2** | **AdamW** |
| Learning rate | 1e-4 | 7e-5 |
| Weight decay | 1.0 | 1.0 |
| Beta1, Beta2 | 0.9, 0.95 | 0.9, 0.95 |
| **Global batch size** | **768** | **64** |
| **Epochs** | **50,000** | **20,000** |
| LR schedule | Cosine w/ warmup | Cosine w/ warmup |
| Warmup steps | 2000 | 500 |
| Gradient clipping | none (AdamATan2 normalises) | 1.0 |
| **EMA on weights** | **yes (rate=0.999)** | **no** |

### Training steps calculation

Both datasets: 1000 base puzzles × 1000 augments = 1M examples.

**TRM total_steps** = `epochs × total_groups × mean_puzzle_examples / global_batch_size`
= 50,000 × 1000 × 1001 / 768 ≈ **65M steps** (but each step is one ACT cycle — see below)

**CORAL total_steps** = 20,000 (configured explicitly)

These are not directly comparable because TRM's step is one ACT segment while CORAL's step
is 16 segments with deep supervision. See Section 5.

---

## 4. Loss Function

| Property | TRM | CORAL |
|----------|-----|-------|
| Task loss type | stablemax cross-entropy (float64) | stablemax cross-entropy (float64) |
| Loss computed on | all 81 cells | all 81 cells |
| Given cells masked | no | no |
| Normalisation | per-sequence mean (`/ loss_counts`) | per-sequence mean + batch mean |
| Q-halt loss | 0.5 × BCE(q_halt, is_correct) | 0.5 × BCE(q_halt, is_correct) |
| Q-continue loss | disabled (`no_ACT_continue=True`) | disabled (fixed in Session 9) |
| PC prediction loss | no | no (baseline mode) |
| Commitment loss | no | no (baseline mode) |
| **Deep supervision** | **no — loss at current step only** | **yes — loss at each of 16 segments** |

### TRM loss structure
TRM calls `model.forward()` once per training step, computing loss for **one ACT segment**.
The carry (z_H, z_L) is detached and passed to the next training step. Each training step
trains on the current segment's output only.

### CORAL loss structure
CORAL runs all 16 segments in one training call, accumulating loss at every segment with
linear weighting (`deep_supervision_weighting: "linear"`). Gradients flow within each
segment, detached between segments.

**This means CORAL receives 16× more gradient signals per forward pass than TRM.**
Whether this helps or hurts depends on whether early-segment predictions help or hurt
the later-segment representations.

---

## 5. Gradient Flow

| Property | TRM | CORAL |
|----------|-----|-------|
| Gradient within segment | through 1/H_cycles of inner steps (12 backbone apps) | through all 21 inner steps |
| Gradient between segments | detached | detached |
| Carry persistence | **across training batches** | **reset each batch** |

### TRM's persistent carry (CRITICAL)

TRM's `train_state.carry` persists across training batches:
```python
# Training loop:
if train_state.carry is None:
    train_state.carry = train_state.model.initial_carry(batch)  # only once

train_state.carry, loss, ... = train_state.model(carry=train_state.carry, batch=batch)
```

The carry is reset only when a sequence halts (`halted=True` triggers `reset_carry`). This
means TRM can accumulate state across many passes through the same puzzle over multiple
training steps. The model learns to iteratively refine its state over time, not just within
a single forward pass.

CORAL resets its state every batch. All 16 segments of refinement happen within a single
forward call.

---

## 6. Input Injection

| Property | TRM | CORAL |
|----------|-----|-------|
| Injection point | before transformer layers in L_level (inside `ReasoningModule.forward`) | added to `backbone_in` before backbone (similar) |
| Injection signal | `z_H + input_embeddings` (note: includes current H state) | `input_injection` (original encoder output only) |
| Scaling | ×embed_scale (√512 ≈ 22.6) applied to input_embeddings at encoding | no scaling |
| Applied every step? | yes, every L-cycle call | yes, every inner step |

TRM scales input embeddings by √hidden_size = √512 ≈ 22.6. CORAL does not scale the
injection signal. At 22.6× the scale of an unscaled embedding, TRM's injection is much
stronger relative to the accumulated state.

---

## 7. Evaluation Metrics

| Metric | TRM | CORAL |
|--------|-----|-------|
| `accuracy` | correct cells / all 81 cells (given + empty) | `eval/token_accuracy`: correct / all 81 cells |
| `exact_accuracy` | all 81 cells correct | `eval/exact_accuracy`: same |
| Separate given/empty breakdown | no | available but not primary metric |

Both evaluate on all 81 cells. CORAL's observation of "100% given, 45% empty" comes from
manual analysis of the trainer's eval_step output, not the primary evaluator.

**CORAL's ~0% exact accuracy** is consistent with 45% empty-cell accuracy — getting all
~41 empty cells correct requires per-cell accuracy well above 99% for meaningful exact rates.

**TRM's 87% exact accuracy** implies per-empty-cell accuracy is very high (~97–99%+
per-cell, since 0.99^41 ≈ 66%, 0.995^41 ≈ 81%, 0.998^41 ≈ 92%).

---

## 8. Top Differences Ranked by Likely Impact

### #1 — Training Epochs + Batch Size (2.5× more data, 12× larger batches)

TRM trains 2.5× longer with batches 12× larger. At 768 batch size, each step sees much
more diverse examples and gives more stable gradient estimates. This alone likely accounts
for significant performance difference.

**Experiment:** Train CORAL for 50K epochs with batch_size=256 or 512.

### #2 — Per-Puzzle Embeddings (TRM only)

TRM's 512-dim per-puzzle embedding (16 prefix tokens) acts as a learnable "hint" that
the model can query to retrieve puzzle-specific patterns. This is trained with sign-SGD at
lr=1e-2, effectively memorising solution structure per puzzle.

**This may be the single most powerful mechanism.** Without it, TRM-attention gets ~75%.
With it, TRM-mlp gets 87%.

**Experiment:** Add a per-puzzle embedding to CORAL (sparse, trained separately with higher lr).

### #3 — Persistent Carry Across Batches (TRM) vs Fresh Start Each Batch (CORAL)

TRM accumulates state over many training steps for the same puzzle. Over 50K epochs, the
same 1000 base puzzles are seen thousands of times, and the model progressively improves
its state representations. CORAL starts fresh each batch, relying entirely on 16 segments
of in-batch reasoning.

**Experiment:** Increase K_max (segments) significantly, or implement a carry-over mechanism.

### #4 — AdamATan2 vs AdamW

AdamATan2 normalises gradient updates by angle rather than magnitude, similar to sign-SGD.
It avoids the need for gradient clipping and can be more stable on transformer training.
CORAL uses AdamW with gradient clipping at 1.0.

**Experiment:** Switch to AdamATan2 (available as `adam_atan2` package on PyPI).

### #5 — EMA on Weights

TRM evaluates with an EMA model (rate=0.999). EMA typically improves generalisation by
smoothing out training oscillations. CORAL evaluates with the current (non-EMA) weights.

**Experiment:** Add EMA to CORAL (straightforward to implement).

### #6 — Input Embedding Scale (22.6×)

TRM scales input embeddings by √512 ≈ 22.6 before injection. This makes the injection
signal much stronger relative to the accumulated state. CORAL injects unscaled embeddings.
If the injection is too weak, the model loses the input constraints during recurrent reasoning.

**Experiment:** Scale CORAL's `input_injection` by √backbone_dim.

### #7 — Two-Level Hierarchy vs Single Level

TRM separates reasoning into slow (H) and fast (L) streams, with H providing global context
and L doing rapid local refinement. CORAL's baseline is single-level. This structural prior
may help TRM allocate capacity differently.

**Experiment:** Enable CORAL's multi-level mode (n_levels=2 with appropriate config).

### #8 — Partial Gradient (TRM 1/3 with grad) vs Full Gradient (CORAL 21 steps)

TRM backprops through only 1 of 3 H-cycles (6 L-cycles × 2 layers = 12 backbone apps).
This may prevent vanishing/exploding gradients from unrolling 21+ steps. CORAL uses full
backprop through all 21 steps.

**Experiment:** Try `H_cycles=3, L_cycles=7, full_inner_backprop=False` — only backprop
through the last L-cycle pass.

---

## 9. Quick Win Experiments (Priority Order)

| Priority | Experiment | Expected Gain | Complexity |
|----------|-----------|---------------|------------|
| 1 | Train 50K epochs, batch_size≥256 | High (more compute) | Low |
| 2 | Add per-puzzle embeddings (sparse, lr=1e-2) | High | Medium |
| 3 | Scale input_injection by √backbone_dim | Medium | Very Low |
| 4 | Add EMA (rate=0.999) | Medium | Low |
| 5 | Switch to AdamATan2 | Medium | Low |
| 6 | Reduce to 1 with-grad pass (partial grad) | Low-Medium | Low |

---

## 10. File References

### TRM
| File | Purpose |
|------|---------|
| `models/recursive_reasoning/trm.py` | TRM model: two-level hierarchy, input injection, halting |
| `models/losses.py` | ACTLossHead: loss on all 81 cells, no given-cell masking |
| `config/arch/trm.yaml` | Architecture config: H_cycles=3, L_cycles=6, puzzle_emb_ndim=512 |
| `config/cfg_pretrain.yaml` | Default config: global_batch_size=768, epochs=100000 |
| `pretrain.py` | Training loop: persistent carry, AdamATan2, EMA |
| `README.md:64–78` | Sudoku-Extreme commands with expected accuracy |

### CORAL
| File | Purpose |
|------|---------|
| `coral/model/coral_core.py` | CoralCore: single level, full grad, input injection |
| `coral/training/trainer.py` | Deep supervision, fresh carry each batch |
| `coral/training/losses.py` | Loss on all 81 cells (no masking) |
| `configs/phase1_baseline_no_pc.yaml` | 20K steps, batch 64, lr=7e-5, inner_steps=21 |
| `coral/evaluation/evaluator.py` | Accuracy on all 81 cells |
