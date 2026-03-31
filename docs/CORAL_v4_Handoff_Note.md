# CORAL v4 — Handoff Note

## Date: March 29, 2026

## Purpose

This document captures the complete state of the CORAL v4 project to enable
seamless continuation in a new conversation. It covers: what was built, what
works, what doesn't, what's been tried, what to do next, and all known issues.

---

## 1. Project Overview

CORAL (COrtical Reasoning via Abstraction Layers) is a novel neural architecture
for complex reasoning tasks grounded in variational free energy minimisation.
The goal is an **amodal reasoning core** — a modality-agnostic module that takes
embeddings in, reasons through a multi-timescale recurrent hierarchy with
precision-weighted predictive coding, and returns refined embeddings out. It
should interface with language, vision, and sensor inputs through lightweight
adapters, and eventually deploy on devices from cloud GPUs to mobile phones.

**Repository:** `github.com/anwar11235/CORAL-v4` (clean new repo, not based on v3)

**Architecture Spec:** `CORAL_Architecture_Spec_v4.md` (v4.1) — authoritative
reference for all design decisions.

**Build Plan:** `CORAL_v4_Build_Plan.md` — implementation plan with file-by-file
build order and experiment sequence.

**W&B Workspace:** `aktuator-ai`, project `Sudoku-extreme-1k-aug-1000 CORAL-v4`

**Compute:** Vast.ai A100-SXM4-40GB (preferred), SSH access pattern:
`ssh -p <PORT> root@<HOST> -L 8080:localhost:8080`

---

## 2. What Has Been Built

### 2.1 Complete Implementation (Phase 1)

Claude Code built the entire Phase 1 codebase. 35/35 tests pass.

| File | Description |
|------|-------------|
| `coral/config.py` | Hydra dataclasses with all fields defaulted |
| `coral/model/backbone.py` | 2-layer transformer: RoPE, RMSNorm, SwiGLU FFN, PyTorch SDPA |
| `coral/model/level_module.py` | Up/down projections per hierarchy level |
| `coral/model/predictive_coding.py` | Prediction nets, precision nets, error-up projection |
| `coral/model/halting.py` | Q-learning adaptive halting |
| `coral/model/coral_core.py` | Main assembly — full backprop within segments, detach between |
| `coral/adapters/grid.py` | Sudoku encoder/decoder (token + 2D positional embeddings) |
| `coral/training/losses.py` | Stablemax CE (float64) + PC loss + precision regulariser |
| `coral/training/trainer.py` | Deep supervision training loop |
| `coral/evaluation/pareto.py` | Accuracy@K1/K2/K4/K8/K16 Pareto curve |
| `coral/evaluation/evaluator.py` | Accuracy metrics (exact, token) |
| `scripts/train.py` | Hydra entry point with W&B integration |
| `scripts/diagnostic_precision.py` | Quick precision diagnostic (1000 steps) |
| `scripts/diagnostic_no_pc.py` | Baseline diagnostic without PC (500 steps) |
| `configs/exp1_baseline.yaml` | Experiment 1 config |
| `coral/data/build_sudoku_dataset.py` | Copied from HRM repo, generates training data |
| `coral/data/common.py` | Copied from HRM repo, shared utilities |

### 2.2 Key Architecture Parameters

- **Backbone:** 2-layer transformer, d=512, 8 heads, SwiGLU FFN 4×, RMSNorm post-norm, RoPE
- **Hierarchy:** N=2 levels (d₁=512, d₂=256), T=3 timescale multiplier
- **Inner steps per segment:** Level 1 = 3 steps, Level 2 = 1 step (4 total)
- **Segments:** K_max = 16 (deep supervision)
- **Full backprop:** Within each segment, all 4 backbone applications are in the same
  computation graph. Detach only between segments.
- **Total parameters:** 10,232,717
- **Optimizer:** AdamW (lr=7e-5, weight_decay=1.0, betas=(0.9, 0.95), cosine schedule)
- **Precision:** bfloat16 forward, float64 loss (stablemax cross-entropy)

### 2.3 Data Pipeline

- **Dataset:** Sudoku-Extreme-1K (HRM format)
- **Generation:** `python coral/data/build_sudoku_dataset.py --output-dir data/sudoku_extreme_1k --subsample-size 1000 --num-aug 1000`
- **Output:** `.npy` arrays in `data/sudoku_extreme_1k/{train,test}/`
- **Format:** Inputs 1–10, labels 2–10, vocab_size=11 (pad=0, digits 1–10)
- **Splits:** Train dir = `train/`, eval dir = `test/` (not `eval/`), file prefix = `all__`

---

## 3. What Has Been Tested and Fixed

### 3.1 Bugs Found and Fixed

| Bug | Root Cause | Fix | Status |
|-----|-----------|-----|--------|
| CUDA index out of bounds | `vocab_size: 10` in YAML overriding correct default of 11 | Changed YAML to `vocab_size: 11` | ✅ Fixed |
| Eval crash FileNotFoundError | Eval looking for `eval/train__inputs.npy` instead of `test/all__inputs.npy` | Fixed split name and set name in eval code | ✅ Fixed |
| Eval dtype mismatch | `adapter.decode()` outside `torch.autocast` block | Moved decode inside autocast | ✅ Fixed |
| Trainer-core interface mismatch | Trainer manually managed segments; core manages them internally | Fixed trainer to call `core(z1_init, ...)` correctly | ✅ Fixed |
| Precision network outputs constant 0.703 | All weights initialised to zero → constant function | Changed to `xavier_uniform_` init | ✅ Fixed |
| Precision collapses to uniform π=1 | Regulariser overpowers task gradient; no gradient path from task loss through precision | Multiple fixes attempted (see Section 4) | ⚠️ Partially resolved |
| PC conditioning hurts performance | Adding mu to backbone input doubles signal when mu ≈ z1 | Changed to post-backbone residual: `z_new + gate * (mu - z)` | ✅ Fixed |
| Prediction loss explodes | `pi * eps²` summed over 512 dimensions produces huge values | Changed `.sum(dim=-1)` to `.mean(dim=-1)` | ✅ Fixed (needs verification) |
| Lambda_pred too high | At 0.1, prediction loss dominates total loss | Changed to 0.001 | ✅ Fixed (needs verification) |

### 3.2 Vast.ai Instance Setup Procedure

Every new instance requires:

```bash
tmux new -s coral
cd /workspace/CORAL-v4  # or wherever the repo is cloned
export PYTHONPATH=/workspace/CORAL-v4:$PYTHONPATH
pip install wandb hydra-core omegaconf tqdm huggingface_hub argdantic pydantic einops --break-system-packages
python coral/data/build_sudoku_dataset.py --output-dir data/sudoku_extreme_1k --subsample-size 1000 --num-aug 1000
wandb login
```

Note: `pip install -e .` fails on Vast.ai containers due to old pip/setuptools
incompatibility. Use the manual pip install + PYTHONPATH workaround instead.

---

## 4. The Precision Collapse Problem — Full History

This is the central technical challenge. Here is every attempt made, in order:

### Attempt 1: Original design (zero-init precision network)
- **Result:** Precision network output constant 0.703 across all dimensions forever
- **Diagnosis:** All-zero weight init → constant function → symmetric gradient → no differentiation
- **Fix:** Xavier uniform init on precision network weights

### Attempt 2: Xavier init, λ_π=0.01
- **Result:** Precision differentiated initially (std=0.207) but regulariser flattened it within 500 training steps. Task loss stuck at 2.20.
- **Diagnosis:** Regulariser stronger than task gradient through precision

### Attempt 3: Xavier init, λ_π=0.001
- **Result:** Same collapse, just slightly slower
- **Diagnosis:** Even weaker regulariser still wins because task gradient through precision vanishes when prediction error → 0

### Attempt 4: Xavier init, λ_π=0.0 (no regulariser)
- **Result:** Still collapses. Precision std 0.52 → 0.09 over 200 steps.
- **Diagnosis:** Regulariser is NOT the cause. Something else is flattening precision.

### Attempt 5: Precision-gated decode (pi * z_states[0] → decoder)
- **Result:** Task loss moved to 1.84 (first time!) but total loss exploded to 2500+ after step 250
- **Diagnosis:** Feedback loop between precision and logit magnitude creates instability
- **Reverted** — precision-gated decode removed

### Attempt 6: Precision input = cat(z_lower, eps) instead of just z_lower
- **Result:** No improvement. eps → 0 quickly, so cat(z, ~zeros) ≈ z
- **Diagnosis:** The eps input doesn't help because eps itself vanishes

### Attempt 7: Post-backbone residual conditioning (current design)
- **Result:** Task loss reaches 1.61 at step 800 — best PC result. But prediction loss still explodes over time.
- **Status:** Residual conditioning WORKS. This is the correct design for how predictions integrate with the backbone.

### Attempt 8: lambda_pred 0.1 → 0.001 + sum→mean over dims
- **Result:** NOT YET TESTED on GPU. Changes committed to repo but the last diagnostic was interrupted.
- **This is the immediate next step.**

### Key Insight

The fundamental problem is that once prediction error collapses (eps → 0, by step ~300),
precision receives almost no gradient signal:
- Gradient from prediction loss: `dL/dpi = 0.5 * eps²` → vanishes
- Gradient from regulariser: pulls toward uniform π=1
- Gradient from task loss: none (precision doesn't affect output logits)

The residual conditioning fix (Attempt 7) solved the "PC hurts performance" problem.
The lambda_pred + sum→mean fix (Attempt 8) should solve the "auxiliary loss explodes"
problem. But precision still has no strong reason to differentiate long-term.

**The precision collapse may be acceptable.** With the residual conditioning design,
the architecture performs well even with relatively flat precision (task loss 1.61).
The prediction error signal and the residual correction are the mechanisms doing the
work — precision weighting may be less important than originally theorized.

---

## 5. Experimental Results

### 5.1 Completed Training Runs

| Run | Config | Result | W&B |
|-----|--------|--------|-----|
| v4-exp1-baseline-r2 | Original PC, flat precision (zero init bug) | Token acc 51.5% at 15K steps, exact acc 0% | lo0xoplp (incomplete) |
| v4-exp1-baseline-r3 | Same but eval_every=5000 | Same — token acc ~51%, exact acc 0% | 7c7yj7m1 |

### 5.2 Diagnostic Results (Short Runs)

| Config | Steps | Task Loss | Notes |
|--------|-------|-----------|-------|
| No PC (baseline) | 500 | 1.37 | Backbone works well alone |
| PC + residual conditioning, λ_pred=0.1 | 1000 | 1.61 (then explodes) | Best PC task loss, but total loss unstable |
| PC + residual conditioning, λ_pred=0.001, mean-over-dims | 1000 | NOT YET TESTED | **This is the next test** |

### 5.3 Phase 1 Reference (v3 repo)

For comparison, CORAL v3 (Phase 1) achieved:
- 61.1% eval exact accuracy on Sudoku-Extreme-1K
- Used separate H/L modules (not shared backbone), 1-step gradient approx, 27M params
- W&B run: mfno8t1y in the v3 project

---

## 6. Committed Architectural Decisions

These are final and should not be revisited:

- **Shared 2-layer backbone** with level embeddings (not separate per-level modules)
- **Self-attention** (not MLP-mixer) — required for variable sequence lengths
- **T=3 timescale multiplier** — 40 inner steps per segment at N=4
- **Full backprop through inner loop** within segments, detach between segments
- **Precision regulariser:** symmetric log-normal `(λ_π/2)(log π)²`
- **bfloat16 forward, float64 loss**
- **PyTorch SDPA for attention** (not flash-attn package)
- **AdamW optimizer** (fused AdamATan2 as optional fallback if it compiles)
- **Post-backbone residual conditioning:** `z_new = backbone(z); z = z_new + gate * (conditioning - z)`
  This is the correct design — predictions/errors are residual corrections, not backbone input.

---

## 7. Current Architecture Design

### 7.1 Forward Pass (Per Segment)

```
# Level 1: 3 inner steps
for t in 0..2:
    z1 = project_down(backbone(project_up(z1) + level_emb[0] + ts_emb[t]))
    if predictions[0] is not None:
        z1 = z1 + cond_gate[0] * (predictions[0] - z1)  # residual correction

# Compute PC: prediction error, precision, weighted error
mu = prediction_net(z2)
eps = z1 - mu
pi = precision_net(cat(z1, eps))  # precision depends on both state and error
xi = pi * eps
xi_up = error_up_proj(xi)

# Level 2: 1 inner step
z2 = project_down(backbone(project_up(z2) + level_emb[1] + ts_emb[0]))
if xi_up is not None:
    z2 = z2 + cond_gate[1] * (xi_up - z2)  # residual correction from error below

# Decode for deep supervision
logits = decode_fn(z_states[0])  # NO precision gating on decode

# Halting check
h_k = halting_net(z_states)

# Detach for next segment
z_states = [z.detach() for z in z_states]
```

### 7.2 Loss Function

```
L = L_task                                          # stablemax CE, float64
  + λ_pred * mean(pi * eps²)                        # precision-weighted prediction error
  + λ_π * mean((log π)²)                            # precision regulariser
  + L_halt                                           # Q-learning halting
  + λ_amort * amortisation_loss (disabled for Exp 1)
  + λ_crystal * crystallisation_loss (disabled for Exp 1)
```

### 7.3 Key Hyperparameters (Current)

| Parameter | Value | Notes |
|-----------|-------|-------|
| lambda_pred | 0.001 | Reduced from 0.1 to prevent prediction loss explosion |
| lambda_pi | 0.001 | Reduced from 0.01; original 0.01 was too strong |
| epsilon_min | 0.01 | Precision floor |
| vocab_size | 11 | pad=0, digits 1–10 |
| K_max | 16 | Deep supervision segments |
| lr | 7e-5 | AdamW |
| weight_decay | 1.0 | Following HRM |
| batch_size | 64 | |
| training.eval_every | 500 | Quick eval (100 puzzles) |
| training.pareto_eval_every | 5000 | Full Pareto eval |

---

## 8. Immediate Next Steps

### Step 1: Verify Loss Stability (5 minutes GPU)

The last two fixes (lambda_pred=0.001, sum→mean over dims) have NOT been verified
on GPU together. This is the first thing to do:

```bash
cd /workspace/CORAL-v4
python scripts/diagnostic_precision.py data.dataset_path=data/sudoku_extreme_1k
```

**Pass criteria:**
- Task loss below 1.7 at step 500
- Total loss stable through 1000 steps (no component exceeding 10)
- No NaN

### Step 2: Launch Full Experiment 1 (4-6 hours GPU)

If the diagnostic passes:

```bash
python scripts/train.py wandb.disabled=false wandb.run_name=v4-exp1-stable
```

**Target:** ≥70% eval exact accuracy on Sudoku-Extreme-1K at 20K steps.

### Step 3: Analyse Results

Compare against:
- No-PC baseline (diagnostic showed task loss 1.37 at 500 steps)
- Phase 1 v3 result (61.1% exact accuracy, different architecture)
- HRM reported (55%)
- TRM reported (87.4%)

### Step 4: Decision Point

Based on Experiment 1 results:
- **If exact accuracy ≥ 50%:** PC is contributing. Proceed to Experiment 2 (amortisation pressure).
- **If exact accuracy 20-50%:** PC is marginally helping. May need hyperparameter tuning or deeper investigation.
- **If exact accuracy < 20%:** PC is still hurting. Run a full 20K training without PC as a true baseline, then diagnose the gap.

---

## 9. Infrastructure Notes

### 9.1 GPU Efficiency Improvements (Implemented)

- `data.num_workers: 4` (was 1) — keeps GPU fed
- Quick eval (100 puzzles) at step 500, full Pareto at step 5000 — prevents eval from dominating runtime
- `torch.compile` option available (`training.compile_model: true`) but kept off by default

### 9.2 Known Infrastructure Issues

- `pip install -e .` fails on Vast.ai → use manual pip install + PYTHONPATH
- `flash-attn` package fails to compile → use PyTorch SDPA
- `torch.compile` with `dynamic=True` causes slowdown from recompilation → use `dynamic=False` for Sudoku
- "Casting complex values to real" warning from RoPE is harmless
- Eval dataloader uses `test/` split (not `eval/`), file prefix `all__` (not `train__`)

### 9.3 Diagnostic Scripts

- `scripts/diagnostic_precision.py` — 1000 steps, logs precision stats every 100 steps.
  Pass: precision std > 0.01. Accepts `data.dataset_path` as arg.
- `scripts/diagnostic_no_pc.py` — 500 steps without PC, logs task loss every 100 steps.
  Provides the "backbone only" baseline.

Both currently hardcoded to CUDA. A `--device cpu` option was discussed but not
yet implemented.

---

## 10. Competitive Landscape Summary

For context in any architecture decisions:

| Model | Params | Sudoku-Extreme Accuracy | Key Mechanism |
|-------|--------|------------------------|---------------|
| HRM | 27M | 55% (reported) | Two-timescale recurrence |
| CORAL v3 Phase 1 | 27M | 61.1% | Precision-weighted PC (separate H/L modules) |
| TRM | 5-7M | 87.4% | Single tiny network, full backprop, deep recursion |
| Augmented HRM | 27M | 96.9% | Data aug + input perturbation + model bootstrapping |
| GRAM | 10M | 97.0% | Stochastic transitions |

Key findings from the literature:
- Hierarchy (H/L split) is marginal — ARC Prize, Ge et al., TRM all confirm
- Full backprop through recursion gives massive gains (TRM: 56.5% → 87.4%)
- MoE/routing collapses in this regime (TRM, CORAL Phase 2)
- Stochastic transitions help escape spurious fixed points (GRAM, Ren & Liu)
- Deep supervision + iteration depth is the primary driver of performance

---

## 11. Broader Architecture Vision

The v4 spec describes a 4-level hierarchy (N=4) with:
- Progressive dimensionality reduction (512 → 256 → 128 → 64)
- Amortisation-driven crystallisation (learn to skip computation)
- Precision-driven sparsity (replace explicit routing)
- Stochastic variational transitions (precision-gated noise)
- Amodal interface protocol (pluggable adapters for any modality)

Current implementation is N=2 only. Scaling to N=3/N=4 is Experiment 4 in the
build plan, after the core mechanisms are validated at N=2.

The long-term vision: a generalist amodal reasoning core deployable on robots,
phones, and embedded systems. Not a benchmark-optimised model, but a principled
architecture that improves its own efficiency over time through amortisation.

---

## 12. Reference Documents

All three documents should be provided to the new conversation:

1. **CORAL_Architecture_Spec_v4.md** (v4.1) — Architecture specification
2. **CORAL_v4_Build_Plan.md** — Build plan with experiment sequence
3. **This handoff note** — Current state and next steps

The architecture spec and build plan may have minor discrepancies with the actual
implementation (e.g., lambda_pred was 0.1 in the spec, now 0.001). The handoff note
reflects the actual current state. When in conflict, trust this handoff note over
the spec/plan.
