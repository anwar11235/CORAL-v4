# CLAUDE.md — CORAL v4 Repository Context

## What This Project Is

CORAL (COrtical Reasoning via Abstraction Layers) is a brain-inspired neural architecture
for complex reasoning tasks. It is an **amodal reasoning core** — a modality-agnostic
module that takes embeddings in, reasons through a multi-timescale recurrent hierarchy,
and returns refined embeddings out. It interfaces with language, vision, and sensor inputs
through lightweight adapters.

The core thesis: **the goal of a reasoning system is to minimise its own reasoning.**
CORAL progressively amortises expensive recurrent computation into fast codebook-based
recognition, getting more efficient with experience while maintaining accuracy on novel
problems.

The theoretical foundation is variational free energy minimisation (Karl Friston's free
energy principle). The practical foundation is the HRM/TRM lineage of recursive reasoning
models, extended with crystallisation, running-statistics precision, and multi-headed
semantic codebooks.

**Owner:** Muhammad Anwar Ul Haq (Aktuator AI)

---

## Architecture Version

The current architecture is **v4.2** (March 2026). The authoritative reference is
`CORAL_Architecture_Spec_v4.2.md`. If you see references to v4.1 (learned precision
networks, recognition networks, column heads), those are **deprecated** and should not
be implemented.

---

## Session History

### Session 1 — Config Update + Baseline Mode (2026-03-30)

**What was done:**
1. **`coral/config.py`** — Added all v4.2 fields to `ModelConfig`:
   - `codebook_heads` (int, 8), `codebook_entries_per_head` (int, 32)
   - `tau_converge` (float, 0.01), `tau_decrystallise` (float, 0.05), `n_stable` (int, 2)
   - `lambda_dis` (float, 0.01), `precision_momentum` (float, 0.99)
   - `use_local_attention_bias` (bool, True)
   - `mode` (str, "baseline") — controls forward-pass routing in CoralCore
   - `lambda_commit` default updated to 0.25 (crystallisation)
   - `lambda_pi` marked as deprecated in v4.2 comment (kept for backward compat)

2. **`coral/model/coral_core.py`** — Added three-mode forward-pass routing:
   - `mode="baseline"`: no PC, level-0 only, T inner steps per segment
   - `mode="pc_only"`: current PC-with-precision behaviour (unchanged)
   - `mode="full"`: raises `NotImplementedError` (crystallisation wired in Session 5)
   - Backward-compat rule: if `mode="baseline"` but `use_predictive_coding=True`,
     `effective_mode` is promoted to `"pc_only"` so old configs/tests keep working.

3. **`configs/phase1_baseline_no_pc.yaml`** — New Phase 1 config:
   - `model.mode: "baseline"`, `n_levels: 1`, `use_predictive_coding: false`
   - `wandb.tags: ["v4.2", "phase1", "baseline", "no-pc"]`

4. **`tests/test_baseline_mode.py`** — Three new tests (all pass):
   - Output shape `[B, 81, 512]`
   - Forward + backward completes without error
   - Zero PC parameters in baseline mode (`len(core.pc_modules) == 0`)

**Test status:** 36/38 pass (2 pre-existing failures in `test_predictive_coding.py` —
`PrecisionNetwork(dim=...)` API mismatch, not caused by Session 1 changes).
All 35 previously-passing tests continue to pass.

### Session 2 — Backbone Local Attention Bias (2026-03-30)

**What was done:**
1. **`coral/config.py`** — Changed `use_local_attention_bias` default from `True` → `False`
   for backward compat with existing tests (configs that enable the feature set it explicitly).

2. **`coral/model/backbone.py`** — Added local attention bias support to `CoralBackbone`:
   - 3 learnable scalars `row_bias`, `col_bias`, `box_bias` (`nn.Parameter`, init=0.0)
     registered **only** when `use_local_attention_bias=True`.
   - `forward(x, attention_bias=None)` accepts an optional pre-combined `[L, L]` float
     additive bias that is passed to every transformer layer's SDPA call.
   - `RotaryAttention.forward` casts `mask` to query dtype before SDPA (bfloat16 compat).

3. **`coral/adapters/grid.py`** — Added `build_attention_masks(device=None)`:
   - Returns 3 binary `[L, L]` float tensors: `same_row`, `same_col`, `same_box`.
   - Box size is `(grid_height // 3) × (grid_width // 3)` (3×3 for Sudoku).
   - Masks are static (no learnable params); backbone applies its scalars externally.

4. **`coral/model/coral_core.py`** — Wired attention masks through the forward pass:
   - `forward(..., attention_masks=None)` accepts the 3-tuple from the adapter.
   - Combined bias computed once per forward call (not per segment):
     `attn_bias = backbone.row_bias * row + backbone.col_bias * col + backbone.box_bias * box`
   - `_run_level(..., attention_bias=None)` passes bias to every `backbone(...)` call.
   - Skipped entirely when `attention_masks=None` or `use_local_attention_bias=False`.

5. **`tests/test_local_attention_bias.py`** — 11 new tests (all pass):
   - Mask shape, symmetry, binary values, diagonal-ones
   - Same-row / same-col / same-box structural correctness for 9×9 Sudoku
   - Backbone: bias params exist when enabled; init=0.0
   - Backbone: output identical with `attention_bias=None` vs no arg (backward compat)
   - Backbone: output changes with non-zero bias; bias params get gradients
   - CoralCore: integration test with masks; no-masks path unchanged

**Test status:** 49/51 pass (2 pre-existing `PrecisionNetwork` failures unchanged).
All 35 previously-passing tests continue to pass.

### Session 3 — Running-Statistics Precision (2026-03-30)

**What was done:**
1. **`coral/model/predictive_coding.py`** — Replaced learned precision with `RunningPrecision`:
   - `PrecisionNetwork` class **removed** entirely.
   - `RunningPrecision(dim, momentum=0.99, eps=0.01)` added:
     - `register_buffer('ema_var', ones(dim))` — zero learnable parameters.
     - `@torch.no_grad() update(eps)` — EMA update of per-dim variance.
     - `precision` property: `1/(ema_var + eps)` — not in grad graph.
     - `per_head_precision(n_heads=8)` — mean-per-head [H] tensor.
   - `PredictiveCodingModule` updated:
     - `precision_net` replaced by `running_precision: RunningPrecision`.
     - `forward()` calls `running_precision.update(eps)` then reads `precision` as constant.
     - Returns `pi: [dim_lower]` (was `[B, L, dim_lower]`) — broadcasts in loss.
     - `momentum` argument added (default 0.99).
   - `precision_regulariser` function **removed**.
   - `precision_weighted_prediction_loss` updated: now calls `pi.detach()` explicitly.

2. **`coral/training/losses.py`** — Removed precision regulariser:
   - Import of `precision_regulariser` removed.
   - `L_pi` term removed from `CoralLoss.forward`; `breakdown["loss/precision_reg"]` removed.
   - Total loss formula simplified.

3. **`coral/model/coral_core.py`** — Passes `precision_momentum` from config to
   `PredictiveCodingModule` (via `getattr(config, 'precision_momentum', 0.99)`).

4. **`tests/test_predictive_coding.py`** — Fully rewritten for v4.2:
   - Tests for `PrecisionNetwork` replaced by 6 `RunningPrecision` tests.
   - Tests for `precision_regulariser` replaced by precision explosion / NaN tests.
   - 2 previously-failing tests now pass.

5. **`tests/test_coral_core.py`** + **`tests/test_forward_backward.py`** — Updated to
   remove `precision_net` references (replaced with comments noting v4.2 removal).

**Test status:** 56/56 pass — all previously-failing `PrecisionNetwork` tests now pass.

Key v4.2 changes from v4.1:
- Learned precision network → running-statistics precision (EMA, no parameters)
- Learned recognition network → convergence-driven crystallisation (velocity monitoring, no parameters)
- Monolithic codebook → multi-headed semantic codebook (H=8, 32 entries per head)
- Column heads / precision-driven sparsity → dropped entirely
- Research sequence reordered: crystallisation-first, not predictive-coding-first

---

## Repository Layout

```
coral-v4/
├── CLAUDE.md                          # THIS FILE — read every session
├── CORAL_Architecture_Spec_v4.2.md    # Authoritative architecture reference
├── CORAL_v4.2_Build_Plan.md           # Build sequence (8 sessions)
├── CORAL_v4_Handoff_Note.md           # Historical context from v4.0/v4.1 work
│
├── coral/                             # Main package
│   ├── config.py                      # Hydra config dataclasses (ALL fields with defaults)
│   ├── model/
│   │   ├── backbone.py                # 2-layer transformer (RoPE, RMSNorm, SwiGLU, SDPA)
│   │   ├── level_module.py            # Up/down projections, level embeddings
│   │   ├── predictive_coding.py       # Prediction networks + RunningPrecision (NOT learned)
│   │   ├── crystallisation.py         # MultiHeadedCodebook + ConvergenceMonitor + CrystallisationManager
│   │   ├── halting.py                 # Q-learning adaptive halting
│   │   └── coral_core.py              # Main assembly: 3 modes (baseline, pc_only, full)
│   ├── adapters/
│   │   └── grid.py                    # Sudoku encoder/decoder + attention bias masks
│   ├── training/
│   │   ├── losses.py                  # Stablemax CE + prediction loss + commitment + disentanglement
│   │   └── trainer.py                 # Deep supervision loop + W&B logging + repr diagnostics
│   ├── data/
│   │   ├── build_sudoku_dataset.py    # Dataset generation (copied from HRM repo)
│   │   └── common.py                  # Shared data utilities
│   └── evaluation/
│       ├── evaluator.py               # Exact accuracy, token accuracy
│       └── pareto.py                  # Accuracy@K Pareto curves
│
├── configs/                           # Hydra YAML configs (one per experiment phase)
│   ├── phase1_baseline_no_pc.yaml
│   ├── phase3a_crystal_simple.yaml
│   ├── phase3b_crystal_multihead.yaml
│   ├── phase3c_crystal_decrystal.yaml
│   ├── phase4_amort_with_crystal.yaml
│   ├── phase4_amort_no_crystal.yaml
│   └── exp1_baseline.yaml             # Legacy v4.1 config (kept for reference)
│
├── scripts/
│   ├── train.py                       # Main Hydra entry point
│   ├── collect_states.py              # Phase 2: collect representations for analysis
│   ├── codebook_analysis.py           # Phase 2: clustering analysis + bypass accuracy
│   ├── diagnostic_precision.py        # Legacy: precision diagnostic (1000 steps)
│   └── diagnostic_no_pc.py            # Legacy: no-PC baseline diagnostic
│
├── tests/                             # pytest test suite
│   ├── test_backbone.py
│   ├── test_predictive_coding.py
│   ├── test_crystallisation.py
│   ├── test_coral_core.py
│   ├── test_forward_backward.py
│   └── test_losses.py
│
└── data/                              # Data files (gitignored)
    └── sudoku_extreme_1k/
        ├── train/all__inputs.npy, all__labels.npy
        └── test/all__inputs.npy, all__labels.npy
```

---

## Committed Architectural Decisions (DO NOT CHANGE)

These have been validated empirically or are load-bearing design choices. Do not modify
without explicit instruction.

- **Shared 2-layer backbone** with additive level embeddings (not separate per-level modules)
- **Self-attention** via PyTorch SDPA (not flash-attn — it fails to compile on Vast.ai)
- **T=3 timescale multiplier** (40 inner steps per segment at N=4)
- **Full backprop through inner loop** within segments, **detach between segments**
- **bfloat16 forward pass, float64 loss computation** (stablemax cross-entropy)
- **AdamW optimizer** (lr=7e-5, weight_decay=1.0, betas=(0.9, 0.95), cosine schedule)
- **Post-backbone residual conditioning**: `z_new = backbone(z); z = z_new + gate * (conditioning - z)`
  Predictions/errors are residual corrections applied AFTER the backbone, not added to backbone input.
- **Precision is NOT learned** — it is a running EMA statistic with zero learnable parameters
- **Crystallisation is NOT gated by a learned network** — it is triggered by convergence velocity
- **Codebooks are buffers, not parameters** — they update via EMA, not gradient descent

---

## Known Gotchas (READ BEFORE DEBUGGING)

### Data Pipeline
- **vocab_size = 11**, not 10. Pad=0, digits 1–10. Setting vocab_size=10 causes CUDA index-out-of-bounds.
- Eval data lives in `data/sudoku_extreme_1k/test/`, not `eval/`. File prefix is `all__`, not `train__`.
- Dataset generation: `python coral/data/build_sudoku_dataset.py --output-dir data/sudoku_extreme_1k --subsample-size 1000 --num-aug 1000`

### Training Infrastructure
- `pip install -e .` fails on Vast.ai containers. Use `export PYTHONPATH=/workspace/CORAL-v4:$PYTHONPATH` instead.
- `flash-attn` package fails to compile on Vast.ai. Use PyTorch SDPA with flash backend.
- `torch.compile` with `dynamic=True` causes slowdown from recompilation. Use `dynamic=False` for Sudoku (fixed L=81).
- `torch.compile` hangs when tracing crystallisation control flow with Python conditionals and `.item()` calls. Decorate crystallisation methods with `@torch.compiler.disable(recursive=False)` if using compile.
- "Casting complex values to real" warning from RoPE is harmless — ignore it.

### Precision (Historical — v4.1)
- The learned PrecisionNetwork was REMOVED in v4.2. Do not re-add it.
- If you see `PrecisionNetwork` references anywhere, they are dead code — remove them.
- The precision regulariser `(lambda_pi/2)(log pi)^2` was REMOVED. Do not re-add it.
- `lambda_pi` still exists in config for backward compatibility but is unused.

### Numerical Stability
- Cross-entropy MUST use float64 (stablemax). bfloat16 logits → cast to float64 before loss.
- The prediction loss was changed from `.sum(dim=-1)` to `.mean(dim=-1)` to prevent explosion.
- `lambda_pred` was reduced from 0.1 to 0.001. The original value caused prediction loss to dominate.

### Codebooks
- Codebook entries are **registered buffers**, not nn.Parameters. They update via EMA, not optimizer steps.
- Codebook collapse is the default failure mode. Monitor per-head perplexity. If perplexity < 4, increase dead-code restart frequency.
- Dead-code restart runs every 1000 steps by default. Adjust if codebook health is poor.

---

## Testing Requirements

- **Always run tests before pushing:** `pytest tests/ -v`
- **All tests must pass.** Zero failures is the bar.
- Every new module or significant modification must have accompanying tests.
- Test files live in `tests/` and follow the naming convention `test_<module>.py`.
- Tests should run on CPU (no CUDA requirement for unit tests).

---

## How to Run

### Local Tests
```bash
export PYTHONPATH=/path/to/CORAL-v4:$PYTHONPATH
pytest tests/ -v
```

### Training on Vast.ai
```bash
tmux new -s coral
cd /workspace/CORAL-v4
git pull origin main
export PYTHONPATH=/workspace/CORAL-v4:$PYTHONPATH
pip install wandb hydra-core omegaconf tqdm einops --break-system-packages
wandb login

# Phase 1: Baseline
python scripts/train.py --config-name phase1_baseline_no_pc \
    wandb.disabled=false wandb.run_name=v4.2-phase1-baseline-no-pc

# Phase 3b: Multi-headed crystallisation
python scripts/train.py --config-name phase3b_crystal_multihead \
    wandb.disabled=false wandb.run_name=v4.2-phase3b-crystal-multihead
```

### Phase 2 Analysis (after Phase 1 training completes)
```bash
python scripts/collect_states.py \
    --checkpoint checkpoints/phase1_best.pt \
    --data-dir data/sudoku_extreme_1k \
    --output states_phase1.npz

python scripts/codebook_analysis.py \
    --states states_phase1.npz \
    --output-dir analysis/phase2/
```

---

## W&B Configuration

- **Workspace:** `aktuator-ai`
- **Project:** `Sudoku-extreme-1k-aug-1000 CORAL-v4`
- **Run naming:** `v4.2-phase{N}-{descriptor}` (e.g., `v4.2-phase1-baseline-no-pc`)
- **Tags:** Every run tagged with version (`v4.2`), phase (`phase1`), and active components

---

## Code Style

- Type hints on all function signatures
- Docstrings on all classes and public methods (Google style)
- Imports: stdlib first, then third-party (torch, etc.), then coral package
- No wildcard imports
- f-strings for string formatting
- `@torch.no_grad()` on all diagnostic/evaluation code paths
- Codebook and convergence monitor operations must be `@torch.no_grad()` where appropriate
- Constants in UPPER_SNAKE_CASE, classes in PascalCase, functions/methods in snake_case

---

## Key Reference Documents

| Document | Purpose |
|----------|---------|
| `CORAL_Architecture_Spec_v4.2.md` | Authoritative architecture reference — what to build |
| `CORAL_v4.2_Build_Plan.md` | Build sequence — how to build it (8 sessions) |
| `CORAL_v4_Handoff_Note.md` | Historical context — what was tried, what failed, current state |
| `CORAL_v4_Research_Plan_Crystallisation_First.md` | Research phases — what experiments to run |
