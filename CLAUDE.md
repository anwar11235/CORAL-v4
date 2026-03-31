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

### Session 4 — Multi-Headed Semantic Codebooks + Crystallisation (2026-03-30)

**What was done:**
1. **`coral/model/crystallisation.py`** — New file implementing three components:
   - `MultiHeadedCodebook(dim, n_heads, entries_per_head, ema_decay)`:
     - Codebook stored as `register_buffer` — zero learnable parameters, EMA updates only.
     - Shape `[H, M, d_head]` where `d_head = dim / H`.
     - `quantise(z, hard=True)` — hard nearest-neighbour with straight-through gradient.
     - `update_ema(z, indices)` — scatter_add EMA update (no grad).
     - `dead_code_restart(z_buffer, threshold)` — replaces stale entries from rolling buffer.
     - `commitment_loss(z, z_q)` — `||z_h - sg(e_h)||²` per head, averaged.
     - `disentanglement_loss()` — `Σ ||C_h1^T @ C_h2||²_F / M²` over h1<h2 pairs, ×2.
     - `initialise_from_kmeans(states, n_iter)` — offline k-means per head.
     - `get_perplexity()` — `[H]` effective codebook usage (exp of entropy).
   - `ConvergenceMonitor` (plain Python, no nn.Module):
     - Tracks `z_prev`, `consecutive_converged [B,L,H]`, `crystallised [B,L,H]`, `frozen_values [B,L,H,d_head]`.
     - `update_and_crystallise(z, codebook)` — computes per-head L2 velocity; triggers after `n_stable` steps.
     - `check_decrystallisation(z_proposed)` — unfreezes heads where drift > `tau_decrystallise`.
     - `enforce(z)` — overwrites crystallised heads with frozen codebook values.
   - `CrystallisationManager(config)` (nn.Module, not yet wired into CoralCore):
     - Owns one `MultiHeadedCodebook` and one `ConvergenceMonitor`.
     - `step(z, z_prev, segment_idx)` → `(z_crystallised, mask, stats)`.
     - `enforce_after_backbone(z)` — enforce without updating velocity counters.
     - `get_losses()` → `(commitment_loss, disentanglement_loss)`.
     - `get_stats()` → monitoring metrics for W&B.

2. **`tests/test_crystallisation.py`** — 18 tests (all pass):
   - Codebook is a buffer, not a parameter; zero learnable parameters; buffer shape [8,32,64]
   - quantise: output shapes; output differs from input; straight-through gradient
   - EMA update moves codebook entries
   - Dead-code restart replaces unused entries
   - commitment_loss = 0 when z equals codebook entries
   - disentanglement_loss = 0 for orthogonal heads (standard basis construction)
   - disentanglement_loss > 0 for correlated heads
   - ConvergenceMonitor crystallises after n_stable steps; does not crystallise on high velocity
   - De-crystallisation fires on large drift
   - enforce() restores crystallised heads while preserving active heads
   - Partial crystallisation (some heads crystallised, others not)
   - CrystallisationManager end-to-end; get_losses() returns scalars

**Test status:** 74/74 pass. `CrystallisationManager` is NOT yet integrated into `CoralCore`
(that is Session 5). The module is fully tested in isolation.

### Session 5 — Integrate Crystallisation into Core (2026-03-30)

**What was done:**
1. **`coral/model/coral_core.py`** — Wired `CrystallisationManager` into mode="full":
   - Import added; `crystallisation_manager` created in `__init__` when `config.use_crystallisation`.
   - `CoralOutput` extended with `crystal_stats`, `commit_losses`, `dis_losses` fields.
   - `use_crys` flag computed at start of forward (effective_mode=="full" AND manager exists).
   - `monitor.reset(B, L, device)` called once per forward pass before the segment loop.
   - Segment loop: START of segment: `crystal_manager.step()` snaps newly converged heads.
     END of segment: `monitor.check_decrystallisation()` unfreezes drifted heads.
     `get_losses()` collects commit/dis per segment.
   - Decode: training → `decode_fn(z_states[0])` (raw backbone, full gradients);
     eval → `enforce_after_backbone(z_states[0])` then decode (crystallised → codebook value).
   - NotImplementedError for mode="full" **removed**.

2. **`coral/training/losses.py`** — Added `commitment_loss` and `disentanglement_loss` args:
   - Both optional; ignored when `config.use_crystallisation=False`.
   - `L_commit = lambda_commit * commitment_loss`, `L_crystal = lambda_dis * disentanglement_loss`.
   - Both added to total loss and returned in breakdown dict.

3. **`coral/training/trainer.py`** — Crystallisation losses + eval diagnostics:
   - `train_step`: passes `commit_losses[i]` and `dis_losses[i]` to `loss_fn` per segment.
   - `eval_step`: logs `crystal/rate_total`, `crystal/rate_head_*`, `codebook/perplexity_head_*`,
     and `crystal/bypass_accuracy` when `use_crystallisation=True`.

4. **`tests/test_full_mode.py`** — 8 new tests (all pass):
   - Forward pass completes in mode="full"; output shapes unchanged vs mode="pc_only"
   - `crystal_stats` populated when use_crystallisation=True; empty when False
   - Backward: backbone params have non-zero gradients despite crystallisation enforce
   - Commit/dis losses in breakdown; no NaN/Inf gradients; manager gated by config

**Test status:** 82/82 pass.

**CRITICAL gradient invariant (Session 5):** In training, logits are decoded from raw
backbone output BEFORE enforce. `crystal_manager.step()` at the START of the NEXT segment
enforces on the already-detached state, so backbone gradients are never cut during loss computation.

### Session 6 — Representation Diagnostics + State Collection (2026-03-30)

**What was done:**
1. **`coral/training/trainer.py`** — Added `compute_repr_diagnostics(eval_loader, ...)`:
   - Temporarily sets `halting_threshold=2.0` to disable early halting during collection.
   - Collects `z_states[0]` (final segment) across eval batches.
   - Filters to empty cells (token=0) for all metrics.
   - Computes and returns a dict with keys:
     - `repr/inter_position_similarity` — mean pairwise cosine sim (random 500-state sample)
     - `repr/same_digit_similarity` — within-digit cosine sim averaged over digits 1–9
     - `repr/effective_rank` — SVD-based component count explaining 90% variance
     - `repr/state_norm_mean` / `repr/state_norm_std` — L2 norm statistics

2. **`scripts/collect_states.py`** — Phase 2 state collection:
   - CLI: `--checkpoint`, `--data-dir`, `--output`, `--segments` (e.g. "4,8,12,16"), `--device`
   - Loads checkpoint with structured format (`config`, `adapter_state`, `core_state`).
   - Runs one forward pass per target segment (K_max=s, halting disabled) over full eval set.
   - Saves `.npz` with `states [N,81,len(segs),d]`, `labels [N,81]`, `given_mask [N,81]`.

3. **`scripts/codebook_analysis.py`** — Phase 2 representation analysis:
   - CLI: `--states`, `--output-dir`, `--segment-idx`, `--n-heads`, `--tsne-sample`
   - Per-head k-means (k∈{16,32,64,128}): inertia, cluster purity, perplexity per head
   - Whole-vector k-means (k∈{64,128,256,512}): same metrics + bypass accuracy
   - Bypass accuracy = fraction of states whose cluster's majority-vote digit matches true digit
   - t-SNE scatter plot (5000-state sample) coloured by digit, saved as PNG
   - Bypass accuracy vs k curve, saved as PNG
   - Requires `scikit-learn` and `matplotlib`

4. **`tests/test_diagnostics.py`** — 7 tests (all pass):
   - `compute_repr_diagnostics` returns non-NaN, finite values; correct keys; empty dict for empty loader
   - collect_states logic produces .npz with correct shapes
   - `per_head_analysis` and `whole_vector_analysis` return values in valid ranges
   - End-to-end codebook analysis on synthetic .npz

**Test status:** 89/89 pass.

### Session 7 — Updated Configs + Training Scripts (2026-03-30)

**What was done:**
1. **`coral/config.py`** — Added 3 new `TrainingConfig` fields:
   - `codebook_init_from: Optional[str] = None` — path to `.npz` from `collect_states.py` for k-means codebook init
   - `amort_anneal_start: int = 0` — step at which `lambda_amort` ramp begins
   - `amort_anneal_end: int = 0` — step at which `lambda_amort` reaches full value (0 = no annealing)

2. **`coral/training/annealing.py`** — New module:
   - `get_effective_lambda_amort(step, base, anneal_start, anneal_end) -> float`
   - Linearly ramps from 0 → `base` over `[anneal_start, anneal_end]`; immediate when `anneal_end=0`.

3. **Config YAML files** — 5 new phase configs created:
   - `configs/phase3a_crystal_simple.yaml` — mode="full", n_levels=1, codebook_heads=1, entries=256, lambda_dis=0.0
   - `configs/phase3b_crystal_multihead.yaml` — same + codebook_heads=8, entries=32, lambda_dis=0.01
   - `configs/phase3c_crystal_decrystal.yaml` — same as 3b with tau_converge=0.005, tau_decrystallise=0.1, n_stable=3
   - `configs/phase4_amort_with_crystal.yaml` — same as 3b + use_amort=True, lambda_amort=0.01, anneal 2000→8000
   - `configs/phase4_amort_no_crystal.yaml` — mode="baseline", use_crystallisation=False + use_amort=True, annealing

4. **`scripts/train.py`** — Updated with:
   - Import `get_effective_lambda_amort` from `coral.training.annealing`
   - Codebook init from k-means: after `TrainerV4` build, loads `.npz` if `training.codebook_init_from` is set,
     extracts last-segment states, calls `core.crystallisation_manager.codebook.initialise_from_kmeans(states)`.
   - Amortisation annealing in training loop: calls `get_effective_lambda_amort` each step,
     sets `trainer.loss_fn.config.lambda_amort = eff_lambda` dynamically.
   - Logs `train/lambda_amort` when `use_amort=True`.

5. **`tests/test_configs.py`** — 22 tests (all pass):
   - All 5 YAML files load without error into `CoralConfig`
   - New `TrainingConfig` fields present with correct types
   - Per-config assertions: codebook_heads, lambda_dis, mode, use_crystallisation, annealing params
   - mode="full" forward: correct output shape, crystal_stats populated, backward completes
   - Annealing: boundary conditions (before start, midpoint, at end, disabled, zero base)

**Test status:** 111/111 pass.

### Session 8 — Full Test Suite Update (2026-03-30)

**What was done:**
1. **`tests/test_backbone.py`** — Added 3 attention bias tests:
   - `test_backbone_attention_bias_changes_output` — non-uniform bias changes outputs
   - `test_backbone_none_bias_identical_to_no_bias` — `attention_bias=None` backward compatible
   - `test_grid_attention_masks_structure_9x9` — 9×9 mask shape/symmetry/binary/diagonal checks

2. **`tests/test_crystallisation.py`** — Added 2 missing tests:
   - `test_initialise_from_kmeans_changes_codebook` — k-means init replaces random entries
   - `test_get_perplexity_shape_and_range` — perplexity is [H] tensor in [1, entries_per_head]

3. **`tests/test_coral_core.py`** — Added 6 tests covering all three modes:
   - `test_all_modes_output_shape[baseline/pc_only/full]` — correct z_states[0] shape
   - `test_all_modes_backward[baseline/pc_only/full]` — backward completes, backbone grads present
   - `test_full_mode_crystal_stats_in_coral_core` — crystal_stats length == num_segments
   - `test_baseline_mode_no_pc_modules` — baseline has zero PC modules
   - `test_full_mode_no_crystallisation_manager_when_disabled` — manager is None for pc_only

4. **`tests/test_forward_backward.py`** — Added 3 per-mode gradient tests:
   - `test_baseline_mode_grads_reach_backbone` — backbone grads non-zero in baseline
   - `test_pc_only_mode_grads_reach_backbone_and_prediction_net` — grads in backbone + pred net
   - `test_full_mode_commit_loss_affects_encoder` — backbone + adapter grads in full mode

5. **`tests/test_losses.py`** — New file, 12 tests:
   - stablemax CE: finite, non-negative, ignored positions = 0
   - commitment_loss and disentanglement_loss non-negative
   - Baseline mode: PC, crystallisation, amort losses all = 0; task loss > 0
   - `loss/precision_reg` NOT in breakdown (confirms v4.2 removal)
   - Full mode: commit + dis increase total; total = sum of parts
   - amortisation_loss: non-negative, empty → 0
   - End-to-end forward + loss finite in both baseline and full mode

**Test status:** 140/140 pass.

**Parameter counts (N=1 single-level, 9×9 Sudoku):**
- Baseline mode:           8,492,945 total params (adapter + core)
- Full mode (+ crystal):   8,492,945 total params (codebook = 16,640 BUFFERS, not params)
- Codebook learnable params: 0 (confirmed buffers only)

**Smoke tests (programmatic, synthetic data):**
- phase1_baseline_no_pc: 5 training steps complete, loss = ~2.9 ✓
- phase3b_crystal_multihead: 5 training steps complete, loss = ~3.1 ✓
  (crystal stats keys confirmed in metrics)

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
│   │   ├── annealing.py               # get_effective_lambda_amort (linear ramp schedule)
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
├── tests/                             # pytest test suite (140 tests, all pass)
│   ├── test_backbone.py               # backbone shape, gradient, attention bias
│   ├── test_predictive_coding.py      # RunningPrecision, PC module, loss
│   ├── test_crystallisation.py        # codebook, convergence, decrystallisation, manager
│   ├── test_coral_core.py             # all 3 modes, shapes, backward
│   ├── test_forward_backward.py       # per-mode gradient flow
│   ├── test_losses.py                 # stablemax CE, commit, dis, baseline/full mode
│   ├── test_full_mode.py              # session 5 full-mode integration tests
│   ├── test_local_attention_bias.py   # session 2 attention bias tests
│   ├── test_baseline_mode.py          # session 1 baseline mode tests
│   ├── test_adapters.py               # grid adapter encode/decode
│   ├── test_configs.py                # YAML config loading + annealing
│   └── test_diagnostics.py            # repr diagnostics + codebook analysis
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
`documents at 'CORAL-v4\docs\`