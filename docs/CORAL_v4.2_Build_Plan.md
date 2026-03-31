# CORAL v4.2 — Build Plan for Claude Code

## Modifying the Existing CORAL-v4 Repository

**Date:** March 30, 2026

**Repository:** `github.com/anwar11235/CORAL-v4` (existing repo, NOT a new one)

**Reference Document:** `CORAL_Architecture_Spec_v4.2.md` (March 2026)

---

## Current State of the Codebase

The v4 repo has a complete Phase 1 implementation (35/35 tests passing) with:

| File | Status | v4.2 Action |
|------|--------|-------------|
| `coral/config.py` | Complete | **MODIFY** — add v4.2 fields, remove obsolete fields |
| `coral/model/backbone.py` | Complete | **MODIFY** — add local attention bias (3 params) |
| `coral/model/level_module.py` | Complete | **KEEP** — unchanged |
| `coral/model/predictive_coding.py` | Complete | **MODIFY** — remove PrecisionNetwork, add RunningPrecision |
| `coral/model/halting.py` | Complete | **KEEP** — unchanged |
| `coral/model/coral_core.py` | Complete | **MODIFY** — add baseline mode, crystallisation integration |
| `coral/model/crystallisation.py` | Does not exist | **CREATE** — convergence monitor, codebooks, partial crystallisation |
| `coral/adapters/grid.py` | Complete | **KEEP** — unchanged |
| `coral/training/losses.py` | Complete | **MODIFY** — remove precision reg, add commitment + disentanglement |
| `coral/training/trainer.py` | Complete | **MODIFY** — add representation diagnostics logging |
| `coral/evaluation/pareto.py` | Complete | **KEEP** — unchanged |
| `coral/evaluation/evaluator.py` | Complete | **KEEP** — unchanged |
| `scripts/train.py` | Complete | **MODIFY** — support new config fields |
| `scripts/collect_states.py` | Does not exist | **CREATE** — Phase 2 state collection |
| `scripts/codebook_analysis.py` | Does not exist | **CREATE** — Phase 2 clustering analysis |
| `configs/exp1_baseline.yaml` | Complete | **KEEP** — this is the v4.1 config |
| `configs/phase1_baseline_no_pc.yaml` | Does not exist | **CREATE** — Phase 1 no-PC config |
| `configs/phase3_crystal.yaml` | Does not exist | **CREATE** — Phase 3 crystallisation config |
| Tests | 35/35 passing | **UPDATE** — add tests for new components, update existing |

---

## Build Sessions

The build is divided into 8 discrete sessions. Each session produces a testable,
committable unit of work. Sessions are ordered by dependency — later sessions depend
on earlier ones.

---

### Session 1: Config Update + Baseline Mode

**Goal:** Update the config schema for v4.2 and add a `mode='baseline'` path to
coral_core that bypasses all predictive coding. This enables Phase 1 (baseline
without PC) immediately.

**Files to modify:**
- `coral/config.py` — add new v4.2 fields, deprecate obsolete ones
- `coral/model/coral_core.py` — add baseline mode (skip PC when disabled)
- `configs/phase1_baseline_no_pc.yaml` — new config for Phase 1

**Files to create:**
- None

**Tests to update:**
- Existing tests should still pass with default config
- Add test: coral_core in baseline mode produces correct shapes
- Add test: baseline mode has no PC parameters in computation graph

**Verification:**
- All existing 35 tests pass
- New baseline mode test passes
- `python scripts/train.py --config-name phase1_baseline_no_pc` runs without error
  for 10 steps on CPU

---

### Session 2: Backbone Local Attention Bias

**Goal:** Add 3 learned scalar biases to the backbone's attention computation that
upweight same-row, same-column, and same-box position pairs in Sudoku.

**Files to modify:**
- `coral/model/backbone.py` — add local_attention_bias parameter and application
- `coral/config.py` — add `use_local_attention_bias: bool = True`

**Design:**
- The backbone receives an optional `attention_bias` tensor of shape `[L, L]`
- The adapter (grid.py) is responsible for constructing this tensor based on the
  grid structure (which positions share row, column, box)
- Three learned scalars (row_bias, col_bias, box_bias) are added to the attention
  logits before softmax: `attn_logits += row_bias * row_mask + col_bias * col_mask + box_bias * box_mask`
- The masks are precomputed by the adapter and passed through coral_core to the backbone

**Files to modify additionally:**
- `coral/adapters/grid.py` — add method to generate Sudoku attention bias masks
- `coral/model/coral_core.py` — pass attention_bias through to backbone

**Tests:**
- Test: backbone output changes when local_attention_bias is enabled vs disabled
- Test: attention bias masks have correct shape and structure for 9×9 grid
- Existing backbone tests still pass

---

### Session 3: Running-Statistics Precision

**Goal:** Replace the learned PrecisionNetwork with RunningPrecision. This is the
critical change that resolves the precision collapse problem.

**Files to modify:**
- `coral/model/predictive_coding.py` — remove PrecisionNetwork class, add
  RunningPrecision class, update PredictiveCodingModule to use running precision
- `coral/training/losses.py` — remove precision regulariser (L_pi), update
  prediction loss to use running precision as a constant
- `coral/config.py` — remove `lambda_pi`, add `precision_momentum: float = 0.99`

**Design:**
```python
class RunningPrecision:
    def __init__(self, dim, momentum=0.99, eps=0.01):
        self.register_buffer('ema_var', torch.ones(dim))
        self.momentum = momentum
        self.eps = eps

    @torch.no_grad()
    def update(self, prediction_error):
        batch_var = prediction_error.var(dim=(0, 1))
        self.ema_var = self.momentum * self.ema_var + (1 - self.momentum) * batch_var

    @property
    def precision(self):
        return 1.0 / (self.ema_var + self.eps)
```

Key: RunningPrecision should be an nn.Module with `register_buffer` so the EMA state
is saved/loaded with checkpoints and moves to the correct device automatically.
But it has NO learnable parameters.

**Tests:**
- Test: RunningPrecision starts at uniform precision (all 1/(1+eps))
- Test: After updating with non-uniform error variance, precision differentiates
- Test: RunningPrecision has zero learnable parameters
- Test: precision is detached from computation graph (no gradient flows through it)
- Test: PredictiveCodingModule works with RunningPrecision (shapes, forward pass)
- All existing PC tests still pass (with PrecisionNetwork replaced)

---

### Session 4: Multi-Headed Semantic Codebooks

**Goal:** Create the multi-headed codebook module — the core new component of v4.2.

**Files to create:**
- `coral/model/crystallisation.py` — contains MultiHeadedCodebook,
  ConvergenceMonitor, and CrystallisationManager

**Design:**

```python
class MultiHeadedCodebook(nn.Module):
    """
    H independent codebooks, each operating on d_head dimensions.
    Supports: nearest-neighbour lookup, EMA update, dead-code restart,
    Gumbel-Softmax soft assignment, commitment loss, disentanglement loss.
    """
    def __init__(self, dim, n_heads=8, entries_per_head=32, ema_decay=0.99):
        # dim=512, n_heads=8 → d_head=64
        # codebooks: [H, M, d_head]
        # usage_count: [H, M] for dead-code detection

    def quantise(self, z, temperature=1.0, hard=False):
        """
        z: [B, L, dim]
        Returns: z_quantised [B, L, dim], indices [B, L, H], per_head_distances [B, L, H]
        """

    def update_ema(self, z, indices):
        """EMA update of codebook entries from assigned states."""

    def dead_code_restart(self, z_buffer):
        """Replace unused entries with random samples from buffer."""

    def commitment_loss(self, z, z_quantised):
        """Per-head commitment loss: ||z_h - sg(e_h)||²"""

    def disentanglement_loss(self):
        """Cross-head orthogonality: Σ_{h1≠h2} ||C_h1^T · C_h2||²_F"""

    def initialise_from_kmeans(self, states):
        """Initialise codebooks from k-means on collected states."""


class ConvergenceMonitor:
    """
    Tracks per-position, per-head velocity (rate of state change).
    No learnable parameters.
    """
    def __init__(self, n_heads, tau_converge=0.01, tau_decrystallise=0.05,
                 n_stable=2):
        # Buffers for previous states and consecutive-converged counters

    def update(self, z_current, z_previous):
        """Compute velocity per head, update convergence counters."""

    def get_crystallisation_mask(self):
        """Returns bool tensor [B, L, H] of which heads are crystallised."""

    def check_decrystallisation(self, z_proposed, z_frozen):
        """Check if any crystallised heads should be unfrozen."""


class CrystallisationManager(nn.Module):
    """
    Orchestrates codebook + convergence monitor + crystallisation logic.
    This is the main interface that coral_core.py interacts with.
    """
    def __init__(self, config):
        self.codebook = MultiHeadedCodebook(...)
        self.monitor = ConvergenceMonitor(...)

    def crystallise(self, z, z_prev):
        """
        1. Update convergence monitor
        2. Get crystallisation mask
        3. Snap newly crystallised heads to codebook
        4. Check de-crystallisation for previously crystallised heads
        5. Return updated z with frozen heads, mask, stats
        """

    def enforce_crystallisation(self, z, mask):
        """After backbone pass: overwrite crystallised heads with codebook values."""

    def get_stats(self):
        """Return crystallisation rate, codebook perplexity, etc. for logging."""
```

**Tests:**
- Test: MultiHeadedCodebook forward produces correct shapes
- Test: quantise returns nearest-neighbour indices
- Test: EMA update moves codebook entries toward assigned states
- Test: dead_code_restart replaces entries with zero usage
- Test: commitment_loss is zero when z equals codebook entries
- Test: disentanglement_loss is zero when codebooks are orthogonal
- Test: ConvergenceMonitor detects convergence after N_stable low-velocity segments
- Test: ConvergenceMonitor does NOT crystallise when velocity is above threshold
- Test: de-crystallisation fires when drift exceeds threshold
- Test: CrystallisationManager full pipeline (crystallise → enforce → stats)
- Test: partial crystallisation works (some heads frozen, others active)
- Test: codebook has correct parameter count (~16K for dim=512, H=8, M=32)

---

### Session 5: Integrate Crystallisation into Core

**Goal:** Wire the CrystallisationManager into coral_core.py so that crystallisation
operates during the forward pass. The core should support three modes:
1. `baseline` — no PC, no crystallisation (Phase 1)
2. `pc_only` — PC with running precision, no crystallisation (Phase 5)
3. `full` — PC + crystallisation (Phase 3+)

**Files to modify:**
- `coral/model/coral_core.py` — integrate CrystallisationManager into the segment
  loop. After each backbone pass, call `enforce_crystallisation`. Between segments,
  call `crystallise` to update convergence monitor and trigger new crystallisations.
- `coral/config.py` — add `use_crystallisation: bool = False`, crystallisation
  thresholds, codebook parameters

**Critical design point:** During training, the task loss is ALWAYS computed from
the full-recursion state (all backbone passes, no bypass). Crystallisation affects
which heads get snapped to codebook values (providing EMA signal to codebook) and
provides monitoring stats, but does NOT reduce the computation graph during training.
During eval, crystallised positions can genuinely skip computation.

**Files to modify additionally:**
- `coral/training/losses.py` — add commitment_loss and disentanglement_loss to
  the loss computation
- `coral/training/trainer.py` — log crystallisation stats to W&B

**Tests:**
- Test: coral_core forward pass works in all three modes
- Test: crystallisation does not change output shapes
- Test: commitment loss and disentanglement loss are computed and non-negative
- Test: crystallisation stats (rate, perplexity) are returned and loggable
- Test: in eval mode, crystallised positions have frozen states
- Test: in training mode, all positions get gradient signal
- All existing 35 tests still pass

---

### Session 6: Representation Diagnostics + State Collection

**Goal:** Add the representation diagnostics needed during Phase 1 training, and
create the state collection script needed for Phase 2 analysis.

**Files to modify:**
- `coral/training/trainer.py` — add representation diagnostic logging at eval steps:
  inter-position cosine similarity, same-digit cross-puzzle similarity,
  effective rank (PCA), per-segment state norms

**Files to create:**
- `scripts/collect_states.py` — runs a trained model on the eval set and saves
  per-position, per-segment states to disk as numpy arrays. Also saves metadata
  (ground-truth digit, position index, whether given or empty, segment index).

- `scripts/codebook_analysis.py` — loads collected states, runs clustering analysis:
  k-means at multiple k values (per-head and whole-vector), HDBSCAN, computes
  bypass accuracy, cluster purity, codebook perplexity, and the codebook-to-accuracy
  curve. Outputs a summary report and saves plots.

**Design for collect_states.py:**
```python
# Input: trained checkpoint, eval dataset
# Output: states.npz containing:
#   'states': [N_puzzles, 81, N_segments, 512] — per-position per-segment states
#   'labels': [N_puzzles, 81] — ground-truth digits
#   'given_mask': [N_puzzles, 81] — bool, True if position was given (not empty)
#   'segments_collected': list of segment indices (e.g., [4, 8, 12, 16])
```

**Design for codebook_analysis.py:**
```python
# Input: states.npz from collect_states.py
# Analyses:
#   1. Per-head k-means (split 512-d into 8×64, cluster each)
#   2. Whole-vector k-means (cluster full 512-d)
#   3. Bypass accuracy at each k (replace with centroid, decode, check)
#   4. Cluster purity (fraction of same-digit states per cluster)
#   5. Codebook perplexity (entropy of usage distribution)
#   6. t-SNE/UMAP visualisation coloured by digit, segment, difficulty
# Output: printed report + saved figures
```

**Tests:**
- Test: collect_states produces correct output shapes
- Test: codebook_analysis runs on synthetic data without error

---

### Session 7: Updated Configs + Training Scripts

**Goal:** Create all config files needed for Phases 1–4 of the research plan, and
update the training script to handle the new config fields cleanly.

**Files to create:**
- `configs/phase1_baseline_no_pc.yaml` — N=1, no PC, no crystallisation
- `configs/phase3a_crystal_simple.yaml` — N=1, no PC, monolithic codebook crystallisation
- `configs/phase3b_crystal_multihead.yaml` — N=1, no PC, multi-headed codebook
- `configs/phase3c_crystal_decrystal.yaml` — N=1, no PC, multi-headed + de-crystallisation
- `configs/phase4_amort.yaml` — N=1, no PC, crystallisation + amortisation pressure

**Files to modify:**
- `scripts/train.py` — ensure it handles all new config fields, supports mode
  switching (baseline/pc_only/full), properly initialises codebooks when
  crystallisation is enabled

**Key config differences:**

phase1_baseline_no_pc.yaml:
```yaml
model:
  n_levels: 1
  use_predictive_coding: false
  use_crystallisation: false
  use_local_attention_bias: true
training:
  epochs: 20000
wandb:
  tags: ["v4.2", "phase1", "baseline", "no-pc"]
```

phase3b_crystal_multihead.yaml:
```yaml
model:
  n_levels: 1
  use_predictive_coding: false
  use_crystallisation: true
  codebook_heads: 8
  codebook_entries_per_head: 32
  tau_converge: 0.01
  tau_decrystallise: 0.05
  n_stable: 2
  lambda_commit: 0.25
  lambda_dis: 0.01
training:
  epochs: 20000
  codebook_init_from: null  # or path to k-means centroids
wandb:
  tags: ["v4.2", "phase3b", "crystallisation", "multihead"]
```

**Tests:**
- Test: each config file loads without Hydra errors
- Test: each config creates a valid model that can run a forward pass

---

### Session 8: Full Test Suite Update

**Goal:** Ensure the complete test suite covers all v4.2 components and that all
tests pass. This is the final verification before Phase 1 training.

**Files to create/update:**
- `tests/test_backbone.py` — add test for local attention bias
- `tests/test_predictive_coding.py` — update for RunningPrecision, remove
  PrecisionNetwork tests
- `tests/test_crystallisation.py` — comprehensive tests for MultiHeadedCodebook,
  ConvergenceMonitor, CrystallisationManager
- `tests/test_coral_core.py` — test all three modes (baseline, pc_only, full)
- `tests/test_forward_backward.py` — verify gradient flow in all modes
- `tests/test_losses.py` — test commitment loss, disentanglement loss, verify
  precision regulariser is removed

**Verification criteria:**
- ALL tests pass
- `pytest tests/ -v` completes with 0 failures
- Model parameter count matches expected (~7.2M for core at N=4, ~6.8M at N=1)
- Forward pass produces correct output shapes in all modes
- Backward pass completes without error in all modes
- Crystallisation stats are computed and non-degenerate

---

## Session Dependency Graph

```
Session 1 (Config + Baseline Mode)
    ├── Session 2 (Local Attention Bias)
    ├── Session 3 (Running-Statistics Precision)
    │       └── Session 5 (Integrate Crystallisation into Core)
    └── Session 4 (Multi-Headed Codebooks)
            └── Session 5 (Integrate Crystallisation into Core)
                    ├── Session 6 (Diagnostics + State Collection)
                    ├── Session 7 (Configs + Scripts)
                    └── Session 8 (Full Test Suite)
```

Sessions 2, 3, 4 can be done in parallel after Session 1.
Session 5 requires Sessions 3 and 4.
Sessions 6, 7, 8 require Session 5.

---

## Post-Build: Phase 1 Launch Checklist

After all 8 sessions complete and all tests pass:

```bash
# On Vast.ai A100 instance:
tmux new -s coral
cd /workspace/CORAL-v4
git pull origin main
export PYTHONPATH=/workspace/CORAL-v4:$PYTHONPATH
pip install wandb hydra-core omegaconf tqdm einops --break-system-packages

# Generate dataset (if not already present)
python coral/data/build_sudoku_dataset.py --output-dir data/sudoku_extreme_1k \
    --subsample-size 1000 --num-aug 1000

wandb login

# Run all tests
pytest tests/ -v

# Phase 1: Baseline without PC
python scripts/train.py --config-name phase1_baseline_no_pc \
    wandb.disabled=false \
    wandb.run_name=v4.2-phase1-baseline-no-pc

# Monitor: target ≥70% eval exact accuracy at 20K steps
```
