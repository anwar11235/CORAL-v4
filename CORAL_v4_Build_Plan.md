# CORAL v4 — Build Plan for Claude Code

## Project: COrtical Reasoning via Abstraction Layers (v4)
## Purpose: Implementation plan for an amodal reasoning core

**Repository:** `github.com/anwar11235/CORAL-v4` (new repo, clean start)

**Files copied from v3 repo** (`github.com/anwar11235/CORAL-v3`):
- `coral/data/sudoku_dataset.py` — SudokuAugmenter + dataset loading (~300 lines)
- Evaluation metric functions (exact accuracy, token accuracy — ~50 lines)
- Fused CUDA AdamATan2 kernel source (if compilation succeeds on Day 0)
- Sudoku-Extreme-1K data files (training + eval splits)

Everything else is written from scratch. The v3 repo is archived as a reference but
not used as a base. This avoids inheriting dead code, stale configs, and architecture
assumptions from the v3 design.

**Reference Document:** `CORAL_Architecture_Spec_v4.md` (v4.1, March 2026)

**Compute:** Vast.ai A100-SXM4-40GB (primary), tmux for persistent sessions

**Experiment Tracking:** Weights & Biases (`aktuator-ai` workspace)

**Prior Work:** Phases 0–2 in the v3 repo (`github.com/anwar11235/CORAL-v3`)
established the empirical foundations: precision-weighted predictive coding works
(61.1% on Sudoku-Extreme-1K), columnar routing collapses, and the precision
regulariser sign matters. The v4 repo builds on these findings but does not inherit
the v3 codebase.

---

## Benchmarks

### Development Benchmark: Sudoku-Extreme-1K

**Purpose:** Fast iteration, direct comparison to Phase 0–2 results and literature
(HRM 55%, TRM 87.4%, GRAM 97.0%).

**Dataset:**
- Training: 1,000 Sudoku puzzles with 50–60 cells removed
- Augmentation: 1,000× via SudokuAugmenter (digit relabelling, band/within-band
  permutation, transpose) — already implemented in the v3 codebase
- Evaluation: 1,000 puzzles (Sudoku-Extreme-1K eval split)
- Full test: 423,168 puzzles (Sudoku-Extreme full test set, used for final numbers only)

**Grid:** 9×9 = 81 positions, vocabulary size 10 (digits 0–9, where 0 = empty)

**Why this first:** ~4 hours per training run on A100. Fast enough to iterate on
architecture decisions. Prior baselines available for controlled comparison.

### Validation Benchmark: ARC-AGI-1

**Purpose:** Proves generality. Each task is novel (inductive generalisation from 2–3
examples). Variable grid sizes up to 30×30. 400+ distinct task types.

**Dataset:**
- Training: ~800 tasks from ARC-AGI-1 training set + 160 ConceptARC tasks
- Augmentation: 1,000× (colour permutation, dihedral-group transformations, translations)
  — following TRM protocol
- Evaluation: ARC-AGI-1 public evaluation set (400 tasks)
- Each task: 2–3 input/output demonstration pairs, 1–2 test inputs

**Grid:** Variable, up to 30×30 = 900 positions, vocabulary size 11 (10 colours + background)

**Why this second:** ~3 days on 4 H100s per run (following TRM's reported compute).
Only run after core dynamics are validated on Sudoku. Provides the headline result
for the paper.

### Future Benchmarks (Post-Validation)

- **Maze-Hard:** 30×30 mazes, pathfinding. Tests search/backtracking capability.
- **GSM8K/MATH:** Language-grounded reasoning. Requires language adapter implementation.
- **ACRE/RAVEN:** Visual abstract reasoning. Tests vision adapter.

---

## Repository Structure

```
coral-v4/                              # NEW repo — clean start
├── README.md
├── requirements.txt
├── setup.py
│
├── coral/                             # main package
│   ├── __init__.py
│   ├── config.py                      # Hydra config dataclasses
│   │
│   ├── model/                         # architecture
│   │   ├── __init__.py
│   │   ├── backbone.py                # Shared 2-layer transformer backbone
│   │   ├── level_module.py            # Level modulation (up/down projections, level embeddings)
│   │   ├── predictive_coding.py       # Prediction networks, precision networks, error computation
│   │   ├── crystallisation.py         # Recognition networks, codebooks, consolidation
│   │   ├── column_heads.py            # Precision-driven sparsity (S=8 column heads per level)
│   │   ├── stochastic.py              # Precision-gated stochastic transitions
│   │   ├── halting.py                 # Q-learning adaptive halting
│   │   └── coral_core.py              # Main reasoning core: assembles all components
│   │
│   ├── adapters/                      # amodal interface adapters
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract adapter interface
│   │   ├── grid.py                    # Grid encoder/decoder (Sudoku, ARC, Maze)
│   │   ├── language.py                # LLM interface adapter (future)
│   │   └── vision.py                  # Vision encoder adapter (future)
│   │
│   ├── training/                      # training infrastructure
│   │   ├── __init__.py
│   │   ├── optimizer.py               # Optimizer selection (see Build Step 0)
│   │   ├── losses.py                  # Unified loss function
│   │   └── trainer.py                 # Training loop with deep supervision
│   │
│   ├── data/                          # data pipeline
│   │   ├── __init__.py
│   │   ├── sudoku_dataset.py          # SudokuAugmenter + dataset (copied from v3, verified)
│   │   └── arc_dataset.py             # ARC-AGI dataset + augmentation (Phase 6)
│   │
│   └── evaluation/                    # evaluation infrastructure
│       ├── __init__.py
│       ├── evaluator.py               # Accuracy metrics (exact, token)
│       └── pareto.py                  # Accuracy-depth Pareto curve evaluation
│
├── configs/                           # Hydra configs
│   ├── exp1_baseline.yaml             # Experiment 1: shared backbone + full backprop
│   ├── exp2_amort.yaml                # Experiment 2: + amortisation pressure
│   ├── exp3_crystal.yaml              # Experiment 3: + crystallisation
│   ├── exp4_n3.yaml                   # Experiment 4: N=3 hierarchy
│   ├── exp4_n4.yaml                   # Experiment 4: N=4 hierarchy
│   ├── exp5_stochastic.yaml           # Experiment 5: + stochastic transitions
│   ├── exp6_ablations.yaml            # Experiment 6: ablation matrix
│   └── arc_validation.yaml            # ARC-AGI validation run
│
├── scripts/
│   ├── train.py                       # Main training entry point
│   ├── evaluate.py                    # Standalone evaluation
│   └── evaluate_pareto.py             # Accuracy-depth Pareto evaluation
│
├── data/                              # data files (gitignored, downloaded separately)
│   ├── sudoku_extreme_1k/
│   └── arc_agi_1/
│
└── tests/                             # unit and integration tests
    ├── test_backbone.py
    ├── test_predictive_coding.py
    ├── test_crystallisation.py
    ├── test_coral_core.py
    ├── test_forward_backward.py        # Verify full backprop through inner loop
    └── test_adapters.py
```

---

## Build Phases

The build is structured in 6 phases. Each phase produces a runnable, testable system.
No phase begins until the previous phase passes its verification criteria.

---

### Phase 1: Core Backbone and Predictive Coding (Experiment 1)

**Goal:** Reproduce and exceed Phase 1 results (61.1%) with the new architecture:
shared 2-layer backbone, T=3 timescale, full backprop through inner loop.

**Priority:** This is the most important phase. If the shared backbone + full backprop
doesn't improve over Phase 1, the entire v4 design needs diagnosis.

#### Step 1.1: Backbone Implementation

**File:** `coral/v4/backbone.py`

```python
class CoralBackbone(nn.Module):
    """
    Shared 2-layer transformer backbone.
    Weight-shared across all hierarchy levels and recursion steps.
    Level-specific behaviour induced by additive level embeddings.
    """
```

**Specifications:**
- 2 transformer layers
- d_model = 512, n_heads = 8, d_k = 64
- SwiGLU FFN with 4× expansion (d_ff = 2048, but SwiGLU has gate+up+down
  so actual intermediate = 2048 for gate and 2048 for up)
- RMSNorm (post-norm, no learnable parameters)
- Rotary position encoding (RoPE)
- Input: `[B, L, 512]` (always d=512 after level projection)
- Output: `[B, L, 512]`

**Additional inputs (additive):**
- `level_emb`: learned embedding, shape `[512]`, one per level (N=4 total)
- `timescale_emb`: sinusoidal encoding of the step index within the current level

**Implementation notes:**
- Use `torch.nn.functional.scaled_dot_product_attention` (PyTorch SDPA) for
  flash-attention compatibility. Do NOT use the `flash-attn` package (fails to
  compile on Vast.ai containers — known issue from Phase 1).
- The backbone must support `torch.compile` with `dynamic=False` (fixed sequence
  length for Sudoku, dynamic=True for ARC later).
- All parameters in bfloat16. Loss computation in float64.

**Verification:**
- Unit test: forward pass produces correct output shape
- Unit test: gradient flows through all parameters
- Unit test: level embedding changes output (backbone is not ignoring it)
- Smoke test: overfit on 1 Sudoku puzzle (loss → 0 within 100 steps)

#### Step 1.2: Level Module and Inter-Level Projections

**File:** `coral/v4/level_module.py`

```python
class LevelModule(nn.Module):
    """
    Manages state at one hierarchy level.
    Contains up/down projections to backbone dim (512),
    prediction network, precision network.
    """
```

**For each level l (l=1..N):**
- `W_up[l]`: Linear(d_l, 512, bias=False) — project up to backbone dim
- `W_down[l]`: Linear(512, d_l, bias=False) — project back to level dim
- Level 1 (d=512): projections are identity (no-op)
- Level 2 (d=256): 256→512 up, 512→256 down
- Level 3 (d=128): 128→512 up, 512→128 down
- Level 4 (d=64): 64→512 up, 512→64 down

**Verification:**
- Unit test: round-trip projection preserves information (up then down ≈ identity
  for level 1)
- Unit test: correct dimensions at each level

#### Step 1.3: Predictive Coding

**File:** `coral/v4/predictive_coding.py`

```python
class PredictionNetwork(nn.Module):
    """Top-down prediction: level l+1 predicts level l's state."""
    # 2-layer MLP: d_{l+1} → 2*d_l → d_l (GELU activation)

class PrecisionNetwork(nn.Module):
    """Produces per-dimension precision vector."""
    # 2-layer MLP: d_l → d_l → d_l (GELU activation, then softplus + eps_min)

class PredictiveCodingModule(nn.Module):
    """
    Computes prediction, error, precision-weighted error for one level pair.
    """
    def forward(self, z_l, z_l_plus_1):
        mu_l = self.prediction_net(z_l_plus_1)       # top-down prediction
        eps_l = z_l - mu_l                             # prediction error
        pi_l = self.precision_net(z_l)                 # precision vector
        xi_l = pi_l * eps_l                            # precision-weighted error
        return mu_l, eps_l, pi_l, xi_l
```

**Critical implementation detail — precision regulariser:**
```python
# CORRECT: symmetric log-normal prior centred at pi=1
L_pi = (lambda_pi / 2) * (torch.log(pi_l)).pow(2).sum()

# WRONG (causes precision explosion): L_pi = -0.5 * torch.log(pi_l).sum()
```

**Error projection upward:**
- `W_error_up[l]`: Linear(d_l, d_{l+1}, bias=False) — projects precision-weighted
  error from level l into level l+1's space
- Bias-free so zero error maps to zero update

**Verification:**
- Unit test: prediction error is zero when prediction equals actual state
- Unit test: precision regulariser has minimum at pi=1
- Unit test: precision explosion does NOT occur (train for 1000 steps, assert
  pi_mean < 10)
- Comparison test: match Phase 1 precision dynamics (spike at step ~2500, then
  settling to ~0.04)

#### Step 1.4: Halting Mechanism

**File:** `coral/v4/halting.py`

Reuse the Q-learning halting mechanism from v3 with minimal changes. The halting
network takes the concatenation of all level states (projected to a common dim)
and produces a halting probability.

**Verification:**
- Unit test: halting probability is in [0, 1]
- Integration test: with untrained model, halting does not fire (all 16 segments used)

#### Step 1.5: Core Assembly (N=2 Only)

**File:** `coral/v4/coral_core.py`

```python
class CoralCore(nn.Module):
    """
    The amodal reasoning core.
    Takes embeddings in, returns refined embeddings out.
    """
    def __init__(self, config):
        # Shared backbone
        self.backbone = CoralBackbone(config)

        # Level modules (N=2 for Experiment 1)
        self.levels = nn.ModuleList([
            LevelModule(level=1, dim=512, config=config),
            LevelModule(level=2, dim=256, config=config),
        ])

        # Predictive coding (1 inter-level pair for N=2)
        self.pc = PredictiveCodingModule(dim_l=512, dim_l_plus_1=256, config=config)

        # Halting
        self.halting = HaltingNetwork(config)

    def forward(self, z1_init, K_max, training=True):
        """
        z1_init: [B, L, 512] — from adapter
        Returns: [B, L, 512] — refined solution state, plus loss components
        """
```

**Inner loop structure (T=3, N=2):**

```
For each segment k in 1..K_max:
    # Top-down prediction
    mu_1 = prediction_net(z_2)

    # Level 1: T^0 × T = 3 inner steps (fast)
    for t in 1..3:
        backbone_in = W_up[1](z_1) + level_emb[1] + timescale_emb[t] + mu_1_projected
        z_1 = W_down[1](backbone(backbone_in))

    # Precision-weighted error
    eps_1 = z_1 - mu_1
    pi_1 = precision_net(z_1)
    xi_1 = pi_1 * eps_1

    # Level 2: 1 inner step (slow)
    xi_1_up = W_error_up(xi_1)
    backbone_in = W_up[2](z_2) + level_emb[2] + timescale_emb[0] + xi_1_up
    z_2 = W_down[2](backbone(backbone_in))

    # Halting check
    h_k = halting_net(concat(z_1, project(z_2)))

    # Detach for next segment
    z_1, z_2 = z_1.detach(), z_2.detach()
```

**CRITICAL — Full backprop within segment:**
All 3+1=4 backbone applications within a single segment are in the same computation
graph. Gradients flow through all of them. The `detach()` happens only between
segments (deep supervision boundary).

**Verification:**
- Integration test: full forward pass produces correct output shape
- Integration test: `loss.backward()` completes without error
- Integration test: all parameters receive non-zero gradients
- Memory test: peak GPU memory at batch=64, L=81 is <20GB (fits on A100-40GB)
- Gradient flow test: verify that level 2 parameters receive gradients from
  level 1's prediction error (the predictive coding chain is connected)

#### Step 1.6: Grid Adapter (Sudoku)

**File:** `coral/v4/adapters/grid.py`

```python
class GridAdapter(nn.Module):
    """
    Encoder: grid tokens → d₁=512 embeddings
    Decoder: d₁=512 embeddings → token logits
    """
```

Reuse the grid encoding logic from v3 (token embedding + 2D position embedding).
The decoder is a linear projection from d₁=512 to vocabulary size (10 for Sudoku).

**Verification:**
- Unit test: encode → decode cycle on random grid produces valid logits
- Smoke test: overfit on 1 puzzle through the full pipeline (adapter + core + adapter)

#### Step 1.7: Loss Function

**File:** `coral/v4/losses.py`

```python
class CoralLoss(nn.Module):
    """
    Unified loss for CORAL v4.
    Components enabled/disabled based on training phase config.
    """
    def forward(self, outputs, targets, config):
        # Always active
        L_task = cross_entropy_stablemax(outputs.logits, targets)  # float64
        L_pred = config.lambda_pred * precision_weighted_prediction_loss(outputs)
        L_pi = config.lambda_pi * precision_regulariser(outputs.precisions)
        L_halt = halting_loss(outputs.halting_probs, outputs.logits, targets)

        # Enabled in Experiment 2+
        L_amort = config.lambda_amort * amortisation_loss(outputs) if config.use_amort else 0

        # Enabled in Experiment 3+
        L_crystal = config.lambda_crystal * crystallisation_loss(outputs) if config.use_crystal else 0
        L_commit = config.lambda_commit * commitment_loss(outputs) if config.use_crystal else 0

        total = L_task + L_pred + L_pi + L_halt + L_amort + L_crystal + L_commit

        # Return breakdown for W&B logging
        return total, {
            'loss/total': total,
            'loss/task': L_task,
            'loss/prediction': L_pred,
            'loss/precision_reg': L_pi,
            'loss/halting': L_halt,
            'loss/amortisation': L_amort,
            'loss/crystallisation': L_crystal,
            'loss/commitment': L_commit,
        }
```

**Cross-entropy with stablemax:** Use the stablemax normalisation in float64,
following v3 implementation. This prevents numerical instability in the softmax
with large logit values.

#### Step 1.8: Training Loop

**File:** `coral/training/trainer_v4.py`

The training loop implements deep supervision with K_max segments per batch:

```python
for batch in dataloader:
    x, y = batch
    z1_init = grid_adapter.encode(x)  # [B, 81, 512]

    total_loss = 0
    z_states = initial_states(z1_init)

    for segment in range(K_max):
        # Forward through core (full backprop within segment)
        z_states, outputs = coral_core(z_states, segment_idx=segment)

        # Compute loss for this segment
        logits = grid_adapter.decode(z_states[1])
        segment_loss, breakdown = loss_fn(logits, y, outputs, config)
        total_loss += segment_loss

        # Log per-segment metrics
        log_segment_metrics(segment, breakdown)

        # Detach for next segment
        z_states = {l: z.detach() for l, z in z_states.items()}

        # Q-learning halting (training mode)
        if should_halt(outputs.halting_probs, config):
            break

    # Backward and step
    total_loss.backward()
    clip_grad_norm_(parameters, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

**Optimizer — resolve before first run (Build Step 0):**

The optimizer choice is a Day 0 decision. The priority order is:

1. **Fused CUDA AdamATan2** (best performance). Attempt compilation on the Vast.ai
   container. If it compiles and passes a 100-step smoke test, use it.
   ```bash
   # Test compilation on fresh Vast.ai instance
   cd coral/training && python -c "from fused_adam_atan2 import FusedAdamATan2; print('OK')"
   ```

2. **torch.optim.AdamW** (proven baseline). If fused AdamATan2 fails to compile,
   use AdamW with lr=7e-5, weight_decay=1.0, betas=(0.9, 0.95). This is what
   TRM, GRAM, and every competitive model uses. It is a better choice than a slow
   pure-PyTorch AdamATan2.

3. **Pure PyTorch AdamATan2 with torch.compile** (last resort). Only if there is a
   specific reason to prefer scale-invariant updates over AdamW AND fused compilation
   fails. Wrap the optimizer step in `torch.compile` to partially close the
   performance gap.

Do NOT default to the pure PyTorch AdamATan2 from v3 without compilation — it is
measurably slower and the v3 baseline gap (41.2% vs HRM's reported 55%) may be
partly attributable to optimizer performance.

**Whichever optimizer is selected:** use lr=7e-5, weight_decay=1.0, cosine schedule,
gradient clipping at max_norm=1.0.

#### Step 1.9: Evaluation Infrastructure

**File:** `coral/evaluation/pareto.py`

```python
def evaluate_pareto(model, eval_dataloader, K_values=[1, 2, 4, 8, 16]):
    """
    Evaluate accuracy at multiple forced depth limits.
    Returns accuracy-depth curve for Pareto analysis.
    """
    results = {}
    for K in K_values:
        accuracy = evaluate_at_depth(model, eval_dataloader, max_segments=K)
        results[f'eval/accuracy@K{K}'] = accuracy

    # Pareto area: normalised area under the accuracy-depth curve
    results['eval/pareto_area'] = compute_pareto_area(results, K_values)
    return results
```

**W&B logging at every eval step:**
- `eval/exact_accuracy` (full depth, with halting)
- `eval/token_accuracy` (per-cell accuracy)
- `eval/accuracy@K1`, `eval/accuracy@K2`, `eval/accuracy@K4`, `eval/accuracy@K8`,
  `eval/accuracy@K16`
- `eval/pareto_area`
- `eval/avg_halting_step`
- `precision/mean`, `precision/std`, `precision/min`, `precision/max`
- `prediction_error/mean`, `prediction_error/max`
- All loss components from the breakdown dict

#### Step 1.10: Config and Entry Point

**File:** `configs/exp1_baseline.yaml`

```yaml
# CORAL v4 — Experiment 1: Shared Backbone + Full Backprop + Predictive Coding
model:
  n_levels: 2
  level_dims: [512, 256]
  backbone_layers: 2
  backbone_dim: 512
  n_heads: 8
  ffn_expansion: 4
  timescale_base: 3          # T=3

  # Predictive coding (ON)
  use_predictive_coding: true
  lambda_pred: 0.1
  lambda_pi: 0.01
  epsilon_min: 0.01

  # Crystallisation (OFF for Exp 1)
  use_crystallisation: false

  # Column heads (OFF for Exp 1 — standard backbone only)
  use_column_heads: false

  # Stochastic transitions (OFF for Exp 1)
  use_stochastic: false

  # Amortisation pressure (OFF for Exp 1)
  use_amort: false
  lambda_amort: 0.0

  # Halting
  K_max: 16
  halting_threshold: 0.95
  halting_exploration_prob: 0.1

training:
  epochs: 20000
  batch_size: 64
  learning_rate: 7e-5
  weight_decay: 1.0
  gradient_clip: 1.0
  scheduler: cosine
  precision: bfloat16
  loss_precision: float64

  # Full backprop within segment (NEW — this is the key change from v3)
  full_inner_backprop: true

data:
  dataset: sudoku_extreme_1k
  augmentation_factor: 1000
  eval_size: 1000

wandb:
  project: "Sudoku-extreme-1k-aug-1000 CORAL-v4"
  tags: ["v4", "exp1", "shared-backbone", "full-backprop", "predictive-coding"]
```

**Entry point:** `scripts/train.py`

```python
@hydra.main(config_path="../configs", config_name="exp1_baseline")
def main(cfg):
    # Build model
    adapter = GridAdapter(cfg)
    core = CoralCore(cfg)
    model = nn.ModuleDict({'adapter': adapter, 'core': core})

    # Build trainer
    trainer = TrainerV4(model, cfg)

    # Train
    trainer.train()
```

#### Phase 1 Acceptance Criteria

- [ ] Eval exact accuracy ≥ 70% on Sudoku-Extreme-1K (target: >75%)
- [ ] Precision dynamics show the phase transition (error collapse + precision spike
      at step ~2000-4000, consistent with Phase 1 findings)
- [ ] Full backprop does NOT cause OOM at batch=64, L=81 on A100-40GB
- [ ] Training wall-clock time ≤ 6 hours (comparable to Phase 1's ~3.8h; may be
      slightly longer due to full backprop)
- [ ] All loss components log correctly to W&B
- [ ] Pareto evaluation runs successfully (accuracy@K1 through accuracy@K16)

**If this phase fails:** Diagnose by running two ablations:
1. Shared backbone + 1-step gradient approx (isolates backbone sharing effect)
2. Separate H/L modules + full backprop (isolates full backprop effect)

---

### Phase 2: Amortisation Pressure (Experiment 2)

**Goal:** Add computation cost pressure. Verify that the model learns to front-load
reasoning (Pareto curve improvement).

**Changes from Phase 1:**
- Enable `use_amort: true`
- `lambda_amort`: anneal from 0 to 0.01 over epochs 5000–15000

**New code:**
- `amortisation_loss()` in `losses.py`: sum of ||eps_l^t||² across all inner steps
  and levels within each segment

**Acceptance criteria:**
- [ ] Model B (with L_amort) Pareto curve dominates Model A (without) — see
      Section 10.6 of the spec for exact protocol
- [ ] At K=4 (25% of max depth), Model B accuracy is ≥5% higher than Model A
- [ ] Full-depth accuracy degrades by ≤2% vs Phase 1

**Compute:** ~8 hours (two runs: with and without amortisation)

---

### Phase 3: Crystallisation (Experiment 3)

**Goal:** Add the recognition network, codebooks, and adaptive depth per-position.

**Changes from Phase 2:**
- Enable `use_crystallisation: true`
- `lambda_crystal`: anneal from 0 to 0.1 over epochs 15000–25000
- `lambda_commit`: 0.25

**New code:**
- `coral/v4/crystallisation.py`: RecognitionNetwork, Codebook, consolidation logic
- Modify `coral_core.py` to support per-position depth allocation
- Codebook initialisation via k-means on state buffer after Phase A

**Acceptance criteria:**
- [ ] Crystallisation rate > 10% at any level by end of training
- [ ] Codebook usage entropy > 2.0 (no degenerate codebooks)
- [ ] Higher levels crystallise at higher rates than lower levels
- [ ] Accuracy at 50% average depth ≥ 90% of full-depth accuracy
- [ ] eval/pareto_area improves over Phase 2

**Compute:** ~6 hours

---

### Phase 4: Hierarchy Scaling (Experiment 4)

**Goal:** Scale from N=2 to N=3 and N=4. Validate that deeper hierarchy with
progressive bottleneck produces genuine abstraction.

**Changes from Phase 3:**
- Add level modules for N=3 (dims: 512, 256, 128) and N=4 (dims: 512, 256, 128, 64)
- Add prediction/precision networks for new inter-level pairs
- Update T=3 nested execution for N=3 (13 inner steps) and N=4 (40 inner steps)

**Key diagnostic — differential crystallisation:**
If higher levels crystallise faster than lower levels, the hierarchy is creating
genuine abstraction (higher levels see more compressible patterns). If all levels
crystallise at the same rate, the hierarchy is not doing useful work.

**Acceptance criteria:**
- [ ] N=3 or N=4 matches or exceeds N=2 accuracy
- [ ] N=3 or N=4 uses fewer total inner steps than N=2 at matched accuracy
      (due to higher-level crystallisation)
- [ ] Level 4 (d=64) crystallisation rate > Level 1 (d=512) crystallisation rate
- [ ] Memory fits on A100-40GB at batch=64, L=81

**Compute:** ~8 hours (N=3 and N=4 runs)

---

### Phase 5: Stochastic Transitions (Experiment 5)

**Goal:** Add precision-gated noise. Measure accuracy boost from multi-trajectory
voting.

**Changes from Phase 4:**
- Enable `use_stochastic: true`
- `sigma_l`: learned per-level noise scale, initialised at 0.01
- Noise gated by `(1 - h_k)` and scaled by `1/sqrt(pi_l)`

**New code:**
- `coral/v4/stochastic.py`: stochastic transition logic
- Multi-trajectory evaluation: sample K trajectories, majority vote

**Acceptance criteria:**
- [ ] Stochastic single-pass accuracy ≥ Phase 4 deterministic accuracy
- [ ] Multi-trajectory (K=8) accuracy > single-pass accuracy by ≥ 5%
- [ ] Multi-trajectory (K=8) accuracy ≥ 85% on Sudoku-Extreme-1K

**Compute:** ~6 hours

---

### Phase 6: Full Ablation + ARC-AGI Validation (Experiment 6)

**Goal:** Complete ablation matrix on Sudoku, then validate on ARC-AGI.

#### 6a: Ablation Matrix (Sudoku)

Run the full ablation matrix from Section 16.1 of the spec:

| Variant | Config Changes |
|---------|---------------|
| Full CORAL v4 | All components on |
| − Precision weighting | `use_predictive_coding: false` (raw state passing) |
| − Crystallisation | `use_crystallisation: false` |
| − Amortisation pressure | `lambda_amort: 0` |
| − Stochastic transitions | `use_stochastic: false` |
| − Hierarchy (N=2) | `n_levels: 2` |
| − Hierarchy (N=1, flat) | `n_levels: 1` (single level, TRM-like) |
| TRM reproduction | Single 2-layer network, TRM protocol |

**Compute:** ~12 hours (8 runs × ~1.5 hours each, can parallelise on multiple GPUs)

#### 6b: ARC-AGI Validation

**New code:**
- `coral/data/arc_dataset.py`: ARC-AGI data loading + augmentation
  - Colour permutation (10! = 3.6M possible, sample 1000)
  - Dihedral group (8 transformations: 4 rotations × 2 flips)
  - Translations (shift grid within 30×30 canvas)
  - Per-puzzle embedding (following TRM protocol)
- Update `GridAdapter` to handle variable grid sizes (pad to 30×30)
- Majority voting across augmentations at test time

**Protocol:**
- Train on ~960 tasks (800 ARC + 160 ConceptARC)
- 1000× augmentation per task
- Evaluate on ARC-AGI-1 public evaluation set (400 tasks, 2 attempts per task)

**Acceptance criteria:**
- [ ] ARC-AGI-1 eval accuracy > 40% (exceeds HRM's reported 40.3%)
- [ ] ARC-AGI-1 eval accuracy > 35% (exceeds HRM's verified 32% on semi-private)

**Compute:** ~3 days on 4 H100s (or ~12 days on single A100)

---

## Implementation Order (Detailed)

Within Phase 1, the implementation order matters. Build from the bottom up, testing
at each step:

```
Day 0:  REPO SETUP + OPTIMIZER RESOLUTION (Build Step 0)
        - Create new repo: github.com/anwar11235/coral-v4
        - Set up project structure (directories, __init__.py files, requirements.txt)
        - Copy sudoku_dataset.py and eval metrics from v3 repo, verify they run standalone
        - Copy Sudoku-Extreme-1K data files into data/
        - Spin up Vast.ai A100 instance
        - Attempt fused CUDA AdamATan2 compilation
        - If success: copy kernel into coral/training/fused_adam_atan2/, record in config
        - If failure: configure torch.optim.AdamW (lr=7e-5, wd=1.0, betas=(0.9,0.95))
        - Run 100-step smoke test with chosen optimizer on a dummy model
        - Commit decision to configs/exp1_baseline.yaml

Day 1:  backbone.py + tests → verify forward/backward, shapes, memory
Day 2:  level_module.py + tests → verify projections, level embedding effect
Day 3:  predictive_coding.py + tests → verify error computation, precision reg
Day 4:  halting.py (port from v3) + tests
Day 5:  coral_core.py (N=2 assembly) + integration tests
Day 6:  adapters/grid.py (port from v3) + losses.py + tests
Day 7:  trainer_v4.py + train_v4.py + smoke test (overfit 1 puzzle)
Day 8:  eval infrastructure (pareto.py) + W&B integration
Day 9:  Full Experiment 1 run (configs, launch, monitor)
Day 10: Analyse results, decide go/no-go for Phase 2
```

---

## Key Implementation Warnings

### 1. Precision Regulariser Sign

The single most dangerous bug in this codebase. The correct regulariser is:

```python
L_pi = (lambda_pi / 2) * (torch.log(pi)).pow(2).mean()
```

NOT:

```python
L_pi = -0.5 * torch.log(pi).mean()  # WRONG — causes precision explosion
```

Add an assertion in the loss function that checks `pi.mean() < 100` and raises
an error if violated. This catches the bug within the first few hundred steps.

### 2. Full Backprop Memory

With T=3 and N=2, each segment has 4 backbone applications in the computation
graph. At batch=64, L=81, d=512, bfloat16, each backbone activation is:
64 × 81 × 512 × 2 bytes = ~5.3 MB. With 4 applications plus intermediate
activations, expect ~50-100MB per segment. With K_max=16 segments and
detach between them, only 1 segment's activations are live at a time.
Total peak: ~100MB for activations + ~25MB for parameters = well within A100's
40GB.

However: if `torch.compile` is used, the compiled graph may hold references
differently. Test memory with and without compile.

### 3. torch.compile and Variable Sub-Batch Sizes

From Phase 2 experience: `torch.compile` with `dynamic=True` causes ~3 it/s
slowdown due to recompilation from variable sub-batch sizes (ACT halting changes
batch composition). Use `dynamic=False` for Sudoku (fixed L=81). For ARC
(variable grid sizes), either pad to 30×30 or use `dynamic=True` with the
performance cost.

### 4. Detach Between Segments

The `detach()` between deep supervision segments is critical. Without it,
backpropagation would attempt to go through all K_max segments, requiring
K_max × (memory per segment) — which WILL OOM.

```python
# CORRECT: detach between segments
z_states = {l: z.detach() for l, z in z_states.items()}

# WRONG: no detach
# z_states = z_states  # OOM after a few segments
```

### 5. float64 Loss Computation

From v3 experience: cross-entropy with stablemax in bfloat16 can produce NaN
due to limited dynamic range. Cast logits to float64 before loss computation:

```python
logits_f64 = logits.to(torch.float64)
loss = F.cross_entropy(logits_f64.view(-1, V), targets.view(-1))
```

### 6. SudokuAugmenter Integration

The existing SudokuAugmenter from v3 should be reused without modification.
It produces augmented (input, target) pairs on-the-fly during training.
Verify that the augmenter is producing the expected 1000× factor by checking
the effective dataset size in the first epoch.

### 7. Hydra Config: Use + Prefix for New Fields

From v3 experience: Hydra requires `+` prefix for config fields not in the base
struct. Since v4 has a new config schema, define all fields in the base dataclass
to avoid this issue:

```python
@dataclass
class CoralV4Config:
    n_levels: int = 2
    level_dims: List[int] = field(default_factory=lambda: [512, 256])
    timescale_base: int = 3
    use_predictive_coding: bool = True
    use_crystallisation: bool = False
    use_stochastic: bool = False
    use_amort: bool = False
    # ... all fields with defaults
```

---

## W&B Project Structure

**Project name:** `Sudoku-extreme-1k-aug-1000 CORAL-v4`

**Run naming convention:** `v4-exp{N}-{descriptor}`

Examples:
- `v4-exp1-baseline` — Phase 1, shared backbone + full backprop + PC
- `v4-exp2-amort` — Phase 2, + amortisation pressure
- `v4-exp2-no-amort` — Phase 2, comparison without amortisation
- `v4-exp3-crystal` — Phase 3, + crystallisation
- `v4-exp4-n3` — Phase 4, N=3 hierarchy
- `v4-exp4-n4` — Phase 4, N=4 hierarchy
- `v4-exp5-stochastic` — Phase 5, + stochastic transitions
- `v4-exp6-ablation-{variant}` — Phase 6, ablation matrix

**Tags:** Every run should be tagged with:
- Version: `v4`
- Experiment number: `exp1`, `exp2`, etc.
- Components active: `predictive-coding`, `crystallisation`, `stochastic`, `amort`
- Hierarchy: `n2`, `n3`, `n4`

---

## Dependencies

### Python Packages (install in Vast.ai container)

```bash
pip install torch torchvision --break-system-packages  # if not pre-installed
pip install wandb hydra-core omegaconf --break-system-packages
pip install einops --break-system-packages  # for rotary embeddings
```

### Files to Copy from v3 Repo

These files are copied into the new repo, verified, and adapted as needed:

- `sudoku_dataset.py` → `coral/data/sudoku_dataset.py`
  SudokuAugmenter + dataset loading (~300 lines). Verify augmentation factor
  produces expected dataset size. No architectural dependencies.

- Evaluation metric functions (exact accuracy, token accuracy) → `coral/evaluation/evaluator.py`
  ~50 lines. Pure functions, no dependencies on v3 model.

- Fused CUDA AdamATan2 kernel source → `coral/training/fused_adam_atan2/`
  Only if compilation succeeds on Day 0. Self-contained CUDA extension.

- Sudoku-Extreme-1K data files → `data/sudoku_extreme_1k/`
  Training and eval splits. Binary data, no code dependencies.

### Nothing Else from v3

Everything else is written from scratch. In particular, do NOT copy:

- Any model code (`coral/model/` from v3) — entirely replaced
- Any loss functions (`free_energy.py` from v3) — entirely replaced
- Any training loops — entirely replaced
- The pure PyTorch AdamATan2 fallback — too slow; use AdamW instead
- Any Hydra configs — schema is completely different
- Any flash-attn imports — use PyTorch SDPA

---

## Success Definition

**The v4 build is successful if:**

1. **Experiment 1** demonstrates that shared backbone + full backprop + predictive
   coding exceeds Phase 1's 61.1% on Sudoku-Extreme-1K (target: ≥70%).

2. **Experiment 2** demonstrates that amortisation pressure produces a dominant
   Pareto curve (same accuracy at lower depth).

3. **Experiment 3** demonstrates that crystallisation activates meaningfully
   (rate > 10%) and reduces average computation without degrading accuracy.

4. **Experiment 4** demonstrates that N=3 or N=4 provides a benefit over N=2
   (differential crystallisation rates across levels).

5. **Experiment 5** demonstrates that stochastic transitions + multi-trajectory
   voting pushes accuracy above 85% on Sudoku-Extreme-1K.

6. **Experiment 6** demonstrates generality on ARC-AGI-1 (>40% accuracy).

**The v4 build produces a paper-ready result if** items 1–5 pass and item 6
shows competitive performance. The paper's contribution is not primarily about
SOTA accuracy — it is about providing a principled theoretical framework (free
energy) that explains why recursive reasoning works, with amortisation as the
unique mechanism that no competing architecture provides.
