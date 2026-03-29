# Claude Code Starting Prompt — CORAL v4

You are building CORAL v4 (COrtical Reasoning via Abstraction Layers), an amodal
reasoning core based on variational free energy minimisation. This is a clean
implementation in a new repository — do not look for or depend on any existing
codebase.

## Context Documents

Two reference documents are provided:

1. **CORAL_Architecture_Spec_v4.md** — The authoritative architecture specification.
   Contains the complete mathematical framework, component designs, design decisions
   with rationale, and hyperparameter defaults. When in doubt about any architectural
   decision, this document is the source of truth.

2. **CORAL_v4_Build_Plan.md** — The implementation plan. Contains the repo structure,
   file-by-file build order, verification criteria for each step, implementation
   warnings, and experiment configs. Follow this plan sequentially.

Read both documents in full before writing any code.

## What We Are Building

An amodal reasoning core that:

- Takes embedding vectors as input, returns refined embedding vectors as output
- Reasons through precision-weighted predictive coding in a multi-timescale hierarchy
- Learns to minimise its own computational cost over training (amortisation)
- Interfaces with any modality (language, vision, grids) through pluggable adapters
- Is grounded in a single theoretical principle: variational free energy minimisation

## Committed Architectural Decisions (Do Not Revisit)

These decisions have been made through analysis of the competitive landscape (TRM,
GRAM, Augmented HRM, CMM, RSM) and our own Phase 0–2 experimental results. They
are final.

- **Shared backbone**: single 2-layer transformer, weight-shared across all hierarchy
  levels and recursion steps. Level-specific behaviour via additive level embeddings.
- **Self-attention** (not MLP-mixer): required for variable sequence lengths across
  modalities.
- **T=3 timescale multiplier**: 40 inner steps per segment with N=4 (1+3+9+27).
- **Full backpropagation through the inner loop** within each deep-supervision segment.
  Detach only between segments. This is the single highest-impact change from v3.
- **Precision regulariser**: symmetric log-normal `(λ_π/2) * (log π)²` with minimum
  at π=1. NOT the naive `−½ log π` which causes unbounded precision growth.
- **bfloat16 forward, float64 loss**: prevents NaN in stablemax cross-entropy.
- **PyTorch SDPA for attention**: do not use the flash-attn package (compilation
  issues on Vast.ai).

## Current Phase: Phase 1 (Experiment 1)

We are building Phase 1: shared backbone + full backprop + precision-weighted
predictive coding, tested on Sudoku-Extreme-1K.

Start with **Day 0** from the build plan:

1. Create the repository structure (all directories, `__init__.py` files,
   `requirements.txt`, `setup.py`)
2. Set up the Hydra config dataclass in `coral/config.py` with ALL fields defined
   with defaults (to avoid the `+` prefix issue)
3. Copy the SudokuAugmenter and dataset code from the v3 repo at
   `github.com/anwar11235/CORAL-v3` — files: `coral/data/sudoku_dataset.py` and
   related data loading utilities. Verify they run standalone.
4. Set up the optimizer — attempt fused CUDA AdamATan2 compilation first. If it
   fails, use `torch.optim.AdamW` with `lr=7e-5, weight_decay=1.0,
   betas=(0.9, 0.95)`. Do NOT use the pure PyTorch AdamATan2 fallback from v3
   (it is too slow).

Then proceed to **Day 1**: implement `coral/model/backbone.py` with unit tests.

## Build Order (Phase 1)

Follow this order strictly. Each step must pass its verification criteria before
moving to the next.

```
Day 0:  Repo setup + optimizer resolution
Day 1:  backbone.py + tests
Day 2:  level_module.py + tests
Day 3:  predictive_coding.py + tests
Day 4:  halting.py + tests
Day 5:  coral_core.py (N=2 assembly) + integration tests
Day 6:  adapters/grid.py + losses.py + tests
Day 7:  trainer.py + train.py + smoke test (overfit 1 puzzle)
Day 8:  evaluation infrastructure (pareto.py) + W&B integration
Day 9:  Full Experiment 1 run
Day 10: Analyse results
```

## Critical Implementation Warnings

1. **Precision regulariser sign** — the single most dangerous bug. Add a runtime
   assertion `assert pi.mean() < 100` in the loss function. If precision explodes,
   you have the wrong regulariser.

2. **Detach between segments** — without `z.detach()` at segment boundaries, backprop
   will attempt to go through all K_max segments and OOM.

3. **float64 for loss** — cast logits to float64 before cross-entropy. bfloat16
   stablemax produces NaN.

4. **torch.compile** — use `dynamic=False` for Sudoku (fixed L=81). Do NOT use
   `dynamic=True` unless switching to ARC (variable grid sizes).

5. **Full backprop within segment** — all T+1 backbone applications within a single
   segment must be in the same computation graph. This is the key difference from v3.
   Test explicitly: after `loss.backward()`, verify that backbone parameters have
   non-zero `.grad` and that level 2 parameters receive gradients that depend on
   level 1 computation.

## W&B Configuration

- Workspace: `aktuator-ai`
- Project: `Sudoku-extreme-1k-aug-1000 CORAL-v4`
- Run name: `v4-exp1-baseline`
- Tags: `["v4", "exp1", "shared-backbone", "full-backprop", "predictive-coding"]`

Log at every eval step:
- `eval/exact_accuracy`, `eval/token_accuracy`
- `eval/accuracy@K1`, `eval/accuracy@K2`, `eval/accuracy@K4`, `eval/accuracy@K8`,
  `eval/accuracy@K16` (Pareto curve)
- `eval/pareto_area`, `eval/avg_halting_step`
- `precision/mean`, `precision/std`
- `prediction_error/mean`, `prediction_error/max`
- All loss component breakdowns

## Phase 1 Success Criteria

- [ ] Eval exact accuracy ≥ 70% on Sudoku-Extreme-1K
- [ ] Precision dynamics show phase transition (error collapse + precision spike
      around step 2000–4000)
- [ ] No OOM at batch=64, L=81 on A100-40GB
- [ ] Training completes in ≤ 6 hours
- [ ] All loss components log correctly to W&B
- [ ] Pareto evaluation produces accuracy@K1 through accuracy@K16

## Style and Practices

- One file, one concern. Each module should be testable in isolation.
- Type hints on all public interfaces.
- Docstrings on all classes and public methods.
- Config-driven: all hyperparameters flow from the Hydra config, not hardcoded.
- Test-driven: write the test, then the implementation.
- Commit after each day's work passes verification.

## Hardware

- Primary: Vast.ai A100-SXM4-40GB
- Use tmux for persistent sessions
- All training in bfloat16, loss in float64
- Expect ~4-6 hours per Experiment 1 training run
