# CORAL v4 — COrtical Reasoning via Abstraction Layers

An amodal reasoning core based on variational free energy minimisation.

**Status:** Phase 1 implementation — shared backbone + full backprop + predictive coding

## Architecture

CORAL v4 is a multi-timescale recurrent hierarchy that:
- Takes embedding vectors as input, returns refined embedding vectors as output
- Reasons through precision-weighted predictive coding
- Learns to minimise its own computational cost over training (amortisation)
- Interfaces with any modality through pluggable adapters

### Key design decisions
| Decision | Value | Rationale |
|----------|-------|-----------|
| Backbone | Shared 2-layer transformer | Weight sharing forces generalisation (TRM finding) |
| Attention | Self-attention (SDPA) | Required for variable sequence lengths |
| Timescale | T=3 (3/1 steps for N=2) | Best compute/depth tradeoff |
| Backprop | Full within segment | Single biggest improvement from v3 (TRM Table 1) |
| Precision reg | `(λ_π/2)(log π)²` | Symmetric log-normal; no explosion |
| Precision | bfloat16 fwd / float64 loss | Prevents NaN in stablemax CE |

## Repository Structure

```
coral/
├── config.py              # Hydra config dataclasses
├── model/
│   ├── backbone.py        # Shared 2-layer transformer (RoPE, SwiGLU, SDPA)
│   ├── level_module.py    # Level projections (up/down)
│   ├── predictive_coding.py  # Prediction, precision, error
│   ├── halting.py         # Q-learning adaptive halting
│   └── coral_core.py      # Main assembly
├── adapters/
│   ├── base.py            # Abstract adapter interface
│   └── grid.py            # Grid encoder/decoder (Sudoku, ARC, Maze)
├── training/
│   ├── optimizer.py       # Optimizer selection
│   ├── losses.py          # Unified loss (stablemax CE + PC + halting)
│   └── trainer.py         # Training loop with deep supervision
├── data/
│   └── sudoku_dataset.py  # HRM-format Sudoku dataset loader
└── evaluation/
    ├── evaluator.py       # Exact/token accuracy
    └── pareto.py          # Accuracy-depth Pareto curve
configs/
    exp1_baseline.yaml     # Experiment 1 config
scripts/
    train.py               # Main entry point
    evaluate.py            # Standalone evaluation
    evaluate_pareto.py     # Pareto curve evaluation
tests/
    test_backbone.py
    test_predictive_coding.py
    test_coral_core.py
    test_forward_backward.py
    test_adapters.py
```

## Setup

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Training (Vast.ai A100)

```bash
# Start tmux session
tmux new-session -s training

# Train Experiment 1
cd /workspace/CORAL-v4
python scripts/train.py

# Override config fields
python scripts/train.py training.batch_size=32 training.learning_rate=1e-4
```

## Evaluation

```bash
python scripts/evaluate_pareto.py checkpoint=checkpoints/best.pt
```

## Tests

```bash
pytest tests/ -v
```

## Phase 1 Success Criteria

- [ ] Eval exact accuracy ≥ 70% on Sudoku-Extreme-1K
- [ ] Precision dynamics show phase transition (step ~2000–4000)
- [ ] No OOM at batch=64, L=81 on A100-40GB
- [ ] Training completes in ≤ 6 hours
- [ ] All loss components log correctly to W&B
- [ ] Pareto evaluation: accuracy@K1 through accuracy@K16

## W&B Configuration

- Workspace: `aktuator-ai`
- Project: `Sudoku-extreme-1k-aug-1000 CORAL-v4`
- Run name: `v4-exp1-baseline`

## Critical Implementation Warnings

1. **Precision regulariser sign** — uses `(log π)²`, NOT `-log π`. Runtime assertion: `pi.mean() < 100`.
2. **Detach between segments** — `z.detach()` at segment boundaries prevents OOM.
3. **float64 for loss** — cast logits to float64 before stablemax cross-entropy.
4. **torch.compile** — use `dynamic=False` for Sudoku (fixed L=81).
5. **Full backprop within segment** — all backbone applications in one segment share the same computation graph.
