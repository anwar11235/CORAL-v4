# CORAL v4 — Architecture Spec Addendum (v4.2)

## Train Deep, Infer Shallow

**Date:** March 31, 2026

**Status:** This addendum supersedes specific sections of the v4.1 Architecture Spec
based on empirical findings from Session 2. The core theoretical framework (free energy
minimisation, amodal interface, deployment profiles) is unchanged. What changes is the
build order, the role of predictive coding, and the training-time compute model.

---

## 1. Central Paradigm Shift: Train Deep, Infer Shallow

### 1.1 The Finding

Empirical results from Session 2 established:

- The shared 2-layer backbone works well (60% token accuracy with 4 inner steps,
  task loss 1.05 with 21 inner steps at only 700 training steps)
- Every configuration of predictive coding HURT training performance
- The model actively learned to suppress PC conditioning when given a learnable gate
- The bottleneck at 4 inner steps was per-segment depth, not the PC mechanism

### 1.2 The Revised Paradigm

**Training time:** Invest maximum computational depth. N=1, 21+ inner steps per
segment, 16 segments with deep supervision. Full backprop through all inner steps
within each segment. The goal is to learn the richest possible representations and
constraint propagation patterns. No shortcuts during training.

**Inference time:** The trained model is progressively made more efficient through:

1. **Crystallisation** — recognise familiar patterns and skip to codebook answers
2. **Amortisation pressure** — explicit loss term that rewards faster convergence
3. **Adaptive halting** — stop early when the answer has converged
4. **Precision-driven focus** — attend only to uncertain dimensions
5. **Multi-level hierarchy** — higher levels crystallise faster, providing warm starts

The brain analogy: a child learning Sudoku takes many deliberate passes through each
constraint. An expert glances at the grid and fills in easy cells instantly (System 1),
reserving deliberation for hard cells (System 2). The expert's fast recognition was
built on the child's slow practice. CORAL follows the same trajectory.

### 1.3 Why This Order Matters

The original spec attempted to build all mechanisms simultaneously. This failed because:

- PC conditioning blurred the backbone's representations before the backbone had
  learned to produce sharp, position-specific features
- Precision had no useful gradient signal because prediction errors collapsed before
  the task loss could differentiate precision across dimensions
- The efficiency mechanisms (crystallisation, amortisation) had nothing to amortise
  because the backbone couldn't solve puzzles yet

The correct order: build a backbone that solves the task → layer communication
mechanisms on top → add efficiency mechanisms last.

---

## 2. Revised Component Roles

### 2.1 Shared Backbone (Unchanged)

2-layer transformer, d=512, 8 heads, SwiGLU, RMSNorm, RoPE. Weight-shared across
all hierarchy levels and recursion steps. Self-attention for variable sequence lengths.

**Training depth:** 21 inner steps per segment at N=1 (matching TRM's effective depth).
With N=2+, level 1 gets the majority of inner steps.

### 2.2 Predictive Coding (Role Changed)

**Previous role (v4.1):** Top-down conditioning on level 1. Precision-weighted error
propagation upward.

**Revised role (v4.2):** Inter-level error communication ONLY. No top-down conditioning
on level 1.

The forward pass becomes:

```
# Level 1 runs UNCONDITIONED
for t in 1..T:
    z1 = backbone(z1 + level_emb + ts_emb)      # no prediction added

# AFTER level 1 finishes, compute error
mu = prediction_net(z2)                           # level 2 predicts level 1
eps = z1 - mu                                     # what level 2 got wrong
xi_up = error_up_proj(eps)                        # project to level 2 space

# Level 2 receives error signal
z2 = backbone(z2 + level_emb + ts_emb + xi_up)   # error IS conditioning for level 2
```

Key changes from v4.1:
- Level 1 never receives `mu` as conditioning (the diagnostic proved this hurts)
- Precision weighting on the error signal is optional (running-statistics precision
  from v4.2 spec may be used instead of a learned precision network)
- Level 2 receives raw or precision-weighted error as its conditioning signal
- The prediction network's role is to compute what level 2 expected, so the error
  captures what level 2 needs to learn

### 2.3 Precision (Mechanism Changed)

**Previous approach (v4.1):** Learned precision network, backprop through pi.

**Revised approach (v4.2):** Running-statistics precision (EMA of prediction error
variance per dimension). No learnable parameters, no gradient involvement.

```python
class RunningPrecision:
    def __init__(self, dim, momentum=0.99, eps=0.01):
        self.ema_var = torch.ones(dim)
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

This sidesteps the precision collapse problem entirely. Precision is computed from
observed error statistics, not learned via backpropagation. It operates on a slower
timescale (EMA with momentum 0.99) and outside the computation graph.

Precision is used for:
- Weighting the error signal sent to level 2 (optional)
- Scaling noise in stochastic transitions (1/sqrt(pi))
- Monitoring convergence quality (high precision + high error = spurious fixed point)

### 2.4 Crystallisation (Unchanged in Concept, Reordered)

Crystallisation remains CORAL's key differentiator — no other architecture learns to
need less computation over time. But it is now the THIRD mechanism added, not the first.

Crystallisation requires:
- A backbone that can solve puzzles (so there are patterns to crystallise)
- Multiple training runs with the backbone producing codebook-worthy states
- A recognition network that has seen enough examples to judge confidence

None of these prerequisites exist until the backbone is proven.

### 2.5 Hierarchy (N=2, N=3, N=4)

Hierarchy is added AFTER the backbone works at N=1. The purpose of hierarchy is not
to improve per-segment accuracy (the backbone does that alone) but to:

1. Enable crystallisation at different abstraction levels (meta-strategy crystallises
   before cell-level solutions)
2. Provide better initialisation across segments via level 2+ strategic state
3. Enable the amortisation lifecycle (encounter → recognise → decide → compute →
   consolidate)

With the revised PC role (error communication only), the hierarchy doesn't interfere
with level 1's backbone processing. Level 2+ receives error signals, builds abstract
representations, and contributes only via better initialisation at the next segment
boundary (after detach).

---

## 3. Revised Training Depth Model

### 3.1 Per-Segment Inner Steps

| Config | Level 1 Steps | Level 2 Steps | Total | Use Case |
|--------|--------------|--------------|-------|----------|
| N=1, d=21 (TRM-matched) | 21 | - | 21 | Backbone validation |
| N=1, d=42 (deep) | 42 | - | 42 | Maximum training depth |
| N=2, T=5 | 25 | 5 | 30 | Hierarchy with deep L1 |
| N=2, T=7 | 49 | 7 | 56 | Deep hierarchy |

The `inner_steps_override` config field allows setting level 1 steps independently
of the timescale formula.

### 3.2 Memory Budget

Full backprop through 21 inner steps at batch=64, L=81, d=512, bfloat16:
- Per backbone application: ~5.3 MB activations
- 21 applications: ~111 MB
- With K_max=16 segments (detach between): only 1 segment live at a time
- Total peak: ~150 MB activations + ~25 MB parameters = well within A100 40GB
- Can increase to 42 inner steps without memory issues

### 3.3 Training Speed

At 21 inner steps per segment, 16 segments, batch=64:
- ~50 steps per 15 seconds on A100-SXM4
- 20K steps ≈ 100 minutes
- Full experiment cost: ~$0.80 on Vast.ai at $0.48/hr

---

## 4. Revised Experiment Sequence

### Experiment 1: Backbone Validation (N=1, d=21, No PC)
**Currently running.** Validates the shared backbone at TRM-matched depth.
Target: >70% token accuracy, >10% exact accuracy at 20K steps.

### Experiment 2: Deeper Backbone (N=1, d=42, No PC)
If Exp 1 plateaus below TRM's 87%, try doubling depth. Tests whether more
inner steps continue to help or hit diminishing returns.

### Experiment 3: Add Hierarchy (N=2, PC as Error Communication)
Add level 2 with revised PC (error communication only, no top-down conditioning).
Level 1 gets majority of inner steps. Test whether level 2 helps with harder puzzles.

### Experiment 4: Add Amortisation Pressure
With a model that solves puzzles, add L_amort to incentivise faster convergence.
Measure accuracy-depth Pareto curve.

### Experiment 5: Add Crystallisation
Add recognition network and codebooks. Measure whether the model learns to skip
computation for familiar patterns.

### Experiment 6: Add Stochastic Transitions
Precision-gated noise for escaping spurious fixed points. Multi-trajectory voting.

### Experiment 7: Full Ablation + ARC-AGI
Complete ablation matrix on Sudoku. Validate on ARC-AGI with variable grid sizes.

---

## 5. Config Changes

### New Config Fields
- `model.inner_steps_override: Optional[int] = None` — overrides computed level_steps[0]

### Changed Defaults
- `model.lambda_pred: 0.001` (was 0.1)
- `model.lambda_pi: 0.01` (was 0.01, temporarily changed during debugging, restored)
- Conditioning gate init: 0.01 (was 1.0)

### Unchanged
- All backbone parameters (2 layers, d=512, 8 heads, SwiGLU, RMSNorm, RoPE)
- Training parameters (lr=7e-5, AdamW, cosine schedule, weight_decay=1.0)
- Data pipeline (Sudoku-Extreme-1K, 1000× augmentation)
- Evaluation infrastructure (quick eval + Pareto eval)

---

## 6. Relationship to Original Spec

| Spec Section | Status | Notes |
|-------------|--------|-------|
| 1. Vision & Philosophy | Unchanged | Train-deep-infer-shallow strengthens the amortisation thesis |
| 2. Lessons from Field | Unchanged | TRM's depth finding now directly applied |
| 3. Architecture Overview | Updated | Build order changed, PC role changed |
| 4. Shared Backbone | Unchanged | Validated — works well |
| 5. Progressive Bottleneck | Unchanged | Applied in Experiment 3+ |
| 6. Predictive Coding | **Changed** | Error communication only, no L1 conditioning |
| 7. Crystallisation | Unchanged in concept | Reordered to Experiment 5 |
| 8. Precision-Driven Sparsity | **Changed** | Running statistics, not learned network |
| 9. Stochastic Transitions | Unchanged | Applied in Experiment 6 |
| 10. Adaptive Depth | Unchanged | |
| 11. Training Objective | Updated | lambda_pred reduced, precision reg updated |
| 12. Training Phases | **Changed** | New experiment sequence |
| 13. Amodal Interface | Unchanged | |
| 14. Deployment Profiles | Unchanged | |
| 15. Parameter Budget | Updated | inner_steps_override changes effective compute |
| 16. Experiment Plan | **Changed** | New sequence |
| 17. Theoretical Analysis | Unchanged | Free energy framework still applies |
| 18. Relationship to Prior Work | Unchanged | |
| 19. Patent Considerations | Updated | Add train-deep-infer-shallow claim |
