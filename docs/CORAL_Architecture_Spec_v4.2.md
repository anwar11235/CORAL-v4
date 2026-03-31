# CORAL v4.2 — Architecture Specification

## COrtical Reasoning via Abstraction Layers

### An Amodal Reasoning Core for Efficient, Deployable Intelligence

**Version 4.2 — March 2026**

**Author:** Muhammad Anwar Ul Haq

**Status:** CONFIDENTIAL / PRE-PATENT

---

## Document Purpose

This specification defines CORAL v4.2, a fundamental redesign informed by nine months
of empirical results (Phases 0–2), the competitive landscape (TRM, GRAM, RIM, Augmented
HRM, CMM, RSM), independent analyses (ARC Prize Foundation, Ge et al., Ren & Liu), and
the neuroscience of cortical microcircuits and neuromodulatory systems.

CORAL v4.2 supersedes v4.1. It preserves the core theoretical commitment — variational
free energy minimisation as the unifying principle — but restructures three key
mechanisms based on empirical failures and biological evidence:

1. **Precision** is reconceived as running statistics (neuromodulatory analogue) rather
   than a learned network (which collapses due to vanishing gradients through ε²).
2. **Crystallisation** uses convergence-driven triggers (speed cell analogue) and
   multi-headed semantic codebooks (grid cell analogue) rather than a learned
   recognition network.
3. **Precision-driven sparsity** (column heads) is dropped — it depended on learned
   precision that does not work.

The central insight remains: **the goal of a reasoning system is not to maximise
recursion depth but to minimise it**, progressively amortising expensive recurrent
computation into fast recognition while maintaining accuracy on novel problems.

---

## Table of Contents

1. Vision and Design Philosophy
2. Lessons from the Field
3. Architecture Overview
4. Component 1: Shared Backbone with Level Modulation
5. Component 2: Progressive Information Bottleneck
6. Component 3: Predictive Coding with Running-Statistics Precision
7. Component 4: Multi-Headed Semantic Codebooks
8. Component 5: Convergence-Driven Crystallisation
9. Component 6: Stochastic Variational Transitions
10. Component 7: Adaptive Depth Allocation
11. Unified Training Objective
12. Research Phases (Crystallisation-First)
13. Amodal Interface Protocol
14. Deployment Profiles
15. Parameter Budget
16. Theoretical Analysis
17. Relationship to Prior Work
18. Patent Considerations

---

## 1. Vision and Design Philosophy

### 1.1 The Problem

Current reasoning architectures face a fundamental tension. Large language models achieve
broad competence through massive parameterisation but reason inefficiently — externalising
thought as token sequences, consuming thousands of tokens per problem, and requiring
billions of parameters. Small recursive models (HRM, TRM) achieve remarkable accuracy
on structured reasoning tasks with far fewer parameters, but they do so through
brute-force iteration: applying the same tiny network hundreds of times, with no mechanism
to become more efficient over time.

Neither approach mirrors biological intelligence. The human brain does not generate
thousands of reasoning tokens, nor does it apply the same circuit 672 times in a tight
loop. It maintains a hierarchical generative model of the world, processes most inputs
through fast recognition (requiring 1–3 cortical passes), and reserves expensive
deliberative reasoning for genuinely novel situations — all within a 20-watt power budget.

### 1.2 The CORAL Thesis

CORAL's thesis is that a single computational principle — **variational free energy
minimisation** — instantiated across a multi-timescale recurrent hierarchy, produces an
architecture that:

1. **Reasons deeply when necessary**: novel problems trigger extended recurrence across
   multiple abstraction levels, with precision-weighted prediction errors focusing
   computation on what is surprising.

2. **Recognises quickly when possible**: familiar patterns are amortised into fast
   codebook lookups that bypass expensive recurrence, with the architecture explicitly
   learning to minimise its own computational cost.

3. **Improves its efficiency over time**: the free energy objective creates direct
   pressure to crystallise — because maintaining an unconverged representation across
   many recurrence steps incurs complexity cost.

4. **Operates as an amodal core**: the reasoning module never sees raw tokens, pixels,
   or sensor readings. It operates on embeddings in a modality-agnostic latent space,
   interfacing with the outside world through lightweight, task-specific encoder/decoder
   adapters.

### 1.3 Design Principles

**P1 — Amortisation First.** The architecture is designed around the lifecycle of a
computation: Encounter → Recognise → Decide → Compute → Consolidate. Every design
decision is evaluated against the question: "Does this help the system learn to need less
computation over time?"

**P2 — Hierarchical Generative Modelling.** The architecture maintains an explicit
generative model organised as a hierarchy of abstraction levels with progressively tighter
information bottlenecks. Higher levels encode more abstract, lower-dimensional
representations that predict the level below. Only precision-weighted prediction errors
propagate upward.

**P3 — Temporal Separation.** Different levels operate at geometrically separated
timescales. Higher levels update slowly (goals, strategies, constraints); lower levels
update rapidly (specific moves, value assignments). This is not arbitrary — it is what
the free energy framework prescribes when the generative model has multi-scale structure.

**P4 — Free Energy Minimisation.** All learning, inference, structural adaptation, and
depth allocation are driven by a single objective: minimise total variational free energy
across the hierarchy. Crystallisation, adaptive depth, and precision dynamics emerge as
consequences, not as engineered additions.

**P5 — Amodal Reasoning.** The reasoning core operates in a modality-agnostic embedding
space. It knows nothing about language, vision, or any specific sensor modality.
Interfacing with the world is the responsibility of lightweight, pluggable
encoder/decoder modules.

**P6 — Deployability as a Downstream Goal.** The architecture should be compact enough
to eventually run on mobile GPUs, edge TPUs, and embedded accelerators — but parameter
count is not a research-phase constraint. Compression for deployment comes later through
distillation, quantisation, and crystallised-only mode.

**P7 — Separation of Concerns.** Prediction, error computation, precision estimation,
and crystallisation are implemented by separate mechanisms with separate learning rules,
mirroring the brain's separation of cortical microcircuits (prediction/error) from
neuromodulatory systems (precision) from hippocampal consolidation (crystallisation).
A single gradient should not be asked to learn all of these simultaneously.

**P8 — Convergence as the Primary Signal.** The decision to crystallise is driven by
observed convergence dynamics (rate of state change), not by a learned gating network.
This mirrors the brain's use of speed cells and error-detection neurons — hardwired
monitors of processing dynamics rather than learned classifiers.

**P9 — Compositionality of Recognition.** Crystallised representations are composed
from independent semantic factors (multi-headed codebooks), not stored as monolithic
patterns. This mirrors the brain's factored representation: place cells compose from
grid cells at multiple scales; concept cells respond to specific conjunctions of
features rather than holistic templates.

### 1.4 What CORAL Is Not

CORAL is not a language model. It does not generate text, predict next tokens, or
maintain a vocabulary. It is a reasoning engine that takes structured embeddings as input
and produces refined embeddings as output.

CORAL is not a replacement for LLMs. It is a complementary module that provides the
kind of structured, multi-step reasoning that LLMs struggle with (constraint
satisfaction, planning, abstract pattern recognition), while LLMs provide the broad
knowledge and linguistic competence that CORAL does not attempt.

---

## 2. Lessons from the Field

### 2.1 Iteration Depth Is the Primary Driver

**Finding:** Deep recursive refinement with weight sharing is the dominant factor in
reasoning accuracy. A tiny network applied many times beats a big network applied once.

**Sources:** ARC Prize Foundation (Aug 2025), TRM (Jolicoeur-Martineau, Oct 2025),
Ge et al. (Sep 2025).

**Implication:** CORAL must support deep recursion. However, the goal is to learn to
need *less* of it, not to maximise it.

### 2.2 The H/L Hierarchy — As Implemented — Is Marginal

**Finding:** Replacing HRM's two separate H/L modules with a single expanded transformer
that updates at every timestep produces similar performance.

**Sources:** ARC Prize Foundation, Ge et al. (8-layer L-only HRM), TRM (single network).

**Implication:** The hierarchy must create genuine abstraction pressure through
progressive dimensionality reduction and precision-weighted communication.

### 2.3 Full Backpropagation Through Recursion Is Critical

**Finding:** TRM's single largest improvement came from backpropagating through the full
recursion process rather than using the 1-step gradient approximation (56.5% → 87.4%).

**Source:** TRM (Table 1 ablation).

**Implication:** CORAL uses full backpropagation through the inner loop within each
deep-supervision segment. Between segments, states are detached.

### 2.4 Smaller Networks + More Recursion Beats Larger Networks

**Finding:** A 2-layer network with n=6 recursions outperforms a 4-layer network with
n=3 recursions at matched effective depth.

**Source:** TRM (Table 1: 4-layers 79.5% vs 2-layers 87.4%).

**Implication:** CORAL uses a shared 2-layer backbone. Capacity comes from recursion
depth and hierarchical structure, not from wider or deeper individual networks.

### 2.5 MoE-Style Routing Collapses

**Finding:** Both TRM (MoE) and CORAL Phase 2 (columnar routing) observed severe
performance degradation from mixture-of-experts approaches.

**Sources:** TRM (Section 6), CORAL Phase 2 (14.7% with column collapse).

**Implication:** CORAL v4.2 avoids explicit routing entirely. Column heads (from v4.1)
are dropped.

### 2.6 HRM Gets Trapped in Spurious Fixed Points

**Finding:** HRM converges to incorrect solutions (spurious attractors) and cannot escape
them.

**Source:** Ren & Liu, "Are Your Reasoning Models Reasoning or Guessing?" (Jan 2026).

**Implication:** Stochastic variational transitions with precision-gated noise provide
a mechanism for escaping spurious attractors.

### 2.7 Stochastic Transitions Improve Accuracy

**Finding:** GRAM achieved 97.0% on Sudoku-Extreme (vs TRM's 87.4%) by injecting
stochastic guidance at each recursion step.

**Source:** GRAM (ICLR 2026 Workshop).

**Implication:** CORAL incorporates stochasticity derived from the variational framework,
with noise magnitude inversely proportional to precision.

### 2.8 Precision-Weighted Predictive Coding Works — But Learned Precision Does Not

**Finding:** Replacing HRM's raw state passing with precision-weighted prediction errors
yields a 48% relative improvement (41.2% → 61.1%). However, eight attempts to learn
per-dimension precision via backpropagation in v4 all failed due to vanishing gradients
(dL/dπ ∝ ε², which vanishes when prediction error collapses).

**Sources:** CORAL Phase 1 (W&B run mfno8t1y), CORAL v4 handoff notes (8 attempts
documented).

**Implication:** The predictive coding mechanism works. Precision must be implemented
through a separate pathway — running statistics rather than a learned network — mirroring
the brain's separation of neuromodulatory precision from cortical prediction circuits.

### 2.9 Data Augmentation Is Essential

**Finding:** Augmentation is critical for generalisation. Without it, all models overfit
catastrophically.

**Sources:** HRM, TRM, CORAL Phases 0–1.

**Implication:** Continue using SudokuAugmenter (1000× augmentation) for Sudoku.

### 2.10 Recursive Inference Machines Extend TRM

**Finding:** RIM (March 2026) formalises TRM within an inference machine framework and
shows that adding a reweighting component improves performance on ARC-AGI and Sudoku.

**Source:** RIM (arXiv 2603.05234).

**Implication:** The field is converging on principled extensions of recursive reasoning.
CORAL's contribution is the amortisation/crystallisation axis, which RIM does not address.

---

## 3. Architecture Overview

### 3.1 System Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │            AMODAL REASONING CORE            │
                    │                                             │
                    │  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
                    │  │ Level 4 │→→│ Level 3 │→→│ Level 2 │→→  │
                    │  │ d=64    │  │ d=128   │  │ d=256   │    │
                    │  │ Meta-   │  │ Strate- │  │ Tacti-  │    │
                    │  │ strate- │  │ gic     │  │ cal     │    │
                    │  │ gic     │  │         │  │         │    │
                    │  └────┬────┘  └────┬────┘  └────┬────┘    │
                    │       │pred        │pred        │pred      │
                    │       ↓ξ↑          ↓ξ↑          ↓ξ↑        │
                    │  ┌─────────────────────────────────────┐   │
                    │  │          Level 1 (d=512)            │   │
                    │  │       Solution / Detail State       │   │
                    │  │  [8-head semantic codebook per pos] │   │
                    │  └─────────────────────────────────────┘   │
                    │       ↑                         ↓          │
                    └───────┼─────────────────────────┼──────────┘
                            │                         │
                    ┌───────┴───────┐         ┌───────┴───────┐
                    │ Input Adapter │         │Output Adapter  │
                    └───────┬───────┘         └───────┬───────┘
                            │                         │
              ┌─────────────┼─────────────┐           │
              │             │             │           │
         ┌────┴────┐  ┌────┴────┐  ┌────┴────┐     Output
         │Language │  │ Vision  │  │  Grid   │   (task-specific)
         │Encoder  │  │Encoder  │  │Encoder  │
         └─────────┘  └─────────┘  └─────────┘
```

### 3.2 Component Summary

| Component | Est. Parameters | Role |
|-----------|----------------|------|
| Shared Backbone | ~6.3M | Single 2-layer transformer, weight-shared across all levels |
| Level Embeddings | ~4K | Tells the backbone which abstraction level it is operating at |
| Prediction Networks | ~0.4M | Per-level MLPs: project from level l+1 to level l |
| Multi-Headed Codebooks | ~16K/level | H=8 heads × M=32 entries × 64-d per level |
| Halting Network | ~10K | Q-learning head for adaptive depth |
| Running Precision | 0 | EMA statistics, no learnable parameters |
| Convergence Monitor | 0 | Velocity tracking, no learnable parameters |
| **Reasoning Core Total** | **~7.2M** | |
| Input Adapter (task-specific) | 0.5–1M | Projects from source modality to d₁=512 |
| Output Adapter (task-specific) | 0.5–1M | Projects from d₁=512 to task output space |
| **Deployed Total** | **~8–9M** | |

### 3.3 Forward Pass — High-Level Pseudocode

```
def coral_forward(x_emb, K_max, N_levels=4):
    """
    x_emb: [B, L, d1] — input embeddings from adapter
    Returns: [B, L, d1] — refined solution state
    """
    # Initialise level states and tracking
    z = {1: x_emb}
    for l in 2..N:
        z[l] = project_down(z[l-1])
    z_prev = copy(z)                    # for velocity tracking
    crystallised = zeros(N, L, H, bool) # per-level, per-position, per-head

    # Deep supervision segments
    for segment in 1..K_max:

        # --- Top-down pass: predictions flow downward ---
        for l in (N-1) down to 1:
            mu[l] = predict(z[l+1], level=l)

        # --- Bottom-up pass: recurrence + error propagation ---
        for l in 1..N:
            # Recurrence (full backprop within segment)
            for t in 1..T_steps[l]:
                z[l] = backbone(z[l], level_emb=l, timescale_emb=t)
                z[l] = z[l] + cond_gate[l] * (mu[l] - z[l])  # residual correction

                # Enforce crystallisation: overwrite frozen heads
                for h in 1..H:
                    if crystallised[l, :, h]:
                        z[l][:, h_dims] = codebook_values[l, :, h]

            # Compute prediction error and precision-weighted error
            eps[l] = z[l] - mu[l]
            pi[l] = running_precision[l].precision  # no gradient, slow timescale
            xi[l] = pi[l] * eps[l]
            running_precision[l].update(eps[l])     # update EMA

        # --- Convergence check (per-position, per-head) ---
        for l in 1..N:
            for h in 1..H:
                velocity = ||z[l][:, h_dims] - z_prev[l][:, h_dims]||
                if velocity < tau_converge for N_stable consecutive:
                    snap to nearest codebook entry, set crystallised[l, :, h] = True

        # --- De-crystallisation check ---
        for each crystallised head:
            drift = ||backbone_proposed - codebook_frozen||
            if drift > tau_decrystallise:
                unfreeze head

        # --- Halting check ---
        h_k = halt_net(concat(z[1], ..., z[N]))
        if eval and h_k > 0.95:
            break

        # --- Codebook update (EMA, during training) ---
        update_codebooks_ema(z, crystallised)

        # --- Store for velocity tracking, detach for next segment ---
        z_prev = copy(z)
        z = {l: z[l].detach() for l in 1..N}

    return z[1]
```

---

## 4. Component 1: Shared Backbone with Level Modulation

### 4.1 Motivation

TRM demonstrated that a single 2-layer network outperforms two separate 4-layer networks
(87.4% vs 55.0% on Sudoku-Extreme) while using half the parameters. Weight sharing across
recursion steps provides implicit regularisation that prevents overfitting on small
training sets.

CORAL extends this principle: a single backbone is shared not only across recursion
steps but across hierarchy levels.

### 4.2 Architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Layers | 2 | TRM finding: 2 layers optimal |
| Hidden dim | 512 | Matches d₁ (solution level) |
| Attention | Self-attention with 8 heads | Required for variable sequence lengths |
| FFN expansion | 4× (SwiGLU) | Following HRM/TRM |
| Normalisation | RMSNorm (post-norm) | Following HRM |
| Position encoding | Rotary (RoPE) | Supports variable lengths |
| Level embedding | Additive, learned | Tells backbone which level |
| Timescale embedding | Sinusoidal | Encodes step index within level |
| Local attention bias | 3 learned scalars | Upweight same-row, same-column, same-box pairs (Sudoku) |

**Local attention bias (new in v4.2):** Three learned scalar biases added to attention
logits for same-row, same-column, and same-box position pairs. This nudges
representations toward local-context encoding without constraining global attention,
encouraging representations that cluster by local constraint pattern — which aids
codebook formation.

### 4.3 Level Modulation

For inputs at level l with dimension d_l < 512, the input is projected up to d=512
via a learned linear projection before entering the backbone, and the output is
projected back down to d_l.

```
backbone_input = W_up[l] @ z[l] + level_emb[l] + timescale_emb[t]
backbone_output = backbone(backbone_input)
z[l]_updated = W_down[l] @ backbone_output
```

---

## 5. Component 2: Progressive Information Bottleneck

### 5.1 Level Configuration

| Level | Dimension (d_l) | Update Period | Representation Content |
|-------|-----------------|---------------|----------------------|
| 1 (fastest) | 512 | Every step | Solution state: cell values, path segments |
| 2 | 256 | Every T steps | Tactical: local constraint patterns, subgoal progress |
| 3 | 128 | Every T² steps | Strategic: which region to work on, which strategy |
| 4 (slowest) | 64 | Every T³ steps | Meta-strategic: when to backtrack, problem classification |

With T=3: total inner steps per segment = 1 + 3 + 9 + 27 = 40 backbone applications
(with full backpropagation).

### 5.2 Inter-Level Projection

Each level l+1 communicates with level l through a prediction network:

```
f_pred[l](z_{l+1}) = W2 · GELU(W1 · z_{l+1} + b1) + b2
```

The reverse direction (error propagation upward) uses a separate bias-free linear
projection to ensure zero error maps to zero update.

### 5.3 Why Progressive Reduction

1. **Forces abstraction.** Level 4 at d=64 cannot represent individual cell values —
   it must encode abstract properties.
2. **Reduces computation at higher levels.** Cost is dominated by level 1.
3. **Creates meaningful prediction error.** The error captures what the higher level
   does not yet understand.

---

## 6. Component 3: Predictive Coding with Running-Statistics Precision

### 6.1 Core Mechanism

At each level l, the communication protocol between adjacent levels follows predictive
coding dynamics:

**Top-down prediction:** Level l+1 generates a prediction of level l's state:
```
μ_l = f_pred[l](z_{l+1})
```

**Prediction error:** The discrepancy between prediction and reality:
```
ε_l = z_l - μ_l
```

**Precision-weighted error:** Only unexpected, reliable information propagates upward:
```
ξ_l = π_l ⊙ ε_l
```

**Post-backbone residual conditioning:** Predictions integrate with the backbone via
residual correction (validated in v4 handoff notes):
```
z_new = backbone(z) + cond_gate * (mu - z)
```

### 6.2 Running-Statistics Precision

Precision is NOT a learned network. It is an exponential moving average of prediction
error variance per dimension, with no learnable parameters:

```python
class RunningPrecision:
    """
    Neuromodulatory analogue: tracks error statistics on a slow timescale.
    No learnable parameters. No gradient involvement.
    """
    def __init__(self, dim, momentum=0.99, eps=0.01):
        self.ema_var = torch.ones(dim)
        self.momentum = momentum
        self.eps = eps

    @torch.no_grad()
    def update(self, prediction_error):
        """Called once per segment, not per inner step (slow timescale)."""
        batch_var = prediction_error.var(dim=(0, 1))
        self.ema_var = self.momentum * self.ema_var + (1 - self.momentum) * batch_var

    @property
    def precision(self):
        return 1.0 / (self.ema_var + self.eps)
```

### 6.3 Biological Basis for Running-Statistics Precision

In the brain, precision is mediated by neuromodulatory systems that are architecturally
separate from cortical prediction circuits:

**Acetylcholine** (from basal nucleus of Meynert) modulates gain of sensory prediction
errors. High ACh = trust your senses more than your priors.

**Dopamine** (from VTA/substantia nigra) modulates precision of higher-level prediction
errors. Dysfunction leads to psychosis via aberrant precision-weighting.

**Noradrenaline** (from locus coeruleus) modulates precision of transition probabilities
— how predictable the world is.

Key shared properties:
- **Separate pathway** from cortical prediction circuits
- **Slow timescale** (hundreds of milliseconds, not per-pass)
- **Spatially coarse** (per-region, not per-synapse)
- **Different learning rule** (reward/volatility, not prediction-error gradient)

Running-statistics precision mirrors all four properties: it operates outside the
computation graph, updates slowly via momentum, and can be computed per-head (coarse)
rather than per-dimension.

### 6.4 Per-Head Precision

With H=8 multi-headed codebooks (Section 7), precision is also computed per-head:

```python
for h in range(H):
    head_dims = slice(h * d_head, (h + 1) * d_head)
    precision_per_head[h] = 1.0 / (ema_var[head_dims].mean() + eps)
```

Per-head precision (8 values) is used for noise gating in stochastic transitions
and for monitoring which semantic aspects are resolved vs uncertain.

### 6.5 Precision Dynamics: What We Learned from v4.1

From Phase 1 training on Sudoku-Extreme-1K (v3, separate H/L modules):
- Steps 0–2,500: High prediction error (~25), moderate precision (~0.65)
- Steps 2,500–3,000 (phase transition): Error collapses, precision spikes
- Steps 3,000–52,000: Error <1, precision settles to ~0.04

This "Cornsweet mechanism" is an interpretable signal. In v4.2 with running-statistics
precision, we expect analogous dynamics: early in training, high error variance → low
precision. As the model learns to predict, error variance drops on solved dimensions →
precision increases on those dimensions. The running-statistics approach should produce
more stable, differentiated precision than the learned network, which collapsed to
uniform.

### 6.6 Historical Note: Why Learned Precision Failed

Eight experimental attempts documented in the v4 handoff notes established that
precision cannot be learned via backpropagation through the prediction error pathway.
The gradient is dL/dπ ∝ ε², and once prediction error collapses (ε → 0 by step ~300),
precision loses all gradient signal. The regulariser then pulls precision to uniform
(π = 1 everywhere).

This is not a hyperparameter problem — it is a gradient architecture problem.
The brain's solution: use a separate pathway (neuromodulation) with a separate
learning rule (volatility tracking). CORAL v4.2 follows this solution.

---

## 7. Component 4: Multi-Headed Semantic Codebooks

### 7.1 Motivation

The v4.1 design specified a single codebook with M=256 entries of dimension d_l per
hierarchy level. This had three limitations: insufficient capacity (256 distinct states),
all-or-nothing crystallisation granularity, and opacity to analysis. Multi-headed
codebooks solve all three.

### 7.2 Biological Grounding

**Grid cells** provide the precedent for multi-scale factored representation. A specific
location in space is represented by the intersection of multiple grid cell modules at
different spatial frequencies — not by a single monolithic code. The same principle
applies: a reasoning state is represented by the intersection of multiple codebook
modules, each capturing a different semantic aspect.

**Concept cells** provide the precedent for codebook entries: stable, discrete, amodal
representations that activate when a specific high-level pattern is recognised. Fast,
sparse, stable.

**Border cells** motivate a constraint-topology head: encoding structural features of
the constraint environment rather than solution values.

**Context neurons** motivate a reasoning-phase head: encoding the current rule set or
problem-level context that gates lower-level processing.

### 7.3 Architecture

```
Full state:     z ∈ R^{512}
Split into:     z = [z_1, z_2, ..., z_H]  where z_h ∈ R^{d_head}
                H = 8 heads, d_head = 64

Per-head codebook:  C_h ∈ R^{M_h × d_head}   with M_h = 32 entries

Effective capacity: 32^8 ≈ 10^12 composite states
Parameter cost:     8 × 32 × 64 = 16,384 parameters per level
```

### 7.4 Suggested Head Semantics

Heads are not hard-wired to specific roles. The disentanglement regulariser encourages
specialisation; the specific semantics emerge from training. Expected roles:

| Head | Expected Role | Brain Analogue |
|------|--------------|----------------|
| 1 | Value identity | Concept cells |
| 2 | Row context | Grid cells (one axis) |
| 3 | Column context | Grid cells (other axis) |
| 4 | Box/region context | Place cells |
| 5 | Constraint density | Border cells |
| 6 | Solution progress | Time cells |
| 7 | Reasoning phase | Context neurons |
| 8 | Residual | — |

### 7.5 Disentanglement Regulariser

Without explicit pressure, heads may learn redundant representations. Following FQGAN:

```
L_dis = λ_dis · Σ_{h1 ≠ h2} ||C_{h1}^T · C_{h2}||²_F / (M_{h1} · M_{h2})
```

This encourages codebook subspaces to be approximately orthogonal. λ_dis = 0.01.

### 7.6 Codebook Management

**Initialisation.** After Phase 1 baseline trains to convergence:
1. Collect ~50K late-segment states (segment 16) from empty positions
2. Split each into H=8 groups of 64 dimensions
3. Run k-means independently per group with k=32
4. Use centroids to initialise codebooks

**Update rule.** EMA: `C_h[j] ← 0.99 · C_h[j] + 0.01 · mean(assigned states)`

**Dead-code restart.** Every 1000 steps, replace unused entries with random samples from
a buffer of recent states.

**Assignment.** Training: Gumbel-Softmax soft assignment (temperature annealed 1.0 → 0.1).
Inference: hard nearest-neighbour.

**Commitment loss.** `L_commit = 0.25 · Σ_h ||z_h - sg(e_h_nearest)||²`

**Health metric.** Per-head codebook perplexity. Healthy: >8 (≥25% of entries used).
If perplexity drops below 4, increase dead-code restart frequency.

---

## 8. Component 5: Convergence-Driven Crystallisation

### 8.1 Central Role

Crystallisation is CORAL's most distinctive mechanism. All existing recursive reasoning
models (HRM, TRM, GRAM, RIM) treat every problem instance identically: run the full
recursion pipeline at maximum depth. There is no mechanism to become more efficient
over time.

CORAL's crystallisation provides this mechanism through convergence-driven adaptive
depth allocation, operating per-position, per-head.

### 8.2 Biological Grounding

**Speed cells** (entorhinal cortex) fire at rates proportional to velocity — they track
the rate of change of the navigational state. CORAL analogue: track the rate of change
of the reasoning state at each position.

**Error-detection neurons** (anterior cingulate cortex) fire the moment the brain
detects an internal mismatch — before external feedback. CORAL analogue: detect when a
crystallised state diverges from what the backbone would produce.

### 8.3 The Convergence Monitor

For each position i, head h, at each segment k:

```
velocity_h[i, k] = ||z_h[i, k] - z_h[i, k-1]||₂
```

No parameters, no gradient — just a norm on states that already exist.

**Crystallisation trigger:** Head h at position i crystallises when:
```
velocity_h[i, k] < τ_converge   for N_stable consecutive segments
```

When triggered:
1. Find nearest codebook entry: `e_h = argmin_j ||z_h[i] - C_h[j]||`
2. Snap to codebook: `z_h[i] ← C_h[e_h]`
3. Freeze: overwrite with codebook value after each subsequent backbone pass

Defaults: τ_converge = 0.01 (relative to initial state norm), N_stable = 2.

### 8.4 Partial Crystallisation

With multi-headed codebooks, crystallisation is partial — per-head, per-position. A
position might have heads 1, 2, 4, 7 crystallised (value, row context, box context,
reasoning phase resolved) while heads 3, 5, 6, 8 continue to be refined.

During the backbone pass, all 512 dimensions are processed. After each pass:
- Crystallised heads: overwrite with codebook values (frozen)
- Active heads: keep the backbone's output (continue refining)

This is closer to human cognition: you resolve "it must be one of {3, 7}" early
(partial digit crystallisation), then "it's 7 because of the column" later (full digit
crystallisation). Reasoning is incremental and aspect-wise, not all-or-nothing.

### 8.5 De-Crystallisation (Error Detection)

Premature crystallisation is the primary failure mode. The error detector:

```
For each crystallised head h at position i:
    z_h_proposed = backbone_output_for_head_h[i]  # what backbone would produce
    z_h_frozen = codebook_entry[h, i]              # what we froze to

    drift = ||z_h_proposed - z_h_frozen||₂

    if drift > τ_decrystallise:
        crystallised[h, i] = False    # unfreeze
        z_h[i] = z_h_proposed         # use backbone's value
```

Default: τ_decrystallise = 5 × τ_converge (hysteresis prevents oscillation).

### 8.6 Training Protocol

**During training:** Crystallisation is active. However, the task loss is always
computed from the full-recursion state to ensure the backbone continues to receive
learning signal for all positions. Crystallisation provides learning signal for the
codebook (via EMA) and convergence/de-crystallisation statistics for monitoring.

**Bypass safety signal (monitoring, not training):**
```
answer_full = decode(z_final_full_recursion)
answer_crystal = decode(z_crystallised)
bypass_safe = (answer_full == answer_crystal)  # task-accuracy criterion
```

This is sharper than distance-based criteria: two representations can be far apart in
L2 but decode to the same answer because the decoder is many-to-one.

### 8.7 The Amortisation Lifecycle (Revised)

```
┌──────────────────────────────────────────────────────────────┐
│                    AMORTISATION LIFECYCLE                     │
│                                                              │
│  1. ENCOUNTER                                                │
│     New input arrives. All heads active, nothing crystallised │
│                                                              │
│  2. REASON                                                   │
│     Backbone iterates. Convergence monitor tracks velocity   │
│     per position, per head.                                  │
│                                                              │
│  3. CRYSTALLISE (incremental)                                │
│     Heads that converge (low velocity for N_stable segments) │
│     snap to nearest codebook entry and freeze.               │
│     Easy heads freeze early. Hard heads keep computing.      │
│                                                              │
│  4. MONITOR                                                  │
│     Error detector watches backbone's proposed updates to    │
│     frozen heads. Large drift → de-crystallise (unfreeze).   │
│                                                              │
│  5. CONSOLIDATE                                              │
│     Codebook entries update via EMA from frozen states.      │
│     Over training: codebook improves, more heads crystallise │
│     earlier, average depth decreases, inference speeds up.   │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Component 6: Stochastic Variational Transitions

### 9.1 Stochastic Update Rule

```
z_h^{t+1} = z_h^t + η · f_update(z_h^t, ξ) + σ · (1 - h_k) · ε_h / sqrt(π_h)
```

where:
- σ is a learned per-level noise scale (init 0.01)
- (1 - h_k) gates noise by convergence: near-converged → suppress noise
- ε_h ~ N(0, I) is standard Gaussian noise per head
- 1/√π_h is the per-head running-statistics precision: uncertain heads get more noise

### 9.2 Per-Head Noise

With per-head precision:
- High-precision heads (low error variance): minimal noise, deterministic refinement
- Low-precision heads (high error variance): significant noise, exploration

This targets exploration exactly where the model is most uncertain.

### 9.3 Multi-Trajectory Inference

**Fast mode (single trajectory):** σ = 0. Minimal latency. For real-time applications.

**Deliberate mode (K trajectories):** Sample K trajectories with noise, majority-vote
per position. Trades K× compute for higher accuracy on hard problems.

---

## 10. Component 7: Adaptive Depth Allocation

### 10.1 Three Tiers of Adaptation

**Tier 1 — Segment-level halting (coarsest).** Q-learning halting network evaluates
global state and decides whether to run another segment.

**Tier 2 — Per-head crystallisation (medium).** Individual heads at individual positions
crystallise independently based on convergence dynamics. A position with 6/8 heads
crystallised is "mostly resolved."

**Tier 3 — Per-position masking (finest).** When ALL heads at a position are crystallised,
the position is fully resolved and can be removed from attention computation in
subsequent segments.

### 10.2 Amortisation Pressure

```
L_amort = λ_amort · Σ_{k=2}^{K_max} Σ_{i=1}^{L} ||z[i,k] - z[i,k-1]||²
```

A crystallised position contributes zero (its state doesn't change). More
crystallisation → lower L_amort → lower total loss. This creates direct gradient
pressure to produce converging representations.

Note: L_amort penalises *cumulative state change*, not *number of steps*. Step-count
penalties can be gamed (per ACT literature). State-change penalties cannot.

### 10.3 Accuracy-Depth Pareto Testing Protocol

At every evaluation step, measure accuracy at forced depth limits K ∈ {1, 2, 4, 8, 16}.
The Pareto area (normalised area under accuracy-depth curve) is the single scalar
summary of amortisation quality.

---

## 11. Unified Training Objective

### 11.1 The Complete Loss

```
L = L_task                                              # stablemax CE, float64
  + λ_pred · mean(π · ε²)                              # prediction error, π from
                                                        #   running stats (constant
                                                        #   for gradient purposes)
  + λ_amort · Σ_{k,i} ||z[i,k] - z[i,k-1]||²          # amortisation pressure
  + λ_commit · Σ_h ||z_h - sg(e_h_nearest)||²          # per-head codebook commitment
  + λ_dis · Σ_{h1≠h2} ||C_{h1}^T · C_{h2}||²_F        # head disentanglement
  + L_halt                                              # Q-learning halting
```

**Removed from v4.1:**
- λ_π precision regulariser (no learnable precision)
- λ_crystal crystallisation BCE (no recognition network)

**Added in v4.2:**
- λ_dis disentanglement regulariser

### 11.2 Hyperparameter Defaults

| Parameter | Default | Status | Notes |
|-----------|---------|--------|-------|
| λ_pred | 0.001 | From v4 handoff | Reduced from 0.1 |
| λ_amort | 0 → 0.01 (annealed) | Tunable | Pareto protocol |
| λ_commit | 0.25 | Default | Standard VQ-VAE |
| λ_dis | 0.01 | Default | Orthogonality push |
| τ_converge | 0.01 | Tunable | Velocity threshold |
| τ_decrystallise | 0.05 | Tunable | 5× crystallisation threshold |
| N_stable | 2 | Default | Consecutive converged segments |
| H (codebook heads) | 8 | Default | 512/8 = 64-d per head |
| M_h (entries/head) | 32 | Default | 32^8 ≈ 10^12 capacity |
| γ (codebook EMA) | 0.99 | Default | Standard VQ-VAE |
| Precision momentum | 0.99 | Default | ~100 update window |
| ε_min | 0.01 | From Phase 1 | Precision floor |
| K_max | 16 | Default | Deep supervision segments |
| T (timescale) | 3 | **Committed** | 40 inner steps per segment at N=4 |
| Backbone layers | 2 | **Committed** | TRM finding |
| Backbone dim | 512 | **Committed** | Matching d₁ |
| Attention heads | 8 | Default | d_k = 64 |
| FFN expansion | 4× (SwiGLU) | Default | Standard |
| Learning rate | 7e-5 | From Phase 1 | AdamW, cosine schedule |
| Weight decay | 1.0 | From Phase 1 | Following HRM |
| Batch size | 64 | Default | Adjust based on memory |
| Gradient clipping | 1.0 | Default | Max norm |
| Forward precision | bfloat16 | **Committed** | |
| Loss precision | float64 | **Committed** | Numerical stability |

---

## 12. Research Phases (Crystallisation-First)

The research sequence validates the most uncertain components first, with crystallisation
(the core differentiator) front-loaded rather than deferred.

### Phase 1: Baseline Without Predictive Coding

**Duration:** 1–2 days. **Compute:** ~4h A100.

N=1, no PC, no precision, no codebooks. Pure backbone + deep supervision + halting.

**Representation diagnostics logged during training:**
- Inter-position cosine similarity (within puzzle, segment 16)
- Same-digit cross-puzzle cosine similarity
- Representation effective rank (PCA, 90% variance)

**Pass criterion:** ≥70% eval exact accuracy.

### Phase 2: Codebook Formation Study

**Duration:** 3–5 days. **Compute:** ~8h A100.

Observational study. No new training.

**Protocol:**
1. Collect ~50K segment-16 states from empty positions
2. Per-head k-means (8 groups × 64-d, k ∈ {16, 32, 64, 128})
3. Whole-vector k-means (k ∈ {64, 128, 256, 512})
4. HDBSCAN on PCA-reduced states

**Key metric:** Per-head bypass accuracy. Replace each head's state with nearest
centroid, decode, check if answer matches full-recursion answer.

**Pass criterion:** Per-head bypass accuracy ≥80% on easy cells at k=32.

### Phase 3: Crystallisation Prototype

**Duration:** 1–2 weeks. **Compute:** ~20h A100.

Three incremental versions:

**Version A — Simplest:** Convergence monitor + monolithic codebook (no multi-head yet).
Pass: crystallisation rate >10%, accuracy within 3%.

**Version B — Multi-headed:** H=8 factored codebook, per-head convergence, partial
crystallisation, orthogonality regulariser.
Pass: heads specialise, accuracy ≥ Version A.

**Version C — Error detection:** De-crystallisation when backbone drift exceeds threshold.
Pass: catches premature crystallisation, reduces accuracy gap.

### Phase 4: Amortisation Pressure

**Duration:** 3–5 days. **Compute:** ~12h A100.

Add L_amort. Three-way comparison. Produce Pareto curves.

**Pass criterion:** Pareto dominance.

### Phase 5: Hierarchy as Earned Complexity

**Duration:** 1–2 weeks. **Compute:** ~20h A100.

Add N=2 with predictive coding + running-statistics precision. Crystallisation at both
levels independently.

**Key diagnostics:**
- Abstraction probes (linear classifiers on L1 vs L2 for fine-grained vs abstract tasks)
- Differential crystallisation (higher levels should crystallise faster)

### Phase 6: Precision Integration

**Duration:** 1–2 weeks. **Compute:** ~15h A100.

Compare: no precision, per-dim running stats, per-head running stats,
task-loss-modulated precision.

### Phase 7: Stochastic Transitions

**Duration:** 3–5 days. **Compute:** ~10h A100.

Per-head precision-gated noise. Single-pass and multi-trajectory evaluation.

### Phase 8: Cross-Task Validation (ARC-AGI)

**Duration:** 2–3 weeks. **Compute:** ~3 days on 4× H100.

Key question: does crystallisation generalise to novel tasks?

### Compute Budget

| Phase | GPU Hours | Running Total |
|-------|-----------|---------------|
| Phase 1 | 4 | 4 |
| Phase 2 | 8 | 12 |
| Phase 3 | 20 | 32 |
| Phase 4 | 12 | 44 |
| Phase 5 | 20 | 64 |
| Phase 6 | 15 | 79 |
| Phase 7 | 10 | 89 |
| Phase 8 | 72–288 | 161–377 |
| **Pre-ARC total** | **~89h** | **~$45 at Vast.ai A100 rates** |

---

## 13. Amodal Interface Protocol

### 13.1 Design Philosophy

The reasoning core is modality-agnostic. It operates on sequences of embedding vectors
and returns refined sequences of embedding vectors. Everything specific to a modality
lives in the adapter layer.

### 13.2 Input/Output Contract

```
Input:  z₁⁰ ∈ R^{B × L × 512}
Output: z₁_final ∈ R^{B × L × 512}
```

### 13.3 Adapter Specifications

**Grid Tasks (Sudoku, ARC-AGI, Mazes):**
Token embedding + 2D position embedding → d₁=512. Decoder: linear projection to
vocabulary size. ~0.3M parameters each.

**Language (LLM interface):**
Projection from LLM hidden states (d_LLM) to d₁=512. Residual addition back to LLM
states. ~0.5M parameters. LLM backbone frozen.

**Vision (ViT interface):**
Projection from patch features to d₁=512. Spatial position embeddings. ~0.3M parameters.
Vision backbone frozen.

**Robotics (sensor fusion):**
Projection from fused sensor state to d₁=512. Entity-type embeddings. Action + value
heads. ~0.5M parameters.

**Multi-modal:** Concatenate adapter outputs along sequence dimension. Cross-modal
attention happens naturally within the backbone.

---

## 14. Deployment Profiles

| Profile | Hardware | Model | Latency | Mode |
|---------|----------|-------|---------|------|
| Cloud (research) | A100 | Full core (~7.2M, FP16) | N/A | Deliberate (K=16) |
| Cloud (production) | T4/L4 | Full core (~7.2M, FP16) | <500ms | Fast (K=1–4) |
| Mobile | Snapdragon NPU | Distilled (INT8) | <200ms | Fast (K=1) |
| Edge (robotics) | Jetson Orin | Full core (FP16) | <100ms | Fast (K=1) |
| Embedded (IoT) | Cortex-M/RISC-V | Codebook-only (~20K, INT4) | <50ms | Lookup only |

**Crystallised-only mode (v4.2 update):** For embedded deployment, only the multi-headed
codebooks are deployed (8 heads × 32 entries × 64-d per level = ~16K parameters per
level). No backbone, no prediction networks. Each position is looked up from the
codebook via nearest-neighbour. Positions with low per-head match confidence are flagged
for offload to a more capable device.

---

## 15. Parameter Budget

### 15.1 Shared Backbone (2-layer Transformer, d=512)

| Sub-component | Parameters |
|---------------|-----------|
| Self-attention Q/K/V (×2 layers) | 1.57M |
| Self-attention output (×2 layers) | 0.52M |
| SwiGLU FFN (×2 layers) | 4.19M |
| Local attention bias | 3 |
| **Backbone total** | **~6.3M** |

### 15.2 Level-Specific Components (N=4 levels)

| Sub-component | Parameters |
|---------------|-----------|
| Level embeddings | 2K |
| Up/down projections (levels 2–4) | 460K |
| Prediction networks (3 inter-level) | 400K |
| Multi-headed codebooks (4 levels × 8 heads × 32 × 64) | 66K |
| Halting network | 10K |
| **Level-specific total** | **~0.9M** |

### 15.3 Totals

| Configuration | Parameters |
|--------------|-----------|
| Reasoning core | ~7.2M |
| + Grid adapter (Sudoku) | ~7.8M |
| + Language adapter (LLM) | ~8.2M |
| + Multi-modal | ~8.5M |

### 15.4 Comparison with v4.1

| Component | v4.1 | v4.2 | Change |
|-----------|------|------|--------|
| Precision networks | 350K | 0 | Running statistics |
| Recognition networks | 80K | 0 | Convergence monitor |
| Column heads | 500K | 0 | Dropped |
| Codebooks | 246K | 66K | Factored: 8×32×64 |
| **Net change** | | | **−1.1M parameters** |

---

## 16. Theoretical Analysis

### 16.1 Why Free Energy Minimisation Explains Recursive Reasoning

The core claim: HRM-style recursive reasoning implements approximate variational
inference in a hierarchical generative model. Each recursion step updates the
approximate posterior q(z|x) toward the true posterior p(z|x, y).

**Formal statement:** For a hierarchical generative model with L levels:

```
p(x, z_1, ..., z_L) = p(x|z_1) ∏_l p(z_l | z_{l+1}) · p(z_L)
```

The variational free energy decomposes as:

```
F = Σ_l [ (π_l/2) ||ε_l||² - (1/2) log|π_l| + β · KL_l ]
```

Minimising F with respect to z_l gives the update:

```
Δz_l ∝ -π_l ⊙ ε_l + π_{l-1} ⊙ ε_{l-1}
```

This is CORAL's update rule: the state at level l is updated by precision-weighted
errors from above and below.

### 16.2 Why Amortisation Is the Right Objective

The free energy framework naturally produces amortisation pressure: each active
recursion step contributes to complexity cost. A system that converges in fewer steps
has lower complexity and therefore lower free energy.

CORAL's crystallisation is the architectural instantiation of this prediction.

### 16.3 Precision: Sign Matters (Historical)

The standard free energy term −½ log|π| is numerically pathological as a gradient-based
loss: it drives precision to infinity. The v4.1 fix (½(log π)²) is correct for learned
precision. In v4.2, this issue is moot — running-statistics precision has no loss term.

### 16.4 Fixed Points and Spurious Attractors

CORAL addresses spurious fixed points through three mechanisms:

1. **Running precision as a convergence quality signal.** When converged to a wrong
   answer, prediction error variance on critical dimensions remains elevated → low
   precision → signals that the "convergence" is unreliable.

2. **Stochastic transitions.** Precision-gated noise perturbs the state on uncertain
   dimensions (low precision → high noise), providing principled exploration.

3. **Multi-trajectory voting.** K independent trajectories with different noise
   realisations are unlikely to all get trapped in the same spurious attractor.

---

## 17. Relationship to Prior Work

### 17.1 Recursive Reasoning Architectures

| Architecture | Year | Key Idea | Limitation CORAL Addresses |
|-------------|------|----------|---------------------------|
| HRM | 2025 | Two-timescale recurrence | No amortisation; spurious fixed points |
| TRM | 2025 | Single tiny network, deep recursion | Fixed compute per problem; no hierarchy |
| GRAM | 2026 | Stochastic transitions | Ad hoc noise; no amortisation |
| RIM | 2026 | Inference machine framework + reweighting | No crystallisation; no adaptive depth |
| CMM | 2026 | Contraction mapping theory | No abstraction hierarchy; no adaptive depth |
| RSM | 2026 | Depth-agnostic training | No crystallisation; no hierarchy |
| Augmented HRM | 2026 | Data aug + perturbation + bootstrap | No architectural change |
| **CORAL v4.2** | **2026** | **Convergence-driven crystallisation with multi-headed codebooks + running-statistics precision + stochastic variational transitions** | — |

### 17.2 Multi-Headed Codebook Precedents

**SVQ (Wu et al., 2024):** First factored semantic codebooks for object-centric
representation. Proved heads specialise without supervision. CORAL borrows the concept
but applies it to adaptive computation allocation, not compression.

**FQGAN (2024):** Disentanglement regularisation via orthogonality loss. CORAL borrows
this directly.

**VQ-VAE (van den Oord et al., 2017):** Codebook management (EMA, commitment loss,
dead-code restart). CORAL borrows all management machinery.

### 17.3 Adaptive Computation Precedents

**ACT (Graves, 2016):** Foundational per-input adaptive depth. CORAL's halting inherits
from HRM's Q-learning variant. Ponder cost cheating motivates state-change amortisation.

**Early-exit networks:** Extensive literature on per-layer exits. CORAL goes beyond by
providing codebook-based warm-start (not just halting) at per-position, per-head
granularity.

---

## 18. Patent Considerations

### New Claims (v4.2-Specific)

1. **Multi-headed semantic codebook for adaptive reasoning depth:** A method where a
   representation vector is decomposed into multiple independent sub-vectors ("heads"),
   each quantised against its own codebook, with the composed representation used to
   determine per-position, per-head computation allocation in a recursive reasoning
   system.

2. **Convergence-driven crystallisation:** A method where the rate of change of a
   reasoning state (velocity) is monitored during recursive computation, and positions
   whose velocity falls below a threshold for consecutive steps are "crystallised" by
   snapping to the nearest codebook entry and freezing, with an error-detection
   mechanism that reverses crystallisation if the frozen state diverges from what
   continued computation would produce.

3. **Running-statistics precision for predictive coding:** A method where the precision
   (inverse variance) weighting of prediction errors in a hierarchical predictive coding
   architecture is computed as an exponential moving average of prediction error variance
   per dimension, operating outside the gradient computation graph and on a slower
   timescale than the prediction-error updates, analogous to neuromodulatory gain control.

4. **Partial crystallisation with factored codebooks:** A method where different semantic
   aspects of a position's representation can crystallise at different times during
   reasoning, with some heads frozen to codebook entries while others continue to be
   refined through recurrent computation.

### Claims Carried Forward from v4.1

5. **Amortisation pressure loss** (cumulative state change penalty).
6. **Shared backbone with level modulation** (single network, level embeddings).
7. **Amodal reasoning core with pluggable adapters** (standardised embedding interface).
8. **Stochastic variational transitions with precision gating.**

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| x | Observed input (task specification) |
| z_l | Latent state at hierarchy level l ∈ {1, ..., N} |
| z_h | Sub-vector of z for codebook head h ∈ {1, ..., H} |
| d_l | Dimensionality of latent representation at level l |
| d_head | Dimensionality per codebook head (d_l / H) |
| μ_l | Top-down prediction of z_l generated by level l+1 |
| ε_l | Prediction error: ε_l = z_l − μ_l |
| π_l | Running-statistics precision (EMA of 1/var(ε_l)) |
| π_h | Per-head precision (mean of π_l within head h's dims) |
| ξ_l | Precision-weighted prediction error: ξ_l = π_l ⊙ ε_l |
| T | Base timescale multiplier between levels |
| K | Number of outer segments (adaptive via halting) |
| h_k | Halting probability at segment k |
| H | Number of codebook heads per level |
| M_h | Codebook size per head |
| C_h | Codebook for head h: {c_1, ..., c_{M_h}} ⊂ R^{d_head} |
| τ_converge | Velocity threshold for crystallisation |
| τ_decrystallise | Drift threshold for de-crystallisation |
| N_stable | Consecutive converged segments to trigger crystallisation |
| σ_l | Learned noise scale at level l |
| λ_amort | Amortisation pressure weight |
| λ_dis | Disentanglement regulariser weight |
| γ | Codebook EMA decay factor |

## Appendix B: Decision Log

| Decision | Rationale | Reversible? |
|----------|-----------|-------------|
| Drop learned precision | 8 failed attempts; gradient vanishes when ε→0 | Yes, if running-stats fails |
| Drop recognition network | Convergence monitor is simpler and parameter-free | Yes, if convergence alone insufficient |
| Drop column heads | Depended on learned precision | Yes, independently |
| Add multi-headed codebooks | Combinatorial capacity, partial crystallisation | Core design |
| Add disentanglement regulariser | FQGAN demonstrated necessity | Can weaken (lower λ_dis) |
| Add local attention bias | Encourages local-context representations that cluster | Can remove (3 params) |
| Convergence-driven crystallisation | Speed cells analogy; no learned params | Core design |
| Running-statistics precision | Neuromodulatory analogy; slow timescale | Can try other non-learned approaches |
| Crystallisation-first research order | Most uncertain, highest payoff component | Can revert to v4.1 order |

## Appendix C: Biological Cell Type Mapping

| Cell Type | Brain Region | CORAL Analogue |
|-----------|-------------|---------------|
| Place cells | Hippocampus | Codebook entries (crystallised patterns) |
| Grid cells | Entorhinal cortex | Multi-headed factored codebook (multi-scale basis) |
| Head direction cells | Multiple | Continuous attractor dynamics (non-codebook state) |
| Border cells | Entorhinal cortex | Constraint-topology codebook head |
| Speed cells | Entorhinal cortex | Convergence monitor (velocity tracking) |
| Goal-direction cells | Retrosplenial/PFC | Prediction error ε = z - μ (directional signal) |
| Concept cells | Medial temporal lobe | Codebook entries (amodal, sparse, stable, fast) |
| Time cells | Hippocampus | Timescale embeddings (position within reasoning sequence) |
| Mirror neurons | Premotor cortex | Bidirectional codebook (recognition + generation) |
| Context neurons | Prefrontal cortex | Reasoning-phase codebook head; Level 4 state |
| RPE cells | VTA/SN | Task-loss-modulated precision (optional) |
| Error-detection neurons | ACC | De-crystallisation trigger (drift monitoring) |

---

**END OF CORAL v4.2 ARCHITECTURE SPECIFICATION — CONFIDENTIAL**
