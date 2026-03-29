# CORAL v4 — Architecture Specification

## COrtical Reasoning via Abstraction Layers

### An Amodal Reasoning Core for Efficient, Deployable Intelligence

**Version 4.1 — March 2026 (Post-Decision Revision)**

**Author:** Muhammad Anwar Ul Haq

**Status:** CONFIDENTIAL / PRE-PATENT

---

## Document Purpose

This specification defines CORAL v4, a fundamental redesign informed by nine months of
empirical results (Phases 0–2), the competitive landscape (TRM, GRAM, Augmented HRM,
CMM, RSM), and independent analyses (ARC Prize Foundation, Ge et al., Ren & Liu).

CORAL v4 is not an incremental revision. It preserves the core theoretical commitment —
variational free energy minimization as the unifying principle — but restructures the
architecture around a new central insight: **the goal of a reasoning system is not to
maximise recursion depth but to minimise it**, progressively amortising expensive
recurrent computation into fast recognition while maintaining accuracy on novel problems.

This document serves as the authoritative reference for implementation, the NeurIPS
submission, and the provisional patent filing.

---

## Table of Contents

1. Vision and Design Philosophy
2. Lessons from the Field
3. Architecture Overview
4. Component 1: Shared Backbone with Level Modulation
5. Component 2: Progressive Information Bottleneck
6. Component 3: Precision-Weighted Predictive Coding
7. Component 4: Amortisation-Driven Crystallisation
8. Component 5: Precision-Driven Sparsity
9. Component 6: Stochastic Variational Transitions
10. Component 7: Adaptive Depth Allocation
11. Unified Training Objective
12. Training Phases
13. Amodal Interface Protocol
14. Deployment Profiles
15. Parameter Budget
16. Experiment Plan
17. Theoretical Analysis
18. Relationship to Prior Work
19. Patent Considerations

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
across the hierarchy. Sparsity, crystallisation, adaptive depth, and precision dynamics
emerge as consequences, not as engineered additions.

**P5 — Amodal Reasoning.** The reasoning core operates in a modality-agnostic embedding
space. It knows nothing about language, vision, or any specific sensor modality.
Interfacing with the world is the responsibility of lightweight, pluggable
encoder/decoder modules. This separation enables deployment across robots, phones,
embedded systems, and cloud — the same reasoning core, different I/O adapters.

**P6 — Deployability as a Downstream Goal.** The architecture should be compact enough
to eventually run on mobile GPUs, edge TPUs, and embedded accelerators — but parameter
count is not a research-phase constraint. The priority is building a reasoning core with
the right dynamics (precision, crystallisation, adaptive depth). Compression for
deployment comes later through distillation, quantisation, and crystallised-only mode.
The research architecture should be whatever size produces the best reasoning behaviour.

### 1.4 What CORAL Is Not

CORAL is not a language model. It does not generate text, predict next tokens, or
maintain a vocabulary. It is a reasoning engine that takes structured embeddings as input
and produces refined embeddings as output. Language capability comes from pairing CORAL
with a language encoder/decoder — the reasoning process itself operates below the level
of language, on amodal representations.

CORAL is not a replacement for LLMs. It is a complementary module that provides the
kind of structured, multi-step reasoning that LLMs struggle with (constraint
satisfaction, planning, abstract pattern recognition), while LLMs provide the broad
knowledge and linguistic competence that CORAL does not attempt.

---

## 2. Lessons from the Field

The following findings from the HRM/TRM literature directly shaped CORAL v4. Each
finding is stated, its source identified, and its architectural implication made explicit.

### 2.1 Iteration Depth Is the Primary Driver

**Finding:** Deep recursive refinement with weight sharing is the dominant factor in
reasoning accuracy. A tiny network applied many times beats a big network applied once.
Removing the outer refinement loop causes catastrophic performance drops; other changes
are secondary.

**Sources:** ARC Prize Foundation (Aug 2025), TRM (Jolicoeur-Martineau, Oct 2025),
Ge et al. (Sep 2025), all independent analyses.

**Implication:** CORAL must support deep recursion. However, the goal is to learn to
need *less* of it, not to maximise it.

### 2.2 The H/L Hierarchy — As Implemented — Is Marginal

**Finding:** Replacing HRM's two separate H/L modules with a single expanded transformer
that updates at every timestep produces similar performance. The architectural hierarchy
contributes little.

**Sources:** ARC Prize Foundation, Ge et al. (8-layer L-only HRM), TRM (single network).

**Implication:** The hierarchy must create genuine abstraction pressure, not just update
frequency differences. CORAL v4 achieves this through progressive dimensionality
reduction (information bottlenecks) and precision-weighted communication, which force
different levels to encode qualitatively different information.

### 2.3 Full Backpropagation Through Recursion Is Critical

**Finding:** TRM's single largest improvement came from backpropagating through the full
recursion process rather than using the 1-step gradient approximation (56.5% → 87.4%).

**Source:** TRM (Table 1 ablation).

**Implication:** CORAL v4 uses full backpropagation through the inner loop within each
deep-supervision segment. Between segments, states are detached (following TRM/HRM).

### 2.4 Smaller Networks + More Recursion Beats Larger Networks

**Finding:** A 2-layer network with n=6 recursions outperforms a 4-layer network with
n=3 recursions at matched effective depth. Adding layers increases overfitting.

**Source:** TRM (Table 1: 4-layers 79.5% vs 2-layers 87.4%).

**Implication:** CORAL v4 uses a shared 2-layer backbone, not separate per-level
modules. Capacity comes from recursion depth and hierarchical structure, not from wider
or deeper individual networks.

### 2.5 MoE-Style Routing Collapses

**Finding:** Both TRM (replacing MLPs with MoE) and CORAL Phase 2 (columnar routing)
observed severe performance degradation from mixture-of-experts approaches. Routing
adds unnecessary capacity that promotes overfitting.

**Sources:** TRM (Section 6, "Ideas that failed"), CORAL Phase 2 (14.7% with column
collapse).

**Implication:** CORAL v4 replaces explicit routing with precision-driven sparsity,
where the precision vector itself modulates which computational pathways contribute,
avoiding the discrete routing decision entirely.

### 2.6 HRM Gets Trapped in Spurious Fixed Points

**Finding:** HRM converges to incorrect solutions (spurious attractors) and cannot escape
them. It can fail on puzzles with a single missing cell. The model "guesses" fixed points
rather than reasoning toward them.

**Source:** Ren & Liu, "Are Your Reasoning Models Reasoning or Guessing?" (Jan 2026).

**Implication:** CORAL v4's stochastic variational transitions (precision-scaled noise)
provide a principled mechanism for escaping spurious attractors: low precision →
high noise → exploration. This directly addresses the fixed-point violation problem
within the free energy framework.

### 2.7 Stochastic Transitions Improve Accuracy

**Finding:** GRAM achieved 97.0% on Sudoku-Extreme (vs TRM's 87.4%) by injecting
stochastic guidance at each recursion step, enabling diverse reasoning trajectories.

**Source:** GRAM (ICLR 2026 Workshop).

**Implication:** CORAL v4 incorporates stochasticity, but derives it from the variational
framework (sampling from the approximate posterior) rather than adding it ad hoc. Noise
magnitude is inversely proportional to precision, providing principled exploration.

### 2.8 Precision-Weighted Predictive Coding Works

**Finding:** Replacing HRM's raw state passing with precision-weighted prediction errors
yields a 48% relative improvement (41.2% → 61.1%) with zero additional parameters.
Interpretable precision dynamics emerge, including a phase transition early in training.

**Source:** CORAL Phase 1 (our own result, W&B run mfno8t1y).

**Implication:** This is CORAL's unique empirical contribution. v4 deepens and extends
this mechanism across N>2 levels with progressive bottlenecks.

### 2.9 Data Augmentation Is Essential

**Finding:** Augmentation (digit relabelling, band/within-band permutation, transpose)
is critical for generalisation. Without it, all models overfit catastrophically.

**Sources:** HRM, TRM, CORAL Phases 0–1 (SudokuAugmenter).

**Implication:** v4 continues to use the SudokuAugmenter (1000× augmentation) for
Sudoku experiments. For the amodal core, the augmentation strategy is task-specific and
lives in the encoder/decoder adapters, not in the core.

---

## 3. Architecture Overview

### 3.1 System Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │            AMODAL REASONING CORE            │
                    │                                             │
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
                    │  └─────────────────────────────────────┘   │
                    │       ↑                         ↓          │
                    └───────┼─────────────────────────┼──────────┘
                            │                         │
                    ┌───────┴───────┐         ┌───────┴───────┐
                    │ Input Adapter │         │Output Adapter  │
                    │  (0.5–1M)    │         │  (0.5–1M)     │
                    └───────┬───────┘         └───────┬───────┘
                            │                         │
              ┌─────────────┼─────────────┐           │
              │             │             │           │
         ┌────┴────┐  ┌────┴────┐  ┌────┴────┐     Output
         │Language │  │ Vision  │  │  Grid   │   (task-specific)
         │Encoder  │  │Encoder  │  │Encoder  │
         │(frozen  │  │(CNN/ViT)│  │(embed)  │
         │ LLM +   │  │         │  │         │
         │ proj)   │  │         │  │         │
         └─────────┘  └─────────┘  └─────────┘
```

### 3.2 Component Summary

| Component | Est. Parameters | Role |
|-----------|----------------|------|
| Shared Backbone | ~6.3M | Single 2-layer transformer, weight-shared across all levels |
| Level Embeddings | ~4K | Tells the backbone which abstraction level it is operating at |
| Prediction Networks | ~0.5M | Per-level MLPs: project from level l+1 to level l |
| Precision Networks | ~0.5M | Per-level MLPs: produce per-dimension precision vectors |
| Recognition Networks | ~0.5M | Per-level: predict crystallisation confidence before compute |
| Codebooks | ~0.2M | Per-level discrete embedding tables (M=256 entries each) |
| Column Heads | ~2M | S=8 output heads per level, modulated by precision |
| Halting Network | ~10K | Q-learning head for adaptive depth |
| **Reasoning Core Total** | **~10M** | |
| Input Adapter (task-specific) | 0.5–1M | Projects from source modality to d₁=512 |
| Output Adapter (task-specific) | 0.5–1M | Projects from d₁=512 to task output space |
| **Deployed Total** | **~11–12M** | |

Note: these parameter counts are descriptive, not targets. The research priority is
correct reasoning dynamics. If experiments reveal that a 3-layer backbone or larger
hidden dimension produces meaningfully better behaviour, the architecture should grow
accordingly. Compression for deployment is a downstream concern addressed through
distillation, quantisation, and crystallised-only mode.

### 3.3 Forward Pass — High-Level Pseudocode

```
def coral_forward(x_emb, K_max, N_levels=4):
    """
    x_emb: [B, L, d1] — input embeddings from adapter
    Returns: [B, L, d1] — refined solution state
    """
    # Initialise level states
    z = {1: x_emb}
    for l in 2..N:
        z[l] = project_down(z[l-1])  # or zeros

    # Deep supervision segments
    for segment in 1..K_max:

        # --- Top-down pass: predictions flow downward ---
        for l in (N-1) down to 1:
            mu[l] = predict(z[l+1], level=l)  # level l+1 predicts level l

        # --- Bottom-up pass: recurrence + error propagation ---
        for l in 1..N:
            # Crystallisation check (per-position, per-level)
            conf = recognition_net(z[l], context, level=l)
            crystallised = (conf > tau_c)

            # Set per-position recursion depth
            max_steps[l] = where(crystallised, T_min, T_max[l])

            # Initialise from codebook where crystallised
            z[l] = where(crystallised, nearest_codebook(z[l]), z[l])

            # Recurrence (full backprop within segment)
            for t in 1..max_steps[l]:
                noise = sigma * (1 - h_k) / sqrt(pi[l])  # precision-gated stochastic
                z[l] = backbone(z[l] + mu[l] + x_emb, level_emb=l) + noise

            # Compute precision-weighted prediction error
            eps[l] = z[l] - mu[l]
            pi[l] = softplus(precision_net(z[l], level=l)) + eps_min
            xi[l] = pi[l] * eps[l]  # only unexpected info propagates up

        # --- Halting check ---
        h_k = halt_net(concat(z[1], ..., z[N]))
        if eval and h_k > 0.95:
            break

        # --- Codebook consolidation (periodic, during training) ---
        if training and segment % consolidation_freq == 0:
            update_codebooks(z, crystallised)

        # --- Detach for next segment (deep supervision) ---
        z = {l: z[l].detach() for l in 1..N}

    return z[1]  # solution state
```

---

## 4. Component 1: Shared Backbone with Level Modulation

### 4.1 Motivation

TRM demonstrated that a single 2-layer network outperforms two separate 4-layer networks
(87.4% vs 55.0% on Sudoku-Extreme) while using half the parameters. The key insight is
that weight sharing across recursion steps provides an implicit regularisation that
prevents overfitting on small training sets.

CORAL v4 extends this principle: a single backbone is shared not only across recursion
steps but across hierarchy levels. The backbone is told which level it is operating at
via a level embedding, and its behaviour changes accordingly — but the weights are shared.

### 4.2 Architecture

The backbone is a 2-layer transformer with self-attention:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Layers | 2 | TRM finding: 2 layers optimal; more layers overfit on small data |
| Hidden dim | 512 | Matches d₁ (solution level) |
| Attention | Self-attention with 8 heads | Required for variable sequence lengths (amodal core must handle language, vision, grids of different sizes) |
| FFN expansion | 4× (SwiGLU) | Following HRM/TRM |
| Normalisation | RMSNorm (post-norm) | Following HRM |
| Position encoding | Rotary (RoPE) | Following HRM/TRM; supports variable lengths |
| Level embedding | Additive, learned | New: tells backbone which level |
| Timescale embedding | Sinusoidal | New: encodes step index within level |

**Why self-attention, not MLP-mixer:** The amodal core must handle variable sequence
lengths — language token sequences, vision patch sequences, and grid sequences all
differ in length. MLP-mixer requires a fixed L×L weight matrix and cannot handle
variable lengths without padding hacks. TRM found MLP-mixer better for fixed 9×9 grids
but worse for variable 30×30 grids, confirming that self-attention is the correct choice
for a generalist core. Starting with self-attention avoids re-validation when switching
modalities.

### 4.3 Level Modulation

For inputs at level l with dimension d_l < 512, the input is projected up to d=512
via a learned linear projection before entering the backbone, and the output is
projected back down to d_l. This means the backbone always operates at d=512 internally,
but the effective information content varies by level (higher levels have lower-rank
representations due to the bottleneck).

```
backbone_input = W_up[l] @ z[l] + level_emb[l] + timescale_emb[t]
backbone_output = backbone(backbone_input)
z[l]_updated = W_down[l] @ backbone_output
```

### 4.4 Why Not Separate Networks Per Level

Separate networks per level would give each level its own capacity but would:

- Increase total parameters by ~N× (~6M × 4 = 24M for backbone alone)
- Lose the regularisation benefit of weight sharing
- Not force the levels to "speak the same language" — with a shared backbone, the
  level embedding is the only thing that differentiates levels, which means the
  representations must be comparable

The shared backbone forces the architecture to use the hierarchy for what it is meant
to do — abstraction via information bottleneck — rather than simply giving each level
independent capacity.

---

## 5. Component 2: Progressive Information Bottleneck

### 5.1 Motivation

The current HRM literature shows that hierarchy does not help when both levels have the
same dimensionality and capacity. CORAL v4's thesis is that hierarchy helps when it
creates genuine abstraction pressure: higher levels must compress information into
lower-dimensional spaces while still producing useful predictions of the level below.

This is not an arbitrary design choice — it follows directly from the free energy
objective. The KL divergence term at each level penalises complex representations. With
progressively smaller d_l, higher levels are forced to find maximally compressed
representations that still minimise prediction error at the level below. This is the
information bottleneck principle applied hierarchically.

### 5.2 Level Configuration

| Level | Dimension (d_l) | Update Period | Timescale | Representation Content |
|-------|-----------------|---------------|-----------|----------------------|
| 1 (fastest) | 512 | Every step | ~10ms equivalent | Solution state: cell values, path segments, pixel assignments |
| 2 | 256 | Every T steps | ~40ms equivalent | Tactical: local constraint patterns, subgoal progress, region focus |
| 3 | 128 | Every T² steps | ~160ms equivalent | Strategic: which region to work on, which strategy to pursue |
| 4 (slowest) | 64 | Every T³ steps | ~640ms equivalent | Meta-strategic: when to backtrack, overall approach, problem classification |

With T=3 (the base timescale multiplier):

- Level 1: updates at steps 1, 2, 3, 4, 5, 6, ... (every step)
- Level 2: updates at steps 3, 6, 9, ... (every 3 steps)
- Level 3: updates at steps 9, 18, 27, ... (every 9 steps)
- Level 4: updates at steps 27, 54, ... (every 27 steps)

Total inner steps per segment: 1 + 3 + 9 + 27 = 40 backbone applications (with
full backpropagation).

**Why T=3 (not T=2 or T=4):**

T=3 is the committed timescale multiplier. The reasoning:

- T=2 gives levels 4/3/2/1 = 1/2/4/8 = 15 total steps. Level 3 gets only 2 updates —
  barely enough to set a value and adjust once, not enough for a meaningful
  predict-error-correct cycle. Level 2 at 4 updates is marginal.
- T=3 gives levels 4/3/2/1 = 1/3/9/27 = 40 total steps. Level 3 gets 3 updates —
  enough for at least one full predict-error-correct cycle. Level 2 at 9 updates gets
  genuine iterative refinement. Level 1 at 27 updates is comparable to TRM's effective
  depth per supervision step.
- T=4 gives levels 4/3/2/1 = 1/4/16/64 = 85 total steps. Level 1 at 64 updates is
  more than needed (diminishing returns past ~30), and the memory cost for full backprop
  doubles relative to T=3.

The information rate per level (steps × dimensions) at T=3 is: 27×512 : 9×256 : 3×128
: 1×64 = 13,824 : 2,304 : 384 : 64. This is a ~6× geometric compression per level,
which is aggressive enough to force genuine abstraction while giving each level enough
capacity to contribute meaningfully.

### 5.3 Inter-Level Projection

Each level l+1 communicates with level l through a prediction network:

```
f_pred[l]: R^{d_{l+1}} → R^{d_l}
```

Implemented as a 2-layer MLP with GELU activation:

```
f_pred[l](z_{l+1}) = W2 · GELU(W1 · z_{l+1} + b1) + b2
```

where W1 ∈ R^{2·d_l × d_{l+1}}, W2 ∈ R^{d_l × 2·d_l}. The hidden dimension is 2×d_l
to provide nonlinear capacity while keeping the projection small.

The reverse direction (error propagation upward) uses a separate bias-free linear
projection:

```
W_error_up[l]: R^{d_l} → R^{d_{l+1}}
```

This projects the precision-weighted error from level l into level l+1's space. It is
bias-free to ensure that zero error maps to zero update.

### 5.4 Why Progressive Reduction

The dimensionality reduction serves three purposes:

1. **Forces abstraction.** Level 4 at d=64 cannot represent individual Sudoku cell
   values — it must encode abstract properties like "this region is mostly solved" or
   "a contradiction exists in row 5." This forces the kind of abstract strategic
   representation that the architecture needs.

2. **Reduces computation at higher levels.** Higher levels update less frequently AND
   operate in lower dimensions. The computational cost of the hierarchy is dominated
   by level 1, not by the overhead of levels 2–4.

3. **Creates meaningful prediction error.** When level 3 (d=128) predicts level 2
   (d=256), it must decide what aspects of level 2's state are predictable from the
   strategic representation. The prediction error captures what level 3 does not yet
   understand — which is precisely what it needs to learn.

---

## 6. Component 3: Precision-Weighted Predictive Coding

### 6.1 Core Mechanism

This is CORAL's empirically validated mechanism (Phase 1: 61.1% vs 41.2% baseline).

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

**Precision estimation:** A per-dimension precision (inverse variance) vector:

```
π_l = softplus(g_l(z_l)) + ε_min
```

where g_l is a small MLP (d_l → d_l with one hidden layer) and ε_min = 0.01.

**Precision-weighted error:** Only unexpected, reliable information propagates upward:

```
ξ_l = π_l ⊙ ε_l
```

The H-module (or level l+1) receives ξ_l rather than raw z_l. This means:

- Dimensions where the prediction was accurate (small ε) contribute almost nothing
- Dimensions where the prediction was wrong AND precision is high (the model expected
  certainty but was wrong) contribute strongly
- Dimensions where the prediction was wrong but precision is low (expected variability)
  are suppressed

### 6.2 Precision Regularisation

The precision vector is regularised with a symmetric log-normal prior centred at unit
precision:

```
L_π = (λ_π / 2) · Σ_d (log π_d)²
```

This has minimum at π=1 (where log π = 0) and penalises deviation in both directions.

**Critical implementation note:** The original free energy term −½ log|π| is
mathematically correct but numerically dangerous — it rewards large precision, causing
unbounded growth. The symmetric form (log π)² is the correct regulariser for gradient-
based training. This was discovered during Phase 1 and is a key cautionary finding
(see Section 17.3).

### 6.3 Precision Dynamics (Empirical)

From Phase 1 training on Sudoku-Extreme-1K:

- **Steps 0–2,500:** Prediction error is high (~25), precision is moderate (~0.65).
  The H-module cannot yet predict L-module dynamics.
- **Steps 2,500–3,000 (phase transition):** Prediction error collapses to <1.
  Precision spikes to ~0.8 before settling. The H-module has learned to predict.
- **Steps 3,000–52,000 (stable regime):** Prediction error stays <1. Precision
  settles at ~0.04 with per-dimension variation (σ_π ≈ 0.01). The network has
  learned which dimensions to trust.

This "Cornsweet mechanism" — precision spike during prediction improvement, then
relaxation — is a novel and interpretable signal not reported by any other architecture
in the literature.

### 6.4 Attention as Precision

A key theoretical insight: precision-weighting in predictive coding is mathematically
equivalent to attention. High precision on dimension d means the model "attends" to
prediction errors on dimension d. Low precision means it "ignores" them.

This means CORAL does not need a separate inter-level attention mechanism. Precision-
weighting implements content-dependent, learned attention grounded in the generative
model. The attention pattern changes over the course of reasoning (early: diffuse
precision, late: focused precision on solution-relevant dimensions).

---

## 7. Component 4: Amortisation-Driven Crystallisation

### 7.1 Central Role

Crystallisation is CORAL v4's most distinctive mechanism and the primary differentiator
from every other architecture in the HRM/TRM lineage.

All existing recursive reasoning models (HRM, TRM, GRAM, CMM, RSM) treat every problem
instance identically: run the full recursion pipeline at maximum depth. A trained TRM
spends the same 42 serial passes on a Sudoku with one missing cell as on one with 60
missing cells. There is no mechanism to become more efficient over time.

CORAL's crystallisation provides this mechanism. It is not a bypass (which risks locking
in wrong answers), nor merely a warm start. It is an **adaptive depth allocator** that
operates per-position, per-level, at both training and inference time, with explicit
learning pressure to minimise computation.

### 7.2 The Amortisation Lifecycle

```
┌──────────────────────────────────────────────────────────────┐
│                    AMORTISATION LIFECYCLE                     │
│                                                              │
│  1. ENCOUNTER                                                │
│     New input arrives at level l, position i                 │
│                                                              │
│  2. RECOGNISE                                                │
│     Recognition network examines (z[l,i], context):          │
│     "Have I seen something like this before?"                │
│     → Outputs confidence score c ∈ [0, 1]                    │
│                                                              │
│  3. DECIDE                                                   │
│     if c > τ_c:                                              │
│       Initialise z[l,i] from nearest codebook entry          │
│       Allocate T_min = 1–2 refinement steps                  │
│     else:                                                    │
│       Keep default initialisation                            │
│       Allocate T_max steps (full recurrence)                 │
│                                                              │
│  4. COMPUTE                                                  │
│     Run allocated steps with full precision-weighted         │
│     predictive coding dynamics                               │
│                                                              │
│  5. CONSOLIDATE                                              │
│     If converged (low final prediction error):               │
│       Buffer the final state for codebook update             │
│     Periodically: k-means consolidation of buffer → codebook │
│                                                              │
│  After training: the codebook grows, recognition improves,   │
│  more positions get T_min, average depth decreases,          │
│  inference gets faster — while accuracy is maintained.       │
└──────────────────────────────────────────────────────────────┘
```

### 7.3 Recognition Network

The recognition network is lightweight — its job is to compare the current state against
the codebook, not to solve the problem:

```
RecognitionNet(z, context, level):
    # Compute distances to all codebook entries
    distances = ||z - C[l]||²  # [M] distances
    nearest_dist = min(distances)
    nearest_idx = argmin(distances)

    # Context-dependent confidence
    context_feat = linear(concat(z, context_summary))
    confidence = sigmoid(linear(concat(nearest_dist, context_feat)))

    return confidence, C[l][nearest_idx]
```

Parameters per level: ~20K (two small linear layers + codebook lookup).

### 7.4 Per-Position, Per-Level Granularity

Crystallisation operates at the finest granularity the architecture supports:

- **Per-position:** In a Sudoku grid with 81 positions, some cells may be crystallised
  (the recognition network recognises the local pattern) while others require full
  computation. This maps to the biological analogy: a chess grandmaster recognises
  standard opening patterns instantly but thinks deeply about novel middlegame positions.

- **Per-level:** Level 4 (meta-strategic, d=64) crystallises faster than level 1
  (solution detail, d=512) because higher levels see more abstract, more compressible
  patterns. Problem-level classification ("this is a constraint-propagation problem")
  crystallises before cell-level solution ("this cell is 7").

### 7.5 Training the Recognition Network

During training, full recurrence always executes (to provide learning signal). The
recognition network is trained in hindsight:

```
# After full recurrence at level l, position i:
z_final = result of full recurrence
z_codebook = nearest codebook entry

# Would bypass have been safe?
bypass_safe = (||z_final - z_codebook|| < threshold)

# Train recognition network via BCE
L_crystal = BCE(confidence, bypass_safe)
```

This means the recognition network learns to predict when bypass *would have been safe*
without ever actually bypassing during training. At inference time, the trained
recognition network fires and reduces computation.

### 7.6 Codebook Management

**Initialisation:** Codebook entries are initialised via k-means on a buffer of level
states from the first few training epochs.

**Update rule:** Exponential moving average of states assigned to each entry, following
VQ-VAE convention:

```
C[l][j] ← γ · C[l][j] + (1 - γ) · mean(states assigned to j)
```

with γ = 0.99. This is conservative — codebook entries change slowly, providing stable
recognition targets.

**Codebook size:** M = 256 per level. For 4 levels: 256 × 4 = 1024 total entries.
At d_avg ≈ 240 (average across levels), this is ~1M total codebook parameters — well
within budget.

**Diversity maintenance:** A commitment loss encourages encoder outputs to stay close
to their assigned codes:

```
L_commit = ||z_continuous - sg(e_nearest)||²
```

where sg() is the stop-gradient operator. Unused codebook entries are periodically
replaced with random samples from the state buffer (dead-code restart).

---

## 8. Component 5: Precision-Driven Sparsity

### 8.1 Motivation

Explicit routing (selecting which expert/column processes each input) consistently fails
in the small-data recursive reasoning regime: CORAL Phase 2 showed column collapse,
TRM showed MoE degradation. The per-example incentive to select winning columns
overpowers the batch-level load-balancing loss.

CORAL v4 replaces explicit routing with an implicit mechanism: **precision itself
determines which computational pathways contribute.**

### 8.2 Mechanism

At each level, the shared backbone has S=8 independent "column heads" — small linear
projections that each produce a partial update to the level state:

```
For column s in 1..S:
    delta_s = column_head[s](backbone_output, level_emb, column_emb[s])
```

Each column head produces an update in d_l dimensions. The final update is the
precision-weighted sum:

```
delta_total = Σ_s (pi_column[s] ⊙ delta_s) / Σ_s pi_column[s]
```

where pi_column[s] is the precision of the dimensions that column s primarily affects.
(In practice, each column head's output is in the full d_l space, and precision
naturally selects which columns' contributions matter.)

### 8.3 Why This Avoids Collapse

There is no discrete routing decision. Every column always contributes. But the
precision vector determines the magnitude of each contribution:

- A column that produces updates in high-precision dimensions contributes strongly
- A column that produces updates in low-precision dimensions contributes weakly

Specialisation emerges because different input patterns produce different precision
profiles — and the column heads learn to produce useful updates for the precision
profiles they encounter most frequently.

The load-balancing problem disappears because there is no binary select/don't-select
decision. The gradient flows through all columns continuously, weighted by precision.
There is no discontinuity for the optimiser to exploit.

### 8.4 Effective Sparsity

Even though all columns technically contribute, the precision distribution is typically
peaked: a few dimensions have high precision (the model is confident), most have low
precision (the model is uncertain or indifferent). This means 2–3 columns dominate any
given update, achieving ~60–75% effective sparsity without explicit top-k selection.

The sparsity level is not a hyperparameter — it emerges from the task and the training
stage. Early in training (precision is diffuse), all columns contribute roughly equally.
Late in training (precision is selective), a few columns dominate per-input, yielding
sparse computation.

---

## 9. Component 6: Stochastic Variational Transitions

### 9.1 Motivation

GRAM (97.0% on Sudoku-Extreme) showed that injecting noise into recursive transitions
enables diverse reasoning trajectories and escapes spurious fixed points. However,
GRAM's noise is ad hoc — there is no principled basis for its magnitude or when it
should be applied.

CORAL v4 derives stochasticity from the variational framework: each update step samples
from the approximate posterior rather than taking its mean. The noise magnitude is
determined by the posterior variance, which is the inverse of precision.

### 9.2 Stochastic Update Rule

The deterministic update rule:

```
z_l^{t+1} = z_l^t + η_l · f_update(z_l^t, ξ_l)
```

becomes stochastic:

```
z_l^{t+1} = z_l^t + η_l · f_update(z_l^t, ξ_l) + σ_l · (1 - h_k) · ε / sqrt(π_l)
```

where:

- σ_l is a learned per-level noise scale (initialised small, e.g., 0.01)
- (1 - h_k) gates noise by confidence: when the halting probability is high
  (near convergence), noise is suppressed
- ε ~ N(0, I) is standard Gaussian noise
- 1/sqrt(π_l) scales noise inversely with precision: uncertain dimensions get
  more exploration, certain dimensions are left alone

### 9.3 Precision Gating

The (1 - h_k) term is critical for the low-recursion regime. CORAL v4 targets 5–15
passes for hard problems — it cannot afford to waste passes on noise-driven exploration
that goes nowhere. The gating ensures:

- **Early in reasoning** (h_k ≈ 0, far from converged): full noise, maximum exploration.
  This is when the model might be stuck in a spurious attractor.
- **Late in reasoning** (h_k ≈ 0.9, near converged): almost no noise, deterministic
  refinement. The model is on the right track and should not be perturbed.

### 9.4 Multi-Trajectory Inference

At inference time, CORAL can operate in two modes:

**Fast mode (single trajectory):** Use the deterministic mean (σ_l = 0). This gives a
single answer with minimal latency. Suitable for real-time applications (robotics,
dialogue) where latency matters more than accuracy on hard problems.

**Deliberate mode (K trajectories):** Sample K independent trajectories with noise
enabled, then vote on the answer. This trades K× compute for higher accuracy on hard
problems. K is a user-controllable knob: K=1 for easy/time-sensitive, K=8–16 for hard.

The precision gating means early steps produce genuinely diverse trajectories (exploring
different regions of solution space), which then converge independently (each trajectory
refines toward its own answer). This is more efficient than uniform noise throughout.

---

## 10. Component 7: Adaptive Depth Allocation

### 10.1 Motivation

Every existing recursive reasoning model uses fixed maximum depth at inference time
(TRM: T=3 × n=6 × N_sup=16, GRAM: similar). The halting mechanism (ACT) can terminate
early, but the maximum is always set to 16 supervision segments.

CORAL v4's adaptive depth operates at three levels of granularity:

### 10.2 Segment-Level Halting (Inherited from HRM)

At each outer segment k, the halting network evaluates convergence:

```
h_k = σ(w_h · concat(z[1], ..., z[N]) + b_h)
```

If h_k > 0.95, computation terminates. This is the coarsest level of depth adaptation —
it determines how many total segments to run.

Training uses Q-learning with exploration probability 0.1, following HRM.

### 10.3 Level-Level Depth (New)

Within each segment, different levels may need different numbers of inner steps. Level 4
(meta-strategic) may crystallise after 1 step while level 1 (solution detail) needs the
full T steps. The per-level crystallisation mechanism (Section 7) handles this: levels
with high crystallisation rates execute fewer steps.

### 10.4 Position-Level Depth (New)

Within a level, different positions may need different computation. This is the finest
granularity, enabled by per-position crystallisation. In practice, this is implemented
via masking: crystallised positions receive their codebook-initialised states and skip
further updates, while non-crystallised positions continue computing.

Implementation note: to avoid variable-length computation (which breaks batching), all
positions always "run" all steps, but crystallised positions receive zero-gradient
updates (their states are frozen after initialisation from the codebook). The computation
cost is still incurred, but the gradient cost is eliminated, and at inference time with
proper masking, the computation can be genuinely skipped.

### 10.5 Amortisation Pressure Loss

The training objective includes an explicit term that penalises computation:

```
L_amort = λ_a · Σ_l Σ_t ||ε_l^t||²
```

This sums the prediction error across all active steps at all levels. A model that
converges in 3 steps pays for 3 steps of error. A model that converges in 15 steps pays
for 15. Over training, this creates direct gradient pressure to:

- Improve predictions (so error starts lower)
- Crystallise (so fewer steps are needed)
- Halt earlier (so fewer segments execute)

The balance between L_task (pushing for accuracy, favouring more computation) and L_amort
(pushing for efficiency, favouring less computation) is controlled by λ_a. This should
be annealed: start with λ_a = 0 (pure accuracy), gradually increase to target value
(accuracy-efficiency tradeoff).

### 10.6 Testing Protocol: Accuracy-Depth Pareto Curve

The amortisation pressure mechanism has a clean, unambiguous test protocol based on the
**accuracy-depth Pareto curve**.

**Measurement procedure:** At any evaluation checkpoint, measure accuracy at multiple
forced depth limits by overriding the halting mechanism:

```
For K_forced in {1, 2, 4, 8, 16}:
    accuracy[K_forced] = evaluate(model, max_segments=K_forced, force_no_halt=True)
```

This produces a curve: accuracy as a function of maximum allowed segments. A model with
good amortisation has a curve that rises steeply (most accuracy is captured in early
segments). A model without amortisation has a curve that rises slowly (all 16 segments
are needed).

**Pass/fail criterion:** Train two models under identical conditions except:
- Model A: λ_a = 0 throughout (no amortisation pressure)
- Model B: λ_a annealed from 0 → 0.01

Model B passes if its Pareto curve dominates Model A's — specifically, if at every
depth K, Model B's accuracy is within 2% of Model A's, AND at K ≤ 4, Model B's
accuracy is strictly higher. This means Model B has learned to front-load its reasoning
without sacrificing final accuracy.

**W&B logging:** At every evaluation step, log:
- `eval/accuracy@K1`, `eval/accuracy@K2`, `eval/accuracy@K4`, `eval/accuracy@K8`,
  `eval/accuracy@K16`
- `eval/avg_halting_step` (with halting enabled)
- `eval/crystallisation_rate` (fraction of positions bypassed, once enabled)
- `eval/pareto_area` (area under the accuracy-depth curve, normalised — a single
  scalar summary of amortisation quality)

This logging protocol ensures that every run produces the data needed to evaluate
amortisation quality without requiring separate evaluation jobs.

---

## 11. Unified Training Objective

### 11.1 The Complete Loss

All CORAL parameters are trained to minimise:

```
L = L_task
  + λ_pred · Σ_{k,l} (π_l / 2) · ||ε_{l,k}||²     # precision-weighted prediction error
  + λ_π · (1/2) · Σ_d (log π_d)²                    # precision regulariser
  + λ_amort · Σ_{k,l,t} ||ε_{l,k}^t||²              # amortisation pressure
  + λ_crystal · L_crystal                             # crystallisation BCE
  + λ_commit · ||z_cont - sg(e_nearest)||²            # codebook commitment
  + L_halt                                             # Q-learning halting loss
```

### 11.2 Hyperparameter Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| λ_pred | 0.1 | From Phase 1 experiments |
| λ_π | 0.01 | From Phase 1 (symmetric log-normal) |
| λ_amort | 0 → 0.01 (annealed) | Start pure accuracy, add pressure |
| λ_crystal | 0 → 0.1 (annealed) | Enabled after Phase A training |
| λ_commit | 0.25 | Standard VQ-VAE value |
| L_halt | Q-learning (ε=0.1) | Following HRM |

### 11.3 Free Energy Interpretation

The loss decomposes as variational free energy:

- **Accuracy:** L_task (how well the model explains the data)
- **Complexity (representation):** Precision-weighted prediction error + precision
  regulariser (cost of maintaining the posterior)
- **Complexity (computation):** L_amort (cost of reasoning steps)
- **Complexity (structure):** L_crystal + L_commit (cost of the codebook vs
  continuous representation)
- **Complexity (halting):** L_halt (cost of deciding when to stop)

All terms reduce to "minimise the total description length of the model's
representation, computation, and output." This is formally equivalent to the Minimum
Description Length (MDL) principle.

---

## 12. Training Phases

### Phase A: Foundation (Epochs 1–10,000)

**Goal:** Learn basic generative model and prediction functions.

**Configuration:**
- λ_amort = 0 (no efficiency pressure)
- Crystallisation disabled (full recurrence always)
- Stochastic transitions disabled (deterministic)
- Full inner backpropagation enabled
- All column heads active (no sparsity)

**Success criterion:** Eval accuracy surpasses Phase 1 result (61.1%) at matched
step count. If not, diagnose before proceeding.

### Phase B: Precision & Depth (Epochs 10,001–30,000)

**Goal:** Develop precision dynamics and learn halting.

**Configuration:**
- Enable stochastic transitions (σ_l initialised at 0.01)
- λ_amort anneals from 0 to 0.005
- Crystallisation still disabled
- Monitor: precision dynamics, halting step distribution, accuracy

**Success criterion:** Average halting steps decrease while accuracy maintains or
improves. Precision shows per-dimension specialisation.

### Phase C: Crystallisation (Epochs 30,001–50,000)

**Goal:** Learn to amortise. Codebook entries emerge, recognition network activates.

**Configuration:**
- Enable crystallisation (recognition network active, codebook updates begin)
- λ_crystal anneals from 0 to 0.1
- λ_amort increases to target value (0.01)
- Monitor: crystallisation rate per level, codebook usage, accuracy vs avg depth

**Success criterion:** Crystallisation rate increases over training. Higher levels
crystallise faster than lower levels. Average depth per segment decreases. Accuracy
does not degrade by more than 2% from Phase B peak.

### Phase D: Refinement (Epochs 50,001–60,000)

**Goal:** Full objective with all components active. Fine-tune the accuracy-efficiency
tradeoff.

**Configuration:**
- All components active at target values
- Multi-trajectory evaluation (K=8) to measure deliberate-mode accuracy

**Success criterion:** Final evaluation with full metrics (accuracy, average depth,
crystallisation rate, effective sparsity, per-difficulty depth allocation).

---

## 13. Amodal Interface Protocol

### 13.1 Design Philosophy

The reasoning core is modality-agnostic. It operates on sequences of embedding vectors
and returns refined sequences of embedding vectors. Everything specific to a modality
(tokenisation, feature extraction, output formatting) lives in the adapter layer.

This separation is strict: the core has no vocabulary, no convolutional filters, no
task-specific heads. It has one input contract and one output contract.

### 13.2 Input Contract

```
Input:  z₁⁰ ∈ R^{B × L × d₁}
        B = batch size
        L = sequence length (variable, padded within batch)
        d₁ = 512 (Level 1 dimension)
```

The adapter is responsible for producing this tensor. How it does so depends on the
modality.

### 13.3 Output Contract

```
Output: z₁_final ∈ R^{B × L × d₁}
        Same shape as input
        Refined solution-level embeddings
```

A task-specific decoder maps this to the output format (grid values, path, text tokens,
action commands, etc.).

### 13.4 Adapter Specifications

#### 13.4.1 Grid Tasks (Sudoku, ARC-AGI, Mazes)

```
Encoder:
    Input: grid ∈ Z^{H × W} (integer token grid)
    Process:
        1. Token embedding: embed(grid) → R^{H×W × d₁}
        2. Position embedding: 2D sinusoidal or learned
        3. z₁⁰ = token_emb + pos_emb
    Parameters: ~0.3M (embedding table + position)

Decoder:
    Input: z₁_final ∈ R^{H×W × d₁}
    Process:
        1. Linear projection: W_out · z₁_final → R^{H×W × V}
           where V = vocabulary size
        2. Softmax → per-cell token distribution
    Parameters: ~0.3M
```

#### 13.4.2 Language (Interfacing with LLMs)

```
Encoder:
    Input: LLM hidden states h ∈ R^{L × d_LLM}
    Process:
        1. Projection: W_in · h → R^{L × d₁}
           where W_in ∈ R^{d₁ × d_LLM}
        2. Optional: lightweight cross-attention adapter (LoRA-style)
    Parameters: ~0.5M (for d_LLM = 4096, d₁ = 512)
    LLM backbone: frozen

Decoder:
    Input: z₁_final ∈ R^{L × d₁}
    Process:
        1. Projection: W_out · z₁_final → R^{L × d_LLM}
        2. Add back to LLM hidden states (residual)
        3. LLM generates output tokens from modified hidden states
    Parameters: ~0.5M
```

This interface allows the reasoning core to be inserted as a "thinking module"
between layers of a frozen LLM. The LLM handles tokenisation, knowledge, and
language generation; CORAL handles multi-step reasoning in latent space.

#### 13.4.3 Vision (Interfacing with Vision Encoders)

```
Encoder:
    Input: Image features f ∈ R^{P × d_vision}
           (P = number of patches, from ViT or CNN backbone)
    Process:
        1. Projection: W_in · f → R^{P × d₁}
        2. Add spatial position embeddings
    Parameters: ~0.3M
    Vision backbone: frozen

Decoder:
    Input: z₁_final ∈ R^{P × d₁}
    Process:
        Task-specific head:
        - Classification: pool + linear
        - Detection: per-patch linear
        - Reasoning (ARC-style): reshape + per-position linear
    Parameters: ~0.3M
```

#### 13.4.4 Robotics (Interfacing with Sensor Fusion)

```
Encoder:
    Input: Fused sensor state s ∈ R^{N × d_sensor}
           (N = number of entities/objects in scene representation)
    Process:
        1. Projection: W_in · s → R^{N × d₁}
        2. Add entity-type embeddings
    Parameters: ~0.3M

Decoder:
    Input: z₁_final ∈ R^{N × d₁}
    Process:
        1. Pool relevant entities
        2. Action head: MLP → action space
        3. Value head: MLP → scalar value estimate
    Parameters: ~0.5M
```

### 13.5 Adapter Interoperability

The same reasoning core can be connected to multiple adapters simultaneously for
multi-modal reasoning:

```
z₁⁰ = concat(
    language_adapter(text_tokens),    # L₁ language positions
    vision_adapter(image_patches),     # L₂ vision positions
    sensor_adapter(sensor_readings)    # L₃ sensor positions
)
# Total sequence: L = L₁ + L₂ + L₃
# The core reasons over the concatenated sequence
# Cross-modal attention happens naturally within the backbone
```

The modality information is encoded in the position embeddings and (optionally) a
modality-type embedding, not in the core architecture.

---

## 14. Deployment Profiles

### 14.1 Profile Matrix

These profiles represent downstream deployment targets, not research-phase constraints.
The research architecture (full core ~8.4M + adapter) is the starting point;
deployment-optimised variants are derived through distillation, quantisation,
and crystallised-only mode after the core dynamics are validated.

| Profile | Hardware | Model | Latency Target | Mode |
|---------|----------|-------|---------------|------|
| Cloud (research) | A100 | Full core + adapter (~10M, FP16) | N/A | Deliberate (K=16) |
| Cloud (production) | T4/L4 | Full core + adapter (~10M, FP16) | <500ms | Fast (K=1–4) |
| Mobile (on-device) | Snapdragon NPU | Distilled core (INT8) | <200ms | Fast (K=1) |
| Edge (robotics) | Jetson Orin | Full core (FP16) | <100ms | Fast (K=1) |
| Embedded (IoT) | Cortex-M/RISC-V | Crystallised-only (~1–2M, INT4) | <50ms | Codebook lookup only |

### 14.2 Embedded Profile: Crystallised-Only Mode

For the most constrained environments, CORAL can run in a "crystallised-only" mode where:

- Only the recognition network and codebooks are deployed (no backbone)
- Every position is looked up from the codebook
- If recognition confidence is below threshold, the input is flagged for offload to
  a more capable device

This mode uses approximately 1–2M parameters and requires only codebook lookup +
nearest-neighbour search, which can run on microcontrollers. It is useful for:

- Sensor pre-processing (classify known patterns, flag anomalies)
- Fast pre-filtering before more expensive reasoning
- Offline-trained pattern recognition deployed to resource-constrained devices

### 14.3 Inference Optimisations

**Torch.compile:** The shared backbone with fixed graph structure is highly amenable
to torch.compile. Variable per-position depth is handled via masking, not dynamic
control flow.

**Quantisation:** INT8 quantisation of the backbone reduces memory by 2× and compute
by ~2× on supported hardware (mobile NPUs, Tensor Cores). Codebook entries can be
quantised to INT8 with minimal accuracy loss.

**KV-cache reuse:** Within a segment, attention KV-caches from earlier inner steps can
be reused (the sequence length doesn't change, only the values are updated).

**Batched multi-trajectory:** For deliberate mode, K trajectories can be batched along
the batch dimension, giving K× throughput on parallel hardware with the same latency as
a single trajectory.

---

## 15. Parameter Budget (Detailed)

### 15.1 Shared Backbone (2-layer Transformer, d=512)

| Sub-component | Parameters | Notes |
|---------------|-----------|-------|
| Self-attention Q/K/V projections (×2 layers) | 2 × 3 × 512 × 512 = 1.57M | 8 heads, d_k=64 |
| Self-attention output projection (×2 layers) | 2 × 512 × 512 = 0.52M | |
| SwiGLU FFN (×2 layers) | 2 × (512 × 2048 + 2048 × 512 + 512 × 2048) = 4.19M | gate + up + down |
| RMSNorm (×4) | negligible | No learnable params |
| **Backbone total** | **~6.3M** | 2-layer, d=512, committed configuration |

### 15.2 Level-Specific Components (N=4 levels)

| Sub-component | Parameters | Notes |
|---------------|-----------|-------|
| Level embeddings | 4 × 512 = 2K | One per level |
| Up-projections (levels 2–4 → 512) | 256×512 + 128×512 + 64×512 = 230K | |
| Down-projections (512 → levels 2–4) | 512×256 + 512×128 + 512×64 = 230K | |
| Prediction networks (3 inter-level) | 3 × (d_{l+1}×2d_l + 2d_l×d_l) ≈ 400K | 2-layer MLPs |
| Precision networks (4 levels) | 4 × (d_l × d_l + d_l × d_l) ≈ 350K | 2-layer MLPs per level |
| Recognition networks (4 levels) | 4 × ~20K = 80K | Small classifiers |
| Codebooks (4 levels × 256 entries) | 256×(512+256+128+64) = 246K | |
| Column heads (8 per level × 4 levels) | 32 × (512 × d_l / 8) ≈ 500K | Partial-dim outputs |
| Halting network | 10K | Linear on concat states |
| **Level-specific total** | **~2.1M** | |

### 15.3 Total Reasoning Core

| Component | Parameters |
|-----------|-----------|
| Shared backbone (2-layer) | 6.3M |
| Level-specific components | 2.1M |
| **Total reasoning core** | **~8.4M** |

### 15.4 With Adapters

| Configuration | Total Parameters |
|--------------|-----------------|
| Core + Grid adapter | ~9M |
| Core + Language adapter (LLM interface) | ~9.5M |
| Core + Vision adapter (ViT interface) | ~9M |
| Core + Robotics adapter (sensor fusion) | ~9.5M |
| Core + Multi-modal (language + vision) | ~10M |

These counts will shift if experiments reveal that a larger backbone or hidden
dimension produces meaningfully better dynamics. The numbers above reflect the
starting configuration for Experiment 1.

---

## 16. Experiment Plan

### 16.1 Experiment Sequence

Each experiment is incremental, has a clear pass/fail criterion, and builds on the
previous result. No experiment proceeds unless the previous one passes.

#### Experiment 1: Shared Backbone + Full Inner Backprop

**What:** Replace CORAL's separate H/L modules with a shared 2-layer backbone (d=512,
self-attention, T=3). Replace 1-step gradient approximation with full backprop through
the inner loop. Keep N=2 and precision-weighted predictive coding from Phase 1.

**Baseline:** CORAL Phase 1 (61.1% eval exact accuracy, 27M params, 1-step gradient
approx, separate H/L modules).

**Pass criterion:** ≥70% eval accuracy on Sudoku-Extreme-1K. This would confirm that
the shared-backbone + full-backprop combination improves over Phase 1 (61.1%), as
TRM's evidence strongly predicts. If accuracy does not improve, diagnose whether the
issue is the shared backbone (test: separate backbones with full backprop) or the
full backprop (test: shared backbone with 1-step approx) before proceeding.

**Compute:** ~4 hours on A100 (single run).

#### Experiment 2: Amortisation Pressure

**What:** Add L_amort to the loss (λ_a annealed from 0 to 0.01 over training).
Everything else unchanged from Experiment 1. This experiment uses the accuracy-depth
Pareto testing protocol defined in Section 10.6.

**Metrics:** Accuracy-depth Pareto curve at K ∈ {1, 2, 4, 8, 16}, average halting
steps, `eval/pareto_area` summary scalar. All logged to W&B at every eval step.

**Protocol:** Train two models under identical conditions:
- Model A: λ_a = 0 throughout (Experiment 1 configuration, baseline)
- Model B: λ_a annealed from 0 → 0.01

**Pass criterion:** Model B's Pareto curve dominates Model A's — specifically:
(a) at every depth K, Model B's accuracy is within 2% of Model A's, AND
(b) at K ≤ 4, Model B's accuracy is strictly higher than Model A's.
Additionally, Model B's average halting steps should decrease by ≥20% over training.

**Compute:** ~4 hours on A100.

#### Experiment 3: Crystallisation-as-Adaptive-Depth

**What:** Add per-position recognition network and codebooks. Enable crystallisation
training (Phase C configuration). Measure whether the model learns to allocate fewer
steps to easy positions.

**Metrics:** Crystallisation rate per level, codebook usage entropy, accuracy at
matched average depth, accuracy vs average depth Pareto curve.

**Pass criterion:** Crystallisation rate > 10% at any level. Codebook entries are
non-degenerate (usage entropy > 2.0). Accuracy at 50% average depth ≥ 90% of full-
depth accuracy.

**Compute:** ~6 hours on A100.

#### Experiment 4: Scale to N=3, N=4

**What:** Add hierarchy levels with progressive dimensionality reduction. With
amortisation pressure active, test whether deeper hierarchy produces genuinely
different abstraction levels.

**Key diagnostic:** Do higher levels crystallise faster than lower levels? (They
should — higher levels see more abstract, more compressible patterns.) If yes,
the hierarchy is doing real work.

**Pass criterion:** N=3 or N=4 matches or exceeds N=2 accuracy while using fewer
total inner steps (due to higher-level crystallisation).

**Compute:** ~8 hours on A100 (multiple N configurations).

#### Experiment 5: Stochastic Variational Transitions

**What:** Add precision-gated noise to state updates. Compare deterministic single-
pass, stochastic single-pass, and stochastic multi-trajectory (K=4, K=8, K=16).

**Pass criterion:** Multi-trajectory (K=8) accuracy ≥ 90% on Sudoku-Extreme-1K.
(GRAM achieves 97% on the full 423K test set.)

**Compute:** ~6 hours on A100.

#### Experiment 6: Full Ablation Matrix

**What:** Factorial ablation of all mechanisms on the best N configuration.

| Variant | Description |
|---------|-------------|
| Full CORAL v4 | All components |
| − Precision weighting | Raw state passing (π=1) |
| − Crystallisation | Full recurrence always |
| − Amortisation pressure | λ_amort = 0 |
| − Stochastic transitions | Deterministic (σ=0) |
| − Hierarchy (N=2) | Two levels only |
| − Hierarchy (N=1, flat) | Single level, TRM-like |
| TRM reproduction | Single 2-layer network, matched compute |
| HRM reproduction | Two 4-layer networks, matched params |

**Compute:** ~12 hours on A100 (8 runs × ~1.5 hours each).

### 16.2 Total Compute Budget

| Experiment | GPU Hours | Running Total |
|-----------|-----------|---------------|
| Exp 1 | 4 | 4 |
| Exp 2 | 4 | 8 |
| Exp 3 | 6 | 14 |
| Exp 4 | 8 | 22 |
| Exp 5 | 6 | 28 |
| Exp 6 | 12 | 40 |
| **Total** | **~40 hours** | ~$20 at Vast.ai A100 rates |

---

## 17. Theoretical Analysis

### 17.1 Why Free Energy Minimisation Explains Recursive Reasoning

The core theoretical claim: HRM-style recursive reasoning implements approximate
variational inference in a hierarchical generative model. Each recursion step updates
the approximate posterior q(z|x) toward the true posterior p(z|x, y). Making this
process explicit through precision-weighted predictive coding improves it for the same
reason that a principled optimisation algorithm outperforms random search — it focuses
updates on the most informative dimensions.

**Formal statement:** Consider a hierarchical generative model with L levels:

```
p(x, z_1, ..., z_L) = p(x|z_1) ∏_l p(z_l | z_{l+1}) · p(z_L)
```

The variational free energy decomposes as:

```
F = Σ_l [ (π_l/2) ||ε_l||² - (1/2) log|π_l| + β · KL_l ]
```

Minimising F with respect to z_l (the E-step of variational EM) gives the update:

```
Δz_l ∝ -π_l ⊙ ε_l + π_{l-1} ⊙ ε_{l-1}
```

This is exactly CORAL's update rule: the state at level l is updated by subtracting
the precision-weighted error from above and adding the precision-weighted error from
below. Each recursion step is one step of coordinate-descent variational inference.

**What this explains:**
- Why iteration helps: each step refines the posterior
- Why hierarchy helps (with bottlenecks): it decomposes the posterior into factors
  at different abstraction levels, making each factor easier to optimise
- Why precision helps: it focuses each update step on the most informative dimensions
- Why crystallisation helps: when the posterior has converged (low entropy), further
  iteration wastes compute

**What competing frameworks explain less well:**
- TRM's "less is more" does not explain why iteration works at all — it is purely
  empirical
- GRAM's stochastic transitions do not explain when to explore vs exploit
- CMM's contraction mapping theory explains convergence but not abstraction or
  adaptive depth
- The diffusion model analogy (Ge et al.) explains training but not the inference-
  time dynamics

### 17.2 Why Amortisation Is the Right Objective

The free energy framework naturally produces amortisation pressure through the
complexity term. Each active recursion step contributes to the complexity cost (via
the KL term for maintaining an unconverged posterior). A system that can converge in
fewer steps has lower complexity — and therefore lower free energy — than one that
requires many steps.

This is why biological neural systems evolved to amortise: maintaining an active
inference process (recurrence, sustained neural activity) is metabolically expensive.
The free energy principle predicts that organisms should minimise the expected free
energy of their inference processes, which means learning to recognise patterns that
short-circuit extended inference.

CORAL's crystallisation mechanism is the architectural instantiation of this prediction.

### 17.3 Precision Regulariser: Sign Matters

**The bug:** The standard free energy decomposition includes −½ log|π|, which prevents
precision from collapsing to zero. Implemented naively as a loss term, this becomes
"minimise −½ log π" = "maximise ½ log π" = "maximise π." The optimiser increases
precision without bound.

**The fix:** Replace with ½(log π)², which has its minimum at π=1 and penalises
deviation in both directions.

**The lesson:** Free energy terms that are mathematically correct as parts of an
objective can be numerically pathological when used as gradient-based loss terms.
The log-barrier −log π prevents π→0 in an interior-point optimisation context, but
gradient descent does not respect barriers — it follows the gradient, which points
toward π→∞.

This finding should be reported prominently in any paper using precision-weighted
architectures.

### 17.4 Fixed Points and Spurious Attractors

Ren & Liu (2026) showed that HRM converges to spurious fixed points (wrong solutions
that are stable under continued iteration). This is a fundamental problem for any
deterministic recursive system.

CORAL addresses this through three mechanisms:

1. **Precision as a convergence quality signal.** When the model has converged to a
   wrong answer, the prediction error at higher levels should remain elevated (the
   abstract representation is inconsistent with the detailed solution). Precision on
   these dimensions would be high (the model expected the solution to be consistent),
   creating a strong error signal that flags the spurious fixed point.

2. **Stochastic transitions.** Even if the deterministic dynamics have converged,
   the precision-scaled noise can perturb the state away from a spurious attractor.
   Low precision (high uncertainty) on specific dimensions means large noise on those
   dimensions — exactly where exploration is most needed.

3. **Multi-trajectory voting.** Running K trajectories with different noise
   realisations provides multiple independent "guesses." A spurious attractor that
   traps one trajectory is unlikely to trap all K, especially when noise is precision-
   gated to target uncertain dimensions.

---

## 18. Relationship to Prior Work

| Architecture | Year | Key Idea | Limitation CORAL Addresses |
|-------------|------|----------|---------------------------|
| HRM | 2025 | Two-timescale recurrence | No principled objective; no hierarchy benefit; spurious fixed points |
| TRM | 2025 | Single tiny network, deep recursion | No amortisation; fixed compute per problem; no hierarchy |
| GRAM | 2026 | Stochastic transitions | Ad hoc noise; no precision gating; no amortisation |
| CMM | 2026 | Contraction mapping theory | No abstraction hierarchy; no adaptive depth |
| RSM | 2026 | Depth-agnostic training | No precision; no crystallisation; no hierarchy |
| CGAR | 2025 | Curriculum on recursion depth | Training speedup only; no architectural innovation |
| Augmented HRM | 2026 | Data aug + perturbation + bootstrap | No architectural change; brute-force diversity |
| **CORAL v4** | **2026** | **Free energy unification: precision-weighted predictive coding + amortisation-driven crystallisation + stochastic variational transitions** | — |

CORAL v4's unique contribution is the combination of:

1. A principled theoretical framework (free energy) that explains why recursive
   reasoning works and predicts specific dynamics (precision evolution, crystallisation
   rates, adaptive depth)
2. An amortisation mechanism that no other architecture provides — the system learns
   to need less computation over time
3. Interpretable dynamics (precision, crystallisation rate, per-level depth) that
   provide a window into the reasoning process

---

## 19. Patent Considerations

The provisional patent filing (CORAL Patent Claims Draft) should be updated to cover
v4-specific innovations:

### New Claims to Add

1. **Amortisation-driven crystallisation with per-position, per-level granularity:**
   A method where a recognition network independently determines, for each position
   in a sequence and each level of a hierarchy, whether to bypass recurrent computation
   via codebook initialisation, with the recognition network trained by comparing
   full-recurrence outputs against codebook entries.

2. **Amortisation pressure loss:** A training objective term that penalises the
   cumulative prediction error across active recursion steps, creating explicit
   gradient pressure to reduce the number of steps needed for convergence.

3. **Precision-driven sparsity (replacing explicit routing):** A method where the
   learned precision vector of a predictive coding architecture modulates the
   contribution of multiple parallel computational sub-modules, achieving sparse
   computation without discrete routing decisions.

4. **Stochastic variational transitions with precision gating:** A method where noise
   injected into recursive state updates is scaled inversely by the learned precision
   and gated by the halting probability, providing principled exploration that
   diminishes as the model approaches convergence.

5. **Shared backbone with level modulation:** A method where a single neural network
   backbone is shared across all levels of a hierarchical architecture, with level-
   specific behaviour induced by additive level embeddings and inter-level projection
   networks.

6. **Amodal reasoning core with pluggable adapters:** A system comprising a modality-
   agnostic reasoning module operating on embedding vectors, connected to task-specific
   encoder/decoder adapters via a standardised interface, enabling deployment of the
   same reasoning core across multiple modalities and hardware platforms.

### Existing Claims to Update

Claims 1–3 and 8 from the original draft should be updated to reflect the shared
backbone (replacing separate per-level modules), full backpropagation through inner
loops (replacing 1-step gradient approximation), and the amortisation lifecycle
(replacing simple entropy-gated interpolation).

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| x | Observed input (task specification) |
| z_l | Latent state at hierarchy level l ∈ {1, ..., N} |
| d_l | Dimensionality of latent representation at level l |
| μ_l | Top-down prediction of z_l generated by level l+1 |
| ε_l | Prediction error at level l: ε_l = z_l − μ_l |
| π_l | Learned precision (inverse variance) at level l |
| ξ_l | Precision-weighted prediction error: ξ_l = π_l ⊙ ε_l |
| T | Base timescale multiplier between levels |
| K | Number of outer segments (adaptive, controlled by halting) |
| h_k | Halting probability at segment k |
| F | Total variational free energy |
| θ | Learnable parameters of the generative model |
| φ | Learnable parameters of the recognition model |
| S | Number of column heads per level |
| M | Codebook size per level |
| τ_c | Crystallisation confidence threshold |
| σ_l | Learned noise scale at level l |
| λ_a | Amortisation pressure weight |
| C_l | Codebook at level l: {c_1, ..., c_M} ⊂ R^{d_l} |
| γ | Codebook EMA decay factor |

## Appendix B: Key Hyperparameter Defaults

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| N (hierarchy levels) | 4 | Start N=2, scale in Exp 4 | Progressive: d=512, 256, 128, 64 |
| T (base timescale) | 3 | **Committed** | 40 inner steps per segment; see Section 5.2 |
| Backbone layers | 2 | **Committed** | Self-attention; see Section 4.2 |
| Backbone dim | 512 | **Committed** | Matching d₁ |
| Attention type | Self-attention | **Committed** | Required for variable sequence lengths (amodal core) |
| Attention heads | 8 | Default | d_k = 64 |
| FFN expansion | 4× (SwiGLU) | Default | Standard |
| S (column heads) | 8 | Default | Per level; modulated by precision |
| M (codebook size) | 256 | Default | Per level |
| K_max (segments) | 16 | Default | Deep supervision |
| τ_c (crystallisation threshold) | 0.8 | Tunable | Conservative; tune lower if crystallisation rate too low |
| σ_l (noise init) | 0.01 | Tunable | Small; learned during training |
| ε_min (precision floor) | 0.01 | From Phase 1 | Prevents division by zero |
| η_l (step size init) | 0.1 | From Phase 1 | Learnable per level |
| λ_pred (prediction loss) | 0.1 | From Phase 1 | Precision-weighted prediction error weight |
| λ_π (precision reg) | 0.01 | From Phase 1 | Symmetric log-normal prior |
| λ_amort (amortisation) | 0 → 0.01 | Tunable (Exp 2) | Annealed; tested via Pareto protocol (Section 10.6) |
| λ_crystal (crystallisation) | 0 → 0.1 | Tunable (Exp 3) | Annealed; enabled in Phase C |
| λ_commit (codebook) | 0.25 | Default | Standard VQ-VAE value |
| Learning rate | 7e-5 | From Phase 1 | Fused AdamATan2 if compilable, else AdamW; cosine schedule |
| Weight decay | 1.0 | From Phase 1 | Following HRM |
| Batch size | 64 | Default | Adjust based on memory with full backprop |
| Gradient clipping | 1.0 (max norm) | Default | Standard |
| Precision | bfloat16 forward, float64 loss | **Committed** | Numerical stability |

**Status key:**
- **Committed** = decided based on analysis; do not revisit without strong evidence
- From Phase 1 = validated in prior experiments; carry forward
- Default = reasonable starting point; adjust based on experimental signal
- Tunable = explicitly designed to be tuned; has a defined testing protocol

---

**END OF CORAL v4 ARCHITECTURE SPECIFICATION — CONFIDENTIAL**
