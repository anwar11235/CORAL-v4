# CORAL v4 — Session 2 Handoff Note

## Date: March 31, 2026

## Session Summary

This session covered: launching training runs, debugging multiple issues,
discovering that predictive coding hurts performance in its current form,
diagnosing why, and arriving at a fundamental architectural reordering —
train deep, infer shallow.

---

## 1. Key Architectural Discovery: Train Deep, Infer Shallow

The most important finding of this session:

**Predictive coding, precision weighting, and hierarchy should NOT be built
alongside the backbone. They should be layered on top of a backbone that
already works.**

The evidence:
- No-PC baseline (N=2, 4 inner steps): 60% token accuracy, task loss 0.93
- Every PC configuration: 15-18% token accuracy, task loss 1.65+ (stuck)
- PC diagnostic proved the conditioning is blurred and overwhelms backbone output
- The model actively learned to suppress PC conditioning when given a learnable gate

**The correct build order is:**

1. **Backbone first.** N=1, 21 inner steps, no PC, no hierarchy. Match TRM's
   effective depth. Prove the backbone can solve puzzles. (Currently running)

2. **Hierarchy second.** Add N=2+ with PC operating as inter-level error
   communication only — NOT as top-down conditioning on level 1.

3. **Crystallisation third.** Once the model can solve puzzles, teach it to
   recognise when it doesn't need full depth. This is where inference speedup
   comes from.

4. **Stochastic transitions fourth.** For hard problems where the model gets
   stuck.

5. **ARC-AGI validation fifth.** Same core, different adapter.

This is a different order from the original build plan and spec. The efficiency
mechanisms (PC, crystallisation, amortisation) sit on top of a proven backbone,
not alongside it.

---

## 2. Currently Running Experiment

**Run:** `v4-backbone-n1-d21`
**W&B:** In project `Sudoku-extreme-1k-aug-1000 CORAL-v4`
**Config:** N=1, inner_steps_override=21, no PC, AdamW lr=7e-5

**Status at session end (step ~700):**
- Task loss: 1.05 (dropping fast)
- Token accuracy: 16.4% at step 500 (likely climbing rapidly)
- Training speed: ~50 steps per 15 seconds → 20K steps in ~100 minutes
- GPU: A100-SXM4-40GB (Sweden instance, $0.478/hr)

**Expected trajectory:** Based on the N=2 baseline reaching 60% token accuracy
at step 4500 with only 4 inner steps, the N=1 d=21 run should significantly
exceed this. Target: 80%+ token accuracy by 20K steps.

---

## 3. All Bugs Found and Fixed This Session

### 3.1 Vocab Size YAML Override
- **Bug:** `configs/exp1_baseline.yaml` had `vocab_size: 10`, overriding the
  correct default of 11 in config.py
- **Fix:** Changed YAML to `vocab_size: 11`
- **Lesson:** Hydra YAML always overrides dataclass defaults

### 3.2 Eval Data Path
- **Bug:** Eval looked for `eval/train__inputs.npy` instead of `test/all__inputs.npy`
- **Fix:** Corrected split name and set name in eval code

### 3.3 Eval Dtype Mismatch
- **Bug:** `adapter.decode()` was outside `torch.autocast` block
- **Fix:** Moved decode inside autocast

### 3.4 Trainer-Core Interface Mismatch
- **Bug:** Trainer manually managed segments; core manages them internally
- **Fix:** Fixed trainer to call `core(z1_init, ...)` correctly

### 3.5 Precision Network Zero Init
- **Bug:** All weights initialised to zero → constant output 0.703
- **Fix:** Xavier uniform init on precision network weights

### 3.6 Precision Collapse
- **Bug:** Regulariser overpowers task gradient; precision flattens to uniform
- **Attempted fixes:** λ_π reduction, removing regulariser, precision-gated decode,
  eps-dependent precision input
- **Root cause:** No gradient path from task loss through precision once prediction
  error collapses
- **Status:** Partially mitigated but not fully solved (see Section 4)

### 3.7 PC Conditioning Overwhelms Backbone
- **Bug:** Additive prediction conditioning doubles signal when mu ≈ z1
- **Fix:** Changed to post-backbone residual: `z_new + gate * (mu - z)`
- **Further finding:** Even with residual, conditioning vector is 1.5× larger than
  backbone output. Gate init at 1.0 means conditioning dominates.
- **Fix:** Gate init changed from 1.0 to 0.01. Model learned to push gate negative,
  confirming conditioning is not useful.

### 3.8 Prediction Loss Explosion
- **Bug:** `pi * eps²` summed over 512 dims produces huge values
- **Fix:** Changed `.sum(dim=-1)` to `.mean(dim=-1)` in both prediction loss and
  precision regulariser. Also reduced λ_pred from 0.1 to 0.001.

### 3.9 Inner Steps Formula for N=1
- **Bug:** `level_steps = T^(N-1-i)` gives 1 step for N=1 regardless of T
- **Fix:** Added `model.inner_steps_override` config field that overrides computed
  level_steps[0]

---

## 4. The Precision Collapse Problem — Complete Record

Eight attempts were made to get precision to differentiate and stay differentiated.
All failed to varying degrees. Full record in the Session 1 handoff note (Section 4).

**The fundamental issue:** Precision receives gradient through `dL/dpi = 0.5 * eps²`.
Once prediction error collapses (eps → 0 by step ~300), this gradient vanishes.
The regulariser is then the only force on precision and it pulls everything to uniform.

**Current assessment:** Precision collapse may not be the right problem to solve.
The backbone works well without precision. The efficiency mechanisms (crystallisation,
amortisation) don't require per-dimension precision — they can operate on convergence
signals (velocity of state change) instead. The v4.2 spec's proposal to use
running-statistics precision (EMA of error variance, no learned network) may be the
right approach — it sidesteps the gradient problem entirely.

---

## 5. PC Diagnostic Results — What We Learned

A diagnostic script (`scripts/diagnostic_pc_analysis.py`) was created that measures:
1. Representation sharpness (per-position L2 norm variance)
2. Prediction blur (mu variance vs z1 variance)
3. Gate values
4. Conditioning magnitude relative to z1
5. Token accuracy before and after conditioning

**Key findings:**

| Metric | Value | Implication |
|--------|-------|-------------|
| Blur ratio (mu/z1 variance) | 0.40 | Prediction is 60% less varied than backbone state |
| Conditioning magnitude / z1 | 1.53 | Conditioning vector is larger than the representation |
| Gate (init=1.0) | Stayed at 1.0 | Model couldn't overcome the conditioning |
| Gate (init=0.01) | Went negative (-0.006) | Model actively rejects conditioning |
| Acc before conditioning | 38-43% | Backbone produces good predictions |
| Acc after conditioning | 38-43% | Conditioning is neutral at best |

**Conclusion:** Top-down prediction as conditioning on level 1 is not useful in its
current form. The prediction network produces blurred, position-averaged outputs that
damage the backbone's sharp per-position representations. PC's future role should be
limited to inter-level error communication (bottom-up), not top-down conditioning.

---

## 6. Experimental Results Summary

### Training Runs

| Run | Config | Steps | Token Acc | Exact Acc | Task Loss | Status |
|-----|--------|-------|-----------|-----------|-----------|--------|
| v4-exp1-baseline-r2 | PC, flat precision (zero init) | 15K | 51.5% | 0% | ~1.0 | Completed, poor |
| v4-exp1-baseline-r3 | Same, eval_every=5000 | 15K | ~51% | 0% | ~1.0 | Completed, poor |
| v4-exp1-stable | PC, all fixes, gate=0.01 | ~1K | 18% | 0% | 1.34 | Killed (PC hurting) |
| v4-exp1-no-pc-baseline | No PC, N=2, 4 inner steps | 4.5K | 60% | 0% | 0.93 | Killed (plateaued) |
| v4-backbone-n1-d21 | No PC, N=1, 21 inner steps | ~700 | 16.4% | 0% | 1.05 | **Running** |

### Diagnostic Runs (Short)

| Config | Steps | Task Loss | Notes |
|--------|-------|-----------|-------|
| No PC, N=2, 4 steps | 500 | 1.37 | Backbone works |
| PC + residual + gate=1.0 | 1000 | 1.58 (then explodes) | Best PC task loss |
| PC + residual + gate=0.01 | 500 | ~1.65 | Gate went negative |
| No PC, N=1, 21 steps | 700 | 1.05 | **Best so far, still dropping** |

---

## 7. Infrastructure State

### Repository
- **Repo:** `github.com/anwar11235/CORAL-v4`
- **Latest commits include:** all bug fixes, diagnostic scripts, inner_steps_override,
  gate init change, lambda_pred/lambda_pi fixes, sum→mean fixes

### Data
- Generated at `data/sudoku_extreme_1k/{train,test}/`
- Must regenerate on each new Vast.ai instance:
  ```bash
  python coral/data/build_sudoku_dataset.py --output-dir data/sudoku_extreme_1k --subsample-size 1000 --num-aug 1000
  ```

### Instance Setup (every new Vast.ai instance)
```bash
tmux new -s coral
cd /workspace
git clone https://github.com/anwar11235/CORAL-v4.git
cd CORAL-v4
export PYTHONPATH=/workspace/CORAL-v4:$PYTHONPATH
pip install wandb hydra-core omegaconf tqdm huggingface_hub argdantic pydantic einops --break-system-packages
python coral/data/build_sudoku_dataset.py --output-dir data/sudoku_extreme_1k --subsample-size 1000 --num-aug 1000
wandb login
```

Note: `pip install -e .` fails on Vast.ai. Use manual install + PYTHONPATH.

### Diagnostic Scripts
- `scripts/diagnostic_precision.py` — 1000 steps, precision stats (GPU only)
- `scripts/diagnostic_no_pc.py` — 500 steps, no-PC baseline (GPU only)
- `scripts/diagnostic_pc_analysis.py` — 500 steps, detailed PC analysis with
  before/after conditioning accuracy (GPU, use batch_size=16 if running alongside
  another training job)

---

## 8. Immediate Next Steps for Session 3

### Step 1: Check the Running Experiment
The `v4-backbone-n1-d21` run should still be training. Check:
- Token accuracy at step 5000 (should be >50%)
- Exact accuracy (should start appearing once token acc >90%)
- Task loss trajectory

### Step 2: Decide Based on Results
- **If token acc >70% at 20K:** Backbone validated. Proceed to add hierarchy + PC
  as error communication.
- **If token acc 50-70%:** May need learning rate tuning (try 1e-4 like TRM) or
  more training steps.
- **If token acc <50%:** Something else is wrong. Compare against TRM's exact
  configuration more carefully.

### Step 3: Design PC as Error Communication Only
When ready to add PC back:
- Level 1 runs unconditioned (no top-down prediction on level 1)
- PC computes error and precision after level 1 finishes
- Error signal propagates up to level 2 only
- Level 2 builds strategic representations from error patterns
- Level 2's value: better initialisation for next segment (after detach)
- No conditioning gate on level 1 needed

### Step 4: Update Architecture Spec and Build Plan
Both documents need updating to reflect:
- Train-deep-infer-shallow as the central paradigm
- Revised build order (backbone → hierarchy → crystallisation → stochastic)
- PC as inter-level communication, not top-down conditioning
- inner_steps_override for N=1 training

---

## 9. Key Hyperparameters (Current)

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_levels | 1 | For current backbone validation run |
| inner_steps_override | 21 | Matches TRM effective depth |
| use_predictive_coding | false | Disabled for backbone validation |
| lambda_pred | 0.001 | Reduced from 0.1 |
| lambda_pi | 0.01 | Restored after sum→mean fix |
| cond_gate init | 0.01 | Reduced from 1.0 (for when PC is re-enabled) |
| lr | 7e-5 | May increase to 1e-4 if needed |
| batch_size | 64 | |
| K_max | 16 | Deep supervision segments |
| vocab_size | 11 | HRM format: pad=0, digits 1-10 |

---

## 10. Competitive Landscape Reference

| Model | Params | Sudoku-Extreme | Key Feature |
|-------|--------|---------------|-------------|
| HRM | 27M | 55% | Two-timescale recurrence |
| CORAL v3 Phase 1 | 27M | 61.1% exact | PC with separate H/L modules |
| TRM | 5-7M | 87.4% | Single network, 21 inner steps, full backprop |
| Augmented HRM | 27M | 96.9% | Data aug + perturbation + bootstrapping |
| GRAM | 10M | 97.0% | Stochastic transitions |

Our current backbone run (N=1, 21 inner steps, ~8.9M params) is the closest
configuration to TRM. If it approaches TRM's 87%, the backbone is validated
and we can proceed to layer the efficiency mechanisms on top.
