# CORAL v4.2 — Claude Code Prompts

One prompt per build session. Provide each prompt to Claude Code sequentially.

---

## Session 1 Prompt

```
I am upgrading the CORAL v4 codebase to v4.2. This is Session 1 of 8: Config Update + Baseline Mode.

CONTEXT:
- The repo is at github.com/anwar11235/CORAL-v4
- 35/35 tests currently pass
- The v4.2 architecture spec is attached (docs/CORAL_Architecture_Spec_v4.2.md)
- The handoff note from v4 is attached (CORAL_v4_Handoff_Note.md)

TASK:
1. Update `coral/config.py` to add v4.2 fields and deprecate obsolete ones:
   - ADD: use_crystallisation (bool, default False), codebook_heads (int, default 8), codebook_entries_per_head (int, default 32), tau_converge (float, default 0.01), tau_decrystallise (float, default 0.05), n_stable (int, default 2), lambda_commit (float, default 0.25), lambda_dis (float, default 0.01), precision_momentum (float, default 0.99), use_local_attention_bias (bool, default True), mode (str, default "baseline" — options: "baseline", "pc_only", "full")
   - KEEP but note as deprecated in comments: lambda_pi (kept for backward compat but unused in v4.2)
   - All existing fields must keep their defaults so existing configs don't break

2. Modify `coral/model/coral_core.py` to support a mode='baseline' path:
   - When config.mode == "baseline": skip ALL predictive coding logic. The forward pass becomes: for each segment, run backbone for T inner steps at level 1 only (N=1), decode, compute halting, detach.
   - When config.mode == "pc_only": current behavior (PC with precision)
   - When config.mode == "full": PC + crystallisation (crystallisation will be wired in Session 5; for now, just ensure the code path exists and raises NotImplementedError)
   - The baseline mode should work with n_levels=1 (single level, d=512, no hierarchy)

3. Create `configs/phase1_baseline_no_pc.yaml`:
   - model.mode: "baseline"
   - model.n_levels: 1
   - model.use_predictive_coding: false
   - model.use_crystallisation: false
   - model.use_local_attention_bias: true
   - model.vocab_size: 11
   - training.epochs: 20000
   - All other training params same as exp1_baseline.yaml
   - wandb.tags: ["v4.2", "phase1", "baseline", "no-pc"]

4. Add tests:
   - Test that coral_core in baseline mode produces correct output shape [B, 81, 512]
   - Test that baseline mode forward + backward completes without error
   - Test that baseline mode has zero prediction/precision network parameters in the computation graph

5. Verify ALL existing 35 tests still pass.

CONSTRAINTS:
- Do not modify backbone.py, level_module.py, halting.py, grid.py, or any evaluation code
- Do not break any existing config or test
- Update CLAUDE.md with all that was accomplished in this session
- Push to git when done
```

---

## Session 2 Prompt

```
I am upgrading CORAL v4 to v4.2. This is Session 2 of 8: Backbone Local Attention Bias.

Session 1 has been completed — config.py has v4.2 fields including use_local_attention_bias.

CONTEXT:
- The v4.2 architecture spec is attached (docs/CORAL_Architecture_Spec_v4.2.md), see Section 4.2
- The backbone is in coral/model/backbone.py — a 2-layer transformer with RoPE, RMSNorm, SwiGLU, PyTorch SDPA

TASK:
1. Modify `coral/model/backbone.py`:
   - Add 3 learnable scalar parameters: row_bias, col_bias, box_bias (nn.Parameter, init to 0.0)
   - The backbone's forward method should accept an optional `attention_bias: Optional[Tensor]` parameter of shape [L, L]
   - Before the SDPA call, if attention_bias is provided, add it to the attention logits
   - If use_local_attention_bias is False in config (or attention_bias is None), skip this entirely — backward compatible
   - NOTE: PyTorch SDPA accepts an `attn_mask` parameter. Use this to pass the bias. Make sure the bias is broadcastable across batch and head dimensions.

2. Modify `coral/adapters/grid.py`:
   - Add a method `build_attention_bias(grid_size=9) -> Tensor` that creates the [81, 81] bias mask
   - For each pair of positions (i, j): mask[i,j] = row_bias * same_row(i,j) + col_bias * same_col(i,j) + box_bias * same_box(i,j)
   - same_row, same_col, same_box are binary (1 if same row/col/box, 0 otherwise)
   - The bias scalars (row_bias, col_bias, box_bias) live in the backbone, not the adapter. The adapter just provides the binary masks. The backbone multiplies and adds.
   - So actually: the adapter returns 3 binary mask tensors [L, L], and the backbone applies its 3 learned scalars to them.

3. Modify `coral/model/coral_core.py`:
   - Pass the attention bias masks from the adapter through to the backbone during forward
   - The adapter should compute masks once per forward call (they don't change across segments)

4. Tests:
   - Test: backbone output changes when local_attention_bias params are non-zero
   - Test: attention bias masks have correct shape [81, 81] and correct structure (e.g., positions 0 and 1 are in the same row, positions 0 and 9 are in the same column, positions 0 and 10 are in the same box)
   - Test: backbone with attention_bias=None produces same result as without the feature (backward compatible)
   - All existing tests still pass

5. Push to git.
```

---

## Session 3 Prompt

```
I am upgrading CORAL v4 to v4.2. This is Session 3 of 8: Running-Statistics Precision.

Sessions 1-2 have been completed.

CONTEXT:
- The v4.2 architecture spec is attached (docs/CORAL_Architecture_Spec_v4.2.md), see Sections 6.2-6.6
- The current precision implementation is in coral/model/predictive_coding.py as PrecisionNetwork
- The v4 handoff note documents 8 failed attempts to learn precision via backprop
- The v4.2 design replaces learned precision with running EMA statistics

TASK:
1. Modify `coral/model/predictive_coding.py`:
   - REMOVE the PrecisionNetwork class entirely
   - ADD a RunningPrecision class (nn.Module with register_buffer, NO learnable parameters):
     - __init__(self, dim, momentum=0.99, eps=0.01): register_buffer 'ema_var' as ones(dim)
     - update(self, prediction_error): @torch.no_grad(), compute per-dim variance of prediction_error across batch and position dims, EMA update ema_var
     - precision property: return 1.0 / (self.ema_var + self.eps)
     - per_head_precision(self, n_heads=8): compute mean variance within each head's dimensions, return H precision values
   - UPDATE PredictiveCodingModule to use RunningPrecision instead of PrecisionNetwork:
     - The precision is treated as a CONSTANT for gradient purposes (already detached via @torch.no_grad)
     - Precision is used to weight the prediction error: xi = precision * eps
     - RunningPrecision.update() is called once per segment, not per inner step
     - The precision_net input used to be cat(z_lower, eps) — this is no longer needed since RunningPrecision just tracks statistics

2. Modify `coral/training/losses.py`:
   - REMOVE the precision regulariser loss (L_pi = lambda_pi/2 * (log pi)^2)
   - UPDATE the prediction loss to use running precision as a constant multiplier: L_pred = lambda_pred * mean(pi.detach() * eps^2)
   - The precision value should come from RunningPrecision.precision, detached

3. Modify `coral/config.py`:
   - Add precision_momentum: float = 0.99 (if not already added in Session 1)
   - Mark lambda_pi as deprecated with a comment

4. Tests:
   - Test: RunningPrecision initialises with uniform precision ≈ 1/(1+eps)
   - Test: After feeding non-uniform error variance, precision differentiates (std > 0.01)
   - Test: RunningPrecision has exactly 0 learnable parameters (len(list(module.parameters())) == 0)
   - Test: precision tensor is not in the computation graph (precision.requires_grad == False)
   - Test: per_head_precision returns H values
   - Test: PredictiveCodingModule forward pass works with RunningPrecision
   - Test: prediction loss computes without NaN
   - All existing tests pass (update any that tested PrecisionNetwork directly)

5. Push to git and update Claude.md
```

---

## Session 4 Prompt

```
I am upgrading CORAL v4 to v4.2. This is Session 4 of 8: Multi-Headed Semantic Codebooks.

Sessions 1-3 have been completed.

CONTEXT:
- The v4.2 architecture spec is attached (CORAL_Architecture_Spec_v4.2.md), see Sections 7 and 8
- This session creates the NEW crystallisation module — the core v4.2 component
- No integration with coral_core yet (that's Session 5)

TASK:
Create `coral/model/crystallisation.py` with three classes:

1. MultiHeadedCodebook(nn.Module):
   - __init__(dim=512, n_heads=8, entries_per_head=32, ema_decay=0.99)
   - Stores codebooks as buffer: [H, M, d_head] where d_head = dim // n_heads
   - Stores usage_count as buffer: [H, M]
   - quantise(z, temperature=1.0, hard=True):
     - Split z [B, L, dim] into [B, L, H, d_head]
     - For each head: compute distances to all entries, find nearest
     - If hard: return nearest entries (straight-through gradient)
     - If not hard: Gumbel-Softmax soft assignment at given temperature
     - Return: z_quantised [B, L, dim], indices [B, L, H], per_head_distances [B, L, H]
   - update_ema(z, indices): update codebook entries toward assigned states
   - dead_code_restart(z_buffer, threshold=0): replace entries with usage_count <= threshold using random samples from z_buffer
   - commitment_loss(z, z_quantised): per-head ||z_h - sg(e_h)||^2
   - disentanglement_loss(): Σ_{h1≠h2} ||C_h1^T · C_h2||²_F / (M_h1 * M_h2)
   - initialise_from_kmeans(states): split states into heads, run k-means per head, set codebook entries to centroids
   - get_perplexity(): per-head codebook usage entropy

2. ConvergenceMonitor (plain class, not nn.Module — no parameters):
   - __init__(n_heads, d_head, tau_converge=0.01, tau_decrystallise=0.05, n_stable=2)
   - Maintains state: z_prev [B, L, dim], consecutive_converged [B, L, H], crystallised [B, L, H]
   - reset(batch_size, seq_len): reset all tracking state for new batch
   - update_and_crystallise(z_current, codebook):
     - Compute per-head velocity: ||z_h_current - z_h_prev||
     - Update consecutive_converged counter: increment where velocity < tau, reset where >=
     - Newly crystallised = (consecutive_converged >= n_stable) & (not already crystallised)
     - For newly crystallised heads: snap to nearest codebook entry
     - Return: crystallisation_mask [B, L, H], newly_crystallised [B, L, H]
   - check_decrystallisation(z_proposed, codebook_values):
     - For crystallised heads: compute drift = ||z_proposed_h - codebook_h||
     - Where drift > tau_decrystallise: unfreeze (set crystallised = False)
     - Return: decrystallised [B, L, H]
   - enforce(z, codebook_values):
     - Overwrite crystallised heads with their codebook values
     - Return: z with frozen heads restored

3. CrystallisationManager(nn.Module):
   - __init__(config): creates MultiHeadedCodebook and ConvergenceMonitor
   - step(z, z_prev, segment_idx):
     - Calls monitor.update_and_crystallise
     - If training: also updates codebook EMA
     - Every 1000 steps: dead_code_restart
     - Returns: z_crystallised, mask, stats_dict
   - enforce_after_backbone(z):
     - Calls monitor.enforce
   - get_losses(): returns commitment_loss + disentanglement_loss
   - get_stats(): returns dict with crystallisation_rate, per_head_rates, perplexity, etc.

TESTS (comprehensive):
- Test: MultiHeadedCodebook has correct parameter/buffer counts (codebooks are buffers, not parameters — they update via EMA, not gradient)
- Test: quantise produces correct shapes [B, L, dim] and [B, L, H]
- Test: quantised output differs from input (nearest-neighbour is not identity)
- Test: EMA update moves codebook entries (check that entries change after update)
- Test: dead_code_restart replaces unused entries
- Test: commitment_loss = 0 when z equals codebook entries
- Test: disentanglement_loss = 0 when codebook heads are orthogonal
- Test: disentanglement_loss > 0 when codebook heads are correlated
- Test: ConvergenceMonitor correctly detects convergence after n_stable steps of low velocity
- Test: ConvergenceMonitor does NOT crystallise when velocity > threshold
- Test: de-crystallisation fires when drift > tau_decrystallise
- Test: enforce correctly overwrites crystallised heads while preserving active heads
- Test: partial crystallisation — some heads crystallised, others not, in the same position
- Test: CrystallisationManager full pipeline works end-to-end
- Test: codebook_entries_per_head=32, n_heads=8, dim=512 gives total buffer size [8, 32, 64]

Push to git.
```

---

## Session 5 Prompt

```
I am upgrading CORAL v4 to v4.2. This is Session 5 of 8: Integrate Crystallisation into Core.

Sessions 1-4 have been completed. The crystallisation module exists in coral/model/crystallisation.py but is not yet wired into coral_core.py.

CONTEXT:
- The v4.2 architecture spec is attached (CORAL_Architecture_Spec_v4.2.md)
- coral_core.py currently supports mode="baseline" (Session 1) and mode="pc_only"
- This session wires crystallisation into mode="full" and updates losses/trainer for logging

TASK:
1. Modify `coral/model/coral_core.py`:
   - Import CrystallisationManager from crystallisation.py
   - In __init__: if config.use_crystallisation, create CrystallisationManager
   - In forward pass, when mode="full":
     - At the START of each segment: call crystallisation_manager.step(z, z_prev) to detect convergence and trigger crystallisation
     - After EACH backbone pass within a segment: call crystallisation_manager.enforce_after_backbone(z) to overwrite crystallised heads
     - At the END of each segment: call check_decrystallisation with the backbone's proposed output
     - Collect crystallisation stats into the output dict
   - CRITICAL: During training, the task loss must be computed from the FULL state (after backbone, before enforce). The enforce operation is applied to the state that carries forward to the next segment, but the logits for loss come from the unmodified backbone output. This ensures the backbone always gets gradient signal.
   - During eval: enforce is applied before decode, so crystallised positions use codebook states for the final answer.
   - The crystallisation losses (commitment + disentanglement) should be returned in the output dict so the trainer can add them to total loss.

2. Modify `coral/training/losses.py`:
   - Add commitment_loss and disentanglement_loss to the total loss computation
   - These are gated by config.use_crystallisation
   - commitment_loss weighted by config.lambda_commit (default 0.25)
   - disentanglement_loss weighted by config.lambda_dis (default 0.01)

3. Modify `coral/training/trainer.py`:
   - Log crystallisation stats to W&B at every eval step:
     crystal/rate_total, crystal/rate_per_head (as a histogram or per-head scalars),
     crystal/bypass_accuracy, crystal/decrystallisation_rate,
     codebook/perplexity_per_head, codebook/dead_codes_per_head
   - Compute bypass_accuracy: compare decode(z_full) vs decode(z_crystallised) at eval time

4. Tests:
   - Test: coral_core forward works in mode="full" with crystallisation enabled
   - Test: output shapes are unchanged regardless of crystallisation
   - Test: crystallisation stats are present in output dict when use_crystallisation=True
   - Test: crystallisation stats are absent when use_crystallisation=False
   - Test: backward pass works in mode="full" (gradients flow through backbone despite enforce)
   - Test: commitment_loss and disentanglement_loss appear in total loss
   - ALL existing tests still pass

Push to git.
```

---

## Session 6 Prompt

```
I am upgrading CORAL v4 to v4.2. This is Session 6 of 8: Representation Diagnostics + State Collection.

Sessions 1-5 have been completed.

CONTEXT:
- The v4.2 architecture spec is attached (CORAL_Architecture_Spec_v4.2.md), see Sections 12 (Phase 1, Phase 2)
- The research plan (CORAL_v4_Research_Plan_Crystallisation_First.md) is attached
- Phase 1 training will produce a baseline model; Phase 2 analyses its representations

TASK:
1. Modify `coral/training/trainer.py` — add representation diagnostics at eval steps:
   - At every eval step, compute and log to W&B:
     a) repr/inter_position_similarity: mean pairwise cosine similarity between empty-cell states at segment 16, averaged across eval puzzles
     b) repr/same_digit_similarity: for each digit 1-9, mean cosine similarity between states that have that ground-truth digit, across puzzles. Log the mean across digits.
     c) repr/effective_rank: run PCA on a sample of segment-16 states (~5000 vectors), compute the number of components explaining 90% of variance
     d) repr/state_norm_mean and repr/state_norm_std: mean and std of L2 norms of segment-16 states
   - These should only be computed at eval steps, not every training step (too expensive)
   - Use @torch.no_grad() for all diagnostics

2. Create `scripts/collect_states.py`:
   - CLI args: --checkpoint (path to trained model), --data-dir (path to eval data), --output (path to save .npz), --segments (comma-separated list of segment indices to collect, default "4,8,12,16"), --device (cuda/cpu)
   - Loads the trained model and eval dataset
   - Runs forward pass on entire eval set (1000 puzzles)
   - At specified segments, records z_states for level 1
   - Saves to .npz with arrays:
     'states': [N_puzzles, 81, len(segments), 512]
     'labels': [N_puzzles, 81] — ground truth digits
     'given_mask': [N_puzzles, 81] — True where cell is a given (clue)
     'segment_indices': the list of segment indices collected
   - Memory-efficient: process puzzles in batches, accumulate on CPU

3. Create `scripts/codebook_analysis.py`:
   - CLI args: --states (path to .npz from collect_states), --output-dir (path to save figures and report), --segment-idx (which collected segment to analyse, default -1 = last)
   - Loads states, filters to empty cells only, uses the specified segment
   - Runs these analyses:
     a) Per-head k-means: split 512-d into 8 groups of 64-d, run k-means with k in {16, 32, 64, 128} on each group. Report per-head: inertia, cluster purity (fraction of states in each cluster sharing the most common digit), codebook perplexity.
     b) Whole-vector k-means: run k-means with k in {64, 128, 256, 512} on full 512-d vectors. Same metrics.
     c) Bypass accuracy: for each k in per-head analysis, replace each state with its nearest centroid, decode (linear projection using saved decoder weights or just argmax of the centroid → check if the cluster's majority digit matches the true digit), report accuracy.
     d) Codebook-to-accuracy curve: plot bypass accuracy vs k
     e) Cluster visualisation: t-SNE on a random 5000-state sample, coloured by digit. Save as PNG.
   - Output: printed summary report + saved plots to output_dir
   - Dependencies: sklearn for k-means and t-SNE, matplotlib for plots

4. Tests:
   - Test: collect_states runs on a tiny synthetic dataset (2 puzzles) without error
   - Test: codebook_analysis runs on synthetic .npz data without error
   - Test: representation diagnostics compute valid numbers (not NaN, not zero)

Push to git.
```

---

## Session 7 Prompt

```
I am upgrading CORAL v4 to v4.2. This is Session 7 of 8: Updated Configs + Training Scripts.

Sessions 1-6 have been completed.

CONTEXT:
- The v4.2 architecture spec is attached (CORAL_Architecture_Spec_v4.2.md)
- The build plan is attached (CORAL_v4.2_Build_Plan.md)
- All model code is complete. This session creates the remaining configs and ensures the training script handles everything cleanly.

TASK:
1. Create config files:

   a) `configs/phase3a_crystal_simple.yaml`:
      - model.mode: "full"
      - model.n_levels: 1
      - model.use_predictive_coding: false
      - model.use_crystallisation: true
      - model.codebook_heads: 1 (monolithic — single head, 256 entries, dim 512)
      - model.codebook_entries_per_head: 256
      - model.tau_converge: 0.01
      - model.tau_decrystallise: 0.05
      - model.n_stable: 2
      - model.lambda_commit: 0.25
      - model.lambda_dis: 0.0 (disabled — only 1 head)
      - wandb.tags: ["v4.2", "phase3a", "crystallisation", "monolithic"]

   b) `configs/phase3b_crystal_multihead.yaml`:
      - Same as 3a but: codebook_heads: 8, codebook_entries_per_head: 32
      - model.lambda_dis: 0.01
      - wandb.tags: ["v4.2", "phase3b", "crystallisation", "multihead"]

   c) `configs/phase3c_crystal_decrystal.yaml`:
      - Same as 3b (de-crystallisation is always on, but this config notes it explicitly)
      - wandb.tags: ["v4.2", "phase3c", "crystallisation", "decrystallisation"]

   d) `configs/phase4_amort_with_crystal.yaml`:
      - Same as 3b + model.lambda_amort: 0.01 (or annealed — use training.amort_anneal_start and amort_anneal_end if the trainer supports it)
      - wandb.tags: ["v4.2", "phase4", "amortisation", "crystallisation"]

   e) `configs/phase4_amort_no_crystal.yaml`:
      - Same as phase1_baseline_no_pc but with lambda_amort: 0.01
      - wandb.tags: ["v4.2", "phase4", "amortisation", "no-crystallisation"]

2. Modify `scripts/train.py`:
   - Ensure it cleanly handles ALL config fields from all phase configs
   - If config.use_crystallisation and config.training.codebook_init_from is set (a path to a .npz of k-means centroids), load the centroids and call codebook.initialise_from_kmeans() before training starts
   - Add amortisation loss annealing: if config.training.amort_anneal_start and amort_anneal_end are set, linearly anneal lambda_amort from 0 to config.model.lambda_amort over that step range
   - Log the effective lambda_amort to W&B at each step (so the anneal schedule is visible)

3. Tests:
   - Test: each config file loads without Hydra errors
   - Test: each config creates a valid model that produces correct output shapes
   - Test: amortisation annealing produces correct lambda values at start, middle, and end of anneal range

Push to git.
```

---

## Session 8 Prompt

```
I am upgrading CORAL v4 to v4.2. This is Session 8 of 8: Full Test Suite Update.

Sessions 1-7 have been completed. This is the final verification session.

CONTEXT:
- The v4.2 architecture spec is attached (CORAL_Architecture_Spec_v4.2.md)
- All code changes are complete. This session ensures comprehensive test coverage and that everything works together.

TASK:
1. Update `tests/test_backbone.py`:
   - Add test: local attention bias changes output when enabled
   - Add test: local attention bias with None mask produces identical output to no-bias backbone
   - Add test: attention bias masks for 9×9 grid have correct structure

2. Update `tests/test_predictive_coding.py`:
   - REMOVE all PrecisionNetwork tests
   - Add tests for RunningPrecision:
     - Uniform initialisation
     - Differentiation after non-uniform updates
     - Zero learnable parameters
     - No gradient flow through precision
     - per_head_precision correctness
   - Update PredictiveCodingModule tests to use RunningPrecision

3. Create `tests/test_crystallisation.py`:
   - MultiHeadedCodebook: shapes, quantise, EMA update, dead_code_restart, commitment_loss, disentanglement_loss, initialise_from_kmeans, get_perplexity
   - ConvergenceMonitor: convergence detection, non-convergence, de-crystallisation, enforce, partial crystallisation
   - CrystallisationManager: full pipeline end-to-end

4. Update `tests/test_coral_core.py`:
   - Test all three modes: baseline, pc_only, full
   - Test that each mode produces correct output shapes
   - Test that crystallisation stats are present in full mode
   - Test that backward pass works in all modes

5. Update `tests/test_forward_backward.py`:
   - Verify gradient flow in baseline mode (grads reach backbone)
   - Verify gradient flow in pc_only mode (grads reach backbone AND prediction networks)
   - Verify gradient flow in full mode (grads reach backbone, prediction networks, and codebook commitment loss affects encoder)

6. Create or update `tests/test_losses.py`:
   - Test: commitment_loss is non-negative
   - Test: disentanglement_loss is non-negative
   - Test: total loss in full mode includes commitment and disentanglement
   - Test: total loss in baseline mode does NOT include PC or crystallisation losses
   - Test: precision regulariser is NOT in the loss (verify it was removed)

7. Run the full test suite and fix any failures:
   - `pytest tests/ -v` must show ALL PASS, ZERO failures
   - Report the total test count (should be >50 given all the new tests)

8. Verify parameter counts:
   - In baseline mode (N=1): report exact parameter count
   - In full mode (N=1 + crystallisation): report exact parameter count
   - Confirm codebook parameters are BUFFERS not PARAMETERS (they don't count toward optimizer params)

9. Final smoke test:
   - Run `python scripts/train.py --config-name phase1_baseline_no_pc training.epochs=5 wandb.disabled=true` and verify it completes without error
   - Run `python scripts/train.py --config-name phase3b_crystal_multihead training.epochs=5 wandb.disabled=true` and verify it completes without error

Push to git with commit message: "CORAL v4.2: complete codebase upgrade — crystallisation, running precision, multi-headed codebooks"
```
