# Session J Audit Report

Pre-run audit of CORAL v4.2 architecture mechanisms.
Inspected at commit `f521b78` (latest master).
All findings are read-only; zero source files modified.

---

## Verified Working

- **Item 6 (Attention bias):** Correctly gated.  When `use_local_attention_bias=True`,
  `TrainerV4.__init__` calls `adapter.build_attention_masks()` once and stores them as
  `self._attention_masks`; those masks are passed to every `core.forward()` call in
  `train_step`, `eval_step`, and `evaluate_accuracy`.  When the flag is False,
  `attn_bias=None` and SDPA runs as vanilla dot-product.
  `trainer.py:83–84`, `coral_core.py:355–361`.

- **Item 8 (Float64 task loss):** `stablemax_cross_entropy` casts logits to `float64`
  before computing log-stablemax (`losses.py:65`).  `per_token` is float64; `L_task`
  converts back to float32 before mean (`losses.py:183`), which is correct — float32 is
  fine for the optimizer.  Float64 guard is exactly where NaN risk is highest.

- **Item 9 (Detach between segments):** `z_states = [z.detach() for z in z_states]`
  is at `coral_core.py:594`, placed *after* the halting check and *before* the next
  segment begins.  When halting fires early the break happens before the detach, which
  is correct — there is no next segment to detach for.

- **Item 10 (Optimizer weight decay exclusions):** `build_optimizer` uses the keywords
  `("emb", "embedding", "norm")` to route parameters to the no-decay group
  (`optimizer.py:68`).  `pos_emb.weight` contains `"emb"` → no-decay.  `token_emb`,
  `level_emb`, `timescale_emb` all match.  Confirmed correct after Session H's
  row+col → joint pos_emb refactor.

- **Item 11 (Position embedding dimensions):** `GridAdapter` computes `pos_indices =
  rows * grid_width + cols` in row-major order (`grid.py:79`), covering exactly
  `grid_height * grid_width` unique values matching `pos_emb`'s Embedding size.
  No off-by-one or transposition errors.  Works correctly for both Sudoku (9×9=81)
  and maze (30×30=900).

- **Item 13 (Eval/train forward consistency):** `evaluate_accuracy` calls
  `core(z1, K_max=K_max, training=False, decode_fn=adapter.decode,
  attention_masks=attention_masks, ...)` — the same `core.forward()` path as training.
  The old manual segment-loop bug (Session 11) is gone.  `evaluator.py:222–232`.

---

## Broken or Suspicious

### Item 1 — PC: loss/prediction = 0 is correct; premise of the audit question is wrong

`phase1_n2_partial_grad.yaml` sets `use_predictive_coding: false` (`yaml:31`).  The
audit question assumed this config has PC enabled — it does not.  `loss/prediction =
0.0000` at every step is *correct* behaviour, not a bug.

No config in the current repo that is actually being run has PC enabled.  If/when a
config with `use_predictive_coding: true` is run, the loss wiring is correct:

- `coral_core.py:503–511` stores `eps` and `pi` in `seg_eps` / `seg_pi` dicts.
- `trainer.py:122–124` passes them to `loss_fn`.
- `losses.py:193–200` multiplies by `lambda_pred`.

PC loss IS in `total` and will receive gradients.  The only concern is numerical scale:
`lambda_pred: 0.001` means `L_pred ≈ 0.001 * (pi * eps²).mean()`.  With initial
`pi ≈ 1.0` and small errors, logged values may round to `0.0000` at 4 decimal places
without being literally zero — this is a display artefact, not a gradient cut.

### Item 3 — Amortisation: BROKEN for `phase4_amort_no_crystal.yaml`

`phase4_amort_no_crystal.yaml` sets `use_amort: true`, `lambda_amort: 0.01`, and
`use_predictive_coding: false` (mode="baseline").

In `losses.py:231`:
```python
if self.config.use_amort and all_pred_errors is not None:
    L_amort = self.config.lambda_amort * amortisation_loss(all_pred_errors)
```

In baseline mode, `all_pred_errors` is a list of *empty* dicts `[{}, {}, ..., {}]` — one
per segment, but no level keys because no PC is running.  `amortisation_loss` never
enters its inner loop and returns `torch.tensor(0.0)` — a **CPU tensor** with no device
argument (`losses.py:99`).

`L_amort = 0.01 * CPU_tensor(0.0)` stays on CPU.  Then:

```python
total = L_task.float() + L_pred + L_halt + L_amort + ...
```

`L_task`, `L_pred`, `L_halt` are all on CUDA.  Adding `L_amort` (CPU) to CUDA tensors
raises `RuntimeError: Expected all tensors to be on the same device`.

**This config will crash on the first training step.**

Fix: in `amortisation_loss`, change the initialisation to
`total = torch.tensor(0.0, device=<device>)`, or return a zero tensor on the correct
device when `pred_errors` contains no actual error tensors.

### Item 14 — Checkpoint: `collect_states.py` uses wrong keys and wrong config type

`train.py` saves checkpoints as (`train.py:263–276`):
```python
{
    "model_state_dict": {
        "adapter": trainer.adapter.state_dict(),
        "core":    trainer.core.state_dict(),
    },
    "optimizer_state_dict": ...,
    "step": step,
    "config": OmegaConf.to_container(cfg, resolve=True),  # plain dict, not CoralConfig
}
```

`collect_states.py` expects (`collect_states.py:43–66`):
```python
ckpt["adapter_state"]   # key does not exist
ckpt["core_state"]      # key does not exist
config: CoralConfig = ckpt["config"]   # ckpt["config"] is a plain dict, not a dataclass
config.model            # AttributeError on plain dict
```

Two failures:
1. The `if "adapter_state" in ckpt:` guard silently skips loading weights.  The adapter
   and core are returned with random initialisation and no error is raised.
2. `config.model` raises `AttributeError` immediately because the saved config is a
   plain Python dict (from `OmegaConf.to_container`), not a `CoralConfig` dataclass.

`diagnose_rank_collapse.py` uses the correct keys `ckpt["model_state_dict"]["adapter"]`
and `ckpt["model_state_dict"]["core"]` (`diagnose_rank_collapse.py:196–197`), so there
are two divergent loading implementations.

**`collect_states.py` will always crash or silently produce wrong results.**

---

## Unclear (needs further inspection)

### Item 2 — Crystallisation: works in isolation; missing compiler guard

Crystallisation forward path, commitment/dis losses, and EMA codebook update are all
wired and unit-tested (74 tests).  However, `CLAUDE.md` explicitly documents:

> `torch.compile` hangs when tracing crystallisation control flow with Python
> conditionals and `.item()` calls.  Decorate crystallisation methods with
> `@torch.compiler.disable(recursive=False)` if using compile.

No `@torch.compiler.disable` annotation exists anywhere in `coral/model/crystallisation.py`
(grep confirms zero matches).  All current phase3/phase4 configs have `compile_model:
false`, so this cannot bite today.  If someone enables `torch.compile` with
crystallisation in the future, they will get a silent hang.

**Verdict:** correct for current runs; latent footgun for future compile use.

### Item 4 — Halting: K_max always — by design, but halting loss has a learning asymmetry

`avg_halting_step = 16.0` at every eval step is expected during early training.  The
halting network's output layer is initialised to zeros (`halting.py:67–68`), so
`q_halt_logit = 0` and `h = sigmoid(0) = 0.5 < 0.95`.  The threshold is never reached.

The halting loss (loss/halting) is correctly wired in `losses.py:205–225` and
`L_halt = 0.5 * BCE(q_halt, seq_is_correct)` does receive gradients.

Learning asymmetry concern: early in training `seq_is_correct ≈ 0` for all samples, so
the loss pushes `q_halt_logit → −∞` (never halt).  This is self-consistent but creates
a regime where the halting head only learns to halt once accuracy is already high.  The
halting head may effectively stay inactive for the entire bs=64 run.  Not a bug, but
worth monitoring `loss/halting` and `cond_gate` to detect if the head is ever activated.

**Verdict:** no bug; whether the halting head becomes useful depends on training trajectory.

### Item 5 — Inner steps: grad split is correct; consolidation step adds +1 uncounted

For `phase1_n2_partial_grad.yaml`:
- `inner_steps_override: 18`, `grad_inner_steps: 6`
- In `_run_level`: `no_grad_steps = 18 − 6 = 12` warmup + 6 grad steps. ✓
- For level 1 (1 step): `grad_inner_steps=6 ≥ 1` → `no_grad_steps = 0`, all in-graph. ✓

However, `use_consolidation_step` defaults to `True` in `config.py:94` and this config
does NOT override it.  `_run_level` appends an extra no-injection backbone pass when
`input_injection is not None` (`coral_core.py:309–312`).  Level 0 gets injection, so
it runs **19** backbone applications per segment (18 + 1 consolidation), not 18 as the
config comment states.  Level 1 has no injection in baseline mode → no consolidation → 1 step.

Total backbone apps/segment = 19 + 1 = 20 (config comment says 19).

This is a documentation discrepancy, not a gradient bug.  The consolidation step runs
outside the `no_grad_steps` block (`coral_core.py:309`) so it always has gradient.

**Verdict:** partial-grad split is correct; total backbone count is off-by-one vs
documented.

### Item 7 — Precision regulariser: removed in v4.2 — no assertion present (intentional)

The `(λ_π/2)(log π)²` regulariser was removed in v4.2 (`losses.py:187–201`).
`lambda_pi` still exists in config for backward compat but is read by zero code paths.
No assertion `pi.mean() < 100` exists — none is needed because `RunningPrecision.precision`
is bounded: `pi = 1 / (ema_var + eps)` with `eps = 0.01`, so `pi ≤ 1/0.01 = 100`
by construction.  An explicit assertion would be redundant but not harmful.

**Verdict:** correct for v4.2; the audit question's framing is based on v4.1 behaviour
that no longer exists.

### Item 12 — Label semantics: correct by inspection; not exercised by static analysis

In `SudokuDataset._collate`, the `ignore_label_id` remap (`sudoku_dataset.py:179–181`)
replaces any label matching `metadata.ignore_label_id` with `-100`.  The default
metadata has `ignore_label_id: None` (no remapping).  Whether given-digit positions have
their correct label vs IGNORE_LABEL_ID depends on how `build_sudoku_dataset.py` writes
the `.npy` files — this cannot be verified by reading the loader code alone.  From the
Session 9 fixes and test suite passing at 223/223, labels are assumed correct.

**Verdict:** static analysis is inconclusive; test suite acceptance is the evidence.

---

## Incidental Findings

### Dead code: `evaluator.py:104` — unreachable `return "60_plus"`

```python
# evaluator.py  (function _summarize_norm_trace, line 103–104)
    return metrics
    return "60_plus"   # <-- UNREACHABLE
```

`return "60_plus"` is the default fallback from the `_bucket_key` function that was
accidentally placed two lines after `return metrics` inside `_summarize_norm_trace`.
This is never executed but will confuse any reader.

### Dead config fields: `use_column_heads`, `use_stochastic`, `lambda_crystal`, `full_inner_backprop`

Grep across `coral/` finds zero references to these fields outside of `config.py` itself:
- `config.py:42` — `use_column_heads: bool = False` — never read in any model code
- `config.py:43` — `use_stochastic: bool = False` — never read
- `config.py:46` — `lambda_crystal: float = 0.0` — never read; `losses.py` uses
  `config.lambda_dis` for the disentanglement loss (logged as `loss/crystallisation`),
  making `lambda_crystal` a misleadingly named no-op
- `config.py:122` — `full_inner_backprop: bool = True` — never read in training code

### Diagnostic function `_log_precision_metrics` silently ignores `inner_steps_override`

`train.py:305`:
```python
z_states[i] = trainer.core._run_level(z_states[i], i, cfg.timescale_base, None)
```

This passes `cfg.timescale_base = 3` as `n_steps` instead of
`self.core.level_steps[i]` (which would be 18 for level 0 with `inner_steps_override=18`).
The diagnostic runs only 3 inner steps to collect precision/error stats, while training
runs 18.  The logged `precision/level0_*` and `prediction_error/level0_*` values reflect
a dramatically different number of refinement steps than actual training uses.  This makes
the diagnostic misleading, not the training itself.

### `input_injection` reaches ALL levels in `pc_only`/`full` mode

In baseline mode, only level 0 receives `input_injection` (`coral_core.py:447`).  In
`pc_only` / `full` mode, ALL levels receive it (`coral_core.py:495–499`):

```python
z_states[i] = self._run_level(
    z_states[i], i, n_steps, conditioning=cond, ...,
    input_injection=input_signal,   # i=0 and i=1 both get injection
)
```

TRM injects embeddings into the L-level only, not the H-level.  Injecting the raw input
into level 1 may interfere with its role as an abstract context, but this is a design
question, not a definitive bug.  Worth testing whether blocking level 1 injection
improves accuracy when PC is enabled.

---

## Summary Table

| # | Name | Status |
|---|------|--------|
| 1 | Predictive coding | **No bug** — PC disabled in current config; loss=0 is correct |
| 2 | Crystallisation | **Latent** — `@torch.compiler.disable` missing; safe while compile=False |
| 3 | Amortisation | **BROKEN** — `phase4_amort_no_crystal` crashes: CPU tensor device mismatch |
| 4 | Halting | **By design** — K_max always; learning asymmetry worth monitoring |
| 5 | Inner steps | **Correct** — grad split OK; consolidation adds +1 uncounted in comments |
| 6 | Attention bias | **Correct** — gated cleanly; masks passed to all forward calls |
| 7 | Precision regulariser | **N/A** — removed in v4.2; pi is bounded by construction |
| 8 | Float64 loss | **Correct** — float64 at log-stablemax; float32 for optimizer |
| 9 | Detach between segments | **Correct** — `z_states = [z.detach() ...]` at line 594 |
| 10 | WD exclusions | **Correct** — pos_emb via "emb" keyword match |
| 11 | Position embedding | **Correct** — row-major pos_indices, exact H×W size |
| 12 | Label semantics | **Unclear** — static analysis inconclusive; tests pass |
| 13 | Eval/train consistency | **Correct** — evaluator calls core.forward() directly |
| 14 | Checkpoint loading | **BROKEN** — collect_states.py: wrong keys, wrong config type |
| 15 | Incidentals | See above: dead code, dead config fields, misleading diagnostic |
