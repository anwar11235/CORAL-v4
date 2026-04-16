# GridAdapter ARC audit

**Date:** 2026-04-16
**Author:** Claude Sonnet 4.6 (with Anwar Haq)
**Scope:** Read-only audit of `coral/adapters/grid.py` and related files.
No code was changed. All line numbers reference the current HEAD (Phase 2 commit).

---

## Sudoku-specific assumptions identified

### 1. Grid dimensions default to 9×9

**Location:** `coral/adapters/grid.py:46-47`
```python
grid_height: int = 9,
grid_width:  int = 9,
```
**Location:** `coral/config.py:174-176`
```python
seq_len:     int = 81       # 9x9 Sudoku
grid_height: int = 9
grid_width:  int = 9
```

**Current Sudoku behavior:** `GridAdapter` defaults to 9×9. `DataConfig` also defaults
`seq_len=81`, `grid_height=9`, `grid_width=9`. Any code that constructs
`GridAdapter` without explicit overrides silently targets Sudoku dimensions.

**Required for ARC:** `grid_height=30, grid_width=30, seq_len=900`. These need to be
set explicitly in both the config and the adapter constructor call in `scripts/train.py`
(which already reads from `config.data.grid_height/grid_width`, so no train.py change
is needed — only the YAML config).

**Change scope:** Small — config-gated already; Maze-Hard at 30×30 set the precedent.
The pattern `GridAdapter(coral_config, vocab_size=11, grid_height=30, grid_width=30)`
already exists in `test_maze_support.py`.

---

### 2. `learned_z_init_seq_len` hardcodes 81

**Location:** `coral/config.py:110-111`
```python
# adapter's seq_len (81 for Sudoku). Read at CoralCore construction time.
learned_z_init_seq_len: int = 81
```
**Location:** `coral/model/coral_core.py:189`
```python
torch.zeros(config.learned_z_init_seq_len, config.backbone_dim)
```

**Current Sudoku behavior:** When `use_learned_z_init=True`, a parameter of shape
`[81, backbone_dim]` is allocated and added to the initial state before the first
segment. The default `81` is Sudoku-specific.

**Required for ARC:** Set `learned_z_init_seq_len=900` in the ARC config (if
`use_learned_z_init` is enabled at all). The model would fail at runtime with a
shape mismatch (`coral_core.py:386`) if `learned_z_init_seq_len != L` and
`use_learned_z_init=True`.

**Change scope:** Small — config field, ARC config must set it to 900. Currently
`use_learned_z_init` defaults to `False`, so this is only relevant if enabled.

---

### 3. RoPE `max_seq_len` is 1024 — safely covers 900 but undocumented

**Location:** `coral/model/backbone.py:91`
```python
def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024) -> None:
```

**Current Sudoku behavior:** RoPE positional frequencies are precomputed for up to 1024
positions. Sudoku uses 81.

**Required for ARC:** ARC with 30×30 padding needs seq_len=900. The current default
of 1024 is already sufficient. No change required, but the `max_seq_len=1024` comment
should note it covers ARC (900 < 1024). If any config tried to use full-attention
over a batch of ARC tasks with a longer sequence concatenating demos, 1024 could be
exceeded — but that is not the current architecture.

**Change scope:** None for the standard single-grid encoding. Informational.

---

### 4. `build_attention_masks` uses a Sudoku box concept (3×3 sub-grids)

**Location:** `coral/adapters/grid.py:129-178`

```python
# Box size is grid_height // 3 × grid_width // 3 (i.e. 3×3 for Sudoku).
if self.grid_height % 3 == 0 and self.grid_width % 3 == 0:
    box_h = self.grid_height // 3
    box_w = self.grid_width // 3
    ...
    box_idx = (row // box_h) * n_box_cols + (col // box_w)
else:
    box_idx = row   # fallback: each row is its own "box"
```

**Current Sudoku behavior:** For 9×9 Sudoku the "box" mask encodes the 9 disjoint
3×3 sub-grids that are a fundamental Sudoku constraint. The backbone learns a scalar
`box_bias` weight to attend within the same sub-grid.

**Required for ARC:** ARC has no box constraint. The masks would still be *generated*
(30 is divisible by 3, so box_h=10, box_w=10 producing 9 "boxes" of 10×10 cells),
but the `box_bias` scalar would be learning an arbitrary grouping with no semantic
meaning. More critically, for an ARC grid with padding, the padded positions (tokens
at row ≥ H_actual or col ≥ W_actual) are structurally unrelated to the real content
but would be grouped with real cells in the same row/box masks.

Two actual problems:
1. `box_bias` learns noise (minor — can set to 0 or disable `use_local_attention_bias`).
2. There is no **padding attention mask**: padded positions (token=10, outside the actual
   grid) currently receive the same attention treatment as real positions. The backbone
   has no way to know which positions are padding. See item 6.

**Change scope:** Small to medium. For ARC: set `use_local_attention_bias=False` in
config (same as `phase1_maze_baseline.yaml`). This zeroes out row/col/box bias scalars
entirely (they're never applied when the config flag is off). The padding mask issue
(item 6) requires separate work.

---

### 5. `encode` takes a flat `[B, seq_len]` tensor — no native 2D or masked input

**Location:** `coral/adapters/grid.py:87-108`
```python
def encode(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, seq_len] integer token tensor (e.g., [B, 81] for Sudoku)
    tok = self.token_emb(x)                    # [B, L, d_model]
    pos = self.pos_emb(self.pos_indices)...    # [1, L, d_model]
    emb = tok + pos
```

**Current Sudoku behavior:** Encodes a flattened row-major grid. For Sudoku all 81
positions are valid (no padding), so the flat encoding is lossless.

**Required for ARC:** The `ARCTaskDataset` loader (Part 1 of this session) returns
`inputs` as a padded 30×30 tensor. Before calling `encode`, the caller must flatten
it to `[B, 900]` via `.reshape(B, -1)`. This is the same operation needed for Maze-
Hard (also 30×30 flat). No change to `grid.py` is needed — the reshape happens in
the training script or dataset collate function.

The positional embeddings correctly correspond to `pos_idx = row * 30 + col` for all
900 positions, so padded positions at rows ≥ H_actual get valid (though meaningless)
positional embeddings. This is acceptable if a padding attention mask is also added
(see item 6).

**Change scope:** None to `grid.py`. Data pipeline reshapes 2D → 1D before encoding.

---

### 6. No padding attention mask — all positions treated as valid

**Location:** `coral/adapters/grid.py:129-178` (`build_attention_masks`), and the
calling site in `coral/training/trainer.py:~40` and `coral/evaluation/evaluator.py:181`

**Current Sudoku behavior:** Sudoku has no padding — every position in the 81-token
sequence is a real cell. All positions attend to each other freely. The three masks
returned by `build_attention_masks` are structural (row/col/box), not validity masks.

**Required for ARC:** ARC grids are variable-size padded to 30×30. A 3×3 actual grid
has 9 real positions and 891 padding positions. The backbone should not attend from
or to padding positions. Without an explicit padding mask:
- Backbone self-attention treats padding tokens as content.
- The positional embedding for padding positions is meaningful (row 0, col 3–29),
  which may confuse the model.
- Gate MLP takes padded positions as input and produces gate values for them,
  which then affect the state at those positions.

This is the **critical blocker** for ARC. The current architecture has no mechanism
to pass a per-token validity mask into the backbone SDPA call. The `mask` argument in
`RotaryAttention.forward` (`coral/model/backbone.py:105`) is an additive attention
bias, not a binary key-padding mask.

**Change scope:** Medium. Requires:
1. Adding a `padding_mask: Optional[torch.Tensor]` argument to `GridAdapter.encode`
   (to zero-out or mask pad token embeddings).
2. Passing the `input_mask` from `ARCTaskDataset` through the training pipeline to
   `CoralCore.forward` as an additive attention bias (−∞ at padding positions).
3. Excluding padded positions from the loss computation in `CoralLoss`.
4. Excluding padded positions from eval metrics.

The additive attention bias approach (set bias to −∞ at pad positions) is the cleanest
path and is consistent with how `build_attention_masks` already works.

---

### 7. Eval metrics are Sudoku-specific (difficulty buckets, empty-cell accuracy)

**Location:** `coral/evaluation/evaluator.py:16-23` (bucket definitions)
```python
_DIFFICULTY_BUCKETS = [
    ("0_29",    0,  29),
    ("30_49",  30,  49),
    ("50_59",  50,  59),
    ("60_plus", 60, 10_000),
]
```

**Location:** `coral/evaluation/evaluator.py:243`
```python
empty_mask = (inputs == 1)   # [B, L] — empty cells in input
```

**Location:** `coral/evaluation/evaluator.py:295-306`
```python
n_empty = int(empty_mask[b].sum().item())
key = _bucket_key(n_empty)
...
acc["empty_correct"] += int((correct_b & empty_b).sum().item())
```

**Current Sudoku behavior:** Puzzles are stratified by the number of pre-filled digits
removed (empty cells = token 1). Metrics include per-bucket token accuracy, empty-cell
accuracy, and exact accuracy. These are meaningful for Sudoku because difficulty
correlates with the number of empty cells.

**Required for ARC:** ARC has no concept of "empty cells" (token 1 means color 1, not
"unknown"). The bucket system is entirely inapplicable. ARC's primary eval metric is:
- **Exact match** on the output grid (excluding padding positions).
- Optionally: Chollet's 2-guess or 3-guess protocol (top-N predictions per task).
- Secondary: token accuracy on non-padded output positions.

The bucket accumulator in `evaluate_accuracy` always runs (lines 172–175 create it
unconditionally), even for non-Sudoku datasets. The `empty_mask = (inputs == 1)` line
(243) is inside the main eval loop and would mis-identify ARC color-1 cells as "empty"
cells, corrupting the bucket metrics.

**Change scope:** Medium. The evaluator's `else` branch (not maze, not Sudoku) falls
through to Sudoku bucket metrics by default. ARC needs a third branch:
```python
elif dataset_name == "arc_agi_1":
    # exact match on non-padded output cells only
    # no bucket metrics, no empty-cell accuracy
```
The `dataset_name` mechanism already exists and is the correct extension point.

---

### 8. Repr diagnostics use Sudoku-specific tokens

**Location:** `coral/training/trainer.py:345-351`
```python
dataset = getattr(self.config.data, "dataset", "sudoku_extreme_1k")
if dataset == "sudoku_extreme_1k":
    interesting_token, mask_from = 1, "inputs"   # token 1 = empty Sudoku cell
elif dataset == "maze_30x30_hard":
    interesting_token, mask_from = 5, "labels"   # token 5 = path
else:
    return {}   # unsupported dataset → no repr diagnostics
```

**Current Sudoku behavior:** Representation diagnostics (`repr/inter_position_similarity`,
`repr/same_digit_similarity`, `repr/effective_rank`, `repr/state_norm_*`) are computed
over states at empty cell positions (token=1 in the input).

**Required for ARC:** The `else: return {}` branch already correctly handles unknown
datasets by returning no diagnostics. No code change is needed — ARC would simply skip
repr diagnostics. If ARC-specific diagnostics are desired later (e.g., clustering by
output color), a new branch can be added.

**Change scope:** None for initial ARC support (diagnostics already silently disabled).
Small if ARC-specific repr diagnostics are desired.

---

### 9. `dataset_name` string is not centrally validated

**Location:** `coral/evaluation/evaluator.py:133` (function signature), `coral/training/
trainer.py:345`, `coral/training/losses.py:173`

**Current Sudoku behavior:** The string `"sudoku_extreme_1k"` is used as the canonical
identifier in multiple places and is set by `config.data.dataset`. If a typo or new
dataset name is introduced, the fallback is silently wrong (Sudoku bucket metrics are
applied to a non-Sudoku dataset, or maze metrics are applied incorrectly).

**Required for ARC:** Add `"arc_agi_1"` to the known dataset names. The risk is that
the fallback `else` branch of `evaluate_accuracy` applies Sudoku bucket metrics to ARC
data (the `empty_mask = (inputs == 1)` line would fire for ARC color-1 cells).

**Change scope:** Small — add an explicit ARC branch to `evaluate_accuracy`'s dataset
dispatch. Consider extracting the dataset name strings to a shared constant.

---

## Summary

### Total change scope

**~3–5 days of focused implementation**, depending on whether the padding attention mask
(item 6) is done robustly or minimally. The architectural plumbing for variable-size
masking is the main new work; everything else is config and eval wiring.

### Critical blockers

Items that would prevent ARC from running at all (or producing valid results):

1. **Padding attention mask (item 6)** — Without this, the backbone attends equally
   to padded and real positions for every 3×3 or small ARC grid. The model would
   learn to predict over 900 positions when only 9 are real. This is a correctness
   blocker, not just a quality issue. **Medium effort.**

2. **Eval metric dispatch for ARC (item 7)** — Without a dataset branch for ARC,
   `evaluate_accuracy` applies `empty_mask = (inputs == 1)` to ARC color-1 cells.
   Bucket metrics would be silently wrong. **Small effort.**

3. **Loss masking for padded positions (item 6, loss side)** — `CoralLoss` computes
   cross-entropy over all `seq_len=900` positions. Padded positions (token=10 in labels)
   would contribute to the loss unless excluded. The current `IGNORE_LABEL_ID=-100`
   mechanism already handles this *if* padded label positions are set to -100 in the
   data pipeline. The `ARCTaskDataset` returns padded positions with `label=10`
   (pad token), not `-100`. A collate function or pre-processing step must remap these.
   **Small effort** (add a remapping step analogous to `_collate` in `sudoku_dataset.py`).

### Nice-to-have (not blockers)

- Disable `use_local_attention_bias` for ARC in config (item 4). Already the default
  for Maze-Hard; just needs the ARC YAML config to set it explicitly.
- ARC-specific repr diagnostics (item 8). The `else: return {}` fallback is safe.
- Centralize dataset name strings as constants (item 9). Not blocking.

### Open design questions

**Q1. How are demo pairs used during training?**
The `ARCTaskDataset` provides `demo_pairs` as a list of (input, output) tensors.
CORAL currently has no mechanism to consume these as task context. Options:
  - Ignore demos for a baseline run (train on test input → test output only).
  - Prepend demos to the input sequence (seq_len grows with n_demos × 900).
  - Use a separate context encoder (architectural change, out of scope for this session).
The baseline (ignore demos) is the fastest path to a first ARC number.

**Q2. What is the ARC eval protocol?**
Chollet's protocol allows up to 3 guesses per task (top-3 from the model). This
changes eval from "argmax accuracy" to "top-K accuracy." The current evaluator has no
top-K prediction mechanism. For a first pass, exact-match of argmax is sufficient.

**Q3. Should ARC grids be flattened row-major or use 2D spatial encoding?**
The current GridAdapter uses row-major flattening with a joint 2D positional embedding
(one learnable vector per (row, col) pair). For ARC's variable-size grids, the
positional embedding always covers the full 30×30 space even for a 3×3 grid — most
positions are permanently mapped to pad-token embeddings. A relative or dynamic
positional scheme would be more natural, but is an architectural change.

**Q4. Is `learned_z_init` useful for ARC?**
For Sudoku, a learned initial state at every position makes sense (each of the 81
positions has a stable semantic role). For ARC, positions beyond the actual grid size
vary per task — a learned initial state for 900 positions would mostly learn a prior
over padded positions. Recommendation: keep `use_learned_z_init=False` for ARC.
