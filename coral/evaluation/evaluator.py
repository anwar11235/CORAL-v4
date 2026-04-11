"""CORAL v4 — Evaluation metrics.

Computes exact accuracy (full puzzle correct) and token accuracy (per-cell)
on the Sudoku-Extreme-1K evaluation set.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from coral.data.sudoku_dataset import IGNORE_LABEL_ID
from coral.evaluation.repr_diagnostics import compute_repr_diagnostics

# Difficulty buckets defined as (key_suffix, min_empty_inclusive, max_empty_inclusive).
# Empty cells are encoded as token value 1 in the input tensor.
_DIFFICULTY_BUCKETS = [
    ("0_29",    0,  29),
    ("30_49",  30,  49),
    ("50_59",  50,  59),
    ("60_plus", 60, 10_000),
]


def _bucket_key(n_empty: int) -> str:
    """Return the bucket key suffix for a puzzle with n_empty empty cells."""
    for key, lo, hi in _DIFFICULTY_BUCKETS:
        if lo <= n_empty <= hi:
            return key


def compute_precision_recall_at_5(
    preds: torch.Tensor,
    labels: torch.Tensor,
    token: int = 5,
) -> Tuple[float, float]:
    """Compute precision and recall for a specific token, aggregated over all positions.

    Counts are summed flat over B×L (not per-sample averaged) so that each cell
    contributes equally regardless of how many path cells a particular maze has.

    Args:
        preds:  [B, L] predicted token ids.
        labels: [B, L] ground-truth token ids.
        token:  The positive-class token. Default 5 (optimal-path in Maze-Hard).

    Returns:
        (precision, recall) as Python floats. Returns (0.0, 0.0) when the model
        never predicts the token (precision) or no ground-truth positives exist
        (recall), to avoid NaN.
    """
    pred_pos = preds == token
    label_pos = labels == token

    tp = int((pred_pos & label_pos).sum().item())
    fp = int((pred_pos & ~label_pos).sum().item())
    fn = int((~pred_pos & label_pos).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall


def _summarize_norm_trace(
    trace: Dict[str, List[float]],
) -> Dict[str, float]:
    """Flatten a state_norm_trace into scalar metrics for logging.

    Emits:
        norms/z1_init             — norm of the initial encoder output
        norms/injection_delta_mean — mean increase from injection (post - pre)
        norms/post_backbone_mean  — mean level-0 norm after each backbone step
        norms/segment_end_mean    — mean level-0 norm at end of each segment
        norms/segment_end_final   — level-0 norm at end of the last segment
        norms/post_detach_final   — level-0 norm just after the last detach

    All values are averages over the recorded steps; missing lists return 0.0.
    """
    metrics: Dict[str, float] = {}

    z1 = trace.get("z1_init", [])
    metrics["norms/z1_init"] = z1[0] if z1 else 0.0

    pre = trace.get("pre_injection", [])
    post = trace.get("post_injection", [])
    if pre and post and len(pre) == len(post):
        deltas = [p - r for p, r in zip(post, pre)]
        metrics["norms/injection_delta_mean"] = sum(deltas) / len(deltas)
    else:
        metrics["norms/injection_delta_mean"] = 0.0

    pb = trace.get("post_backbone", [])
    metrics["norms/post_backbone_mean"] = sum(pb) / len(pb) if pb else 0.0

    se = trace.get("segment_end", [])
    metrics["norms/segment_end_mean"] = sum(se) / len(se) if se else 0.0
    metrics["norms/segment_end_final"] = se[-1] if se else 0.0

    pd_ = trace.get("post_detach", [])
    metrics["norms/post_detach_final"] = pd_[-1] if pd_ else 0.0

    return metrics
    return "60_plus"


def evaluate_accuracy(
    adapter: nn.Module,
    core: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    K_override: Optional[int] = None,
    max_puzzles: Optional[int] = None,
    dataset_name: str = "sudoku_extreme_1k",
    collect_diagnostics: bool = True,
) -> Dict[str, float]:
    """Evaluate exact and token accuracy on a full evaluation dataset.

    Uses core.forward() directly so that input_injection and attention_bias
    are applied identically to the training forward pass.

    Args:
        adapter:      GridAdapter for encode/decode.
        core:         CoralCore for reasoning.
        dataloader:   Evaluation DataLoader.
        device:       Compute device.
        dtype:        Forward pass dtype.
        K_override:   If provided, force exactly K segments.
        max_puzzles:  If provided, stop after this many puzzles (quick eval).
        dataset_name: Dataset identifier. Controls which metrics are emitted.
                      "maze_30x30_hard" → maze metrics (path/wall/non_path accuracy).
                      Anything else → Sudoku bucket metrics (default behaviour).

    Returns:
        Dict with overall metrics (eval/exact_accuracy, eval/token_accuracy,
        eval/avg_halting_step).

        For Sudoku: also includes per-difficulty-bucket metrics:
            eval/bucket_{key}_count, eval/bucket_{key}_token_acc,
            eval/bucket_{key}_empty_acc, eval/bucket_{key}_exact_acc
            for key in {"0_29", "30_49", "50_59", "60_plus"}.

        For maze: also includes eval/path_accuracy, eval/wall_accuracy,
            eval/non_path_accuracy, per-segment path accuracy, inner-step
            velocity, and repr/* diagnostics.

        Repr diagnostics (repr/*) are added for both datasets when
        K_override is None (full-depth eval only, not Pareto forced-K calls).
    """
    adapter.eval()
    core.eval()

    total_puzzles = 0
    exact_correct = 0
    token_correct = 0
    token_total = 0
    total_segments = 0

    # Maze-specific accumulators (only used when dataset_name == "maze_30x30_hard").
    path_correct = 0
    path_total = 0
    wall_correct = 0
    wall_total = 0
    non_path_correct = 0
    non_path_total = 0
    tp_5 = 0   # predicted 5 AND label 5
    fp_5 = 0   # predicted 5 AND label != 5
    fn_5 = 0   # predicted != 5 AND label 5

    # Per-bucket accumulators: puzzles, token correct/total, empty correct/total, exact correct.
    bucket_acc: Dict[str, Dict[str, int]] = {
        key: {"puzzles": 0, "token_correct": 0, "token_total": 0,
              "empty_correct": 0, "empty_total": 0, "exact_correct": 0}
        for key, _, _ in _DIFFICULTY_BUCKETS
    }

    # Build attention masks once (static — depend only on grid shape, not on input).
    # These are the same masks used during training via trainer.py.
    attention_masks: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    if hasattr(adapter, "build_attention_masks") and core.config.use_local_attention_bias:
        attention_masks = adapter.build_attention_masks(device=device)

    K_max = K_override if K_override is not None else core.config.K_max

    # Per-segment / velocity / repr diagnostics are enabled only for full-depth
    # quick eval (K_override=None, collect_diagnostics=True). Pareto eval passes
    # collect_diagnostics=False to avoid the overhead across all K values.
    collect_maze_diag = (
        (dataset_name == "maze_30x30_hard")
        and (K_override is None)
        and collect_diagnostics
    )
    if collect_maze_diag:
        if K_max <= 5:
            seg_checkpoints = list(range(K_max))
        else:
            seg_checkpoints = sorted(set([
                0, K_max // 4, K_max // 2, 3 * K_max // 4, K_max - 1,
            ]))
        seg_path_corr: Dict[int, int] = {s: 0 for s in seg_checkpoints}
        seg_path_tot_d: Dict[int, int] = {s: 0 for s in seg_checkpoints}
        seg_exact_corr: Dict[int, int] = {s: 0 for s in seg_checkpoints}
        seg_exact_tot_d: Dict[int, int] = {s: 0 for s in seg_checkpoints}
        seg_tp_5: Dict[int, int] = {s: 0 for s in seg_checkpoints}
        seg_fp_5: Dict[int, int] = {s: 0 for s in seg_checkpoints}
        seg_fn_5: Dict[int, int] = {s: 0 for s in seg_checkpoints}
        inner_step_delta_accum: List[List[float]] = []

    # State norm trace: collected from the very first batch only (diagnostic, low overhead).
    _first_batch_norm_trace: Optional[Dict[str, List[float]]] = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            B = inputs.shape[0]

            # Collect state norms on the first batch only (when diagnostics enabled).
            _collect_norms_this_batch = collect_diagnostics and (batch_idx == 0)

            with torch.autocast(device_type=device.type, dtype=dtype):
                z1 = adapter.encode(inputs)
                output = core(
                    z1,
                    K_max=K_max,
                    training=False,
                    decode_fn=adapter.decode,
                    attention_masks=attention_masks,
                    collect_inner_step_states=collect_maze_diag,
                    collect_state_norms=_collect_norms_this_batch,
                )

            if _collect_norms_this_batch and output.state_norm_trace is not None:
                _first_batch_norm_trace = output.state_norm_trace

            logits = output.all_logits[-1] if output.all_logits else adapter.decode(output.z_states[0])
            preds = logits.argmax(dim=-1)
            num_segs = output.num_segments

            mask = labels != IGNORE_LABEL_ID          # [B, L] — valid label positions
            correct = (preds == labels) & mask         # [B, L]
            empty_mask = (inputs == 1)                 # [B, L] — empty cells in input

            # Overall accumulators.
            exact_correct += (correct.sum(-1) == mask.sum(-1)).sum().item()
            token_correct += correct.sum().item()
            token_total += mask.sum().item()
            total_puzzles += B
            total_segments += num_segs

            # Maze-specific accumulators.
            if dataset_name == "maze_30x30_hard":
                path_mask = (labels == 5)
                wall_mask = (labels == 2)
                non_path_mask = (labels != 5)
                path_correct += (correct & path_mask).sum().item()
                path_total += path_mask.sum().item()
                wall_correct += (correct & wall_mask).sum().item()
                wall_total += wall_mask.sum().item()
                non_path_correct += (correct & non_path_mask & mask).sum().item()
                non_path_total += (non_path_mask & mask).sum().item()
                pred_5 = (preds == 5)
                tp_5 += int((pred_5 & path_mask).sum().item())
                fp_5 += int((pred_5 & ~path_mask).sum().item())
                fn_5 += int((~pred_5 & path_mask).sum().item())

            # Per-segment + velocity accumulators (maze, full-depth only).
            if collect_maze_diag:
                for seg_idx in seg_checkpoints:
                    if seg_idx < len(output.all_logits):
                        sl = output.all_logits[seg_idx]
                        sp = sl.argmax(dim=-1)
                        pm = (labels == 5)
                        seg_path_corr[seg_idx] += ((sp == labels) & pm).sum().item()
                        seg_path_tot_d[seg_idx] += pm.sum().item()
                        seg_exact_corr[seg_idx] += (sp == labels).all(dim=-1).sum().item()
                        seg_exact_tot_d[seg_idx] += B
                        sp_5 = (sp == 5)
                        seg_tp_5[seg_idx] += int((sp_5 & pm).sum().item())
                        seg_fp_5[seg_idx] += int((sp_5 & ~pm).sum().item())
                        seg_fn_5[seg_idx] += int((~sp_5 & pm).sum().item())

                if (output.last_segment_inner_states is not None
                        and len(output.last_segment_inner_states) >= 2):
                    sts = output.last_segment_inner_states
                    batch_deltas = [
                        (sts[i] - sts[i - 1]).norm(dim=-1).mean().item()
                        for i in range(1, len(sts))
                    ]
                    inner_step_delta_accum.append(batch_deltas)

            # Per-puzzle bucket accumulators.
            for b in range(B):
                n_empty = int(empty_mask[b].sum().item())
                key = _bucket_key(n_empty)
                acc = bucket_acc[key]
                mask_b = mask[b]
                correct_b = correct[b]
                empty_b = empty_mask[b]

                acc["puzzles"] += 1
                acc["token_correct"] += int(correct_b.sum().item())
                acc["token_total"] += int(mask_b.sum().item())
                acc["empty_correct"] += int((correct_b & empty_b).sum().item())
                acc["empty_total"] += int((empty_b & mask_b).sum().item())
                acc["exact_correct"] += int((correct_b.sum() == mask_b.sum()).item())

            if max_puzzles is not None and total_puzzles >= max_puzzles:
                break

    exact_acc = exact_correct / max(total_puzzles, 1)
    token_acc = token_correct / max(token_total, 1)
    avg_halt = total_segments / max(total_puzzles // B if B > 0 else 1, 1)

    if dataset_name == "maze_30x30_hard":
        maze_metrics: Dict[str, float] = {
            "eval/exact_accuracy": exact_acc,
            "eval/token_accuracy": token_acc,
            "eval/path_accuracy": path_correct / max(path_total, 1),
            "eval/wall_accuracy": wall_correct / max(wall_total, 1),
            "eval/non_path_accuracy": non_path_correct / max(non_path_total, 1),
            "eval/avg_halting_step": avg_halt,
            "eval/precision_at_5": tp_5 / max(tp_5 + fp_5, 1),
            "eval/recall_at_5": tp_5 / max(tp_5 + fn_5, 1),
        }
        # Per-segment path accuracy, exact accuracy, and precision/recall@5.
        if collect_maze_diag:
            for seg_idx in seg_checkpoints:
                maze_metrics[f"eval/seg_{seg_idx}_path_accuracy"] = (
                    seg_path_corr[seg_idx] / max(seg_path_tot_d[seg_idx], 1)
                )
                maze_metrics[f"eval/seg_{seg_idx}_exact_accuracy"] = (
                    seg_exact_corr[seg_idx] / max(seg_exact_tot_d[seg_idx], 1)
                )
                maze_metrics[f"eval/seg_{seg_idx}_precision_at_5"] = (
                    seg_tp_5[seg_idx] / max(seg_tp_5[seg_idx] + seg_fp_5[seg_idx], 1)
                )
                maze_metrics[f"eval/seg_{seg_idx}_recall_at_5"] = (
                    seg_tp_5[seg_idx] / max(seg_tp_5[seg_idx] + seg_fn_5[seg_idx], 1)
                )
            # Inner-step velocity statistics.
            if inner_step_delta_accum:
                n_bv = len(inner_step_delta_accum)
                n_sv = len(inner_step_delta_accum[0])
                avg_deltas = [
                    sum(b[i] for b in inner_step_delta_accum) / n_bv
                    for i in range(n_sv)
                ]
                maze_metrics["eval/last_seg_inner_step_delta_mean"] = (
                    sum(avg_deltas) / len(avg_deltas)
                )
                maze_metrics["eval/last_seg_inner_step_delta_final"] = avg_deltas[-1]
                for i, d in enumerate(avg_deltas):
                    maze_metrics[f"eval/last_seg_inner_step_{i}_delta"] = d
            # Repr diagnostics (maze path cells).
            repr_max = 64 if max_puzzles is None else min(max_puzzles, 64)
            repr_m = compute_repr_diagnostics(
                adapter=adapter, core=core, dataloader=dataloader,
                max_puzzles=repr_max, interesting_token=5, mask_from="labels",
                device=device, dtype=dtype,
            )
            maze_metrics.update(repr_m)
        # State norm diagnostics (first batch only, full-depth eval).
        if _first_batch_norm_trace is not None:
            maze_metrics.update(_summarize_norm_trace(_first_batch_norm_trace))
        return maze_metrics

    metrics: Dict[str, float] = {
        "eval/exact_accuracy": exact_acc,
        "eval/token_accuracy": token_acc,
        "eval/avg_halting_step": avg_halt,
    }

    # Flatten per-bucket stats into metric keys.
    for key, _, _ in _DIFFICULTY_BUCKETS:
        acc = bucket_acc[key]
        n = acc["puzzles"]
        metrics[f"eval/bucket_{key}_count"] = n
        metrics[f"eval/bucket_{key}_token_acc"] = (
            acc["token_correct"] / acc["token_total"] if acc["token_total"] > 0 else 0.0
        )
        metrics[f"eval/bucket_{key}_empty_acc"] = (
            acc["empty_correct"] / acc["empty_total"] if acc["empty_total"] > 0 else 0.0
        )
        metrics[f"eval/bucket_{key}_exact_acc"] = acc["exact_correct"] / n if n > 0 else 0.0

    # Repr diagnostics for Sudoku (empty cells) — full-depth quick eval only.
    if K_override is None and collect_diagnostics:
        repr_max = 64 if max_puzzles is None else min(max_puzzles, 64)
        repr_m = compute_repr_diagnostics(
            adapter=adapter, core=core, dataloader=dataloader,
            max_puzzles=repr_max, interesting_token=1, mask_from="inputs",
            device=device, dtype=dtype,
        )
        metrics.update(repr_m)

    # State norm diagnostics (first batch only, full-depth eval).
    if _first_batch_norm_trace is not None:
        metrics.update(_summarize_norm_trace(_first_batch_norm_trace))

    return metrics
