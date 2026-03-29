"""CORAL v4 — Evaluation metrics.

Computes exact accuracy (full puzzle correct) and token accuracy (per-cell)
on the Sudoku-Extreme-1K evaluation set.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from coral.data.sudoku_dataset import IGNORE_LABEL_ID


def evaluate_accuracy(
    adapter: nn.Module,
    core: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    K_override: Optional[int] = None,
    max_puzzles: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate exact and token accuracy on a full evaluation dataset.

    Args:
        adapter:     GridAdapter for encode/decode.
        core:        CoralCore for reasoning.
        dataloader:  Evaluation DataLoader.
        device:      Compute device.
        dtype:       Forward pass dtype.
        K_override:  If provided, force exactly K segments.
        max_puzzles: If provided, stop after this many puzzles (quick eval).

    Returns:
        Dict with "eval/exact_accuracy", "eval/token_accuracy",
        "eval/avg_halting_step".
    """
    adapter.eval()
    core.eval()

    total_puzzles = 0
    exact_correct = 0
    token_correct = 0
    token_total = 0
    total_segments = 0

    from coral.model.halting import should_halt

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            B = inputs.shape[0]

            with torch.autocast(device_type=device.type, dtype=dtype):
                z1 = adapter.encode(inputs)
                L = z1.shape[1]
                z_states = [z1]
                for i in range(1, core.config.n_levels):
                    d_l = core.config.level_dims[i]
                    z_states.append(
                        torch.zeros(B, L, d_l, device=device, dtype=dtype)
                    )

                K_max = K_override if K_override is not None else core.config.K_max
                num_segs = 0

                for seg in range(K_max):
                    n_levels = core.config.n_levels
                    predictions = [None] * n_levels
                    error_signals = [None] * n_levels

                    if core.config.use_predictive_coding:
                        for i in range(n_levels - 2, -1, -1):
                            predictions[i] = core.pc_modules[i].predict(z_states[i + 1])

                    for i in range(n_levels):
                        cond = predictions[i]
                        if i > 0 and error_signals[i] is not None:
                            cond = cond + error_signals[i] if cond is not None else error_signals[i]
                        n_steps = core.level_steps[i]
                        z_states[i] = core._run_level(z_states[i], i, n_steps, conditioning=cond)
                        if core.config.use_predictive_coding and i < n_levels - 1:
                            _, _, _, _, xi_up = core.pc_modules[i](z_states[i], z_states[i + 1])
                            error_signals[i + 1] = xi_up

                    h_k, _, _ = core.halting(z_states)
                    num_segs = seg + 1
                    z_states = [z.detach() for z in z_states]

                    if K_override is None and should_halt(
                        h_k, threshold=core.config.halting_threshold,
                        training=False, exploration_prob=0.0
                    ):
                        break

                logits = adapter.decode(z_states[0])
                preds = logits.argmax(dim=-1)

            mask = labels != IGNORE_LABEL_ID
            correct = (preds == labels) & mask

            exact_correct += (correct.sum(-1) == mask.sum(-1)).sum().item()
            token_correct += correct.sum().item()
            token_total += mask.sum().item()
            total_puzzles += B
            total_segments += num_segs

            if max_puzzles is not None and total_puzzles >= max_puzzles:
                break

    return {
        "eval/exact_accuracy": exact_correct / max(total_puzzles, 1),
        "eval/token_accuracy": token_correct / max(token_total, 1),
        "eval/avg_halting_step": total_segments / max(total_puzzles // B if B > 0 else 1, 1),
    }
