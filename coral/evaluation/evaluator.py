"""CORAL v4 — Evaluation metrics.

Computes exact accuracy (full puzzle correct) and token accuracy (per-cell)
on the Sudoku-Extreme-1K evaluation set.
"""

from typing import Dict, Optional, Tuple

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

    Uses core.forward() directly so that input_injection and attention_bias
    are applied identically to the training forward pass.

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

    # Build attention masks once (static — depend only on grid shape, not on input).
    # These are the same masks used during training via trainer.py.
    attention_masks: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    if hasattr(adapter, "build_attention_masks") and core.config.use_local_attention_bias:
        attention_masks = adapter.build_attention_masks(device=device)

    K_max = K_override if K_override is not None else core.config.K_max

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            B = inputs.shape[0]

            with torch.autocast(device_type=device.type, dtype=dtype):
                z1 = adapter.encode(inputs)
                output = core(
                    z1,
                    K_max=K_max,
                    training=False,
                    decode_fn=adapter.decode,
                    attention_masks=attention_masks,
                )

            logits = output.all_logits[-1] if output.all_logits else adapter.decode(output.z_states[0])
            preds = logits.argmax(dim=-1)
            num_segs = output.num_segments

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
