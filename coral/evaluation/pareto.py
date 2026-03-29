"""CORAL v4 — Accuracy-depth Pareto curve evaluation.

Evaluates model accuracy at forced depth limits K=1,2,4,8,16 to produce
the accuracy-depth Pareto curve. This measures how well the model can
reason with limited computation (amortisation quality).

Also computes normalised area under the Pareto curve (pareto_area).
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from coral.evaluation.evaluator import evaluate_accuracy


def evaluate_pareto(
    adapter: nn.Module,
    core: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    K_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Evaluate accuracy at multiple forced depth limits.

    Args:
        adapter:    GridAdapter.
        core:       CoralCore.
        dataloader: Evaluation DataLoader.
        device:     Compute device.
        dtype:      Forward pass dtype.
        K_values:   List of segment limits to evaluate at.

    Returns:
        Dict with:
          - eval/accuracy@K{k} for each k in K_values
          - eval/pareto_area (normalised area under curve)
          - eval/exact_accuracy (full depth, with halting)
          - eval/token_accuracy (full depth)
          - eval/avg_halting_step
    """
    if K_values is None:
        K_values = [1, 2, 4, 8, 16]

    results: Dict[str, float] = {}

    # Evaluate at each forced depth
    for K in K_values:
        metrics = evaluate_accuracy(
            adapter, core, dataloader, device, dtype, K_override=K
        )
        results[f"eval/accuracy@K{K}"] = metrics["eval/exact_accuracy"]

    # Full-depth evaluation (with adaptive halting)
    full_metrics = evaluate_accuracy(
        adapter, core, dataloader, device, dtype, K_override=None
    )
    results["eval/exact_accuracy"] = full_metrics["eval/exact_accuracy"]
    results["eval/token_accuracy"] = full_metrics["eval/token_accuracy"]
    results["eval/avg_halting_step"] = full_metrics["eval/avg_halting_step"]

    # Normalised area under accuracy-depth curve
    results["eval/pareto_area"] = _compute_pareto_area(results, K_values)

    return results


def _compute_pareto_area(
    results: Dict[str, float],
    K_values: List[int],
) -> float:
    """Compute normalised area under the accuracy-depth curve.

    Uses trapezoidal integration over log2(K) space, normalised to [0,1].

    Args:
        results: Dict with eval/accuracy@K{k} entries.
        K_values: Sorted list of K values.

    Returns:
        Normalised area (float in [0, 1]).
    """
    K_values = sorted(K_values)
    accs = [results.get(f"eval/accuracy@K{K}", 0.0) for K in K_values]
    log_K = [math.log2(K) for K in K_values]

    if len(log_K) < 2:
        return accs[0] if accs else 0.0

    # Trapezoidal integration
    area = 0.0
    for i in range(len(log_K) - 1):
        area += 0.5 * (accs[i] + accs[i + 1]) * (log_K[i + 1] - log_K[i])

    # Normalise by total log-K range
    max_area = log_K[-1] - log_K[0]
    return area / max_area if max_area > 0 else 0.0
