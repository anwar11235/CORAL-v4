"""CORAL v4 — Accuracy-depth Pareto curve evaluation.

Evaluates model accuracy at forced depth limits K=1,2,4,8,16 to produce
the accuracy-depth Pareto curve. This measures how well the model can
reason with limited computation (amortisation quality).

Also computes the pareto area scalar — the single summary metric that
validates amortisation claims (Architecture Spec Section 10.3).
"""

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from coral.evaluation.evaluator import evaluate_accuracy

log = logging.getLogger(__name__)


def evaluate_pareto(
    adapter: nn.Module,
    core: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    K_values: Optional[List[int]] = None,
    dataset_name: str = "sudoku_extreme_1k",
) -> Dict[str, float]:
    """Evaluate accuracy at multiple forced depth limits.

    Args:
        adapter:      GridAdapter.
        core:         CoralCore.
        dataloader:   Evaluation DataLoader.
        device:       Compute device.
        dtype:        Forward pass dtype.
        K_values:     List of segment limits to evaluate at.
        dataset_name: Dataset identifier — controls which per-K metrics are emitted.
                      "maze_30x30_hard" → emits pareto_k{K}_path/wall/non_path/exact/token.
                      Anything else → emits accuracy@K{K} (Sudoku default).

    Returns:
        Dict with per-K accuracy metrics, pareto area scalar(s), and
        full-depth eval/exact_accuracy, eval/token_accuracy, eval/avg_halting_step.

        For maze: eval/pareto_k{K}_path_accuracy, eval/pareto_k{K}_wall_accuracy,
            eval/pareto_k{K}_non_path_accuracy, eval/pareto_k{K}_exact_accuracy,
            eval/pareto_k{K}_token_accuracy, eval/pareto_area_path_accuracy,
            eval/pareto_area_exact_accuracy.

        For Sudoku: eval/accuracy@K{K} (existing), eval/pareto_area (trapezoidal),
            eval/pareto_area_exact_accuracy (mean, comparable to maze scalar).

        K values exceeding core.config.K_max are skipped with a warning.
    """
    if K_values is None:
        K_values = [1, 2, 4, 8, 16]

    results: Dict[str, float] = {}
    valid_Ks: List[int] = []

    # Evaluate at each forced depth.
    for K in sorted(K_values):
        if K > core.config.K_max:
            log.warning(
                f"evaluate_pareto: K={K} exceeds core.K_max={core.config.K_max}; skipping."
            )
            continue
        valid_Ks.append(K)
        metrics = evaluate_accuracy(
            adapter, core, dataloader, device, dtype,
            K_override=K, dataset_name=dataset_name,
        )
        if dataset_name == "maze_30x30_hard":
            results[f"eval/pareto_k{K}_path_accuracy"] = metrics.get("eval/path_accuracy", 0.0)
            results[f"eval/pareto_k{K}_wall_accuracy"] = metrics.get("eval/wall_accuracy", 0.0)
            results[f"eval/pareto_k{K}_non_path_accuracy"] = metrics.get("eval/non_path_accuracy", 0.0)
            results[f"eval/pareto_k{K}_exact_accuracy"] = metrics.get("eval/exact_accuracy", 0.0)
            results[f"eval/pareto_k{K}_token_accuracy"] = metrics.get("eval/token_accuracy", 0.0)
        else:
            results[f"eval/accuracy@K{K}"] = metrics["eval/exact_accuracy"]

    # Full-depth evaluation (adaptive halting, K_override=None).
    full_metrics = evaluate_accuracy(
        adapter, core, dataloader, device, dtype,
        K_override=None, dataset_name=dataset_name,
    )
    results["eval/exact_accuracy"] = full_metrics["eval/exact_accuracy"]
    results["eval/token_accuracy"] = full_metrics["eval/token_accuracy"]
    results["eval/avg_halting_step"] = full_metrics["eval/avg_halting_step"]
    if dataset_name == "maze_30x30_hard":
        results["eval/path_accuracy"] = full_metrics.get("eval/path_accuracy", 0.0)
        results["eval/wall_accuracy"] = full_metrics.get("eval/wall_accuracy", 0.0)
        results["eval/non_path_accuracy"] = full_metrics.get("eval/non_path_accuracy", 0.0)
    # Propagate repr diagnostics from the full-depth call.
    for k, v in full_metrics.items():
        if k.startswith("repr/"):
            results[k] = v

    # Pareto area scalars.
    if valid_Ks:
        if dataset_name == "maze_30x30_hard":
            # Primary area metric: mean path accuracy across forced depths.
            path_accs = [results[f"eval/pareto_k{K}_path_accuracy"] for K in valid_Ks]
            results["eval/pareto_area_path_accuracy"] = sum(path_accs) / len(path_accs)
            # Secondary: mean exact accuracy (comparable across datasets).
            exact_accs = [results[f"eval/pareto_k{K}_exact_accuracy"] for K in valid_Ks]
            results["eval/pareto_area_exact_accuracy"] = sum(exact_accs) / len(exact_accs)
        else:
            # Existing trapezoidal Pareto area (backward compat).
            results["eval/pareto_area"] = _compute_pareto_area(results, valid_Ks)
            # Simple mean exact accuracy (comparable scalar with maze).
            exact_accs = [results[f"eval/accuracy@K{K}"] for K in valid_Ks]
            results["eval/pareto_area_exact_accuracy"] = sum(exact_accs) / len(exact_accs)

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
