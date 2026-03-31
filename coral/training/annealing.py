"""CORAL v4 — Training schedule utilities."""


def get_effective_lambda_amort(
    step: int,
    base: float,
    anneal_start: int,
    anneal_end: int,
) -> float:
    """Compute the effective amortisation loss weight at the given training step.

    When anneal_end == 0 (default), no annealing: returns `base` immediately.
    Otherwise, linearly ramps from 0.0 at `anneal_start` to `base` at `anneal_end`.
    Before `anneal_start` returns 0.0; after `anneal_end` returns `base`.

    Args:
        step:         Current global training step.
        base:         Target lambda_amort value (from model config).
        anneal_start: Step at which to begin the ramp (inclusive).
        anneal_end:   Step at which the ramp reaches `base` (inclusive).

    Returns:
        Effective lambda value in [0.0, base].
    """
    if anneal_end == 0 or base == 0.0:
        return base
    if step < anneal_start:
        return 0.0
    if step >= anneal_end:
        return base
    progress = (step - anneal_start) / max(anneal_end - anneal_start, 1)
    return base * progress
