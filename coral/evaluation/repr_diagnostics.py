"""CORAL v4 — Standalone representation diagnostics.

Computes geometry metrics on the final-segment level-0 states, generalised
to any dataset via an ``interesting_token`` + ``mask_from`` parameter pair.

For Sudoku: ``interesting_token=1, mask_from="inputs"``  (empty cells)
For Maze:   ``interesting_token=5, mask_from="labels"``  (optimal-path cells)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_repr_diagnostics(
    adapter: nn.Module,
    core: nn.Module,
    dataloader: DataLoader,
    max_puzzles: int = 100,
    interesting_token: int = 1,
    mask_from: str = "inputs",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    pca_sample: int = 500,
) -> Dict[str, float]:
    """Compute representation geometry metrics on a sample of eval puzzles.

    Runs the model at full K_max (halting disabled) and analyses the
    final-segment level-0 states on "interesting" cells only.

    Args:
        adapter:           Adapter for encode.
        core:              CoralCore reasoning module.
        dataloader:        Evaluation DataLoader (re-iterable).
        max_puzzles:       Cap on number of puzzles to process.
        interesting_token: Token value that identifies the interesting cells.
        mask_from:         ``"inputs"`` → mask = (inputs == interesting_token);
                           ``"labels"`` → mask = (labels == interesting_token).
        device:            Compute device. Defaults to CPU.
        dtype:             Autocast dtype.
        pca_sample:        Max states sampled for SVD effective-rank estimate.

    Returns:
        Dict with repr/* metrics (empty dict if not enough interesting cells).

        Keys:
          ``repr/inter_position_similarity`` — mean pairwise cosine similarity
          ``repr/same_target_similarity``    — mean within-target-value cosine sim
          ``repr/effective_rank``            — PCA components explaining 90% variance
          ``repr/state_norm_mean``           — mean L2 norm of interesting states
          ``repr/state_norm_std``            — std of L2 norms

    Notes on latent-bug fixes vs the original trainer.py version:
        - Digit loop ``range(1, 10)`` missed digit 10 for Sudoku (vocab_size=11).
          Fixed by iterating ``interesting_labels.unique()`` instead.
        - Division-by-zero in SVD var_ratio when all singular values are zero.
          Fixed with ``.clamp(min=1e-12)``.
        - ``std()`` on a single-element tensor raises RuntimeError.
          Fixed with a size guard.
        - Missing ``@torch.no_grad()`` on the whole function. Added.
    """
    if device is None:
        device = torch.device("cpu")

    adapter.eval()
    core.eval()

    all_states: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []
    n_collected = 0

    # Disable halting so we always collect the full K_max final state.
    orig_threshold = core.config.halting_threshold
    core.config.halting_threshold = 2.0  # sigmoid ∈ (0,1) → never triggers

    try:
        for batch in dataloader:
            if n_collected >= max_puzzles:
                break
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=dtype):
                z1 = adapter.encode(inputs)
                output = core(z1, K_max=core.config.K_max, training=False, decode_fn=None)

            z_final = output.z_states[0].float().cpu()   # [B, L, d]
            labels_cpu = labels.cpu()                    # [B, L]

            if mask_from == "inputs":
                mask = (inputs == interesting_token).cpu()
            else:
                mask = (labels == interesting_token).cpu()

            all_states.append(z_final)
            all_labels.append(labels_cpu)
            all_masks.append(mask)
            n_collected += inputs.shape[0]
    finally:
        core.config.halting_threshold = orig_threshold

    if not all_states:
        return {}

    states = torch.cat(all_states, dim=0)   # [N, L, d]
    labels_t = torch.cat(all_labels, dim=0)  # [N, L]
    masks_t = torch.cat(all_masks, dim=0)    # [N, L]

    N, L, d = states.shape
    states_flat = states.reshape(N * L, d)
    labels_flat = labels_t.reshape(N * L)
    masks_flat = masks_t.reshape(N * L)

    interesting_states = states_flat[masks_flat]   # [E, d]
    interesting_labels = labels_flat[masks_flat]   # [E]

    E = interesting_states.shape[0]
    if E < 2:
        return {}

    metrics: Dict[str, float] = {}

    # L2-normalise for cosine similarity.
    norms = interesting_states.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normed = interesting_states / norms

    # a) Inter-position similarity (random sample to limit O(S²) cost).
    S = min(500, E)
    idx = torch.randperm(E)[:S]
    sample = normed[idx]                              # [S, d]
    sim = sample @ sample.T                           # [S, S]
    off_diag = ~torch.eye(S, dtype=torch.bool)
    metrics["repr/inter_position_similarity"] = sim[off_diag].mean().item()

    # b) Same-target similarity (mean within-group cosine sim across unique targets).
    # Iterates unique values in interesting_labels — generalises to any dataset:
    #   Sudoku: up to 10 values (solution digits 1–10).
    #   Maze:   1 value (5 = optimal path).
    target_sims: List[float] = []
    for val in interesting_labels.unique():
        tmask = (interesting_labels == val)
        if tmask.sum() < 2:
            continue
        tstates = normed[tmask]
        Sv = min(200, tstates.shape[0])
        tidx = torch.randperm(tstates.shape[0])[:Sv]
        tsample = tstates[tidx]                       # [Sv, d]
        tsim = tsample @ tsample.T                    # [Sv, Sv]
        toff = ~torch.eye(Sv, dtype=torch.bool)
        target_sims.append(tsim[toff].mean().item())
    if target_sims:
        metrics["repr/same_target_similarity"] = sum(target_sims) / len(target_sims)

    # c) Effective rank via SVD (90% variance threshold).
    pca_n = min(pca_sample, E)
    pidx = torch.randperm(E)[:pca_n]
    pca_states = interesting_states[pidx].float()     # [pca_n, d]
    centered = pca_states - pca_states.mean(dim=0)
    try:
        _, S_vals, _ = torch.linalg.svd(centered, full_matrices=False)
        total_var = (S_vals ** 2).sum().clamp(min=1e-12)
        var_ratio = (S_vals ** 2).cumsum(0) / total_var
        effective_rank = int((var_ratio < 0.9).sum().item()) + 1
        metrics["repr/effective_rank"] = float(effective_rank)
    except Exception:
        pass

    # d) State norm statistics.
    state_norms = interesting_states.norm(dim=-1)     # [E]
    metrics["repr/state_norm_mean"] = state_norms.mean().item()
    metrics["repr/state_norm_std"] = (
        state_norms.std().item() if state_norms.shape[0] > 1 else 0.0
    )

    return metrics
