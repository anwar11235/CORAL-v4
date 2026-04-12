"""CORAL v4 — Rank-collapse spatial diagnostic.

Three questions this script answers
------------------------------------
Q1 (spatial):  Where does rank collapse? The script registers forward hooks at
               every major stage of the forward pass and reports effective_rank
               at each one. If rank drops from ~512 to ~45 at a specific layer,
               that layer is the bottleneck.

Q2 (temporal): Is rank collapse present at init, or does it develop during
               training? Run on a fresh-init model (omit --checkpoint) and then
               on a trained checkpoint. A small diff between the two traces means
               the collapse is structural (baked into the architecture). A large
               diff means the loss landscape is collapsing rank during training.

Q3 (causal):   Which mechanism is responsible? If the trace shows collapse at the
               codebook, disable it for a smoke test. If rank degrades
               monotonically through the backbone layers, the attention pattern
               is the cause (classic deep-transformer rank collapse, Dong 2021).
               If it collapses at the encoder output, the embedding layer is the
               bottleneck.

Effective-rank formula
-----------------------
Identical to coral/evaluation/repr_diagnostics.py so numbers are directly
comparable to repr/effective_rank in training logs:
  SVD on centred [N, D] matrix → keep components until 90% variance explained.

Hook points (per forward pass)
-------------------------------
  encoder_output         : adapter.encode() output = z1_init [B, L, d]
  backbone_input         : input to CoralBackbone (project_up + emb + injection)
  after_transformer_0    : output of backbone.layers[0]
  after_transformer_1    : output of backbone.layers[1]  (= backbone output)
  after_project_down     : output of levels[0].down_proj  (= z_state after step)
  final_z_states[0]      : output.z_states[0] at end of all K segments

Each backbone stage reports both "first-call rank" (first inner step, segment 1)
and "last-call rank" (last inner step, segment K) so you can see whether
recursion is compressing rank.

Usage
-----
  # 1. Fresh-init — is rank collapse structural?
  python scripts/diagnose_rank_collapse.py --config-name phase1_maze_reference

  # 2. Trained checkpoint — did training move the collapse?
  python scripts/diagnose_rank_collapse.py \\
      --checkpoint outputs/2026-04-12/05-29-32/last.pt \\
      --config-name phase1_maze_reference

  # 3. Diff the two
  diff rank_trace_init.log rank_trace_trained.log

Interpreting the output
-----------------------
  rank=45 at encoder_output (init AND trained)  → structural collapse in embedding
  rank=45 appears after a specific layer         → that layer is the bottleneck
  rank high at init, drops at checkpoint         → loss-landscape collapse
  rank degrades monotonically through backbone   → attention-pattern collapse
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig, DataConfig, ModelConfig, TrainingConfig, WandbConfig
from coral.data.sudoku_dataset import create_sudoku_dataloader
from coral.model.coral_core import CoralCore


# ---------------------------------------------------------------------------
# Effective rank (verbatim from coral/evaluation/repr_diagnostics.py)
# Numbers computed here are directly comparable to repr/effective_rank logs.
# ---------------------------------------------------------------------------


def _effective_rank(states: torch.Tensor, pca_sample: int = 1000) -> int:
    """Effective rank of a [N, D] matrix (90% variance threshold).

    Verbatim formula from coral/evaluation/repr_diagnostics.py.
    """
    E = states.shape[0]
    pca_n = min(pca_sample, E)
    pidx = torch.randperm(E)[:pca_n]
    pca_states = states[pidx].float()
    centered = pca_states - pca_states.mean(dim=0)
    try:
        _, S_vals, _ = torch.linalg.svd(centered, full_matrices=False)
        total_var = (S_vals ** 2).sum().clamp(min=1e-12)
        var_ratio = (S_vals ** 2).cumsum(0) / total_var
        return int((var_ratio < 0.9).sum().item()) + 1
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Activation collector
# ---------------------------------------------------------------------------


class BatchCapture:
    """Collects first and last activations per batch across multiple batches.

    Within one batch's forward pass, the backbone is called many times
    (inner_steps × K segments).  ``start_batch`` / ``end_batch`` delimit each
    batch; ``push`` is called from a registered forward hook on every firing.
    After all batches, ``get_matrix("first")`` / ``get_matrix("last")`` return
    a [N*L, D] tensor suitable for rank computation.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._batch_acts: List[torch.Tensor] = []
        self.first_per_batch: List[torch.Tensor] = []
        self.last_per_batch: List[torch.Tensor] = []

    def start_batch(self) -> None:
        """Reset per-batch buffer.  Call once before each batch's forward."""
        self._batch_acts = []

    def push(self, act: torch.Tensor) -> None:
        """Accept a [B, L, D] activation from a hook or direct capture."""
        if act.dim() == 3:
            self._batch_acts.append(act.detach().float().cpu())

    def end_batch(self) -> None:
        """Commit first and last activations from the finished batch."""
        if self._batch_acts:
            self.first_per_batch.append(self._batch_acts[0])
            self.last_per_batch.append(self._batch_acts[-1])

    def get_matrix(self, which: str = "last") -> Optional[torch.Tensor]:
        """Return [N*L, D] matrix for SVD rank computation."""
        data = self.first_per_batch if which == "first" else self.last_per_batch
        if not data:
            return None
        cat = torch.cat(data, dim=0)   # [N, L, D]
        N, L, D = cat.shape
        return cat.reshape(N * L, D)


# ---------------------------------------------------------------------------
# Config loading (no Hydra required)
# ---------------------------------------------------------------------------


def _load_config(config_name: str) -> CoralConfig:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "configs", f"{config_name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = OmegaConf.load(path)
    return CoralConfig(
        model=ModelConfig(**dict(cfg.model)),
        training=TrainingConfig(**dict(cfg.training)),
        data=DataConfig(**dict(cfg.data)),
        wandb=WandbConfig(**dict(cfg.wandb)),
        seed=int(getattr(cfg, "seed", 42)),
        device=str(getattr(cfg, "device", "cpu")),
        experiment_name=str(getattr(cfg, "experiment_name", config_name)),
    )


# ---------------------------------------------------------------------------
# Model construction and checkpoint loading
# ---------------------------------------------------------------------------


def _build_model(
    config: CoralConfig,
) -> Tuple[GridAdapter, CoralCore]:
    adapter = GridAdapter(
        config,
        grid_height=config.data.grid_height,
        grid_width=config.data.grid_width,
    )
    core = CoralCore(config.model)
    return adapter, core


def _load_checkpoint(
    adapter: GridAdapter,
    core: CoralCore,
    checkpoint_path: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ckpt["model_state_dict"]["adapter"])
    core.load_state_dict(ckpt["model_state_dict"]["core"])


# ---------------------------------------------------------------------------
# Rank trace for one K value
# ---------------------------------------------------------------------------


def _run_rank_trace(
    adapter: GridAdapter,
    core: CoralCore,
    dataloader,
    device: torch.device,
    K: int,
    num_samples: int,
    pca_sample: int = 1000,
) -> List[Dict]:
    """Register hooks, run forward passes, return per-stage rank info.

    Returns a list of dicts:
        name         : str  — hook point label
        shape        : str  — e.g. "57600 × 512"
        rank_first   : int  — rank at first backbone call (first inner step)
        rank_last    : int  — rank at last backbone call (last inner step)
        full_rank    : int  — feature dimension D (max possible rank)

    For encoder_output and final_z_states[0], rank_first == rank_last since
    they are captured exactly once per forward pass.
    """

    # --- capturers (order determines table row order) ---
    enc_cap = BatchCapture("encoder_output  (z1_init)")
    bb_in_cap = BatchCapture("backbone_input  (project_up + emb + inject)")
    tf0_cap = BatchCapture("after_transformer_layer_0")
    tf1_cap = BatchCapture("after_transformer_layer_1  (backbone output)")
    proj_down_cap = BatchCapture("after_project_down  (level-0 state)")
    final_cap = BatchCapture("final_z_states[0]  (end of all segs)")

    all_capturers = [enc_cap, bb_in_cap, tf0_cap, tf1_cap, proj_down_cap, final_cap]

    # Crystallisation codebook (only present when use_crystallisation=True).
    # MultiHeadedCodebook.quantise is called explicitly (not via forward), so
    # we cannot use a standard forward hook.  We monkeypatch it if present.
    crys_cap: Optional[BatchCapture] = None
    original_quantise = None
    codebook = None
    if (
        core.crystallisation_manager is not None
        and hasattr(core.crystallisation_manager, "codebook")
    ):
        crys_cap = BatchCapture("after_codebook_quantise  (crystallisation)")
        all_capturers.insert(-1, crys_cap)
        codebook = core.crystallisation_manager.codebook
        original_quantise = codebook.quantise

        def _patched_quantise(z: torch.Tensor, hard: bool = True) -> torch.Tensor:
            z_q = original_quantise(z, hard=hard)
            crys_cap.push(z_q)
            return z_q

        codebook.quantise = _patched_quantise  # type: ignore[method-assign]

    # --- register forward hooks ---
    handles: List[torch.utils.hooks.RemovableHook] = []

    def _make_out_hook(cap: BatchCapture):
        def _hook(module, inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 3:
                cap.push(out)
        return _hook

    def _backbone_hook(module, inp, out):
        # inp[0] = backbone input tensor [B, L, backbone_dim]
        if isinstance(inp, (tuple, list)) and len(inp) > 0:
            t = inp[0]
            if isinstance(t, torch.Tensor) and t.dim() == 3:
                bb_in_cap.push(t)
        # backbone output is redundant with after_transformer_layer_1
        # (both are the same tensor), so we don't push out separately.

    handles.append(core.backbone.register_forward_hook(_backbone_hook))
    handles.append(core.backbone.layers[0].register_forward_hook(_make_out_hook(tf0_cap)))
    handles.append(core.backbone.layers[1].register_forward_hook(_make_out_hook(tf1_cap)))
    handles.append(
        core.levels.modules_list[0].down_proj.register_forward_hook(
            _make_out_hook(proj_down_cap)
        )
    )

    # --- run forward passes ---
    adapter.eval()
    core.eval()
    n_collected = 0

    # Disable halting so we always complete K_max segments.
    orig_threshold = core.config.halting_threshold
    core.config.halting_threshold = 2.0

    try:
        with torch.no_grad():
            for batch in dataloader:
                if n_collected >= num_samples:
                    break

                inputs = batch["inputs"].to(device)
                B = inputs.shape[0]

                for cap in all_capturers:
                    cap.start_batch()

                # Encoder: capture z1 directly (encode() is a method, not forward())
                z1 = adapter.encode(inputs)
                enc_cap.push(z1)

                # Core forward — hooks fire inside here
                output = core(
                    z1,
                    K_max=K,
                    training=False,
                    decode_fn=adapter.decode,
                )

                # Final state: capture directly from output
                final_cap.push(output.z_states[0])

                for cap in all_capturers:
                    cap.end_batch()

                n_collected += B
    finally:
        core.config.halting_threshold = orig_threshold
        # Restore monkeypatched codebook.quantise
        if codebook is not None and original_quantise is not None:
            codebook.quantise = original_quantise  # type: ignore[method-assign]

    # --- remove hooks ---
    for h in handles:
        h.remove()

    # --- compute ranks ---
    rows: List[Dict] = []
    for cap in all_capturers:
        mat_first = cap.get_matrix("first")
        mat_last = cap.get_matrix("last")

        if mat_last is None and mat_first is None:
            continue

        mat = mat_last if mat_last is not None else mat_first
        D = mat.shape[1]
        n_states = mat.shape[0]

        rank_first = _effective_rank(mat_first, pca_sample) if mat_first is not None else None
        rank_last = _effective_rank(mat_last, pca_sample) if mat_last is not None else None

        rows.append(
            dict(
                name=cap.name,
                n_states=n_states,
                full_dim=D,
                rank_first=rank_first,
                rank_last=rank_last,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------


def _print_table(rows: List[Dict], K: int, num_samples: int) -> None:
    header = (
        f"\n{'=' * 90}\n"
        f"  K = {K}  |  {num_samples} puzzles\n"
        f"{'=' * 90}\n"
    )
    print(header)

    col_name = 50
    col_states = 14
    col_rank = 16
    col_frac = 15

    fmt_hdr = (
        f"{'Stage':<{col_name}}"
        f"{'States x Dim':<{col_states}}"
        f"{'First-step rank':>{col_rank}}"
        f"{'Last-step rank':>{col_rank}}"
        f"{'Frac (last/D)':>{col_frac}}"
    )
    print(fmt_hdr)
    print("-" * (col_name + col_states + col_rank + col_rank + col_frac))

    for r in rows:
        D = r["full_dim"]
        rf = r["rank_first"]
        rl = r["rank_last"]
        frac = rl / D if rl is not None else None

        rf_str = f"{rf:>{col_rank}.2f}" if rf is not None else f"{'—':>{col_rank}}"
        rl_str = f"{rl:>{col_rank}.2f}" if rl is not None else f"{'—':>{col_rank}}"
        fr_str = f"{frac:>{col_frac}.3f}" if frac is not None else f"{'—':>{col_frac}}"

        states_str = f"{r['n_states']}x{D}"
        print(
            f"{r['name']:<{col_name}}"
            f"{states_str:<{col_states}}"
            f"{rf_str}"
            f"{rl_str}"
            f"{fr_str}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank-collapse spatial/temporal diagnostic for CORAL v4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved checkpoint (.pt).  If omitted, uses a fresh-init model.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="phase1_maze_reference",
        help="Hydra config name (without .yaml extension) from configs/.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for data loading.  Smaller = less VRAM.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of puzzles to process (across batches).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (cpu / cuda).",
    )
    parser.add_argument(
        "--pca-sample",
        type=int,
        default=1000,
        help=(
            "Max states sampled for SVD rank estimate.  Default 1000; "
            "use 500 to match training-log repr/effective_rank exactly."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    # ---- config & model ----
    print(f"Loading config: {args.config_name}")
    config = _load_config(args.config_name)
    # Override wandb to avoid accidental init
    config.wandb.disabled = True

    adapter, core = _build_model(config)

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        _load_checkpoint(adapter, core, args.checkpoint)
        model_desc = f"trained checkpoint  ({args.checkpoint})"
    else:
        print("No checkpoint provided — using fresh random-init model.")
        model_desc = "fresh random init"

    adapter = adapter.to(device)
    core = core.to(device)
    adapter.eval()
    core.eval()

    total_params = (
        sum(p.numel() for p in adapter.parameters())
        + sum(p.numel() for p in core.parameters())
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Model: {model_desc}")
    print(f"K_max={config.model.K_max}, inner_steps={core.level_steps[0]}, L={config.data.seq_len}")
    print()

    # ---- data ----
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = config.data.dataset_path
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(repo_root, dataset_path)

    print(f"Loading eval data from: {dataset_path}")
    try:
        eval_loader, _ = create_sudoku_dataloader(
            dataset_path=dataset_path,
            split="test",
            global_batch_size=args.batch_size,
            seed=config.seed,
            test_set_mode=True,
        )
    except Exception as e:
        print(f"\nERROR: Could not load eval data: {e}")
        print("Make sure the dataset exists at the path specified in the config.")
        sys.exit(1)

    # ---- rank traces ----
    print(
        f"Running rank trace: K=1 and K={config.model.K_max}, "
        f"{args.num_samples} samples, pca_sample={args.pca_sample}"
    )
    print()

    for K in [1, config.model.K_max]:
        rows = _run_rank_trace(
            adapter=adapter,
            core=core,
            dataloader=eval_loader,
            device=device,
            K=K,
            num_samples=args.num_samples,
            pca_sample=args.pca_sample,
        )
        _print_table(rows, K=K, num_samples=args.num_samples)

    print("Done.")


if __name__ == "__main__":
    main()
