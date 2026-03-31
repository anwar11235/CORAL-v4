#!/usr/bin/env python3
"""Diagnostic script: why does predictive coding hurt token accuracy?

Trains with PC enabled (mode=pc_only, N=2 levels) for up to --steps gradient
updates, and at each diagnostic checkpoint collects five metrics that test the
"blurred-prediction / representation-smoothing" hypothesis.

Five diagnostics collected at each checkpoint:
  [1] Representation sharpness of z1
        Variance of per-position L2 norms across the 81 cells.
        High variance → cells have distinct representations.
        Low variance  → backbone is producing similar vectors for all cells.

  [2] Prediction blur: z1 vs mu
        Same variance metric computed on mu (the top-down prediction from z2).
        blur_ratio = var(||mu||) / var(||z1||).
        < 1 → mu is LESS varied than z1, confirming the blurring hypothesis.
        > 1 → mu is MORE varied (unusual — would sharpen representations).

  [3] Gate magnitudes
        cond_gate[0] (level-0 correction strength) and cond_gate[1] (level-1).
        Initialised to 1.0; a large positive value amplifies the blur.

  [4] Conditioning magnitude relative to z1
        ||gate * (mu - z1)|| / ||z1|| averaged over all positions.
        > 1 → the residual correction DOMINATES the backbone output.
        ~ 0 → conditioning has negligible effect.

  [5] Token accuracy: backbone output BEFORE vs AFTER applying gate*(mu-z)
        Runs one backbone step on z1, decodes z_new (before correction) and
        z_cond (after correction), computes token accuracy for each.
        Delta = acc_after - acc_before.
        Negative delta → conditioning is actively *hurting* accuracy.

Usage:
    python scripts/diagnostic_pc_analysis.py \\
        --data-dir data/sudoku_extreme_1k \\
        --steps 500 --diag-steps 100 300 500 \\
        --output outputs/pc_diagnostic.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig, ModelConfig, TrainingConfig, DataConfig, WandbConfig
from coral.data.sudoku_dataset import create_sudoku_dataloader
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss
from coral.training.optimizer import build_scheduler
from coral.training.trainer import TrainerV4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def make_pc_config(data_path: str, device: str = "cuda") -> CoralConfig:
    """Build a pc_only config matching Phase 1 training hyperparameters."""
    model = ModelConfig(
        n_levels=2,
        level_dims=[512, 256],
        backbone_layers=2,
        backbone_dim=512,
        n_heads=8,
        d_k=64,
        ffn_expansion=4,
        timescale_base=3,
        use_predictive_coding=True,
        lambda_pred=0.001,
        epsilon_min=0.01,
        precision_momentum=0.99,
        K_max=16,
        halting_threshold=0.95,
        halting_exploration_prob=0.1,
        halting_gamma=0.9,
        use_continue_loss=False,
        use_crystallisation=False,
        use_amort=False,
        lambda_amort=0.0,
        lambda_crystal=0.0,
        lambda_commit=0.25,
        lambda_dis=0.01,
        mode="pc_only",
        vocab_size=11,
        use_local_attention_bias=True,
    )
    training = TrainingConfig(
        epochs=20000,          # will be cut by --steps
        batch_size=64,
        learning_rate=7e-5,
        weight_decay=1.0,
        betas=[0.9, 0.95],
        gradient_clip=1.0,
        scheduler="cosine",
        warmup_steps=100,
        precision="bfloat16",
        log_every=50,
        eval_every=100,
        optimizer="adamw",
        compile_model=False,
    )
    data = DataConfig(
        dataset="sudoku_extreme_1k",
        dataset_path=data_path,
        augmentation_factor=1000,
        eval_size=1000,
    )
    wandb_cfg = WandbConfig(disabled=True)
    return CoralConfig(
        model=model,
        training=training,
        data=data,
        wandb=wandb_cfg,
        device=device,
        experiment_name="pc_diagnostic",
    )


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of non-ignored positions predicted correctly."""
    mask = labels != -100
    if mask.sum() == 0:
        return float("nan")
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & mask
    return (correct.sum().float() / mask.sum().float()).item()


@torch.no_grad()
def run_diagnostic(
    core: CoralCore,
    adapter: GridAdapter,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    step: int,
) -> Dict:
    """Collect all five diagnostics for one batch.

    Strategy:
      1. Encode the input and run K_warm=4 full segments (no_grad) to obtain
         non-trivial level states — the diagnostic measures steady-state behaviour,
         not the raw embedding.
      2. From the warmed-up states, extract z1 (level-0) and z2 (level-1).
      3. Compute mu = prediction_net(z2) — the top-down prediction for level 0.
      4. Run ONE backbone step on z1 (t=0, with input injection, NO conditioning)
         to get z_raw.
      5. Apply the conditioning residual: z_cond = z_raw + gate*(mu - z1).
         Note: the 'z' in the gate formula is the pre-backbone z1, not z_raw —
         this matches the implementation in _run_level exactly.
      6. Decode both z_raw and z_cond and compare token accuracies.
    """
    core.eval()
    adapter.eval()

    inputs = batch["inputs"].to(device)
    labels = batch["labels"].to(device)

    # --- Encode ---
    with torch.autocast(device_type=device.type, dtype=dtype):
        z1_init = adapter.encode(inputs).float()   # [B, L, 512], float32

    # --- Warm up: run K_warm segments to reach a non-trivial state ---
    K_warm = min(4, core.K_max)
    with torch.autocast(device_type=device.type, dtype=dtype):
        warm_out = core(z1_init.to(dtype), K_max=K_warm, training=False, decode_fn=None)

    z1 = warm_out.z_states[0].float()   # [B, L, 512]
    z2 = warm_out.z_states[1].float()   # [B, L, 256]  (always present: n_levels=2)

    # --- Top-down prediction mu ---
    with torch.autocast(device_type=device.type, dtype=dtype):
        mu = core.pc_modules[0].predict(z2.to(dtype)).float()  # [B, L, 512]

    # =========================================================
    # [1] Representation sharpness: variance of per-position
    #     L2 norms across the 81 cells, averaged over the batch.
    # =========================================================
    z1_pos_norms = z1.norm(dim=-1)                          # [B, L]
    z1_norm_var  = z1_pos_norms.var(dim=1).mean().item()    # scalar
    z1_norm_mean = z1_pos_norms.mean().item()

    # =========================================================
    # [2] Prediction blur: same metric for mu.
    #     blur_ratio = var(||mu||) / var(||z1||)
    #     < 1 → mu is blurred (less varied) → confirms hypothesis.
    # =========================================================
    mu_pos_norms = mu.norm(dim=-1)                          # [B, L]
    mu_norm_var  = mu_pos_norms.var(dim=1).mean().item()
    blur_ratio   = mu_norm_var / (z1_norm_var + 1e-8)

    # =========================================================
    # [3] Gate values.
    # =========================================================
    gate_0 = core.cond_gate[0].item()
    gate_1 = core.cond_gate[1].item()

    # =========================================================
    # [4] Conditioning magnitude relative to z1.
    #     ||gate*(mu - z1)|| / ||z1|| per position, mean over batch.
    # =========================================================
    correction      = gate_0 * (mu - z1)                   # [B, L, 512]
    correction_norm = correction.norm(dim=-1)               # [B, L]
    z1_base_norm    = z1.norm(dim=-1).clamp(min=1e-8)      # [B, L]
    cond_mag_rel    = (correction_norm / z1_base_norm).mean().item()

    # Also report absolute correction and absolute z1 magnitude for reference
    cond_mag_abs = correction_norm.mean().item()
    z1_mag_abs   = z1_base_norm.mean().item()

    # =========================================================
    # [5] Token accuracy: backbone output BEFORE vs AFTER conditioning.
    #
    # Run one backbone step (t=0) on z1 with input injection, capturing
    # z_raw BEFORE the residual correction is applied, then z_cond AFTER.
    # The 'z' subtracted in gate*(conditioning - z) is z1 (pre-backbone),
    # exactly matching _run_level's implementation.
    # =========================================================
    level_mod = core.levels[0]
    lev_emb = core.level_emb(0, device).unsqueeze(0).unsqueeze(0).float()  # [1,1,512]
    ts_emb  = core.timescale_emb(0).unsqueeze(0).unsqueeze(0).float()      # [1,1,512]

    backbone_in = level_mod.project_up(z1.to(dtype))
    backbone_in = backbone_in + lev_emb.to(dtype)
    backbone_in = backbone_in + ts_emb.to(dtype)
    backbone_in = backbone_in + z1_init.to(dtype)   # input re-injection (matches training)

    with torch.autocast(device_type=device.type, dtype=dtype):
        z_raw_dt = level_mod.project_down(core.backbone(backbone_in))
    z_raw = z_raw_dt.float()   # [B, L, 512] — backbone output, before conditioning

    # Residual correction (gate uses pre-backbone z1, not z_raw — see _run_level)
    z_cond = z_raw + correction   # correction = gate_0 * (mu - z1), computed above

    with torch.autocast(device_type=device.type, dtype=dtype):
        logits_before = adapter.decode(z_raw.to(dtype))     # decode raw backbone output
        logits_after  = adapter.decode(z_cond.to(dtype))    # decode conditioned output

    acc_before = _token_accuracy(logits_before.float(), labels)
    acc_after  = _token_accuracy(logits_after.float(), labels)
    acc_delta  = acc_after - acc_before

    # --- Compile ---
    return {
        "step": step,
        # [1]
        "z1_norm_mean": z1_norm_mean,
        "z1_pos_norm_variance": z1_norm_var,
        # [2]
        "mu_pos_norm_variance": mu_norm_var,
        "blur_ratio_mu_over_z1": blur_ratio,
        # [3]
        "gate_0": gate_0,
        "gate_1": gate_1,
        # [4]
        "cond_magnitude_relative": cond_mag_rel,
        "cond_magnitude_absolute": cond_mag_abs,
        "z1_magnitude_absolute": z1_mag_abs,
        # [5]
        "token_acc_before_conditioning": acc_before,
        "token_acc_after_conditioning": acc_after,
        "conditioning_delta_accuracy": acc_delta,
    }


def print_diagnostic(r: Dict) -> None:
    """Print a formatted diagnostic report to stdout."""
    s = r["step"]
    width = 62
    print(f"\n{'='*width}")
    print(f"  PC DIAGNOSTIC REPORT  —  training step {s}")
    print(f"{'='*width}")

    print("\n[1] Representation sharpness of z1 (level-0 state):")
    print(f"    Mean L2 norm:                {r['z1_norm_mean']:.4f}")
    print(f"    Variance across 81 positions: {r['z1_pos_norm_variance']:.6f}")
    if r['z1_pos_norm_variance'] < 0.01:
        print("    ⚠  Very low variance — backbone outputs near-identical vectors for all cells")

    print("\n[2] Prediction blur  (mu vs z1 position-norm variance):")
    print(f"    var(||z1||):  {r['z1_pos_norm_variance']:.6f}")
    print(f"    var(||mu||):  {r['mu_pos_norm_variance']:.6f}")
    br = r['blur_ratio_mu_over_z1']
    if br < 0.5:
        interp = "STRONGLY BLURRED — mu is far less varied than z1"
    elif br < 1.0:
        interp = "blurred — mu less varied than z1"
    elif br < 1.5:
        interp = "comparable — mu and z1 have similar spread"
    else:
        interp = "SHARPENED — mu more varied than z1 (unusual)"
    print(f"    blur ratio (mu/z1): {br:.4f}  ←  {interp}")

    print("\n[3] Conditioning gate values (init = 1.0):")
    print(f"    cond_gate[0] (level-0): {r['gate_0']:.4f}")
    print(f"    cond_gate[1] (level-1): {r['gate_1']:.4f}")
    if r['gate_0'] > 1.5:
        print("    ⚠  gate_0 > 1.5 — correction is being AMPLIFIED")
    elif r['gate_0'] < 0.1:
        print("    gate_0 < 0.1 — model has largely learned to suppress conditioning")

    print("\n[4] Conditioning magnitude:")
    print(f"    ||z1||:              {r['z1_magnitude_absolute']:.4f}")
    print(f"    ||gate*(mu-z1)||:    {r['cond_magnitude_absolute']:.4f}")
    rel = r['cond_magnitude_relative']
    if rel > 1.0:
        interp = "DOMINANT — correction vector larger than the representation itself"
    elif rel > 0.5:
        interp = "significant — comparable to z1"
    elif rel > 0.1:
        interp = "moderate"
    else:
        interp = "small — conditioning has minor effect"
    print(f"    Relative magnitude:  {rel:.4f}  ←  {interp}")

    print("\n[5] Token accuracy: before vs after conditioning (one backbone step):")
    print(f"    z_raw (backbone output, no gate):  {r['token_acc_before_conditioning']*100:.2f}%")
    print(f"    z_cond (after gate*(mu-z1)):       {r['token_acc_after_conditioning']*100:.2f}%")
    delta = r['conditioning_delta_accuracy']
    sign = "+" if delta >= 0 else ""
    if delta < -0.02:
        effect = "HURTS  ← conditioning is actively degrading accuracy"
    elif delta < -0.005:
        effect = "mildly hurts"
    elif delta > 0.02:
        effect = "HELPS  ← conditioning improves accuracy"
    elif delta > 0.005:
        effect = "mildly helps"
    else:
        effect = "neutral"
    print(f"    Delta:                             {sign}{delta*100:.2f}pp  ←  {effect}")

    print()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PC representation diagnostic")
    parser.add_argument(
        "--data-dir", default="data/sudoku_extreme_1k",
        help="Path to sudoku_extreme_1k dataset directory",
    )
    parser.add_argument(
        "--output", default="outputs/pc_diagnostic.json",
        help="JSON file to write diagnostic results",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Compute device (cuda or cpu)",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Number of gradient updates to run",
    )
    parser.add_argument(
        "--diag-steps", nargs="+", type=int, default=[100, 300, 500],
        help="Steps at which to run the diagnostic (space-separated)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    config = make_pc_config(args.data_dir, str(device))
    dtype  = torch.bfloat16 if config.training.precision == "bfloat16" else torch.float32

    # ---- Data ----
    log.info("Building data loaders…")
    train_loader, _ = create_sudoku_dataloader(
        dataset_path=config.data.dataset_path,
        split="train",
        global_batch_size=config.training.batch_size,
        seed=args.seed,
        epochs_per_iter=1,
    )
    eval_loader, _ = create_sudoku_dataloader(
        dataset_path=config.data.dataset_path,
        split="test",
        global_batch_size=config.training.batch_size,
        seed=args.seed,
        test_set_mode=True,
    )

    # ---- Model ----
    log.info("Building model (pc_only, N=2 levels, d=[512,256])…")
    adapter  = GridAdapter(config, grid_height=9, grid_width=9)
    core     = CoralCore(config.model)
    loss_fn  = CoralLoss(config.model)

    total_params = (
        sum(p.numel() for p in adapter.parameters())
        + sum(p.numel() for p in core.parameters())
    )
    log.info(f"Total parameters: {total_params:,}")

    trainer = TrainerV4(
        adapter=adapter, core=core, loss_fn=loss_fn,
        config=config, wandb_run=None,
    )

    scheduler = build_scheduler(
        trainer.optimizer,
        total_steps=args.steps,
        warmup_steps=config.training.warmup_steps,
        scheduler_type=config.training.scheduler,
    )

    diag_steps_set = set(args.diag_steps)
    all_diagnostics: List[Dict] = []
    # Keep the last eval batch for diagnostics
    _eval_iter = iter(eval_loader)

    def _next_eval_batch():
        nonlocal _eval_iter
        try:
            return next(_eval_iter)
        except StopIteration:
            _eval_iter = iter(eval_loader)
            return next(_eval_iter)

    # ---- Training ----
    log.info(f"Training for {args.steps} steps with diagnostics at {sorted(diag_steps_set)}…")
    step = 0

    for _epoch in range(100_000):
        for batch in train_loader:
            if step >= args.steps:
                break

            # --- Train step ---
            metrics = trainer.train_step(batch)
            scheduler.step()
            step = trainer.step

            # --- Log ---
            if step % config.training.log_every == 0:
                log.info(
                    f"step={step:4d} | "
                    + " ".join(
                        f"{k}={v:.4f}"
                        for k, v in metrics.items()
                        if "loss" in k
                    )
                )

            # --- Quick eval token accuracy (no_grad) ---
            if step % config.training.eval_every == 0:
                eval_batch = _next_eval_batch()
                eval_inputs  = eval_batch["inputs"].to(device)
                eval_labels  = eval_batch["labels"].to(device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                    z1 = trainer.adapter.encode(eval_inputs)
                    out = trainer.core(z1, K_max=4, training=False, decode_fn=trainer.adapter.decode)
                if out.all_logits:
                    tok_acc = _token_accuracy(out.all_logits[-1].float(), eval_labels)
                    log.info(f"  [eval] step={step} token_acc={tok_acc*100:.2f}%")

            # --- Diagnostic ---
            if step in diag_steps_set:
                log.info(f"  Running diagnostic at step {step}…")
                eval_batch = _next_eval_batch()
                diag = run_diagnostic(
                    core=trainer.core,
                    adapter=trainer.adapter,
                    batch=eval_batch,
                    device=device,
                    dtype=dtype,
                    step=step,
                )
                print_diagnostic(diag)
                all_diagnostics.append(diag)

        if step >= args.steps:
            break

    # ---- Save ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(all_diagnostics, fh, indent=2)
    log.info(f"Saved {len(all_diagnostics)} diagnostic records → {args.output}")

    # ---- Summary table ----
    if all_diagnostics:
        print("\n" + "="*62)
        print("SUMMARY TABLE")
        print("="*62)
        headers = ["step", "z1_var", "mu_var", "blur", "gate0",
                   "cond_rel", "acc_pre%", "acc_post%", "delta_pp"]
        print(f"{'':>5}  {'z1_var':>8}  {'mu_var':>8}  {'blur':>6}  {'gate0':>6}  "
              f"{'cond_rel':>8}  {'acc_pre':>8}  {'acc_post':>9}  {'delta_pp':>8}")
        print("-"*84)
        for r in all_diagnostics:
            print(
                f"{r['step']:>5}  "
                f"{r['z1_pos_norm_variance']:>8.4f}  "
                f"{r['mu_pos_norm_variance']:>8.4f}  "
                f"{r['blur_ratio_mu_over_z1']:>6.3f}  "
                f"{r['gate_0']:>6.3f}  "
                f"{r['cond_magnitude_relative']:>8.4f}  "
                f"{r['token_acc_before_conditioning']*100:>8.2f}  "
                f"{r['token_acc_after_conditioning']*100:>9.2f}  "
                f"{r['conditioning_delta_accuracy']*100:>+8.2f}"
            )
        print()


if __name__ == "__main__":
    main()
