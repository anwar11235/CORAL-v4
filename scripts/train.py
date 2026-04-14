"""CORAL v4 — Main training entry point.

Usage:
    python scripts/train.py                          # uses configs/exp1_baseline.yaml
    python scripts/train.py training.batch_size=32   # override any field
    python scripts/train.py +experiment=exp2_amort   # different config

Run on Vast.ai:
    tmux new-session -s training
    cd /workspace/CORAL-v4
    python scripts/train.py

W&B workspace: aktuator-ai
W&B project:   Sudoku-extreme-1k-aug-1000 CORAL-v4
"""

import logging
import os
import random
import sys

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig, ModelConfig, TrainingConfig, DataConfig, WandbConfig
from coral.data.sudoku_dataset import create_sudoku_dataloader
from coral.evaluation.evaluator import evaluate_accuracy
from coral.evaluation.pareto import evaluate_pareto
from coral.model.coral_core import CoralCore
from coral.training.annealing import get_effective_lambda_amort
from coral.training.losses import CoralLoss
from coral.training.optimizer import build_scheduler
from coral.training.trainer import TrainerV4

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dict_to_config(cfg: DictConfig) -> CoralConfig:
    """Convert Hydra DictConfig to CoralConfig dataclass."""
    model_cfg = ModelConfig(**dict(cfg.model))
    train_cfg = TrainingConfig(**dict(cfg.training))
    data_cfg = DataConfig(**dict(cfg.data))
    wandb_cfg = WandbConfig(**dict(cfg.wandb))
    return CoralConfig(
        model=model_cfg,
        training=train_cfg,
        data=data_cfg,
        wandb=wandb_cfg,
        seed=cfg.seed,
        device=cfg.device,
        experiment_name=cfg.experiment_name,
    )


@hydra.main(config_path="../configs", config_name="exp1_baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    # Convert to typed config
    config = dict_to_config(cfg)
    seed_everything(config.seed)

    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    output_dir = HydraConfig.get().runtime.output_dir

    # ---- W&B initialisation ----
    wandb_run = None
    if not config.wandb.disabled:
        wandb_run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.run_name,
            tags=config.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        log.info(f"W&B run: {wandb_run.url}")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ---- Data loaders ----
    log.info("Building data loaders...")
    train_loader, train_meta = create_sudoku_dataloader(
        dataset_path=config.data.dataset_path,
        split="train",
        global_batch_size=config.training.batch_size,
        seed=config.seed,
        epochs_per_iter=1,
    )
    eval_batch_size = getattr(config.training, "eval_batch_size", config.training.batch_size)
    eval_loader, _ = create_sudoku_dataloader(
        dataset_path=config.data.dataset_path,
        split="test",
        global_batch_size=eval_batch_size,
        seed=config.seed,
        test_set_mode=True,
    )

    # ---- Model ----
    log.info("Building model...")
    adapter = GridAdapter(
        config,
        grid_height=config.data.grid_height,
        grid_width=config.data.grid_width,
    )
    core = CoralCore(config.model)
    loss_fn = CoralLoss(config.model, dataset_name=config.data.dataset)

    total_params = sum(p.numel() for p in adapter.parameters()) + \
                   sum(p.numel() for p in core.parameters())
    log.info(f"Total parameters: {total_params:,}")

    # ---- torch.compile (optional) ----
    if config.training.compile_model:
        log.info("Compiling backbone with torch.compile...")
        core.backbone = torch.compile(core.backbone, dynamic=config.training.compile_dynamic)

    # ---- Trainer ----
    trainer = TrainerV4(
        adapter=adapter,
        core=core,
        loss_fn=loss_fn,
        config=config,
        wandb_run=wandb_run,
    )

    # ---- Codebook initialisation from k-means centroids (optional) ----
    init_path = config.training.codebook_init_from
    if (
        init_path is not None
        and config.model.use_crystallisation
        and core.crystallisation_manager is not None
    ):
        log.info(f"Initialising codebook from k-means centroids: {init_path}")
        try:
            raw = np.load(init_path, mmap_mode=None)
            states_all = raw["states"].copy()   # [N, L, n_segs, d]
            raw.close()

            # Use the last segment's states (most refined representations)
            N, L, n_segs, d = states_all.shape
            states_flat = states_all[:, :, -1, :].reshape(N * L, d)

            # Initialise using per-head k-means (run on CPU, weights on device)
            states_tensor = torch.from_numpy(states_flat).float()
            core.crystallisation_manager.codebook.initialise_from_kmeans(states_tensor)
            log.info(
                f"Codebook initialised from {states_flat.shape[0]} states "
                f"({config.model.codebook_heads} heads × "
                f"{config.model.codebook_entries_per_head} entries)"
            )
        except Exception as e:
            log.warning(f"Codebook initialisation failed (continuing without): {e}")

    # Build LR scheduler
    total_steps = config.training.epochs
    scheduler = build_scheduler(
        trainer.optimizer,
        total_steps=total_steps,
        warmup_steps=config.training.warmup_steps,
        scheduler_type=config.training.scheduler,
    )

    # ---- Training loop ----
    log.info("Starting training...")
    step = 0

    for epoch in range(config.training.epochs):
        for batch in train_loader:
            metrics = trainer.train_step(batch)
            scheduler.step()
            step = trainer.step

            # Amortisation annealing — update loss_fn's effective weight each step
            eff_lambda = 0.0
            if config.model.use_amort:
                eff_lambda = get_effective_lambda_amort(
                    step=step,
                    base=config.model.lambda_amort,
                    anneal_start=config.training.amort_anneal_start,
                    anneal_end=config.training.amort_anneal_end,
                )
                trainer.loss_fn.config.lambda_amort = eff_lambda

            # Log to W&B
            if step % config.training.log_every == 0:
                if config.model.use_amort:
                    metrics["train/lambda_amort"] = eff_lambda
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)
                log.info(
                    f"Step {step}/{total_steps} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )

            # Quick eval — first 100 puzzles, full K_max
            if step % config.training.eval_every == 0:
                log.info(f"Running quick eval at step {step}...")
                quick_metrics = evaluate_accuracy(
                    adapter=trainer.adapter,
                    core=trainer.core,
                    dataloader=eval_loader,
                    device=device,
                    dtype=torch.bfloat16,
                    max_puzzles=config.training.quick_eval_samples,
                    dataset_name=config.data.dataset,
                )
                # Surface conditioning-gate evolution in every quick-eval log line.
                for i, gate in enumerate(trainer.core.cond_gate):
                    quick_metrics[f"cond_gate/level{i}"] = gate.item()
                # Precision dynamics — mirror training-step precision metrics at
                # quick-eval granularity (every eval_every steps).
                for i, pc in enumerate(trainer.core.pc_modules):
                    rp = pc.running_precision
                    pi = rp.precision
                    ev = rp.ema_var
                    eps_rms = ev.sqrt()
                    quick_metrics[f"precision/level{i}_mean"] = pi.mean().item()
                    quick_metrics[f"precision/level{i}_std"] = pi.std().item()
                    quick_metrics[f"precision/level{i}_min"] = pi.min().item()
                    quick_metrics[f"precision/level{i}_max"] = pi.max().item()
                    quick_metrics[f"prediction_error/level{i}_mean"] = eps_rms.mean().item()
                    quick_metrics[f"prediction_error/level{i}_max"] = eps_rms.max().item()
                    quick_metrics[f"precision_ema_var/level{i}_mean"] = ev.mean().item()
                if wandb_run is not None:
                    wandb_run.log(quick_metrics, step=step)
                log.info(
                    f"[QUICK EVAL] step={step} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in quick_metrics.items())
                )
                _save_checkpoint(trainer, cfg, step, output_dir)

            # Full Pareto eval — all puzzles, K=1..16
            if step % config.training.pareto_eval_every == 0:
                log.info(f"Running full Pareto eval at step {step}...")
                pareto_results = evaluate_pareto(
                    adapter=trainer.adapter,
                    core=trainer.core,
                    dataloader=eval_loader,
                    device=device,
                    dtype=torch.bfloat16,
                    max_puzzles=getattr(config.training, "pareto_eval_samples", 100),
                    dataset_name=config.data.dataset,
                )
                _log_eval_diagnostics(trainer, pareto_results)
                if wandb_run is not None:
                    wandb_run.log(pareto_results, step=step)
                log.info(
                    f"[PARETO EVAL] step={step} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in pareto_results.items())
                )

            if step >= total_steps:
                break

        if step >= total_steps:
            break

    log.info("Training complete.")
    _save_checkpoint(trainer, cfg, step, output_dir)
    if wandb_run is not None:
        wandb_run.finish()


def _save_checkpoint(trainer: "TrainerV4", cfg: DictConfig, step: int, output_dir: str) -> None:
    """Overwrite last.pt with current model, optimizer, step, and config."""
    ckpt_path = os.path.join(output_dir, "last.pt")
    torch.save(
        {
            "model_state_dict": {
                "adapter": trainer.adapter.state_dict(),
                "core": trainer.core.state_dict(),
            },
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "step": step,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        ckpt_path,
    )
    log.info(f"Checkpoint saved → {ckpt_path}")


def _log_eval_diagnostics(trainer, out_dict):
    """Read precision/error diagnostics from running_precision buffers and add to out_dict.

    Called after evaluate_pareto, which has already run the real forward pass and updated
    the running_precision EMA buffers.  Reading the buffers here produces the same values
    as training-step precision metrics, making Pareto-time and training-time series
    directly comparable.  No second forward pass is performed.
    """
    for i, pc in enumerate(trainer.core.pc_modules):
        rp = pc.running_precision
        pi = rp.precision          # [dim_lower] — 1/(ema_var + eps_min), no grad
        ev = rp.ema_var            # [dim_lower] — EMA variance of normalized error
        eps_rms = ev.sqrt()        # per-dim RMS prediction-error proxy (matches training-step)
        out_dict[f"precision/level{i}_mean"] = pi.mean().item()
        out_dict[f"precision/level{i}_std"] = pi.std().item()
        out_dict[f"precision/level{i}_min"] = pi.min().item()
        out_dict[f"precision/level{i}_max"] = pi.max().item()
        out_dict[f"prediction_error/level{i}_mean"] = eps_rms.mean().item()
        out_dict[f"prediction_error/level{i}_max"] = eps_rms.max().item()
        out_dict[f"precision_ema_var/level{i}_mean"] = ev.mean().item()

    # Log learnable conditioning gate values
    for i, gate in enumerate(trainer.core.cond_gate):
        out_dict[f"cond_gate/level{i}"] = gate.item()


if __name__ == "__main__":
    main()
