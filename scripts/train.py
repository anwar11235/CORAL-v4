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
from omegaconf import DictConfig, OmegaConf

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig, ModelConfig, TrainingConfig, DataConfig, WandbConfig
from coral.data.sudoku_dataset import create_sudoku_dataloader
from coral.evaluation.evaluator import evaluate_accuracy
from coral.evaluation.pareto import evaluate_pareto
from coral.model.coral_core import CoralCore
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
    eval_loader, _ = create_sudoku_dataloader(
        dataset_path=config.data.dataset_path,
        split="test",
        global_batch_size=config.training.batch_size,
        seed=config.seed,
        test_set_mode=True,
    )

    # ---- Model ----
    log.info("Building model...")
    adapter = GridAdapter(
        config,
        grid_height=9,
        grid_width=9,
    )
    core = CoralCore(config.model)
    loss_fn = CoralLoss(config.model)

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

            # Log to W&B
            if step % config.training.log_every == 0:
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
                )
                if wandb_run is not None:
                    wandb_run.log(quick_metrics, step=step)
                log.info(
                    f"[QUICK EVAL] step={step} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in quick_metrics.items())
                )

            # Full Pareto eval — all puzzles, K=1..16
            if step % config.training.pareto_eval_every == 0:
                log.info(f"Running full Pareto eval at step {step}...")
                pareto_results = evaluate_pareto(
                    adapter=trainer.adapter,
                    core=trainer.core,
                    dataloader=eval_loader,
                    device=device,
                    dtype=torch.bfloat16,
                )
                _log_precision_metrics(trainer, eval_loader, device, pareto_results)
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
    if wandb_run is not None:
        wandb_run.finish()


def _log_precision_metrics(trainer, eval_loader, device, out_dict):
    """Collect precision/error stats from one eval batch and add to out_dict."""
    trainer.adapter.eval()
    trainer.core.eval()
    dtype = trainer.dtype

    try:
        batch = next(iter(eval_loader))
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            z1 = trainer.adapter.encode(inputs)
            B, L, _ = z1.shape
            z_states = [z1]
            cfg = trainer.core.config
            for i in range(1, cfg.n_levels):
                d_l = cfg.level_dims[i]
                z_states.append(torch.zeros(B, L, d_l, device=device, dtype=dtype))

            # Run one segment to get precision stats
            if cfg.use_predictive_coding:
                for i in range(cfg.n_levels - 2, -1, -1):
                    pred = trainer.core.pc_modules[i].predict(z_states[i + 1])
                for i in range(cfg.n_levels):
                    z_states[i] = trainer.core._run_level(z_states[i], i, cfg.timescale_base, None)
                    if i < cfg.n_levels - 1:
                        _, eps, pi, _, _ = trainer.core.pc_modules[i](z_states[i], z_states[i + 1])
                        out_dict[f"precision/level{i}_mean"] = pi.mean().item()
                        out_dict[f"precision/level{i}_std"] = pi.std().item()
                        out_dict[f"prediction_error/level{i}_mean"] = eps.abs().mean().item()
                        out_dict[f"prediction_error/level{i}_max"] = eps.abs().max().item()
    except Exception as e:
        log.warning(f"Precision metric collection failed: {e}")


if __name__ == "__main__":
    main()
