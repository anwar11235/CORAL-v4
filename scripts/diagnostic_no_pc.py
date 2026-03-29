"""Diagnostic: shared backbone without predictive coding.

Trains for 500 steps with use_predictive_coding=False, logging task loss
every 100 steps. Used to isolate whether learning problems are in the
backbone or in the PC mechanism.

Usage:
    python scripts/diagnostic_no_pc.py
"""

import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss
from coral.training.trainer import TrainerV4

logging.basicConfig(level=logging.WARNING, format="%(message)s")

STEPS = 500
BATCH_SIZE = 64
LOG_EVERY = 100


def _synthetic_batch(batch_size: int, device: torch.device) -> dict:
    return {
        "inputs": torch.randint(1, 11, (batch_size, 81), device=device),
        "labels": torch.randint(1, 11, (batch_size, 81), device=device),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = CoralConfig()
    cfg.device = str(device)
    cfg.training.precision = "bfloat16" if device.type == "cuda" else "float32"
    cfg.model.use_predictive_coding = False
    print("use_predictive_coding: False")

    adapter = GridAdapter(cfg)
    core = CoralCore(cfg.model)
    loss_fn = CoralLoss(cfg.model)
    trainer = TrainerV4(adapter, core, loss_fn, cfg)

    # Try real data; fall back to synthetic
    data_iter = None
    loader = None
    use_real = False
    try:
        from coral.data.sudoku_dataset import create_sudoku_dataloader
        loader, _ = create_sudoku_dataloader(
            dataset_path=cfg.data.dataset_path,
            split="train",
            global_batch_size=BATCH_SIZE,
            seed=42,
        )
        data_iter = iter(loader)
        _ = next(data_iter)  # eagerly verify files are readable
        data_iter = iter(loader)  # reset to start
        use_real = True
        print("Using real dataset")
    except Exception:
        data_iter = None
        print("Real dataset not found — using synthetic data")

    def next_batch() -> dict:
        nonlocal data_iter
        if not use_real:
            return _synthetic_batch(BATCH_SIZE, torch.device("cpu"))
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    for step in range(1, STEPS + 1):
        batch = next_batch()
        metrics = trainer.train_step(batch)

        if step % LOG_EVERY == 0:
            print(
                f"step={step:4d} | "
                f"loss={metrics['loss/total']:.4f} | "
                f"task={metrics.get('loss/task', 0.0):.4f} | "
                f"segs={metrics.get('train/num_segments', 0)}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
