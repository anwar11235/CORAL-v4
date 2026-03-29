"""Quick diagnostic: verify precision network produces varying output after training.

Trains for 200 steps and reports whether precision is differentiating.
Every 50 steps prints: precision mean/std/min/max, prediction error mean, task loss.

Exit code 0 if precision std > 0.01 at step 200, exit code 1 otherwise.

Usage:
    python scripts/diagnostic_precision.py
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

STEPS = 200
BATCH_SIZE = 64
LOG_EVERY = 50
PASS_THRESHOLD = 0.01


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

    last_pi_std = 0.0

    for step in range(1, STEPS + 1):
        batch = next_batch()
        metrics = trainer.train_step(batch)

        if step % LOG_EVERY == 0:
            trainer.adapter.eval()
            trainer.core.eval()

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=trainer.dtype):
                inputs = batch["inputs"].to(device)
                z1 = trainer.adapter.encode(inputs)
                output = trainer.core(
                    z1, K_max=1, training=False, decode_fn=trainer.adapter.decode
                )

            if cfg.model.use_predictive_coding and output.precisions:
                pi_tensors = list(output.precisions[0].values())
                eps_tensors = list(output.pred_errors[0].values())
                pi_all = torch.cat([t.float().flatten() for t in pi_tensors])
                eps_all = torch.cat([t.float().flatten() for t in eps_tensors])

                pi_mean = pi_all.mean().item()
                pi_std  = pi_all.std().item()
                pi_min  = pi_all.min().item()
                pi_max  = pi_all.max().item()
                eps_mean = eps_all.abs().mean().item()
                last_pi_std = pi_std

                print(
                    f"step={step:4d} | "
                    f"loss={metrics['loss/total']:.4f} | "
                    f"task={metrics.get('loss/task', 0.0):.4f} | "
                    f"eps_mean={eps_mean:.4f} | "
                    f"pi mean={pi_mean:.4f} std={pi_std:.6f} "
                    f"min={pi_min:.4f} max={pi_max:.4f}"
                )
            else:
                print(f"step={step:4d} | loss={metrics['loss/total']:.4f} (no PC module active)")

            trainer.adapter.train()
            trainer.core.train()

    passed = last_pi_std > PASS_THRESHOLD
    status = "PASS" if passed else "FAIL"
    print(
        f"\n{status}: precision std at step {STEPS} = {last_pi_std:.6f} "
        f"(threshold > {PASS_THRESHOLD})"
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
