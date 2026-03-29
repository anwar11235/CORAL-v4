"""CORAL v4 — Pareto evaluation script.

Evaluates accuracy at K=1,2,4,8,16 segments to produce the accuracy-depth
Pareto curve.
"""

import logging
import sys
import os

import hydra
import torch
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coral.adapters.grid import GridAdapter
from coral.data.sudoku_dataset import create_sudoku_dataloader
from coral.evaluation.pareto import evaluate_pareto
from coral.model.coral_core import CoralCore
from scripts.train import dict_to_config

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="exp1_baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    config = dict_to_config(cfg)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    adapter = GridAdapter(config)
    core = CoralCore(config.model)

    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        adapter.load_state_dict(ckpt["adapter"])
        core.load_state_dict(ckpt["core"])

    adapter = adapter.to(device).to(torch.bfloat16)
    core = core.to(device).to(torch.bfloat16)

    eval_loader, _ = create_sudoku_dataloader(
        dataset_path=config.data.dataset_path,
        split="test",
        global_batch_size=config.training.batch_size,
        test_set_mode=True,
    )

    results = evaluate_pareto(adapter, core, eval_loader, device, dtype=torch.bfloat16)

    print("\n=== CORAL v4 Pareto Evaluation ===")
    print(f"{'K':>4}  {'Accuracy':>10}")
    for K in [1, 2, 4, 8, 16]:
        acc = results.get(f"eval/accuracy@K{K}", 0.0)
        print(f"{K:>4}  {acc:>10.4f}")
    print(f"\nPareto area: {results.get('eval/pareto_area', 0.0):.4f}")
    print(f"Full accuracy: {results.get('eval/exact_accuracy', 0.0):.4f}")
    print(f"Avg halting step: {results.get('eval/avg_halting_step', 0.0):.2f}")


if __name__ == "__main__":
    main()
