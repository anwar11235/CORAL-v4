"""CORAL v4 — Phase 2 state collection script.

Loads a trained checkpoint and runs forward passes on the eval set, recording
level-0 (z_states[0]) at specified segment indices.

Output (.npz):
    states         [N, 81, len(segments), 512]  — collected z_states
    labels         [N, 81]                       — ground-truth digits
    given_mask     [N, 81]                       — True where cell is a given (clue)
    segment_indices [len(segments)]              — the collected segment indices

Usage:
    python scripts/collect_states.py \\
        --checkpoint checkpoints/phase1_best.pt \\
        --data-dir data/sudoku_extreme_1k \\
        --output states_phase1.npz \\
        --segments 4,8,12,16 \\
        --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig, DataConfig, ModelConfig, TrainingConfig, WandbConfig
from coral.data.sudoku_dataset import create_sudoku_dataloader
from coral.model.coral_core import CoralCore


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, config_name: str, device: torch.device):
    """Load a CORAL v4 checkpoint.

    Config is composed from ``configs/<config_name>.yaml`` via OmegaConf —
    not cast from the checkpoint dict (which is a plain dict from
    OmegaConf.to_container, not a CoralConfig dataclass).

    State dicts are loaded with ``strict=False`` to tolerate unexpected keys
    from older checkpoints (e.g. row_bias/col_bias/box_bias written by
    Session-F-era runs that pre-date the current adapter layout).

    Args:
        checkpoint_path: Path to a ``last.pt`` checkpoint saved by train.py.
        config_name:     Name of the config YAML in ``configs/`` (no extension).
        device:          Device to map tensors onto.

    Returns:
        (adapter, core, config)
    """
    # Load config from configs/ (identical pattern to diagnose_rank_collapse.py)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, "configs", f"{config_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    config = CoralConfig(
        model=ModelConfig(**dict(cfg.model)),
        training=TrainingConfig(**dict(cfg.training)),
        data=DataConfig(**dict(cfg.data)),
        wandb=WandbConfig(**dict(cfg.wandb)),
        seed=int(getattr(cfg, "seed", 42)),
        device=str(getattr(cfg, "device", "cpu")),
        experiment_name=str(getattr(cfg, "experiment_name", config_name)),
    )

    # Build model architecture from config
    adapter = GridAdapter(
        config,
        grid_height=config.data.grid_height,
        grid_width=config.data.grid_width,
    )
    core = CoralCore(config.model)

    # Load weights from nested model_state_dict keys (train.py format)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    msd = ckpt["model_state_dict"]

    res_a = adapter.load_state_dict(msd["adapter"], strict=False)
    res_c = core.load_state_dict(msd["core"], strict=False)

    if res_a.unexpected_keys or res_a.missing_keys:
        print(f"  adapter state_dict: unexpected={res_a.unexpected_keys}, "
              f"missing={res_a.missing_keys}")
    if res_c.unexpected_keys or res_c.missing_keys:
        print(f"  core state_dict:    unexpected={res_c.unexpected_keys}, "
              f"missing={res_c.missing_keys}")

    adapter = adapter.to(device)
    core = core.to(device)
    adapter.eval()
    core.eval()

    return adapter, core, config


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_states(
    adapter: GridAdapter,
    core: CoralCore,
    data_dir: str,
    segment_indices: list,
    device: torch.device,
    batch_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Collect level-0 states at specified segment indices over the eval set.

    Runs the model once per target segment index (with K_max=s, no halting)
    so that states at each segment are correctly captured.

    Args:
        adapter:         GridAdapter for encoding.
        core:            CoralCore reasoning module.
        data_dir:        Path to sudoku_extreme_1k data directory.
        segment_indices: List of int segment indices to collect (e.g. [4,8,12,16]).
        device:          Torch device.
        batch_size:      Eval batch size.
        dtype:           Compute dtype (float32 by default for CPU compat).

    Returns:
        Dict with 'states', 'labels', 'given_mask', 'segment_indices'.
    """
    loader, _ = create_sudoku_dataloader(
        dataset_path=data_dir,
        split="test",
        global_batch_size=batch_size,
        seed=0,
        test_set_mode=True,
    )

    # Disable halting so we always reach the requested segment count
    orig_threshold = core.config.halting_threshold
    core.config.halting_threshold = 2.0

    # First pass: collect labels and given_mask (same across all segment runs)
    all_labels = []
    all_given = []

    for batch in loader:
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        all_labels.append(labels.cpu().numpy().astype(np.int16))
        all_given.append((inputs != 0).cpu().numpy())

    labels_arr = np.concatenate(all_labels, axis=0)   # [N, 81]
    given_arr = np.concatenate(all_given, axis=0)     # [N, 81]
    N = labels_arr.shape[0]
    d = core.config.level_dims[0]
    n_segs = len(segment_indices)

    states_arr = np.zeros((N, 81, n_segs, d), dtype=np.float32)

    # One pass per target segment index
    for seg_col, target_seg in enumerate(segment_indices):
        print(f"Collecting states at segment {target_seg} / {segment_indices[-1]} ...", flush=True)

        loader2, _ = create_sudoku_dataloader(
            dataset_path=data_dir,
            split="test",
            global_batch_size=batch_size,
            seed=0,
            test_set_mode=True,
        )

        row = 0
        for batch in loader2:
            inputs = batch["inputs"].to(device)
            B = inputs.shape[0]

            with torch.autocast(device_type=device.type, dtype=dtype):
                z1 = adapter.encode(inputs)
                output = core(
                    z1,
                    K_max=target_seg,
                    training=False,
                    decode_fn=None,
                )

            z = output.z_states[0].float().cpu().numpy()  # [B, 81, d]
            states_arr[row:row + B, :, seg_col, :] = z
            row += B

    core.config.halting_threshold = orig_threshold

    return {
        "states": states_arr,
        "labels": labels_arr,
        "given_mask": given_arr,
        "segment_indices": np.array(segment_indices, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Collect CORAL v4 reasoning states for Phase 2 analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument(
        "--config-name", required=True,
        help="Name of the config YAML in configs/ (without .yaml extension), "
             "e.g. phase1_baseline_no_pc.  Used to reconstruct the model architecture.",
    )
    parser.add_argument("--data-dir", required=True, help="Path to sudoku_extreme_1k data directory")
    parser.add_argument("--output", required=True, help="Output path for .npz file")
    parser.add_argument(
        "--segments", default="4,8,12,16",
        help="Comma-separated list of segment indices to collect (default: 4,8,12,16)"
    )
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Eval batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    segment_indices = [int(s.strip()) for s in args.segments.split(",")]
    print(f"Collecting states at segments: {segment_indices}")

    print(f"Loading checkpoint from {args.checkpoint} (config: {args.config_name}) ...")
    adapter, core, config = load_checkpoint(args.checkpoint, args.config_name, device)

    # Use float32 for CPU compat; bfloat16 on GPU
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("Starting state collection ...")
    result = collect_states(
        adapter=adapter,
        core=core,
        data_dir=args.data_dir,
        segment_indices=segment_indices,
        device=device,
        batch_size=args.batch_size,
        dtype=dtype,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"Saved states to {args.output}")
    print(f"  states shape:    {result['states'].shape}")
    print(f"  labels shape:    {result['labels'].shape}")
    print(f"  given_mask shape:{result['given_mask'].shape}")
    print(f"  segments:        {result['segment_indices'].tolist()}")


if __name__ == "__main__":
    main()
