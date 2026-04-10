"""Tests for Maze-Hard support (Session A).

Verifies:
  - GridAdapter works with 30x30 grid and vocab_size=6
  - Maze config loads correctly with seq_len=900 and vocab_size=6
  - CoralCore forward pass works with maze dimensions
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig, ModelConfig, TrainingConfig, DataConfig, WandbConfig
from coral.model.coral_core import CoralCore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _make_maze_model_config(**overrides) -> ModelConfig:
    """Minimal ModelConfig for a 30x30 maze smoke test."""
    defaults = dict(
        n_levels=1,
        level_dims=[512],
        backbone_dim=512,
        backbone_layers=2,
        n_heads=8,
        d_k=64,
        ffn_expansion=4,
        timescale_base=3,
        use_predictive_coding=False,
        use_crystallisation=False,
        use_amort=False,
        use_local_attention_bias=False,
        K_max=2,
        inner_steps_override=3,
        grad_inner_steps=None,
        vocab_size=6,
        embed_scale=True,
        use_consolidation_step=False,
        mode="baseline",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_coral_config(model_cfg: ModelConfig) -> CoralConfig:
    return CoralConfig(
        model=model_cfg,
        training=TrainingConfig(),
        data=DataConfig(
            dataset="maze_30x30_hard",
            seq_len=900,
            grid_height=30,
            grid_width=30,
        ),
        wandb=WandbConfig(),
    )


# ---------------------------------------------------------------------------
# Test 1: GridAdapter 30x30 encode
# ---------------------------------------------------------------------------

def test_grid_adapter_30x30():
    """GridAdapter encode produces correct shape for 30x30 maze input."""
    model_cfg = _make_maze_model_config()
    config = _make_coral_config(model_cfg)

    adapter = GridAdapter(config, vocab_size=6, grid_height=30, grid_width=30)
    adapter.eval()

    B = 2
    x = torch.randint(0, 6, (B, 900))  # [B, 900] tokens in [0,5]
    with torch.no_grad():
        z = adapter.encode(x)

    assert z.shape == (B, 900, 512), f"Expected (2, 900, 512), got {z.shape}"
    assert z.isfinite().all(), "encode output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 2: GridAdapter 30x30 decode
# ---------------------------------------------------------------------------

def test_grid_adapter_decode_30x30():
    """GridAdapter decode produces correct shape for 30x30 maze embeddings."""
    model_cfg = _make_maze_model_config()
    config = _make_coral_config(model_cfg)

    adapter = GridAdapter(config, vocab_size=6, grid_height=30, grid_width=30)
    adapter.eval()

    B = 2
    z = torch.randn(B, 900, 512)
    with torch.no_grad():
        logits = adapter.decode(z)

    assert logits.shape == (B, 900, 6), f"Expected (2, 900, 6), got {logits.shape}"
    assert logits.isfinite().all(), "decode output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 3: Maze config loads correctly
# ---------------------------------------------------------------------------

def test_maze_config_loads():
    """phase1_maze_baseline.yaml loads into a valid CoralConfig."""
    omegaconf = pytest.importorskip("omegaconf", reason="omegaconf not installed")
    OmegaConf = omegaconf.OmegaConf

    path = os.path.join(_CONFIGS_DIR, "phase1_maze_baseline.yaml")
    assert os.path.exists(path), f"Config file not found: {path}"

    raw = OmegaConf.load(path)
    model_cfg = ModelConfig(**dict(raw.model))
    train_cfg = TrainingConfig(**dict(raw.training))
    data_cfg = DataConfig(**dict(raw.data))
    wandb_cfg = WandbConfig(**dict(raw.wandb))
    config = CoralConfig(
        model=model_cfg,
        training=train_cfg,
        data=data_cfg,
        wandb=wandb_cfg,
        seed=raw.seed,
        device=raw.device,
        experiment_name=raw.experiment_name,
    )

    assert config.data.seq_len == 900
    assert config.model.vocab_size == 6
    assert config.data.grid_height == 30
    assert config.data.grid_width == 30
    assert config.data.dataset == "maze_30x30_hard"
    assert config.model.use_local_attention_bias is False
    assert config.model.mode == "baseline"


# ---------------------------------------------------------------------------
# Test 4: CoralCore forward pass with maze dimensions
# ---------------------------------------------------------------------------

def test_coral_core_forward_maze_shapes():
    """CoralCore forward pass works with maze dimensions (vocab=6, seq_len=900)."""
    model_cfg = _make_maze_model_config(K_max=2, inner_steps_override=3)
    config = _make_coral_config(model_cfg)

    core = CoralCore(model_cfg)
    core.eval()

    B = 2
    z1 = torch.randn(B, 900, 512)

    with torch.no_grad():
        output = core(z1, K_max=2, training=False, decode_fn=None)

    assert output.z_states[0].shape == (B, 900, 512), (
        f"Expected z_states[0] shape (2, 900, 512), got {output.z_states[0].shape}"
    )
    assert output.z_states[0].isfinite().all(), "CoralCore output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 5: evaluate_accuracy returns maze metrics for maze dataset
# ---------------------------------------------------------------------------

def test_evaluate_accuracy_returns_maze_metrics():
    """evaluate_accuracy with dataset_name='maze_30x30_hard' returns maze-specific keys."""
    from coral.evaluation.evaluator import evaluate_accuracy
    from torch.utils.data import DataLoader, Dataset

    model_cfg = ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=3, K_max=2,
        use_predictive_coding=False, use_crystallisation=False,
        use_amort=False, lambda_amort=0.0,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=6, mode="baseline",
        inner_steps_override=2,
        use_local_attention_bias=False,
    )
    cfg = CoralConfig(model=model_cfg, device="cpu")

    adapter = GridAdapter(cfg, vocab_size=6, grid_height=10, grid_width=10)
    core = CoralCore(model_cfg)

    class TinyMaze(Dataset):
        def __len__(self): return 4
        def __getitem__(self, i):
            inp = torch.randint(1, 5, (100,), dtype=torch.long)
            lbl = inp.clone()
            lbl[::10] = 5  # mark some cells as optimal path
            return {"inputs": inp, "labels": lbl}

    loader = DataLoader(TinyMaze(), batch_size=2)
    metrics = evaluate_accuracy(
        adapter=adapter, core=core, dataloader=loader,
        device=torch.device("cpu"), dtype=torch.float32,
        max_puzzles=4, dataset_name="maze_30x30_hard",
    )

    assert "eval/path_accuracy" in metrics, f"Missing eval/path_accuracy. Got: {list(metrics.keys())}"
    assert "eval/wall_accuracy" in metrics
    assert "eval/non_path_accuracy" in metrics
    assert "eval/exact_accuracy" in metrics
    assert "eval/token_accuracy" in metrics
    assert "eval/bucket_60_plus_empty_acc" not in metrics, "Sudoku metrics leaked into maze eval"
    assert "eval/bucket_0_29_token_acc" not in metrics

    for k, v in metrics.items():
        if k != "eval/avg_halting_step":
            assert 0.0 <= v <= 1.0, f"{k}={v} out of range"


# ---------------------------------------------------------------------------
# Test 6: evaluate_accuracy preserves Sudoku metrics by default
# ---------------------------------------------------------------------------

def test_evaluate_accuracy_preserves_sudoku_metrics():
    """evaluate_accuracy with default dataset_name returns Sudoku bucket metrics unchanged."""
    from coral.evaluation.evaluator import evaluate_accuracy
    from torch.utils.data import DataLoader, Dataset

    model_cfg = ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=3, K_max=2,
        use_predictive_coding=False, use_crystallisation=False,
        use_amort=False, lambda_amort=0.0,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=11, mode="baseline",
        inner_steps_override=2,
    )
    cfg = CoralConfig(model=model_cfg, device="cpu")

    adapter = GridAdapter(cfg, vocab_size=11, grid_height=9, grid_width=9)
    core = CoralCore(model_cfg)

    class TinySudoku(Dataset):
        def __len__(self): return 4
        def __getitem__(self, i):
            inp = torch.randint(1, 11, (81,), dtype=torch.long)
            lbl = torch.randint(2, 11, (81,), dtype=torch.long)
            return {"inputs": inp, "labels": lbl}

    loader = DataLoader(TinySudoku(), batch_size=2)
    metrics = evaluate_accuracy(
        adapter=adapter, core=core, dataloader=loader,
        device=torch.device("cpu"), dtype=torch.float32,
        max_puzzles=4,
        # dataset_name defaults to "sudoku_extreme_1k"
    )

    assert "eval/exact_accuracy" in metrics
    assert "eval/token_accuracy" in metrics
    assert "eval/path_accuracy" not in metrics
    assert "eval/wall_accuracy" not in metrics
