"""CORAL v4 — Hydra config dataclasses.

All hyperparameters flow from this config. Every field has a default so
that no `+` prefix is required on the command line.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    # Hierarchy
    n_levels: int = 2
    level_dims: List[int] = field(default_factory=lambda: [512, 256])

    # Backbone
    backbone_layers: int = 2
    backbone_dim: int = 512
    n_heads: int = 8
    d_k: int = 64  # head dimension; backbone_dim must equal n_heads * d_k
    ffn_expansion: int = 4  # SwiGLU expansion factor

    # Timescale: level l gets T^(l-1) inner steps per segment (l=1 is fastest)
    timescale_base: int = 3  # T=3

    # Predictive coding
    use_predictive_coding: bool = True
    lambda_pred: float = 0.1
    lambda_pi: float = 0.01
    epsilon_min: float = 0.01  # minimum precision floor

    # Halting
    K_max: int = 16
    halting_threshold: float = 0.95
    halting_exploration_prob: float = 0.1
    halting_gamma: float = 0.9  # Q-learning discount factor

    # Components disabled for Experiment 1
    use_crystallisation: bool = False
    use_column_heads: bool = False
    use_stochastic: bool = False
    use_amort: bool = False
    lambda_amort: float = 0.0
    lambda_crystal: float = 0.0
    lambda_commit: float = 0.0

    # Adapter
    vocab_size: int = 11  # Sudoku: PAD=0, digits 1-10


@dataclass
class TrainingConfig:
    epochs: int = 20000
    batch_size: int = 64
    learning_rate: float = 7e-5
    weight_decay: float = 1.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    gradient_clip: float = 1.0
    scheduler: str = "cosine"  # "cosine" or "constant"
    warmup_steps: int = 500

    # Mixed precision
    precision: str = "bfloat16"   # bfloat16 forward
    loss_precision: str = "float64"  # float64 loss

    # Full backprop within segment — critical for v4
    full_inner_backprop: bool = True

    # Evaluation
    eval_every: int = 500        # quick eval (first 100 puzzles, full K_max)
    pareto_eval_every: int = 5000  # full Pareto eval (all puzzles, K=1..16)
    log_every: int = 50

    # Optimizer selection
    optimizer: str = "adamw"  # "adamw" or "fused_adam_atan2"

    # torch.compile
    compile_model: bool = False
    compile_dynamic: bool = False  # False for Sudoku (fixed L=81)


@dataclass
class DataConfig:
    dataset: str = "sudoku_extreme_1k"
    dataset_path: str = "data/sudoku_extreme_1k"
    augmentation_factor: int = 1000
    eval_size: int = 1000
    num_workers: int = 1
    seq_len: int = 81  # 9x9 Sudoku


@dataclass
class WandbConfig:
    project: str = "Sudoku-extreme-1k-aug-1000 CORAL-v4"
    entity: str = "aktuator-ai"
    run_name: str = "v4-exp1-baseline"
    tags: List[str] = field(
        default_factory=lambda: ["v4", "exp1", "shared-backbone", "full-backprop", "predictive-coding"]
    )
    log_every: int = 50
    disabled: bool = False


@dataclass
class CoralConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 42
    device: str = "cuda"
    experiment_name: str = "exp1_baseline"
