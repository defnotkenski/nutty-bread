from dataclasses import dataclass
from typing import Optional


@dataclass
class SAINTConfig:
    # Notes
    notes: str = "Running experimental: Attention pooling. Stochastic competition attention."

    # Model hyperparams
    learning_rate: float = 1.0
    d_model: int = 64
    num_block_layers: int = 4
    num_attention_heads: int = 8
    output_size: int = 1
    dropout: float = 0.3
    label_smoothing: bool = True

    # Training hyperparams
    batch_size: int = 32
    num_workers: int = 6
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None
    max_epochs: int = 30
    val_check_interval: float | None = None  # Default: None
    enable_checkpointing: bool | None = True  # Default: None
    precision: str | None = None  # Default: None
    early_stopping: bool = False

    # Attention hyperparams
    num_competitors: int = 4

    # Dataloader hyperparams
    pin_memory: bool = True
    shuffle: bool = False

    # Optimizer hyperparams
    prodigy_use_speed: bool = True
    prodigy_use_orthograd: bool = False
    prodigy_use_focus: bool = False
    weight_decay: float = 0.0
