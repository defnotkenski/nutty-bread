from dataclasses import dataclass
from typing import Optional


@dataclass
class SAINTConfig:
    # Notes
    notes: str = "Running experimental: Attention pooling. Stochastic competition attention."

    # Model hyperparams
    learning_rate: float = 1.0  # Prev. 0.0001
    num_block_layers: int = 4
    d_model: int = 64
    num_attention_heads: int = 4
    output_size: int = 1

    # Training hyperparams
    batch_size: int = 8
    num_workers: int = 6
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None  # Set to None for Prodigy compatibility
    max_epochs: int = 30
    val_check_interval: float = 1.0
    enable_checkpointing: bool = True

    # Attention hyperparams
    num_competitors: int = 4

    # Dataloader hyperparams
    pin_memory: bool = True
    shuffle: bool = False

    # Optimizer hyperparams
    prodigy_use_speed: bool = True
    prodigy_use_orthograd: bool = False
    prodigy_use_focus: bool = False
