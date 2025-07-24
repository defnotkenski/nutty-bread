from dataclasses import dataclass
from typing import Optional


@dataclass
class SAINTConfig:
    # Notes
    notes: str = "This is the ammendment to the original."

    # Model hyperparams
    learning_rate: float = 1e-5  # Set to 1.0 for Prodify otherwise 1e-5 is a good start
    d_model: int = 64
    num_block_layers: int = 4
    num_attention_heads: int = 8
    output_size: int = 1
    dropout: float = 0.3
    label_smoothing: bool = False

    # Training hyperparams
    random_state: int = 777
    batch_size: int = 32
    num_workers: int = 6
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None
    max_epochs: int = 30
    val_check_interval: float | None = None
    enable_checkpointing: bool | None = True
    precision: str | None = None
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
    weight_decay: float = 0.1  # Set to 0.0 if using Prodigy Speed otherwise 0.1 is good
    betas: tuple[float, float] = (0.9, 0.95)
