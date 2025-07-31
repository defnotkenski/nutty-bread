from dataclasses import dataclass
from typing import Literal


@dataclass
class SAINTConfig:
    # Notes
    notes: str = ""

    # EBT hyperparams
    mcmc_num_steps: int = 3
    mcmc_step_size: float = 0.1
    entropy_beta: float = 0.01

    # Langevin hyperparams
    langevin_dynamics_noise: float = 0.1  # Noise std for Langevin dynamics
    langevin_dynamics_noise_learnable: bool = False  # Make noise learnable
    no_langevin_during_eval: bool = True  # Disable noise during validation

    # Replay Buffer Parameters
    mcmc_replay_buffer: bool = True  # Enable replay buffer
    mcmc_replay_buffer_size: int = 192  # Buffer size
    mcmc_replay_buffer_sample_bs_percent: float = 0.5  # % of batch from buffer

    # MCMC Randomization
    randomize_mcmc_num_steps: int = 1  # Max random variation in steps
    randomize_mcmc_num_steps_min: int = 2  # Min steps when randomizing

    # Model hyperparams
    learning_rate: float = 3e-4  # If using prodigy-plus, lr is automatically set to 1.0
    d_model: int = 64
    num_block_layers: int = 4
    num_attention_heads: int = 8
    output_size: int = 1
    dropout: float = 0.3
    label_smoothing: bool = False

    # Logging
    enable_logging: bool = True
    log_every_n_steps: int = 50

    # Training hyperparams
    disable_torch_compile: bool = False
    random_state: int = 777
    batch_size: int = 32
    num_workers: int = 6
    accumulate_grad_batches: int = 1
    gradient_clip_val: float | None = 1.0
    max_epochs: int = 50
    enable_checkpointing: bool = False
    precision: str | None = None
    early_stopping: bool = False

    # Inference hyperparams
    normalize_race_predictions: bool = False

    # Attention hyperparams
    num_competitors: int = 4

    # Dataloader hyperparams
    pin_memory: bool = True
    shuffle: bool = False

    # Optimizer hyperparams
    optimizer: Literal["prodigy-plus", "adamw", "adamw-schedule-free"] = "adamw-schedule-free"
    prodigy_use_speed: bool = True
    prodigy_use_orthograd: bool = False
    prodigy_use_focus: bool = False
    weight_decay: float = 0.01  # Automatically set to 0.0 if using prodigy's speed
    betas: tuple[float, float] = (0.9, 0.999)

    # Scheduler hyperparams
    scheduler: Literal["cosine"] | None = "cosine"
    warmup_pct: float = 0.1
    min_lr_ratio: float = 0.1
