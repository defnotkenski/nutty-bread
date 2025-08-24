from dataclasses import dataclass
from typing import Literal


@dataclass
class SADDLEConfig:
    # --- Notes ---
    notes: str = ""

    # --- EBT hyperparams ---
    mcmc_num_steps: int = 4
    mcmc_step_size: float = 0.06
    entropy_beta: float = 0.01
    num_variants: int = 3
    variant_selection: Literal["lowest_energy"] = "lowest_energy"
    softmax_temperature: float = 0.9

    # --- Langevin hyperparams ---
    langevin_dynamics_noise: float = 0.05  # Noise std for Langevin dynamics
    langevin_dynamics_noise_learnable: bool = True  # Make noise learnable
    no_langevin_during_eval: bool = True  # Disable noise during validation

    # --- Memory Bakery Parameters ---
    mcmc_memory_bakery: bool = True  # Enable memory bakery
    mcmc_memory_bakery_size: int = 192  # Buffer size
    mcmc_memory_bakery_sample_bs_percent: float = 0.5  # % of batch from buffer

    # --- MCMC Randomization ---
    randomize_mcmc_num_steps: int = 1  # Max random variation in steps
    randomize_mcmc_num_steps_min: int = 2  # Min steps when randomizing

    # --- Model hyperparams ---
    learning_rate: float = 3e-4  # If using prodigy-plus, lr is automatically set to 1.0
    d_model: int = 128
    num_block_layers: int = 4
    num_attention_heads: int = 8
    output_size: int = 1
    dropout: float = 0.3
    label_smoothing: bool = False

    # --- Logging ---
    logging_mode: Literal["async", "debug"] = "debug"
    log_every_n_steps: int = 50

    # --- Training hyperparams ---
    disable_torch_compile: bool = False
    random_state: int = 777
    batch_size: int = 128
    num_workers: int = 6
    accumulate_grad_batches: int = 1
    gradient_clip_val: float | None = 1.0
    max_epochs: int = 50
    enable_checkpointing: bool = False
    precision: str | None = None
    early_stopping: bool = False
    target_type: Literal["win", "place", "show"] = "show"

    # --- Attention hyperparams ---
    num_competitors: int = 4

    # --- Dataloader hyperparams ---
    pin_memory: bool = True
    shuffle: bool = False

    # --- Optimizer hyperparams ---
    optimizer: Literal["prodigy-plus", "adamw", "adamw-schedule-free"] = "adamw-schedule-free"
    prodigy_use_speed: bool = True
    prodigy_use_orthograd: bool = False
    prodigy_use_focus: bool = False
    weight_decay: float = 0.01  # Automatically set to 0.0 if using prodigy's speed
    betas: tuple[float, float] = (0.9, 0.999)

    # --- Scheduler hyperparams ---
    scheduler: Literal["cosine"] | None = "cosine"
    warmup_pct: float = 0.1
    min_lr_ratio: float = 0.1

    # --- Validation hyperparams ---
    mc_samples: int = 100  # Old: 20
    mc_use_dropout: bool = True
    uncertainty_thresholds: tuple[float, ...] = (0.30,)
