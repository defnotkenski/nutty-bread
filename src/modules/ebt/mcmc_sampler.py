import torch
import torch.nn as nn
from src.models.saddle.config import SADDLEConfig
from torch import Tensor


class MCMCSampler(nn.Module):
    """
    Markov Chain Monte Carlo sampler that refines predictions using Langevin dynamics.

    Generates multiple prediction variants and iteratively improves them through energy-guided
    gradient descent with stochastic noise. Each MCMC step computes energy gradients to push
    predictions toward more realistic states while adding Langevin noise for exploration.
    Optionally integrates with memory bakery for experience replay during training.

    Input: Feature representations and attention masks
    Output: List of refined prediction tensors (one per MCMC step)
    """

    def __init__(self, energy_fn, config: SADDLEConfig, memory_bakery=None):
        super().__init__()

        self.energy_fn = energy_fn
        self.config = config
        self.memory_bakery = memory_bakery

        # Handle Langevin noise parameter
        if config.langevin_dynamics_noise_learnable:
            self.langevin_noise_std = nn.Parameter(torch.tensor(config.langevin_dynamics_noise))
        else:
            self.register_buffer("langevin_noise_std", torch.tensor(config.langevin_dynamics_noise))

    @staticmethod
    def _init_predictions(num_variants: int, batch_size: int, horse_len: int, device: torch.device):
        """
        Initialize random prediction variants.
        """
        predictions = torch.randn(num_variants, batch_size, horse_len, device=device) * 0.1
        predictions = torch.clamp(predictions, min=-10, max=10)
        predictions.requires_grad_(True)

        return predictions

    def _apply_memory_bakery(self, predictions: Tensor, is_training: bool) -> Tensor:
        """
        Mix predictions with memory bakery samples during training.
        """
        if not (is_training and self.memory_bakery and len(self.memory_bakery) > 0):
            return predictions

        num_variants, batch_size, horse_len = predictions.shape

        total_items = num_variants * batch_size
        num_samples = int(total_items * self.config.mcmc_memory_bakery_sample_bs_percent)
        bakery_samples = self.memory_bakery.sample(num_samples, horse_len, predictions.device)

        if bakery_samples is None:
            return predictions

        actual_num_samples = bakery_samples.shape[0]
        predictions_flat = predictions.view(total_items, horse_len)

        if actual_num_samples > 0:
            predictions_flat = torch.cat((predictions_flat[:-actual_num_samples], bakery_samples), dim=0)

        predictions = predictions_flat.view(num_variants, batch_size, horse_len)

        return predictions

    def _mcmc_step(self, features: Tensor, predictions: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Performs a single MCMC step: energy computation, gradient update, and regularization.
        """

        # --- Convert logits to probabilities for energy/entropy calculations ---
        probs = torch.sigmoid(predictions)

        # --- Energy computation ---
        energy_scores = self.energy_fn(features, probs.unsqueeze(-1))
        masked_energy = energy_scores * attention_mask.float()

        # --- Mean-based energy calculations ---
        energy_per_race = masked_energy.sum(dim=1)
        horses_per_race = attention_mask.sum(dim=1)
        mean_energy_per_race = energy_per_race / horses_per_race

        # --- Binary entropy calculations ---
        entropy = -(probs * torch.log(probs + 1e-7) + (1 - probs) * torch.log(1 - probs + 1e-7))
        masked_entropy = entropy * attention_mask.float()
        entropy_per_race = masked_entropy.sum(dim=1)
        mean_entropy_per_race = entropy_per_race / horses_per_race

        # --- Calculate total energy with entropy ---
        total_energy = mean_energy_per_race.mean() - self.config.entropy_beta * mean_entropy_per_race.mean()

        # --- Gradient computation and update ---
        energy_grad = torch.autograd.grad(total_energy, predictions, create_graph=True)[0]
        energy_grad = energy_grad * attention_mask.float()

        predictions = predictions - self.config.mcmc_step_size * energy_grad

        # --- Langevin Dynamics: Add noise (EBT regularization) ---
        if self.training and self.langevin_noise_std > 0:
            # Only add noise during training (not eval) if configured
            if not (self.config.no_langevin_during_eval and not self.training):
                langevin_noise = torch.randn_like(predictions).detach() * self.langevin_noise_std
                predictions = predictions + langevin_noise

        return predictions

    def forward(self, features: Tensor, attention_mask: Tensor, is_training: bool):
        """
        Main sampling method that orchestrates the entire MCMC process.
        """
        batch_size, horse_len, d_model = features.shape
        num_variants = self.config.num_variants
        device = features.device
        num_mcmc_steps = self.config.mcmc_num_steps

        # --- MCMC Step Randomization (EBT technique) ---
        if self.training and self.config.randomize_mcmc_num_steps > 0:
            random_variation = torch.randint(0, self.config.randomize_mcmc_num_steps + 1, (1,)).item()
            num_mcmc_steps = max(self.config.randomize_mcmc_num_steps_min, self.config.mcmc_num_steps + random_variation)

        # --- Initialize prediction variants ---
        predictions = self._init_predictions(
            num_variants=num_variants, batch_size=batch_size, horse_len=horse_len, device=device
        )

        # --- Apply memory bakery ---
        predictions = self._apply_memory_bakery(predictions=predictions, is_training=is_training)

        # --- MCMC loop ---
        all_step_predictions = []

        for mcmc_step in range(num_mcmc_steps):
            # Expand features and mask to match variants dim
            features_exp = features.unsqueeze(0).expand(num_variants, -1, -1, -1)
            mask_exp = attention_mask.unsqueeze(0).expand(num_variants, -1, -1)

            # Flatten for energy computation (process all variants at once)
            predictions_flat = predictions.view(num_variants * batch_size, horse_len)
            features_flat = features_exp.reshape(num_variants * batch_size, horse_len, d_model)
            mask_flat = mask_exp.reshape(num_variants * batch_size, horse_len)

            # Perform MCMC step
            predictions_flat = self._mcmc_step(
                features=features_flat, predictions=predictions_flat, attention_mask=mask_flat
            )

            # Reshape back to variants format
            predictions = predictions_flat.reshape(num_variants, batch_size, horse_len)

            # Store this step's results
            all_step_predictions.append(predictions.unsqueeze(-1))

        # --- Store predictions in memory bakery for future use ---
        if self.training and self.memory_bakery is not None:
            self.memory_bakery.add(predictions.view(-1, horse_len))

        return all_step_predictions
