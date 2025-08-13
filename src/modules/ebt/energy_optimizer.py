import torch
import torch.nn as nn
from src.models.saddle.config import SADDLEConfig
from src.modules.ebt import EnergyFunction, MCMCSampler
from torch import Tensor
import torch.nn.functional as f


class EnergyOptimizer(nn.Module):
    """
    High-level EBT coordinator that orchestrates MCMC sampling and variant selection.

    Uses MCMCSampler to generate multiple prediction candidates, then applies energy-based
    selection to choose the best variant at each step. Converts raw MCMC outputs into
    race-ready probability distributions via per-race softmax normalization.

    Input: Transformer feature representations
    Output: Final probability distributions for race prediction
    """

    def __init__(self, d_model: int, config: SADDLEConfig, memory_bakery=None):
        super().__init__()

        self.config = config
        self.energy_fn = EnergyFunction(d_model)

        self.memory_bakery = memory_bakery
        self.mcmc_sampler = MCMCSampler(energy_fn=self.energy_fn, config=config, memory_bakery=self.memory_bakery)

    def forward(self, features: Tensor, attention_mask: Tensor, training: bool) -> Tensor:
        # --- MCMC sampler ---
        all_step_logits = self.mcmc_sampler(features, attention_mask, training)

        # --- Select the best variant per batch item ---
        competitive_steps = []
        for step_idx in range(len(all_step_logits)):
            step_logits = all_step_logits[step_idx].squeeze(-1)

            if self.config.num_variants > 1:
                step_selected = self._select_best_variant(step_logits, features, attention_mask)
                step_probs = self._apply_per_race_softmax(step_selected, attention_mask)
            else:
                step_probs = self._apply_per_race_softmax(step_logits.squeeze(0), attention_mask)

            competitive_steps.append(step_probs.unsqueeze(-1))

        return torch.stack(competitive_steps, dim=0)

    @staticmethod
    def _apply_per_race_softmax(logits_tensor: Tensor, attention_mask: Tensor):
        """
        Helper method for applying softmax to a single variant.

        Args:
            logits_tensor: [batch_len, horse_len] - Raw prediction scores for each horse
            attention_mask: [batch_len, horse_len] - Binary mask
        Returns:
            race_probs: [batch_len, horse_len] - Softmax probabilities for each horse within their race
        """

        # Extract dimensions: [batch_len, horse_len]
        batch_len, horse_len = logits_tensor.shape

        # Create output tensor initialized with zeros -> [batch_size, horse_len]
        race_probs = torch.zeros_like(logits_tensor)

        # Loop through each race in the batch
        for race_idx in range(batch_len):
            # Get boolean mask for current race -> [horse_len] as boolean tensor
            # True for real horses, False for padding
            horse_mask = attention_mask[race_idx].bool()

            # Check if current race has multiple horses
            if horse_mask.sum() > 1:
                # Extract logits only for real horses -> [num_real_horses]
                # Variable length depending on how many True values in mask
                race_logits = logits_tensor[race_idx][horse_mask]

                # Apply softmax to compete only within this race -> [num_real_horses]
                race_softmax = f.softmax(race_logits, dim=0)

                # Put softmax results back into output tensor
                race_probs[race_idx][horse_mask] = race_softmax

            # Handle edge case of single horse race
            elif horse_mask.sum() == 1:
                race_probs[race_idx][horse_mask] = 1.0

        return race_probs

    def _select_best_variant(self, variants_preds: Tensor, features: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Selects the best variant per batch item based on config.

        Args:
            variants_preds: [num_variants, batch_len, horse_len] - Raw prediction scores for each variant
            features: [batch_len, horse_len, d_model] - Horse feature representations
            attention_mask: [batch_len, horse_len] - Mask indicating real horses vs padding
        Returns:
            best_preds:[batch_len, horse_len] - Best prediction per race based on lowest energy


        Note: I have no desire to implement other strategies for the sake of simplicity.
        """
        # Extract dimensions: [3, 32, 20]
        num_variants, batch_size, horse_len = variants_preds.shape

        # Create an empty tensor filled with zeros: [32, 20]
        best_preds = torch.zeros(batch_size, horse_len, device=variants_preds.device)

        # Add dimension: [3, 32, 20] -> [3, 32, 20, 1]
        # Apply sigmoid to convert to probabilities
        probs = torch.sigmoid(variants_preds.unsqueeze(-1))

        # Add dimension to features: [batch_len, horse_len, d_model] -> [1, ...]
        # Expand to: [3, ...]
        energies = self.energy_fn(features.unsqueeze(0).expand(num_variants, -1, -1, -1), probs)

        # Add dimension to attention_mask [batch_len, horse_len] -> [1, batch_len, horse_len]
        # Multiply energies by mask to zero out padding horses
        masked_energies = energies * attention_mask.unsqueeze(0).float()

        # Sum energies across horses: [num_variants, batch_len, horse_len] -> [num_variants, batch_len]
        # This gives total energy per race for each variant.
        sum_energies = masked_energies.sum(dim=-1)

        # Sum energies across horses: [batch_len, horse_len] -> [1, batch_len]
        # This gives count of real horses per race. Prevents division by zero if no horses. Adds variant dimension.
        sum_attention_mask = attention_mask.sum(dim=-1).clamp(min=1).unsqueeze(0)

        # Division via broadcasting -> [num_variants, batch_len]
        mean_energies = sum_energies / sum_attention_mask

        # Find the index of the min. value along the variant dimension -> [batch_len]
        best_indices = mean_energies.argmin(dim=0)

        for b in range(batch_size):
            # Index the first dimension of best_preds and replace [horse_len] with winning variant's [horse_len]
            best_preds[b] = variants_preds[best_indices[b], b]

        return best_preds
