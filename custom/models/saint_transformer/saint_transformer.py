import torch
import torch.nn as nn
import torch.nn.functional as f
from custom.blocks.energy_function_blocks import EnergyFunction
from custom.commons.replay_buffer import EBTReplayBuffer
from custom.layers.dual_attention_layer import DualAttentionLayer
from custom.models.saint_transformer.config import SAINTConfig
from custom.commons.batched_embedding import BatchedEmbedding
from custom.blocks.attention_pooling_blocks import AttentionPooling

# Import types
from torch import Tensor


class SAINTTransformer(nn.Module):
    def __init__(
        self,
        continuous_dims,
        categorical_dims: list,
        num_block_layers: int,
        d_model: int,
        num_heads: int,
        output_size: int,
        pos_weight: float,
        config: SAINTConfig,
    ):
        super().__init__()

        # === Configuration ===
        self.config = config
        assert self.config.num_variants >= 1, f"num_variants must be at least 1"

        # === Training Parameters ===
        self.pos_weight = torch.tensor(pos_weight)

        # === Model Architecture ===
        self.embedding_layer = BatchedEmbedding(
            continuous_dim=continuous_dims, categorical_cardinality=categorical_dims or [], embedding_dim=d_model
        )

        total_features = continuous_dims + len(categorical_dims)

        self.race_cls_token = nn.Parameter(torch.randn(1, total_features, d_model) * 0.02)
        self.race_projection = nn.Linear(d_model * 2, d_model)

        self.transformer_blocks = nn.ModuleList(
            [
                DualAttentionLayer(
                    d_model=d_model, num_heads=num_heads, num_competitors=config.num_competitors, config=config
                )
                for _ in range(num_block_layers)
            ]
        )

        self.pooler = AttentionPooling(d_model)
        # self.output_layer = nn.Linear(d_model, output_size)
        self.energy_function = EnergyFunction(d_model)
        self.output_size = output_size

        # === EBT Regularization components ===
        if config.mcmc_replay_buffer:
            self.replay_buffer = EBTReplayBuffer(config.mcmc_replay_buffer_size, config.mcmc_replay_buffer_sample_bs_percent)
        else:
            self.replay_buffer = None

        # Learnable Langevin noise (if enabled)
        if config.langevin_dynamics_noise_learnable:
            self.langevin_noise_std = nn.Parameter(torch.tensor(config.langevin_dynamics_noise))
        else:
            self.register_buffer("langevin_noise_std", torch.tensor(config.langevin_dynamics_noise))

    @staticmethod
    def _create_block_diagonal_mask(attention_mask: Tensor) -> tuple[Tensor, Tensor, int]:
        """
        Creates a block-diagonal attention mask to prevent cross-race attention.

        Args:
            attention_mask: [batch_size, max_horses] - 1 for real horses, 0 for padding
        """
        batch_size, max_horses = attention_mask.shape

        # Calc race lengths (real horses + 1 CLS token per race)
        race_lengths = attention_mask.sum(dim=1).int() + 1
        total_seq = int(race_lengths.sum().item())

        # Calc cumulative start positions for each race
        cum_lengths = torch.cumsum(race_lengths, dim=0)
        starts = torch.cat([torch.tensor([0], device=attention_mask.device), cum_lengths[:-1]])

        # Create block-diagonal mask [total_seq, total_seq]
        block_mask = torch.zeros(total_seq, total_seq, device=attention_mask.device, dtype=torch.bool)

        # Fill diagonal blocks
        for i in range(batch_size):
            start_pos = starts[i].item()
            end_pos = cum_lengths[i].item()

            block_mask[start_pos:end_pos, start_pos:end_pos] = True

        return block_mask, race_lengths, total_seq

    def _flatten_races_with_cls(self, x: Tensor, race_lengths: Tensor) -> Tensor:
        """Flattens all races into a single sequence, adding CLS tokens per race."""
        batch_size, horse_len, num_features, d_model = x.shape
        flattened_sequences = []

        for race_idx in range(batch_size):
            # Get only real horses (no padding)
            num_real_horses = race_lengths[race_idx] - 1
            race_x_real = x[race_idx, :num_real_horses]

            # Add CLS token for this race
            race_cls_token = self.race_cls_token.expand(1, num_features, d_model)
            race_x_with_cls = torch.cat([race_x_real, race_cls_token], dim=0)

            flattened_sequences.append(race_x_with_cls)

        flattened_x = torch.cat(flattened_sequences, dim=0)
        return flattened_x

    def _unflatten_to_race_outputs(
        self, flattened_x: Tensor, race_lengths: Tensor, horse_len: int, d_model: int
    ) -> list[Tensor]:
        """Splits flattened transformer output back into per-race processed results."""
        # Split flattened output back into per-race chunks
        race_chunks = torch.split(flattened_x, race_lengths.tolist(), dim=0)
        race_outputs = []

        for race_idx, race_chunk in enumerate(race_chunks):
            # Last position is CLS token, rest are horses
            race_cls = race_chunk[-1]
            horse_representations = race_chunk[:-1]

            num_real_horses = race_lengths[race_idx] - 1

            # Apply pooling
            horse_features = self.pooler(horse_representations)
            race_context = self.pooler(race_cls.unsqueeze(0)).squeeze(0)

            # Expand race context to match horse count
            race_context_expanded = race_context.unsqueeze(0).expand(num_real_horses, -1)

            # Combine and project
            combined = torch.cat([horse_features, race_context_expanded], dim=-1)
            cls_tokens = self.race_projection(combined)

            # Pad to full batch length
            padded_cls_tokens = torch.zeros(horse_len, d_model, device=cls_tokens.device)
            padded_cls_tokens[:num_real_horses] = cls_tokens

            race_outputs.append(padded_cls_tokens)

        return race_outputs

    def _vectorized_processing(self, x: Tensor, attention_mask: Tensor):
        """Replaces the squential per-race loop with a single call"""
        batch_size, horse_len, num_features, d_model = x.shape

        # Create block-diagonal mask
        block_mask, race_lengths, total_seq = self._create_block_diagonal_mask(attention_mask)

        # Flatten all races into sequence
        flattened_x = self._flatten_races_with_cls(x, race_lengths)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            flattened_x = block(flattened_x, block_mask)

        race_outputs = self._unflatten_to_race_outputs(flattened_x, race_lengths, horse_len, d_model)

        return race_outputs

    def _apply_replay_buffer(self, predictions: Tensor) -> Tensor:
        """
        Mix predictions with replay buffer samples during training.
        """
        num_variants, batch_size, horse_len = predictions.shape

        total_items = num_variants * batch_size
        num_replay = int(total_items * self.config.mcmc_replay_buffer_sample_bs_percent)
        replay_samples = self.replay_buffer.sample(num_replay, horse_len, predictions.device)

        if replay_samples is None:
            return predictions

        actual_num_replay = replay_samples.shape[0]
        predictions_flat = predictions.view(total_items, horse_len)

        if actual_num_replay > 0:
            predictions_flat = torch.cat((predictions_flat[:-actual_num_replay], replay_samples), dim=0)

        predictions = predictions_flat.view(num_variants, batch_size, horse_len)

        return predictions

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
        energies = self.energy_function(features.unsqueeze(0).expand(num_variants, -1, -1, -1), probs)

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

    def _mcmc_step(self, features: Tensor, predictions: Tensor, attention_mask: Tensor) -> Tensor:
        """Performs a single MCMC step: energy computation, gradient update, and regularization."""

        # --- Convert logits to probabilities for energy/entropy calculations ---
        probs = torch.sigmoid(predictions)

        # --- Energy computation ---
        energy_scores = self.energy_function(features, probs.unsqueeze(-1))
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

    def forward(self, x: dict[str, torch.Tensor], attention_mask: torch.Tensor, num_mcmc_steps: int = None):
        if num_mcmc_steps is None:
            num_mcmc_steps = self.config.mcmc_num_steps

        # --- MCMC Step Randomization (EBT technique) ---
        if self.training and self.config.randomize_mcmc_num_steps > 0:
            random_variation = torch.randint(0, self.config.randomize_mcmc_num_steps + 1, (1,)).item()
            num_mcmc_steps = max(self.config.randomize_mcmc_num_steps_min, num_mcmc_steps + random_variation)

        # --- Embeddings ---
        x: torch.Tensor = self.embedding_layer(x)  # Returns: [batch_size, horses_len, num_features, d_model]
        batch_size, horse_len, num_features, d_model = x.shape

        assert (
            num_features == self.race_cls_token.shape[1]
        ), f"Feature mismatch: {num_features} vs {self.race_cls_token.shape[1]}"

        # --- Vectorized processing ---
        race_outputs = self._vectorized_processing(x, attention_mask)
        features = torch.stack(race_outputs)

        # --- Initialize multiple variants ---
        num_variants = self.config.num_variants
        predictions = torch.randn(num_variants, batch_size, horse_len, device=x.device) * 0.1
        predictions = torch.clamp(predictions, min=-10, max=10)
        predictions.requires_grad_(True)

        # --- Replay Buffer: Replace some predictions with stored ones ---
        if self.training and self.replay_buffer is not None and len(self.replay_buffer) > 0:
            predictions = self._apply_replay_buffer(predictions)

        all_step_logits = []
        for mcmc_step in range(num_mcmc_steps):
            # Expand features and mask to match variants dim
            features_exp = features.unsqueeze(0).expand(num_variants, -1, -1, -1)
            mask_exp = attention_mask.unsqueeze(0).expand(num_variants, -1, -1)

            # Flatten for energy computation (process all variants at once)
            predictions_flat = predictions.view(num_variants * batch_size, horse_len)
            features_flat = features_exp.view(num_variants * batch_size, horse_len, d_model)
            mask_flat = mask_exp.view(num_variants * batch_size, horse_len)

            predictions_flat = self._mcmc_step(features_flat, predictions_flat, mask_flat)

            # Reshape back
            predictions = predictions_flat.view(num_variants, batch_size, horse_len)

            all_step_logits.append(predictions.unsqueeze(-1))

        # --- Store predictions in replay buffer for future use ---
        if self.training and self.replay_buffer is not None:
            self.replay_buffer.add(predictions.view(-1, horse_len))

        # --- Select the best variant per batch item ---
        competitive_steps = []
        for step_idx in range(len(all_step_logits)):
            step_logits = all_step_logits[step_idx].squeeze(-1)

            if num_variants > 1:
                step_selected = self._select_best_variant(step_logits, features, attention_mask)
                step_probs = self._apply_per_race_softmax(step_selected, attention_mask)
            else:
                step_probs = self._apply_per_race_softmax(step_logits.squeeze(0), attention_mask)

            competitive_steps.append(step_probs.unsqueeze(-1))

        return torch.stack(competitive_steps, dim=0)

    def compute_step(
        self, batch: tuple[dict[str, Tensor], Tensor, Tensor, Tensor], apply_label_smoothing: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        x, y, attention_mask, winner_indices = batch

        all_step_predictions = self(x, attention_mask)
        num_steps, batch_size, horse_len, _ = all_step_predictions.shape

        total_loss = 0
        final_probs_flat = None
        y_flat = None

        # Compute loss for each MCMC step
        for step_idx in range(num_steps):
            step_probs = all_step_predictions[step_idx].squeeze(-1)  # [batch_size, horse_len]
            step_loss = 0

            # Process each race seperately for cross-entropy
            for race_idx in range(batch_size):
                horse_mask = attention_mask[race_idx].bool()
                num_horses = horse_mask.sum().int()

                # Skip races with < 2 horses
                if num_horses < 2:
                    continue

                # Get race probabilities (already softmaxed from forward)
                race_probs = step_probs[race_idx][horse_mask]

                # Create one-hot target from winner index
                race_target = torch.zeros(num_horses, device=step_probs.device)
                race_target[winner_indices[race_idx]] = 1.0

                if self.config.label_smoothing and apply_label_smoothing:
                    # Decreasing label smoothing: stronger at first step, weaker at final step
                    smoothing_factor = 0.1 * (num_steps - step_idx) / num_steps
                    race_target = race_target * (1 - smoothing_factor) + smoothing_factor / num_horses

                # Compute cross-entropy loss for this race
                # Use log probabilities to avoid numerical issues
                log_probs = torch.log(race_probs + 1e-8)
                race_loss = -torch.sum(race_target * log_probs)
                step_loss += race_loss

            # Average loss over races in batch
            step_loss = step_loss / batch_size if batch_size > 0 else step_loss
            total_loss += step_loss

            # Store final step results for metrics
            if step_idx == num_steps - 1:
                # Flatten valid predictions and targets for metrics
                valid_mask = attention_mask.bool()
                final_probs_flat = step_probs[valid_mask]
                y_flat = y[valid_mask]

        avg_loss = total_loss / num_steps
        return avg_loss, final_probs_flat, y_flat

    def to(self, device: torch.device):
        # Move all standard PyTorch stuff
        super().to(device)

        # Recursively move any raw tensor attributes
        for module in self.modules():
            for name, attr in module.__dict__.items():
                if isinstance(attr, torch.Tensor):
                    setattr(module, name, attr.to(device))

        return self
