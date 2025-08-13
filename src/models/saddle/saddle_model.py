import torch
import torch.nn as nn
from src.commons.memory_bakery import MemoryBakery
from src.layers.dual_attention_layer import DualAttentionLayer
from src.models.saddle.config import SADDLEConfig
from src.modules import BatchedEmbedding, AttentionPooling, EnergyOptimizer
from torch import Tensor


class SaddleModel(nn.Module):
    def __init__(
        self,
        continuous_dims,
        categorical_dims: list,
        num_block_layers: int,
        d_model: int,
        num_heads: int,
        config: SADDLEConfig,
    ):
        super().__init__()

        # === Configuration ===
        self.config = config
        assert self.config.num_variants >= 1, f"num_variants must be at least 1"

        # === Model Architecture ===
        self.embedding_layer = BatchedEmbedding(
            continuous_dim=continuous_dims, categorical_cardinality=categorical_dims or [], embedding_dim=d_model
        )

        total_features = continuous_dims + len(categorical_dims)

        self.race_cls_token = nn.Parameter(torch.randn(1, total_features, d_model) * 0.02)
        self.race_projection = nn.Linear(d_model * 2, d_model)

        self.pooler = AttentionPooling(d_model)

        self.transformer_blocks = nn.ModuleList(
            [
                DualAttentionLayer(
                    d_model=d_model, num_heads=num_heads, num_competitors=config.num_competitors, config=config
                )
                for _ in range(num_block_layers)
            ]
        )

        # === EBT Regularization components ===
        if config.mcmc_memory_bakery:
            self.memory_bakery = MemoryBakery(config.mcmc_memory_bakery_size, config.mcmc_memory_bakery_sample_bs_percent)
        else:
            self.memory_bakery = None

        # === Energy Sampler ===
        self.energy_optimizer = EnergyOptimizer(d_model, config, self.memory_bakery)

    def forward(self, x: dict[str, torch.Tensor], attention_mask: torch.Tensor) -> Tensor:
        """
        Forward pass through the SADDLE model with Energy-Based Training.

        Processes horse racing data through embedding, self-attention, and MCMC sampling
        to produce competitive probability distributions for each MCMC step.
        """

        # --- Embeddings ---
        x: torch.Tensor = self.embedding_layer(x)  # Returns: [batch_size, horses_len, num_features, d_model]
        batch_size, horse_len, num_features, d_model = x.shape

        assert (
            num_features == self.race_cls_token.shape[1]
        ), f"Feature mismatch: {num_features} vs {self.race_cls_token.shape[1]}"

        # --- Vectorized processing ---
        race_outputs = self._vectorized_processing(x, attention_mask)
        features = torch.stack(race_outputs)

        # --- Energy-based optimizing ---
        all_step_predictions = self.energy_optimizer(features, attention_mask, self.training)

        return all_step_predictions

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
