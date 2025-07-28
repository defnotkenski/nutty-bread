import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as f
from custom.blocks.energy_function_blocks import EnergyFunction
from custom.commons.replay_buffer import EBTReplayBuffer
from custom.layers.dual_attention_layer import DualAttentionLayer
from custom.models.saint_transformer.config import SAINTConfig
from custom.commons.batched_embedding import BatchedEmbedding
from custom.blocks.attention_pooling_blocks import AttentionPooling


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

    def forward(
        self,
        x: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        num_mcmc_steps: int = None,
        return_all_steps: bool = False,
    ):
        if num_mcmc_steps is None:
            num_mcmc_steps = self.config.mcmc_num_steps

        # MCMC Step Randomization (EBT technique)
        if self.training and self.config.randomize_mcmc_num_steps > 0:
            random_variation = torch.randint(0, self.config.randomize_mcmc_num_steps + 1, (1,)).item()
            num_mcmc_steps = max(self.config.randomize_mcmc_num_steps_min, num_mcmc_steps + random_variation)

        # --- Embeddings ---
        x: torch.Tensor = self.embedding_layer(x)  # Returns: [batch_size, horses_len, num_features, d_model]
        batch_size, horse_len, num_features, d_model = x.shape

        assert (
            num_features == self.race_cls_token.shape[1]
        ), f"Feature mismatch: {num_features} vs {self.race_cls_token.shape[1]}"

        # --- Process each race seperately to maintain race boundaries ---
        race_outputs = []
        for race_idx in range(batch_size):
            race_mask = attention_mask[race_idx]
            race_x = x[race_idx]

            # --- Add class token ---
            race_cls_token = self.race_cls_token.expand(1, num_features, d_model)
            race_x_with_cls = torch.cat([race_x, race_cls_token], dim=0)

            # --- Create mask for class token ---
            cls_mask = torch.ones(1, device=race_mask.device)
            full_mask = torch.cat([race_mask, cls_mask], dim=0)

            assert (
                full_mask.sum() > 0
            ), f"Race {race_idx}: All positions masked - race_mask sum: {race_mask.sum()}, cls_mask sum: {cls_mask.sum()}"

            # --- Pass through transformer blocks ---
            for block in self.transformer_blocks:
                race_x_with_cls = block(race_x_with_cls, full_mask)

            # --- Extract class token ---
            race_cls = race_x_with_cls[-1]  # class token
            horse_representations = race_x_with_cls[:-1]  # All horses (including padding)

            # --- Apply mask when computing features (only use real horses) ---
            num_real_horses = int(race_mask.sum())
            horse_reps_real = horse_representations[:num_real_horses]

            horse_features: Tensor = self.pooler(horse_reps_real)

            # --- Combine horse features with race context ---
            race_context: Tensor = self.pooler(race_cls.unsqueeze(0)).squeeze(0)
            race_context_expanded = race_context.unsqueeze(0).expand(num_real_horses, -1)

            combined = torch.cat([horse_features, race_context_expanded], dim=-1)
            cls_tokens = self.race_projection(combined)

            # --- Pad to full length ---
            padded_cls_tokens = torch.zeros(horse_len, d_model, device=cls_tokens.device)
            padded_cls_tokens[:num_real_horses] = cls_tokens
            race_outputs.append(padded_cls_tokens)

        features = torch.stack(race_outputs)

        # --- Energy minimization with step tracking ---
        predictions = torch.randn(batch_size, horse_len, device=x.device) * 0.1

        # --- Replay Buffer: Replace some predictions with stored ones ---
        if self.training and self.replay_buffer is not None and len(self.replay_buffer) > 0:
            replay_samples = self.replay_buffer.sample(batch_size, horse_len, x.device)
            if replay_samples is not None:
                num_replay = replay_samples.shape[0]
                predictions[-num_replay:] = replay_samples

        all_step_logits = []

        predictions.requires_grad_(True)
        for mcmc_step in range(num_mcmc_steps):
            energy_scores = self.energy_function(features, predictions.unsqueeze(-1))

            masked_energy = energy_scores * attention_mask.float()

            # total_energy = masked_energy.sum() # Fixed a bug where bias is towards longer races
            energy_per_race = masked_energy.sum(dim=1)
            horses_per_race = attention_mask.sum(dim=1)
            mean_energy_per_race = energy_per_race / horses_per_race
            total_energy = mean_energy_per_race.mean()

            energy_grad = torch.autograd.grad(total_energy, predictions, create_graph=True)[0]
            energy_grad = energy_grad * attention_mask.float()

            predictions = predictions - self.config.mcmc_step_size * energy_grad

            # Langevin Dynamics: Add noise (EBT regularization)
            if self.training and self.langevin_noise_std > 0:
                # Only add noise during training (not eval) if configured
                if not (self.config.no_langevin_during_eval and not self.training):
                    langevin_noise = torch.randn_like(predictions).detach() * self.langevin_noise_std
                    predictions = predictions + langevin_noise

            predictions = (torch.tanh(predictions - 0.5) + 1) / 2 * (1 - 2e-7) + 1e-7

            if return_all_steps:
                all_step_logits.append(predictions.unsqueeze(-1))

        # --- Store predictions in replay buffer for future use ---
        if self.training and self.replay_buffer is not None:
            self.replay_buffer.add(predictions)

        # --- Apply race-level normalization (inference only) ---
        if not self.training and self.config.normalize_race_predictions:
            for race_idx in range(batch_size):
                mask = attention_mask[race_idx].bool()
                if mask.sum() > 1:
                    predictions[race_idx][mask] = f.softmax(predictions[race_idx][mask], dim=0)

        if return_all_steps:
            return torch.stack(all_step_logits, dim=0)
        else:
            # return final_energy.unsqueeze(-1)
            return predictions.unsqueeze(-1)

    def compute_step(
        self, batch: tuple[dict[str, Tensor], Tensor, Tensor], apply_label_smoothing: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        x, y, attention_mask = batch

        all_step_predictions = self(x, attention_mask, return_all_steps=True)
        num_steps, batch_size, horse_len, _ = all_step_predictions.shape

        total_loss = 0
        final_probs = None
        y_masked = None

        # Compute loss for each MCMC step
        for step_idx in range(num_steps):
            step_predictions = all_step_predictions[step_idx].squeeze(-1)

            valid_mask = attention_mask.bool()
            step_pred_masked = step_predictions[valid_mask]
            y_step_masked = y[valid_mask]

            if self.config.label_smoothing and apply_label_smoothing:
                # Decreasing label smoothing: stronger at first step, weaker at final step
                smoothing_factor = 0.1 * (num_steps - step_idx) / num_steps
                y_step_masked = y_step_masked * (1 - smoothing_factor) + (1 - y_step_masked) * smoothing_factor

            # Compute step loss
            # step_loss = f.binary_cross_entropy_with_logits(step_pred_masked, y_step_masked, pos_weight=self.pos_weight)
            weights = self.pos_weight * y_step_masked + (1 - y_step_masked)
            step_loss = f.binary_cross_entropy(step_pred_masked, y_step_masked, weight=weights, reduction="mean")
            total_loss += step_loss

            if step_idx == num_steps - 1:
                # final_probs = torch.sigmoid(step_pred_masked)
                final_probs = step_pred_masked
                y_masked = y_step_masked

        avg_loss = total_loss / num_steps
        return avg_loss, final_probs, y_masked

    def to(self, device: torch.device):
        # Move all standard PyTorch stuff
        super().to(device)

        # Recursively move any raw tensor attributes
        for module in self.modules():
            for name, attr in module.__dict__.items():
                if isinstance(attr, torch.Tensor):
                    setattr(module, name, attr.to(device))

        return self
