import torch
import torch.nn as nn


class EnergyFunction(nn.Module):
    """
    Energy function that scores compatibility between horse features and predictions.
    Lower energy = better compatibility = more likely outcome.
    """

    def __init__(self, d_model: int, num_step_bins: int):
        super().__init__()

        self.feature_projection = nn.Linear(d_model, d_model)
        self.prediction_projection = nn.Linear(1, d_model)
        self.num_step_bins = num_step_bins

        # Energy computation layers
        self.interaction_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.t_film = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.t_film[-1].weight.data.zero_()
        self.t_film[-1].bias.data.zero_()

        self.step_embed = nn.Embedding(self.num_step_bins, d_model)  # Max 8 MCMC steps
        self.film_strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, candidate_predictions, step_idx: tuple[int, int] | None = None):
        """
        Args:
            features: [batch_size, max_horses, d_model] or [V, B, H, D]
            candidate_predictions: [batch_size, max_horses, 1] - Candidate win probabilities
            step_idx: current MCMC step (current_step, total_steps) or None

        Returns:
            energy_scores: [batch_size, max_horses] or [V, B, H] - Energy scores (lower = better)
        """

        # --- Project to same dimensional space ---
        proj_features = self.feature_projection(features)
        proj_predictions = self.prediction_projection(candidate_predictions)

        # --- FiLM timestep embedding ---
        if step_idx is not None:
            current_step, total_steps = step_idx
            normalized_step = current_step / max(total_steps - 1, 1)
            embedded_step = int(normalized_step * (self.num_step_bins - 1))
            embedded_step = max(0, min(self.num_step_bins - 1, embedded_step))

            t_emb = self.step_embed(torch.tensor(embedded_step, device=proj_features.device, dtype=torch.long))

            gamma, beta = self.t_film(t_emb).chunk(2, dim=-1)
            gamma, beta = torch.tanh(gamma), torch.tanh(beta)

            s = torch.sigmoid(self.film_strength) * 0.5 if hasattr(self, "film_strength") else 0.2

            # Broadcast to proj_features rank
            while gamma.dim() < proj_features.dim():
                gamma = gamma.unsqueeze(0)
                beta = beta.unsqueeze(0)

            gamma = gamma.expand(*proj_features.shape[:-1], -1)
            beta = beta.expand(*proj_features.shape[:-1], -1)

            proj_features = proj_features * (1 + s * gamma) + s * beta

        # --- Multiplicative interaction ---
        interaction = proj_features * proj_predictions

        # --- Compute energy scores ---
        energy_scores = self.interaction_layer(interaction).squeeze(-1)

        return energy_scores
