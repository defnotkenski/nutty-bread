import torch
import torch.nn as nn
import torch.nn.functional as f


class EnergyFunction(nn.Module):
    """
    Energy function that scores compatibility between horse features and predictions.
    Lower energy = better compatibility = more likely outcome.
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.feature_projection = nn.Linear(d_model, d_model)
        self.prediction_projection = nn.Linear(1, d_model)

        # Energy computation layers
        self.interaction_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.t_film = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.t_film[-1].weight.data.zero_()
        self.t_film[-1].bias.data.zero_()
        self.film_strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, candidate_predictions, step_idx: int | None = None):
        """
        Args:
            features: [batch_size, max_horses, d_model] or [V, B, H, D]
            candidate_predictions: [batch_size, max_horses, 1] - Candidate win probabilities
            step_idx: current MCMC step (int) or None

        Returns:
            energy_scores: [batch_size, max_horses] or [V, B, H] - Energy scores (lower = better)
        """

        # --- Project to same dimensional space ---
        proj_features = self.feature_projection(features)
        proj_predictions = self.prediction_projection(candidate_predictions)

        # --- FiLM from sinusoidal timestep embedding ---
        if step_idx is not None:
            d_model = proj_features.shape[-1]
            t_emb = self._sinusoidal_timestep_embeddings(int(step_idx), d_model, proj_features.device)

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

    @staticmethod
    def _sinusoidal_timestep_embeddings(step_idx: int, dim: int, device: torch.device, max_period: float = 10000.0):
        # sin/cos over log-spaced frequencies
        half = dim // 2
        if half == 0:
            return torch.zeros(dim, device=device)

        # Compute frequencies
        freq_exponents = torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
        inv_freq = torch.exp(-torch.log(torch.tensor(max_period, device=device)) * freq_exponents)

        # Phase
        phase = step_idx * inv_freq
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=0)

        if emb.shape[0] < dim:
            emb = f.pad(emb, (0, dim - emb.shape[0]))

        return emb
