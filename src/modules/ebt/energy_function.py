import torch.nn as nn


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

    def forward(self, features, candidate_predictions):
        """
        Args:
            features: [batch_size, max_horses, d_model] or [V, B, H, D]
            candidate_predictions: [batch_size, max_horses, 1] - Candidate win probabilities

        Returns:
            energy_scores: [batch_size, max_horses] or [V, B, H] - Energy scores (lower = better)
        """

        # --- Project to same dimensional space ---
        proj_features = self.feature_projection(features)
        proj_predictions = self.prediction_projection(candidate_predictions)

        # --- Multiplicative interaction ---
        interaction = proj_features * proj_predictions

        # --- Compute energy scores ---
        energy_scores = self.interaction_layer(interaction).squeeze(-1)

        return energy_scores
