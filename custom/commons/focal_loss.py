from torch import nn
import torch.nn.functional as f
import torch
import pandas


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        """
        alpha: weight for class 1. Higher alpha = more penalty for class 1 errors.
        gamma: focusing parameter. Higher gamma = more focus on hard examples.
        """

        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        ce = f.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")

        p_t = torch.sigmoid(logits)
        p_t = p_t * targets + (1 - p_t) * (1 - targets)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce

        return loss.mean() if self.reduction == "mean" else loss.sum()


def calculate_optimal_focal_loss_params(
    target_series: pandas.Series, alpha_multiplier: float = 1.0, gamma_boost: float = 0.0
) -> tuple[float, float]:
    # Calculate class distribution
    class_counts = target_series.value_counts().sort_index()
    total_samples = len(target_series)

    # Calculate class frequencies
    class_freqs = class_counts / total_samples

    # Calculate imbalance ratio
    minority_class_freq = min(class_freqs)
    imbalance_ratio = max(class_freqs) / minority_class_freq

    # Find which class is minority
    minority_class = class_freqs.idxmin()

    # Set alpha so minority class gets higher weight
    if minority_class == 1:
        # Class 1 is minority, so alpha should be high
        alpha = 1.0 - minority_class_freq
    else:
        # Class 0 is minority, so alpha should be low
        alpha = minority_class_freq

    # Calculate gamma based on imbalance severity
    if imbalance_ratio <= 2:
        gamma = 1.0
    elif imbalance_ratio <= 5:
        gamma = 2.0
    elif imbalance_ratio <= 10:
        gamma = 3.0
    else:
        gamma = 4.0

    # Apply tuning adjustments
    alpha = min(max(alpha * alpha_multiplier, 0.1), 0.9)  # Keep in reasonable range
    gamma = max(gamma + gamma_boost, 0.5)  # Minimum gamma of 0.5

    return alpha, gamma


def get_class_imbalance_info(target_series: pandas.Series) -> dict:
    """Get detailed class imbalance information."""
    class_counts = target_series.value_counts().sort_index()
    total_samples = len(target_series)
    class_freqs = class_counts / total_samples

    imbalance_ratio = max(class_freqs) / min(class_freqs)

    return {
        "class_counts": class_counts.to_dict(),
        "class_frequencies": class_freqs.to_dict(),
        "imbalance_ratio": imbalance_ratio,
        "total_samples": total_samples,
    }
