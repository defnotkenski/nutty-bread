import torch.nn as nn
import torch


class StableMax(nn.Module):
    """
    StableMax is an alternative to Softmax that helps mitigate numerical
    instabilities. It applies an elementwise transform s(x) and normalizes
    over the specified dimension to produce probabilities that sum to 1.

    s(x) = (x + 1) if x >= 0, or 1 / (1 - x) if x < 0.
    logits are clamped to a minimum value (clamp_min) to avoid extreme negatives.
    """

    def __init__(self, dim: int = -1, clamp_min: float = -10.0, clamp_max: float = 10.0):
        super().__init__()
        self.dim = dim
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Clamp extreme negative logits
        logits = torch.clamp(logits, min=self.clamp_min, max=self.clamp_max)
        s_logits = torch.where(logits >= 0, logits + 1, 1 / (1 - logits))
        s_sum = s_logits.sum(dim=self.dim, keepdim=True)

        return s_logits / (s_sum + 1e-9)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return gate * self.silu(up)
