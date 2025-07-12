import torch
import torch.nn as nn
import torch.nn.functional as fnn
from torch import Tensor


class LWTA(nn.Module):
    def __init__(self, in_features: int, num_competitors: int, temp: float):
        super().__init__()
        self.num_competitors = num_competitors
        self.temp = temp
        self.competitors = nn.ModuleList([nn.Linear(in_features, in_features) for _ in range(num_competitors)])

    def forward(self, x: Tensor):
        # x: (..., in_features) e.g., (batch, seq, d_model)
        activations = torch.stack([comp(x) for comp in self.competitors], dim=-1)  # (..., d_model, num_competitors)

        logits = activations.mean(dim=-2)  # (..., num_competitors) – average over d_model for competition scores

        if self.training:
            gates = fnn.gumbel_softmax(logits, tau=self.temp, hard=False, dim=-1)  # (..., num_competitors)
        else:
            gates = fnn.one_hot(logits.argmax(dim=-1), num_classes=self.num_competitors).float()  # (..., num_competitors)

        # Apply gates by expanding and multiplying, then sum over competitors
        gates_expanded = gates.unsqueeze(-2)  # (..., 1, num_competitors) to broadcast over d_model
        weighted_acts = activations * gates_expanded  # (..., d_model, num_competitors)
        output = weighted_acts.sum(dim=-1)  # (..., d_model) – sum winners' contributions

        return output
