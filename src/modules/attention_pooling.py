import math
import torch
import torch.nn as nn
import torch.nn.functional as f


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)  # Small init

    def forward(self, x: torch.Tensor):  # x: (entities, num_features, d_model), entities can be horses or 1 for CLS
        attn_scores = torch.matmul(x, self.query.unsqueeze(-1)).squeeze(-1) / math.sqrt(x.shape[-1])  # (entities, features)
        attn_weights = f.softmax(attn_scores, dim=-1)
        pooled = torch.einsum("ef,efd->ed", attn_weights, x)  # (entities, d_model)
        return pooled
