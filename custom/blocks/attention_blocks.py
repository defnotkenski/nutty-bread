import torch.nn as nn
import torch.nn.functional as f
import torch
from torch.nn import Linear
import math


def compute_attention(x: torch.Tensor, q_proj: Linear, k_proj: Linear, v_proj: Linear, num_heads: int, dropout: nn.Dropout):
    """Computes attention made into a reusable block for inter and intra attention."""
    batch_size, num_features, d_model = x.shape
    heads_dim = d_model // num_heads

    # Create Q, K, V
    queries = q_proj(x)
    keys = k_proj(x)
    values = v_proj(x)

    # Reshape for multi-head processing
    def reshape_for_heads(tensor: torch.Tensor):
        reshaped = tensor.view(batch_size, num_features, num_heads, heads_dim)
        return reshaped.transpose(1, 2)

    q = reshape_for_heads(queries)
    k = reshape_for_heads(keys)
    v = reshape_for_heads(values)

    # Scaled dot-product attention
    attention_scores = torch.matmul(q, k.transpose(-2, -1))

    scale = math.sqrt(heads_dim)
    attention_scores = attention_scores / scale

    attention_weights = f.softmax(attention_scores, dim=-1)

    # Add attention dropout
    attention_weights = dropout(attention_weights)

    # Apply attention to values
    attended = torch.matmul(attention_weights, v)

    # Concatenate heads back together
    attended_transposed = attended.transpose(1, 2)
    attended_concat = attended_transposed.contiguous().view(batch_size, num_features, d_model)

    return attended_concat


class IntraRowAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        return compute_attention(
            x,
            q_proj=self.query_projection,
            k_proj=self.key_projection,
            v_proj=self.value_projection,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )


class InterRowAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        x_transposed = x.transpose(0, 1)

        attended = compute_attention(
            x=x_transposed,
            q_proj=self.query_projection,
            k_proj=self.key_projection,
            v_proj=self.value_projection,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

        return attended.transpose(0, 1)
