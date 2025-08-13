import torch.nn.functional as f
import torch.nn as nn
import torch
from torch.nn import Linear
import math
from torch import Tensor


def compute_attention(x: Tensor, q_proj: Linear, k_proj: Linear, v_proj: Linear, num_heads: int, dropout: nn.Dropout):
    """Computes attention made into a reusable block for inter and intra attention."""
    horse_len, num_features, d_model = x.shape
    heads_dim = d_model // num_heads

    # Create Q, K, V
    queries = q_proj(x)
    keys = k_proj(x)
    values = v_proj(x)

    # Reshape for multi-head processing
    def reshape_for_heads(tensor: Tensor):
        reshaped = tensor.view(horse_len, num_features, num_heads, heads_dim)
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
    attended_concat = attended_transposed.contiguous().view(horse_len, num_features, d_model)

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

    def forward(self, x: Tensor):
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

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        horse_len, num_features, d_model = x.shape

        # Apply projections to original tensor
        q: Tensor = self.query_projection(x)  # (horse_len, num_features, d_model)
        k: Tensor = self.key_projection(x)  # (horse_len, num_features, d_model)
        v: Tensor = self.value_projection(x)  # (horse_len, num_features, d_model)

        # Flatten each horse's features for intersample attention
        flattened_dim = num_features * d_model
        q_flat = q.view(horse_len, num_features * d_model)  # (horse_len, num_features * d_model)
        k_flat = k.view(horse_len, num_features * d_model)  # (horse_len, num_features * d_model)
        v_flat = v.view(horse_len, num_features * d_model)  # (horse_len, num_features * d_model)

        # Multi-head attention setup
        head_dim = flattened_dim // self.num_heads
        assert flattened_dim % self.num_heads == 0, "flattened_dim must be divisible by num_heads"

        # Reshape for multi-head attention
        q_heads = q_flat.view(horse_len, self.num_heads, head_dim).transpose(0, 1)  # (num_heads, horse_len, head_dim)
        k_heads = k_flat.view(horse_len, self.num_heads, head_dim).transpose(0, 1)  # (num_heads, horse_len, head_dim)
        v_heads = v_flat.view(horse_len, self.num_heads, head_dim).transpose(0, 1)  # (num_heads, horse_len, head_dim)

        # Compute attention for each head across horses (intersample attention)
        attention_scores: Tensor = torch.matmul(q_heads, k_heads.transpose(-2, -1))
        scale = math.sqrt(head_dim)
        attention_scores = attention_scores / scale

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 1:
                # Attention mask shape: (horse_len,) where 1=real, 0=padded
                # Legacy: Create 2D mask: (horse_len, horse_len)
                mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(0)
            else:
                # Vectorized: Use pre-built 2D block-diagonal mask (total_seq, total_seq)
                mask_2d = attention_mask

            # Expand for multihead: (num_heads, horse_len, horse_len)
            mask_expanded = mask_2d.unsqueeze(0).expand(self.num_heads, -1, -1)

            attention_scores = attention_scores.masked_fill(mask_expanded == 0, -1e9)

        attention_weights = f.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended_heads = torch.matmul(attention_weights, v_heads)

        # Concatenate heads
        attended = (
            attended_heads.transpose(0, 1).contiguous().view(horse_len, flattened_dim)
        )  # (horse_len, num_features * d_model)

        # Reshape back to original format
        attended = attended.view(horse_len, num_features, d_model)

        return attended
