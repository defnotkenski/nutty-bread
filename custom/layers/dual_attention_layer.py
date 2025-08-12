import torch.nn as nn
import torch
from custom.blocks.attention_blocks import IntraRowAttention, InterRowAttention
from custom.blocks.lwta_blocks import LWTA
from custom.models.saddle.config import SAINTConfig
from custom.blocks.activation_blocks import SwiGLU


class DualAttentionLayer(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, num_competitors: int = 4, config: SAINTConfig = None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % (num_heads // 2) == 0, f"d_model must be divisible by heads per attention"

        self.intra_attention = IntraRowAttention(d_model=d_model, num_heads=num_heads // 2, dropout=config.dropout)
        self.inter_attention = InterRowAttention(d_model=d_model, num_heads=num_heads // 2, dropout=config.dropout)

        # self.layer_norm_1 = nn.LayerNorm(d_model)
        # self.layer_norm_2 = nn.LayerNorm(d_model)

        self.layer_norm_1 = nn.RMSNorm(d_model)  # Experimenting with Transformer++
        self.layer_norm_2 = nn.RMSNorm(d_model)  # Experimenting with Transformer++

        self.lwta = LWTA(d_model * 4, num_competitors=num_competitors, temp=1.0)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 8, bias=False),  # Change to d_model * 4 if not using SwiGLU based activations
            SwiGLU(),  # Experimenting with Transformer++
            self.lwta,
            nn.Linear(d_model * 4, d_model, bias=False),
        )

        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(d_model * 2, d_model)

    def forward(self, x, attention_mask=None):
        # Dual attention blocks
        intra_out = self.intra_attention(x)
        inter_out = self.inter_attention(x, attention_mask)

        # Combine dual attention and project
        combined = torch.cat([intra_out, inter_out], dim=-1)
        attended = self.output_projection(combined)
        attended = self.dropout(attended)

        norm1 = self.layer_norm_1(x + attended)

        # Feed forward with residual connection
        ff_output = self.feed_forward(norm1)
        ff_output = self.dropout(ff_output)

        norm2 = self.layer_norm_2(norm1 + ff_output)

        return norm2
