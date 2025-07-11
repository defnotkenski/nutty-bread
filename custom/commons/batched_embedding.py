import torch.nn as nn
import torch
from pytorch_tabular.models.common.layers.embeddings import Embedding2dLayer


class BatchedEmbedding(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinality, embedding_dim):
        super().__init__()

        self.embedding_layer = Embedding2dLayer(
            continuous_dim=continuous_dim, categorical_cardinality=categorical_cardinality, embedding_dim=embedding_dim
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        # Handle input dict (continuous/categorical)
        assert isinstance(x, dict), f"Input in the forward method must be a dict of continuous and categorical keys."

        batch_size = next(iter(x.values())).shape[0]  # Get batch size from first tensor
        seq_len = next(iter(x.values())).shape[1]  # Get sequence length

        # Flatten batch and sequence dimensions. Converts (batch_size, seq_len, features) -> (batch_size * seq_len, features)
        x_flat = {k: v.view(-1, v.shape[-1]) for k, v in x.items()}

        # Apply embedding
        embedded: torch.Tensor = self.embedding_layer(x_flat)  # (batch*seq, features, d_model)

        # Reshape back to batched 4D format. Converts (batch_size * seq_len, features, d_model) → (batch_size, seq_len, features, d_model)
        reshaped = embedded.view(batch_size, seq_len, *embedded.shape[1:])

        return reshaped
