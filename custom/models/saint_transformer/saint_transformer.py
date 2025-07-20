import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as f
from custom.layers.dual_attention_layer import DualAttentionLayer
from custom.models.saint_transformer.config import SAINTConfig
from custom.commons.batched_embedding import BatchedEmbedding
from custom.blocks.attention_pooling_blocks import AttentionPooling


class SAINTTransformer(nn.Module):
    def __init__(
        self,
        continuous_dims,
        categorical_dims: list,
        num_block_layers: int,
        d_model: int,
        num_heads: int,
        output_size: int,
        learning_rate: float,
        pos_weight: float,
        config: SAINTConfig,
    ):
        super().__init__()

        # === Configuration ===
        self.config = config

        # === Training Parameters ===
        self.learning_rate = learning_rate
        self.pos_weight = torch.tensor(pos_weight)

        # === Model Architecture ===
        self.embedding_layer = BatchedEmbedding(
            continuous_dim=continuous_dims, categorical_cardinality=categorical_dims or [], embedding_dim=d_model
        )

        total_features = continuous_dims + len(categorical_dims)

        self.race_cls_token = nn.Parameter(torch.randn(1, total_features, d_model) * 0.02)
        self.race_projection = nn.Linear(d_model * 2, d_model)

        self.transformer_blocks = nn.ModuleList(
            [
                DualAttentionLayer(
                    d_model=d_model, num_heads=num_heads, num_competitors=config.num_competitors, config=config
                )
                for _ in range(num_block_layers)
            ]
        )

        self.pooler = AttentionPooling(d_model)
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x: dict[str, torch.Tensor], attention_mask: torch.Tensor):
        # Step 1: Embeddings
        x: torch.Tensor = self.embedding_layer(x)  # Returns: [batch_size, horses_len, num_features, d_model]

        batch_size, horse_len, num_features, d_model = x.shape

        assert (
            num_features == self.race_cls_token.shape[1]
        ), f"Feature mismatch: {num_features} vs {self.race_cls_token.shape[1]}"

        race_outputs = []

        # Process each race seperately to maintain race boundaries
        for race_idx in range(batch_size):
            race_mask = attention_mask[race_idx]  # (max_horses,)

            race_x = x[race_idx]  # (max_horses, num_features, d_model)

            # Add class token
            race_cls_token = self.race_cls_token.expand(1, num_features, d_model)
            race_x_with_cls = torch.cat([race_x, race_cls_token], dim=0)  # (max_horses + 1, num_features, d_model)

            # Create mask for class token
            cls_mask = torch.ones(1, device=race_mask.device)
            full_mask = torch.cat([race_mask, cls_mask], dim=0)  # (max_horses + 1,)

            assert (
                full_mask.sum() > 0
            ), f"Race {race_idx}: All positions masked - race_mask sum: {race_mask.sum()}, cls_mask sum: {cls_mask.sum()}"

            # Pass through transformer blocks
            for block in self.transformer_blocks:
                race_x_with_cls = block(race_x_with_cls, full_mask)

            # Extract class token
            race_cls = race_x_with_cls[-1]  # class token
            horse_representations = race_x_with_cls[:-1]  # All horses (including padding)

            # Apply mask when computing features (only use real horses)
            num_real_horses = int(race_mask.sum())
            horse_reps_real = horse_representations[:num_real_horses]

            # horse_features = horse_representations[:num_real_horses].mean(dim=1)
            horse_features: Tensor = self.pooler(horse_reps_real)

            # race_context = race_cls.mean(dim=0).unsqueeze(0).expand(num_real_horses, -1)
            race_context: Tensor = self.pooler(race_cls.unsqueeze(0)).squeeze(0)

            # Combine horse features with race context
            race_context_expanded = race_context.unsqueeze(0).expand(num_real_horses, -1)

            combined = torch.cat([horse_features, race_context_expanded], dim=-1)

            cls_tokens = self.race_projection(combined)

            padded_cls_tokens = torch.zeros(horse_len, d_model, device=cls_tokens.device)
            padded_cls_tokens[:num_real_horses] = cls_tokens

            race_outputs.append(padded_cls_tokens)

        cls_tokens = torch.stack(race_outputs)

        # Get logits before sigmoid for binary classification
        logits = self.output_layer(cls_tokens)
        return logits

    def compute_step(
        self, batch: tuple[dict[str, Tensor], Tensor, Tensor], apply_label_smoothing: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        x, y, attention_mask = batch

        y_predict: Tensor = self(x, attention_mask)
        y_predict = y_predict.squeeze(-1)

        # Apply attention mask for loss computation
        valid_mask = attention_mask.bool()
        y_predict_masked = y_predict[valid_mask]
        y_masked = y[valid_mask]

        # Label smoothing
        if self.config.label_smoothing and apply_label_smoothing:
            y_masked = y_masked * 0.9 + (1 - y_masked) * 0.1

        # Compute loss
        loss = f.binary_cross_entropy_with_logits(y_predict_masked, y_masked, pos_weight=self.pos_weight)

        # Compute probabilities
        probs = torch.sigmoid(y_predict_masked)

        return loss, probs, y_masked

    def to(self, device: torch.device):
        # Move all standard PyTorch stuff
        super().to(device)

        # Recursively move any raw tensor attributes
        for module in self.modules():
            for name, attr in module.__dict__.items():
                if isinstance(attr, torch.Tensor):
                    setattr(module, name, attr.to(device))

        return self
