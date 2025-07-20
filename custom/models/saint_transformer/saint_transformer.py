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

    # def training_step(self, batch, batch_idx):
    #     loss, probs, y_masked = self._compute_step(batch, apply_label_smoothing=True)
    #
    #     self.train_acc(probs, y_masked.int())
    #
    #     self.log("train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=False)
    #     self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
    #
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     _, probs, y_masked = self._compute_step(batch, apply_label_smoothing=False)
    #
    #     # Get the predictions based on probabilities
    #     preds = (probs > 0.5).float()
    #
    #     # Store results for later collection
    #     result = {"test_probs": probs, "test_preds": preds, "test_targets": y_masked}
    #     self.test_step_outputs.append(result)
    #
    #     return result

    # def on_test_epoch_end(self) -> None:
    #     # Lightning automatically passes collected outputs
    #     all_probs = torch.cat([x["test_probs"] for x in self.test_step_outputs])
    #     all_preds = torch.cat([x["test_preds"] for x in self.test_step_outputs])
    #     all_targets = torch.cat([x["test_targets"] for x in self.test_step_outputs])
    #
    #     # Convert to numpy for sklearn
    #     probs_np = all_probs.cpu().to(torch.float32).numpy()
    #     preds_np = all_preds.cpu().to(torch.float32).numpy()
    #     targets_np = all_targets.cpu().to(torch.float32).numpy()
    #
    #     # Calculate sklearn metrics
    #     accuracy = accuracy_score(targets_np, preds_np)
    #     auroc = roc_auc_score(targets_np, probs_np)
    #     f1 = f1_score(targets_np, preds_np)
    #     kappa = cohen_kappa_score(targets_np, preds_np)
    #     mcc = matthews_corrcoef(targets_np, preds_np)
    #     report = classification_report(targets_np, preds_np, target_names=["Fucked", "Not Fucked"])
    #
    #     self.test_metrics = {"accuracy": accuracy, "auroc": auroc, "f1": f1, "kappa": kappa, "mcc": mcc}
    #
    #     print(f"===== 🪿 Eval Results 🦖 =====")
    #     print(f"Test Accuracy: {accuracy:.4f}")
    #     print(f"Test AUROC: {auroc:.4f}")
    #     print(f"Test F1: {f1:.4f}")
    #     print(f"Test Kappa: {kappa:.4f}")
    #     print(f"Test MCC: {mcc:.4f}")
    #     print(f"Test Report:\n{report}")
    #
    #     self.test_step_outputs.clear()
    #
    #     return

    # def validation_step(self, batch, batch_idx):
    #     loss, probs, y_masked = self._compute_step(batch, apply_label_smoothing=False)
    #
    #     self.val_acc(probs, y_masked.int())
    #     self.val_auroc(probs, y_masked.int())
    #     self.val_f1(probs, y_masked.int())
    #
    #     self.log("val_loss", loss, on_epoch=True, prog_bar=True)
    #     self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
    #     self.log("val_auroc", self.val_auroc, on_epoch=True, prog_bar=False)
    #     self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=False)
    #
    #     return loss

    # def configure_optimizers(self):
    #     # Configure optimizers
    #     optimizer = ProdigyPlusScheduleFree(
    #         self.parameters(),
    #         lr=self.learning_rate,
    #         weight_decay=self.config.weight_decay,
    #         use_speed=self.config.prodigy_use_speed,
    #         use_orthograd=self.config.prodigy_use_orthograd,
    #         use_focus=self.config.prodigy_use_focus,
    #     )
    #
    #     # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.config.weight_decay)
    #
    #     # Configure schedulers
    #     # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     #     optimizer=optimizer, T_max=self.config.max_epochs, eta_min=1e-6
    #     # )
    #
    #     return optimizer  # If using a scheduler, need to return [optimizer], [scheduler] as a tuple

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
