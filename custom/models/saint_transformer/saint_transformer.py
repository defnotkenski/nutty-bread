# import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
import torch.nn as nn
from custom.layers.dual_attention_layer import DualAttentionLayer
from torchmetrics import Accuracy, F1Score, AUROC
import torch.nn.functional as f

# Import the actual components from pytorch-tabular
from pytorch_tabular.models.common.layers.embeddings import Embedding2dLayer
from pytorch_tabular.models.common.layers.transformers import AppendCLSToken


class SAINTTransformer(pl.LightningModule):
    def __init__(
        self,
        continuous_dims,
        categorical_dims: list,
        num_block_layers: int,
        d_model: int,
        num_heads: int,
        output_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.embedding_layer = Embedding2dLayer(
            continuous_dim=continuous_dims, categorical_cardinality=categorical_dims or [], embedding_dim=d_model
        )

        self.add_cls = AppendCLSToken(d_model, initialization="kaiming_uniform")

        # Transformer block creations
        self.transformer_blocks = nn.ModuleList(
            [DualAttentionLayer(d_model=d_model, num_heads=num_heads) for _ in range(num_block_layers)]
        )

        # Classification head
        self.output_layer = nn.Linear(d_model, output_size)

        self.learning_rate = learning_rate

        # Initialize torch metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary")

    def forward(self, x: dict[str, torch.Tensor]):
        # RACE AWARE: Squeeze out the batch dimension added by the dataloader
        x = {k: v.squeeze(0) for k, v in x.items()}

        # Step 1: Embeddings
        x = self.embedding_layer(x)  # Returns: [batch_size, num_features, d_model]

        # Step 2: Add class token
        x = self.add_cls(x)

        # Step 3: Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Step 4: Use class token for prediction
        cls_token = x[:, -1]

        # Step 5: Get logits before sigmoid for binary classification
        logits = self.output_layer(cls_token)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        y = y.squeeze(0)  # RACE AWARE: Remove batch dimension from targets
        y_hat = self(x)
        y_hat = y_hat.squeeze(-1)  # RACE AWARE: Remove the last dimension

        loss = f.binary_cross_entropy_with_logits(y_hat, y)

        # Calculate torch metrics
        probs = torch.sigmoid(y_hat)

        self.train_acc(probs, y.int())
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y = y.squeeze(0)  # RACE AWARE: Remove batch dimension from targets
        y_hat = self(x)
        y_hat = y_hat.squeeze(-1)  # RACE AWARE: Remove the last dimension

        loss = f.binary_cross_entropy_with_logits(y_hat, y)

        # Calculate torch metrics
        probs = torch.sigmoid(y_hat)

        self.val_acc(probs, y.int())
        self.val_auroc(probs, y.int())
        self.val_f1(probs, y.int())

        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_auroc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50)

        return [optimizer], [scheduler]
