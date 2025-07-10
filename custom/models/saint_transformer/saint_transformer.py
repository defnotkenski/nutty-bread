import lightning.pytorch as pl
import torch
import torch.nn as nn
from custom.layers.dual_attention_layer import DualAttentionLayer
from torchmetrics import Accuracy, F1Score, AUROC
import torch.nn.functional as f
from custom.models.saint_transformer.config import SAINTConfig
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
)

# from prodigyopt import Prodigy
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

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
        config: SAINTConfig,
    ):
        super().__init__()

        # self.save_hyperparameters()
        self.config = config

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

        self.test_step_outputs = []
        self.test_metrics = None

    def forward(self, x: dict[str, torch.Tensor]):
        # Squeeze out the batch dimension added by the dataloader
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

        y = y.squeeze(0)  # Remove batch dimension from targets
        y_hat = self(x)
        y_hat = y_hat.squeeze(-1)  # Remove the last dimension

        loss = f.binary_cross_entropy_with_logits(y_hat, y)

        # Calculate torch metrics
        probs = torch.sigmoid(y_hat)

        self.train_acc(probs, y.int())

        self.log("train_acc", self.train_acc, on_step=True, prog_bar=False)
        self.log("train_loss", loss, on_step=True, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y = y.squeeze(0)
        y_hat = self(x)
        y_hat = y_hat.squeeze(-1)

        # Calculate predictions
        probs = torch.sigmoid(y_hat)
        preds = (probs > 0.5).float()

        # Store results for later collection
        result = {"test_probs": probs, "test_preds": preds, "test_targets": y}
        self.test_step_outputs.append(result)

        return result

    def on_test_epoch_end(self) -> None:
        # Lightning automatically passes collected outputs
        all_probs = torch.cat([x["test_probs"] for x in self.test_step_outputs])
        all_preds = torch.cat([x["test_preds"] for x in self.test_step_outputs])
        all_targets = torch.cat([x["test_targets"] for x in self.test_step_outputs])

        # Convert to numpy for sklearn
        probs_np = all_probs.cpu().numpy()
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()

        # Calculate sklearn metrics
        accuracy = accuracy_score(targets_np, preds_np)
        auroc = roc_auc_score(targets_np, probs_np)
        f1 = f1_score(targets_np, preds_np)
        kappa = cohen_kappa_score(targets_np, preds_np)
        mcc = matthews_corrcoef(targets_np, preds_np)
        report = classification_report(targets_np, preds_np, target_names=["Fucked", "Not Fucked"])

        self.test_metrics = {"accuracy": accuracy, "auroc": auroc, "f1": f1, "kappa": kappa, "mcc": mcc}

        print(f"===== 🪿 Eval Results 🦖 =====")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUROC: {auroc:.4f}")
        print(f"Test F1: {f1:.4f}")
        print(f"Test Kappa: {kappa:.4f}")
        print(f"Test MCC: {mcc:.4f}")
        print(f"Test Report:\n{report}")

        self.test_step_outputs.clear()

        return

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

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_auroc, on_epoch=True, prog_bar=False)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        # Configure optimizers
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.05)
        # optimizer = Prodigy(self.parameters(), lr=self.learning_rate, weight_decay=0.05, d_coef=1.0)
        optimizer = ProdigyPlusScheduleFree(
            self.parameters(),
            lr=self.learning_rate,
            use_speed=self.config.prodigy_use_speed,
            use_orthograd=self.config.prodigy_use_orthograd,
            use_focus=self.config.prodigy_use_focus,
        )

        # Configure schedulers
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10)

        return optimizer  # If using a scheduler, need to return [optimizer], [scheduler] as a tuple
