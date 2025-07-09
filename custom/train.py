import lightning.pytorch as pylightning
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
import torch
from torch.utils.data import DataLoader
from custom.models.saint_transformer.data_processing import preprocess_df, SAINTDataset
from custom.models.saint_transformer.saint_transformer import SAINTTransformer
from pathlib import Path
from sklearn.model_selection import train_test_split
import modal

neptune_logger = NeptuneLogger(
    project="toastbutter/diabeticdonkey",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4OTY0NjY1My00ZmViLTQxYzctOWIzNi1mNDJlNDdiNDk2NjIifQ==",
)


modal_app = modal.App("neural-learning")
modal_img = (
    modal.Image.debian_slim(python_version="3.12.2")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir(Path.cwd() / "custom", remote_path="/root/custom")
    .add_local_dir(Path.cwd() / "datasets", remote_path="/root/datasets")
)
# modal_gpu = "A100-40GB"
modal_gpu = "H100"

torch.set_float32_matmul_precision("medium")


class CustomCallback(Callback):
    def on_train_epoch_start(self, trainer: pylightning.Trainer, pl_module: pylightning.LightningModule) -> None:
        epoch = trainer.current_epoch
        print(f"Starting epoch: {epoch}")

        return


def train_model(path_to_csv: Path, perform_eval: bool, quiet_mode: bool) -> None:
    # Test the transformer model
    print("Testing FT-Transformer structure...")

    preprocessed = preprocess_df(df_path=path_to_csv)
    dataset = SAINTDataset(preprocessed)

    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.2, shuffle=False, random_state=42)
    validate_idx, eval_idx = train_test_split(temp_idx, test_size=0.5, shuffle=False, random_state=42)

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, validate_idx)
    eval_dataset = torch.utils.data.Subset(dataset, eval_idx)

    # Check target distribution in validation set
    _val_targets = [dataset[i][1] for i in validate_idx[:10]]  # Check first 10

    # Create dataloaders
    batch_size = 1  # DO NOT CHANGE THE BATCH SIZE HOE.
    num_workers = 6

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create test tabular data
    # batch_size = 2
    # num_continuous_features = 90

    # Categorical features: city (vocab size 5), gender (vocab size 2)
    # categorical_cardinalities = [5, 2]

    # Simulate tabular data
    # test_data = {
    #     "continuous": torch.randn(batch_size, num_continuous_features),
    #     "categorical": torch.tensor([[2, 1], [0, 0]]),
    # }

    # Create tabular data
    # test_data = {
    #     "continuous": preprocessed.continuous_tensor,
    #     "categorical": preprocessed.categorical_tensor,
    # }

    # Create model
    saint_model = SAINTTransformer(
        continuous_dims=preprocessed.continuous_tensor.shape[1],
        categorical_dims=preprocessed.categorical_cardinalities,
        learning_rate=0.0001,
        num_block_layers=4,
        d_model=64,
        num_heads=4,
        output_size=1,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="saint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        verbose=True,
    )

    callbacks_list: list = [checkpoint_callback]
    if quiet_mode:
        callbacks_list.append(CustomCallback())

    trainer = pylightning.Trainer(
        accumulate_grad_batches=8,
        gradient_clip_val=1.0,
        max_epochs=30,
        accelerator="auto",
        devices=1,
        val_check_interval=1.0,
        enable_checkpointing=True,
        logger=neptune_logger,
        enable_progress_bar=not quiet_mode,
        callbacks=callbacks_list,
    )

    trainer.fit(saint_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # Load the best checkpoint for evaluation
    best_model_path = checkpoint_callback.best_model_path

    print(f"Loading best model from: {best_model_path} for evaluation")
    saint_eval_model = SAINTTransformer.load_from_checkpoint(best_model_path)

    # Run evaluations on trained model
    if perform_eval:
        all_eval_probabilities = []
        all_eval_predictions = []
        all_eval_targets = []

        with torch.no_grad():
            for eval_batch in eval_dataloader:
                eval_x, eval_y = eval_batch

                # Get predictions as raw logits
                raw_logits: torch.Tensor = saint_eval_model(eval_x)

                # Run sigmoid to get probabilities from raw logits
                probabilities = torch.sigmoid(raw_logits)

                # Collect probabilities, predictions, and TRUE TARGETS
                all_eval_probabilities.extend(probabilities.squeeze().cpu().numpy())
                all_eval_predictions.extend((probabilities.squeeze() > 0.5).cpu().numpy())
                all_eval_targets.extend(eval_y.squeeze().cpu().numpy())

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            roc_auc_score,
            f1_score,
            classification_report,
            cohen_kappa_score,
            matthews_corrcoef,
        )
        import numpy as np

        eval_targets = np.array(all_eval_targets)
        eval_probs = np.array(all_eval_probabilities)
        eval_preds = np.array(all_eval_predictions)

        # Calculate final metrics
        eval_accuracy = accuracy_score(eval_targets, eval_preds)
        eval_auroc = roc_auc_score(eval_targets, eval_probs)
        eval_f1 = f1_score(eval_targets, eval_preds)
        eval_kappa = cohen_kappa_score(eval_targets, eval_preds)
        eval_mcc = matthews_corrcoef(eval_targets, eval_preds)

        # Print results
        print(f"\n===== Final Eval Results ======")
        print(f"Eval Accuracy: {eval_accuracy:.4f}")
        print(f"Eval AUROC: {eval_auroc:.4f}")
        print(f"Eval F1: {eval_f1:.4f}")
        print(f"Eval Cohen's Kappa: {eval_kappa:.4f}")
        print(f"Eval Matthews Correlation: {eval_mcc:.4f}")

        print(f"\n===== Classification Report ======")
        print(classification_report(eval_targets, eval_preds, target_names=["No Place", "Place"]))

    # If it reaches this point, thank fuckin god
    print(f"Output shape: {raw_logits.shape}")
    print("✅ FT-Transformer structure works!")

    return


@modal_app.function(gpu=modal_gpu, image=modal_img, timeout=3600)
def run_with_modal() -> None:
    has_cuda = torch.cuda.is_available()
    print(f"CUDA status: {has_cuda}")

    modal.interact()

    modal_dataset_path = Path.cwd() / "datasets" / "sample_horses.csv"
    train_model(path_to_csv=modal_dataset_path, perform_eval=True, quiet_mode=True)

    return


@modal_app.local_entrypoint()
def main() -> None:
    run_with_modal.remote()
    return


if __name__ == "__main__":
    pass

    dataset_path = Path.cwd().parent / "datasets" / "sample_horses.csv"
    train_model(path_to_csv=dataset_path, perform_eval=True, quiet_mode=False)
