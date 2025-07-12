import lightning.pytorch as pylightning
import pandas as pd
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
import torch
from torch.utils.data import DataLoader
from custom.models.saint_transformer.data_processing import preprocess_df, SAINTDataset
from custom.models.saint_transformer.saint_transformer import SAINTTransformer
from pathlib import Path
from sklearn.model_selection import train_test_split
import modal
from custom.models.saint_transformer.config import SAINTConfig
from dataclasses import asdict
from custom.models.saint_transformer.data_processing import collate_races


modal_app = modal.App("neural-learning")
modal_img = (
    modal.Image.debian_slim(python_version="3.12.2")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir(Path.cwd() / "custom", remote_path="/root/custom")
    .add_local_dir(Path.cwd() / "datasets", remote_path="/root/datasets")
)
modal_gpu = "A100-40GB"
# modal_gpu = "H100"

torch.set_float32_matmul_precision("medium")


class CustomCallback(Callback):
    def on_train_epoch_start(self, trainer: pylightning.Trainer, pl_module: pylightning.LightningModule) -> None:
        epoch = trainer.current_epoch
        print(f"Starting epoch: {epoch}")

        return


def train_model(path_to_csv: Path, perform_eval: bool, quiet_mode: bool, enable_logging: bool) -> None:
    print("Testing SAINT-Heavy transformer infrastructure...")

    config = SAINTConfig()

    preprocessed = preprocess_df(df_path=path_to_csv)
    dataset = SAINTDataset(preprocessed)

    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.2, shuffle=config.shuffle, random_state=42)
    validate_idx, eval_idx = train_test_split(temp_idx, test_size=0.5, shuffle=config.shuffle, random_state=42)

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, validate_idx)
    eval_dataset = torch.utils.data.Subset(dataset, eval_idx)

    # Check target distribution in validation set
    _val_targets = [dataset[i][1] for i in validate_idx[:10]]  # Check first 10

    # Create dataloaders
    batch_size = config.batch_size  # DO NOT CHANGE THE BATCH SIZE HOE.
    num_workers = config.num_workers
    pin_memory = config.pin_memory
    shuffle = config.shuffle

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_races,
    )
    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_races,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_races,
    )

    # Calculate pos_weight for weighted BCE loss fn
    target_series = pd.Series(preprocessed.target_tensor.numpy().flatten())
    pos_count = target_series.sum()
    neg_count = len(target_series) - pos_count
    pos_weight = neg_count / pos_count

    # Create model
    saint_model = SAINTTransformer(
        continuous_dims=preprocessed.continuous_tensor.shape[1],
        categorical_dims=preprocessed.categorical_cardinalities,
        learning_rate=config.learning_rate,
        num_block_layers=config.num_block_layers,
        d_model=config.d_model,
        num_heads=config.num_attention_heads,
        output_size=config.output_size,
        pos_weight=pos_weight,
        config=config,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints/",
        filename="saint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        verbose=False,
    )

    # Configure logging
    if enable_logging:
        neptune_logger = NeptuneLogger(
            project="toastbutter/diabeticdonkey",
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4OTY0NjY1My00ZmViLTQxYzctOWIzNi1mNDJlNDdiNDk2NjIifQ==",
        )

        config_dict = asdict(config)

        neptune_logger.log_hyperparams(
            {
                "optimizer": "prodigy-plus",
                "attention_activation": "softmax",
                **config_dict,
            }
        )
    else:
        neptune_logger = False

    callbacks_list: list = [checkpoint_callback]
    if quiet_mode:
        callbacks_list.append(CustomCallback())

    # Configure trainer and begin training
    trainer = pylightning.Trainer(
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        val_check_interval=config.val_check_interval,
        enable_checkpointing=config.enable_checkpointing,
        logger=neptune_logger if enable_logging else False,
        enable_progress_bar=not quiet_mode,
        callbacks=callbacks_list,
    )

    trainer.fit(saint_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # Perform evaluations after model training
    if perform_eval:
        # Load the best checkpoint for evaluation
        best_model_path = checkpoint_callback.best_model_path

        print(f"Loading best model from: {best_model_path} for evaluation")
        saint_eval_model = SAINTTransformer.load_from_checkpoint(best_model_path)

        # Run evaluations on trained model
        trainer.test(saint_eval_model, eval_dataloader)

    # If it reaches this point, thank fuckin god
    print("✅ FT-Transformer structure works!")

    return


@modal_app.function(gpu=modal_gpu, image=modal_img, timeout=10800, cpu=2)
def run_with_modal() -> None:
    has_cuda = torch.cuda.is_available()
    print(f"CUDA status: {has_cuda}")

    modal.interact()

    modal_dataset_path = Path.cwd() / "datasets" / "sample_horses_v2.csv"
    train_model(path_to_csv=modal_dataset_path, perform_eval=True, quiet_mode=False, enable_logging=True)

    return


@modal_app.local_entrypoint()
def main() -> None:
    run_with_modal.remote()
    return


if __name__ == "__main__":
    pass

    dataset_path = Path.cwd().parent / "datasets" / "sample_horses_v2.csv"
    train_model(path_to_csv=dataset_path, perform_eval=True, quiet_mode=False, enable_logging=False)
