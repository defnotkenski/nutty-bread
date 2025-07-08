import torch
from torch.utils.data import DataLoader

from custom.data_loader import preprocess_df, CustomTabularDataset
from custom.model import MyFirstTransformer

from pathlib import Path
from sklearn.model_selection import train_test_split
import pytorch_lightning as pylightning


def train_model(path_to_csv: Path, perform_eval: bool) -> None:
    # Test the transformer model
    print("Testing FT-Transformer structure...")

    preprocessed = preprocess_df(df_path=path_to_csv)
    dataset = CustomTabularDataset(preprocessed)

    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.2, shuffle=False, random_state=42)
    validate_idx, eval_idx = train_test_split(temp_idx, test_size=0.5, shuffle=False, random_state=42)

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, validate_idx)
    eval_dataset = torch.utils.data.Subset(dataset, eval_idx)

    # Create dataloaders
    batch_size = 1
    num_workers = 7

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
    my_first_model = MyFirstTransformer(
        continuous_dims=preprocessed.continuous_tensor.shape[1],
        categorical_dims=preprocessed.categorical_cardinalities,
        learning_rate=0.0001,
        num_block_layers=6,
        d_model=64,
        num_heads=8,
        output_size=1,
    )

    trainer = pylightning.Trainer(
        gradient_clip_val=1.0,
        max_epochs=10,
        accelerator="auto",
        devices=1,
        val_check_interval=1.0,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    trainer.fit(my_first_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # Run evaluations on trained model
    if perform_eval:
        all_eval_probabilities = []
        all_eval_predictions = []

        with torch.no_grad():
            for eval_batch in eval_dataloader:
                eval_x, eval_y = eval_batch

                # Get predictions as raw logits
                raw_logits: torch.Tensor = my_first_model(eval_x)

                # Run sigmoid to get probabilities from raw logits
                probabilities = torch.sigmoid(raw_logits)

                # Get probabailities as numpy array and add to list
                all_eval_probabilities.extend(probabilities.squeeze().cpu().numpy())

                # Get predictions as numpy array and add to list
                all_eval_predictions.extend((probabilities.squeeze() > 0.5).cpu().numpy())

            # Print results
            print(f"\n===== Eval results ======\n")

            for i, (prob, pred_class) in enumerate(zip(all_eval_probabilities, all_eval_predictions)):
                print(f"Sample {i}: Probability = {prob}, Prediction = {pred_class}")

    # If it reaches this point, thank fuckin god
    print(f"Output shape: {raw_logits.shape}")
    print("✅ FT-Transformer structure works!")

    return


if __name__ == "__main__":
    dataset_path = Path.cwd().parent / "datasets" / "sample_horses.csv"
    train_model(path_to_csv=dataset_path, perform_eval=True)
