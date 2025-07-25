import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from custom.models.saint_transformer.data_processing import preprocess_df, SAINTDataset
from custom.models.saint_transformer.saint_transformer import SAINTTransformer
from pathlib import Path
from sklearn.model_selection import train_test_split
from custom.models.saint_transformer.config import SAINTConfig
from custom.models.saint_transformer.data_processing import collate_races
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from prodigyplus import ProdigyPlusScheduleFree
import neptune


def configure_logger(config: SAINTConfig) -> neptune.Run:
    run = neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )
    run["config"] = config.__dict__
    return run


def calc_pos_weight(preprocessed):
    # Calculate pos_weight for weighted BCE loss fn
    target_series = pd.Series(preprocessed.target_tensor.numpy().flatten())
    pos_count = target_series.sum()
    neg_count = len(target_series) - pos_count
    pos_weight = neg_count / pos_count

    return pos_weight


def prepare_data(path_to_csv: Path, config: SAINTConfig):
    print("--- Data Preparation ---")

    preprocessed = preprocess_df(df_path=path_to_csv)
    dataset = SAINTDataset(preprocessed)

    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.1, shuffle=config.shuffle, random_state=config.random_state
    )
    validate_idx, eval_idx = train_test_split(
        temp_idx, test_size=0.5, shuffle=config.shuffle, random_state=config.random_state
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, validate_idx)
    eval_dataset = torch.utils.data.Subset(dataset, eval_idx)

    # Create dataloaders
    batch_size = config.batch_size
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
    val_dataloader = DataLoader(
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

    print(
        f"Loaded, processed, and split data: {len(train_dataset)} train, {len(val_dataset)} val, {len(eval_dataset)} test samples."
    )
    print(f"------\n")

    return train_dataloader, val_dataloader, eval_dataloader, preprocessed


def validate_model(model: SAINTTransformer, dataloader: DataLoader, device: torch.device):
    model.eval()

    all_losses = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x, y, attention_mask = batch
            x = {k: v.to(device) for k, v in x.items()}
            y, attention_mask = y.to(device), attention_mask.to(device)

            loss, probs, y_masked = model.compute_step((x, y, attention_mask), False)

            all_losses.append(loss.item())
            all_probs.append(probs.detach().cpu())
            all_targets.append(y_masked.detach().cpu())

    # --- Calculate metrics ---
    avg_loss = sum(all_losses) / len(all_losses)
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    accuracy = accuracy_score(all_targets, (all_probs > 0.5).float())
    auroc = roc_auc_score(all_targets, all_probs)

    return avg_loss, accuracy, auroc


def train_model(path_to_csv: Path, perform_eval: bool) -> None:
    print("\n--- Starting model training ---")
    print(f"CUDA Availability: {torch.cuda.is_available()}")
    print("------\n")

    config = SAINTConfig()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, eval_dataloader, preprocessed = prepare_data(path_to_csv, config)

    # Calculate pos_weight for weighted BCE loss fn
    pos_weight = calc_pos_weight(preprocessed)

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

    # --- Move the model to the GPU if CUDA is available ---
    device = config.device
    saint_model.to(device)

    # --- Create the optimizer ---
    optimizer = None
    scheduler = None

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            saint_model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay
        )

        if config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.max_epochs)
        elif config.scheduler == "lambda":

            def lr_lambda(step):
                warmup_steps = config.max_epochs // 10  # 10% warmup
                if step < warmup_steps:
                    # Linear warmup
                    return step / warmup_steps
                else:
                    # Polynomial decay
                    progress = (step - warmup_steps) / (config.max_epochs - warmup_steps)
                    return max(0.1, (1 - progress) ** 0.5)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif config.optimizer == "prodigy-plus":
        optimizer = ProdigyPlusScheduleFree(
            saint_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            use_speed=config.prodigy_use_speed,
            use_orthograd=config.prodigy_use_orthograd,
            use_focus=config.prodigy_use_focus,
        )
        config.learning_rate = 1.0

    # --- Finetuning and Evaluation Loop ---
    print("--- Starting Finetuning & Evaluation ---")
    print(f"Optimizer: \033[36m{optimizer.__class__.__name__}\033[0m")
    print(f"Scheduler: \033[36m{scheduler.__class__.__name__ if scheduler else None}\033[0m")

    run = None
    if config.enable_logging:
        run = configure_logger(config)

    for epoch in range(config.max_epochs + 1):
        if epoch > 0:
            p_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")

            saint_model.train()
            for batch in p_bar:
                x, y, attention_mask = batch

                # Move to device
                x = {k: v.to(device) for k, v in x.items()}
                y, attention_mask = y.to(device), attention_mask.to(device)

                # Forward pass
                optimizer.zero_grad()
                loss, probs, y_masked = saint_model.compute_step((x, y, attention_mask), True)

                # Log to Neptune
                run["train/loss"].append(loss.item()) if run else None

                # Backward pass and step forward with optimizer or scheduler
                loss.backward()
                optimizer.step()

                p_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Scheduler step after epoch
            if scheduler:
                scheduler.step()
                run["train/lr"].append(scheduler.get_last_lr()[0]) if run else None

        # --- Validation loop ---
        epoch_avg_loss, epoch_accuracy, epoch_auroc = validate_model(saint_model, val_dataloader, config.device)

        # --- Log validation metrics ---
        run["val/loss"].append(epoch_avg_loss) if run else None
        run["val/acc"].append(epoch_accuracy) if run else None
        run["val/auroc"].append(epoch_auroc) if run else None

        status = "Initial" if epoch == 0 else f"Epoch: {epoch}"
        print(
            f"{status} Validation | Accuracy: {epoch_accuracy:.4f}, Avg. Loss: {epoch_avg_loss:.4f}, ROC: {epoch_auroc:.4f}\n"
        )

    # Perform evaluations after model training
    run.stop() if run else None

    if perform_eval:
        print("--- Final Evaluation on Test Set ---")
        test_avg_loss, test_accuracy, test_auroc = validate_model(saint_model, eval_dataloader, config.device)
        print(f"Evaluation | Accuracy: {test_accuracy:.4f}, Avg. Loss: {test_avg_loss:.4f}, ROC: {test_auroc:.4f}\n")

    # If it reaches this point, thank fuckin god
    print("🪿 --- Training finished --- 🪿")
    return
