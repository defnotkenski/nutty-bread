import pandas as pd
import torch
from torch.utils.data import DataLoader
from custom.models.saint_transformer.data_processing import preprocess_df, SAINTDataset, PreProcessor
from custom.models.saint_transformer.saint_transformer import SAINTTransformer
from pathlib import Path
from sklearn.model_selection import train_test_split
from custom.models.saint_transformer.config import SAINTConfig
from custom.models.saint_transformer.data_processing import collate_races
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from prodigyplus import ProdigyPlusScheduleFree
from custom.commons.logger import McLogger
from torch.optim.lr_scheduler import OneCycleLR
import signal

# Type Imports
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def calc_pos_weight(preprocessed):
    # Calculate pos_weight for weighted BCE loss fn
    target_series = pd.Series(preprocessed.target_tensor.numpy().flatten())
    pos_count = target_series.sum()
    neg_count = len(target_series) - pos_count
    pos_weight = neg_count / pos_count

    # Cap pos weight for stability
    pos_weight = min(pos_weight, 5.0)
    return pos_weight


class ModelTrainer:
    def __init__(self, config: SAINTConfig):
        self.config = config

        self.should_stop: bool = False
        self.mclogger: McLogger = McLogger(config)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.model: SAINTTransformer | None = None
        # self.optimizer: Optimizer | ProdigyPlusScheduleFree | None = None
        # self.scheduler: LRScheduler | None = None

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, _signum, _frame):
        print("Received graceful shutdown signal...")
        self.should_stop = True

    def _prepare_data(self, path_to_csv: Path):
        print("--- Data Preparation ---")

        config = self.config
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
            drop_last=True,
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

    def _move_batch_to_device(self, batch):
        x, y, attention_mask, winner_indices = batch

        x = {k: v.to(self.device) for k, v in x.items()}
        y = y.to(self.device)
        attention_mask = attention_mask.to(self.device)
        winner_indices = winner_indices.to(self.device)

        return x, y, attention_mask, winner_indices

    def _prepare_optimizer(
        self, model: SAINTTransformer, train_dataloader: DataLoader
    ) -> tuple[Optimizer | ProdigyPlusScheduleFree, LRScheduler | None]:
        """Create optimizer and scheduler"""
        config = self.config
        optimizer = None
        scheduler = None

        if config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay
            )

            if config.scheduler == "cosine":
                total_steps = len(train_dataloader) * config.max_epochs
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=config.learning_rate,
                    total_steps=total_steps,
                    pct_start=config.warmup_pct,
                    anneal_strategy="cos",
                    final_div_factor=1 / config.min_lr_ratio,
                )

        elif config.optimizer == "prodigy-plus":
            print(f"--- Optimizer info ---")
            print(f"Detected {config.optimizer} optimizer. Setting appropriate hyperparams based on config.")
            print(f"------\n")

            config.learning_rate = 1.0
            config.gradient_clip_val = None

            if config.prodigy_use_speed:
                config.weight_decay = 0.0

            optimizer = ProdigyPlusScheduleFree(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                use_speed=config.prodigy_use_speed,
                use_orthograd=config.prodigy_use_orthograd,
                use_focus=config.prodigy_use_focus,
            )

        # self.optimizer = optimizer
        # self.scheduler = scheduler
        return optimizer, scheduler

    def _setup_model(self, preprocessed: PreProcessor) -> SAINTTransformer:
        config = self.config
        pos_weight = calc_pos_weight(preprocessed)

        model = SAINTTransformer(
            continuous_dims=preprocessed.continuous_tensor.shape[1],
            categorical_dims=preprocessed.categorical_cardinalities,
            num_block_layers=config.num_block_layers,
            d_model=config.d_model,
            num_heads=config.num_attention_heads,
            output_size=config.output_size,
            pos_weight=pos_weight,
            config=config,
        )
        model = torch.compile(model, mode="default", disable=config.disable_torch_compile)
        model.to(self.device)

        return model

    def _train_epoch(
        self,
        model: SAINTTransformer,
        optimizer: Optimizer | ProdigyPlusScheduleFree,
        scheduler: LRScheduler | None,
        epoch: int,
        train_dataloader: DataLoader,
    ) -> None:
        """Train for one epoch"""
        config = self.config
        mclogger = self.mclogger

        if epoch == 0:
            return  # Skip training on epoch 0

        model.train()
        if config.optimizer == "prodigy-plus":
            optimizer.train()

        mclogger.set_context("train")
        p_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")

        for batch in p_bar:
            if self.should_stop:
                print(f"Gracefully stopped at epoch {epoch}")
                break

            # --- Move to device ---
            # x, y, attention_mask, winner_indices = batch
            # x = {k: v.to(self.device) for k, v in x.items()}
            # y, attention_mask = y.to(self.device), attention_mask.to(self.device)
            # winner_indices = winner_indices.to(self.device)
            x, y, attention_mask, winner_indices = self._move_batch_to_device(batch)

            # --- Forward pass ---
            optimizer.zero_grad()
            loss, probs, y_masked = model.compute_step((x, y, attention_mask, winner_indices), True)

            # --- Training metric to log ---
            if mclogger.should_log():
                train_acc = accuracy_score(y_masked.detach().cpu(), (probs > 0.5).detach().cpu())
                mclogger.log("accuracy", train_acc)

            mclogger.log("loss", loss.item())
            mclogger.step()

            # --- Backward pass and step forward with optimizer or scheduler ---
            loss.backward()

            # --- Implement gradient clipping ---
            if config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)

            optimizer.step()

            # --- Scheduler step after batch completion ---
            if scheduler:
                mclogger.log("lr", scheduler.get_last_lr()[0])
                scheduler.step()

            p_bar.set_postfix(loss=f"{loss.item():.4f}")

    def _validate_model(self, model: SAINTTransformer, dataloader: DataLoader):
        model.eval()

        all_losses = []
        all_probs = []
        all_targets = []

        # Collect race-level data
        race_predictions = []
        race_actual = []

        for batch in dataloader:
            # --- Move to device ---
            # x, y, attention_mask, winner_indices = batch
            # x = {k: v.to(self.device) for k, v in x.items()}
            # y, attention_mask = y.to(self.device), attention_mask.to(self.device)
            # winner_indices = winner_indices.to(self.device)
            x, y, attention_mask, winner_indices = self._move_batch_to_device(batch)

            loss, probs, y_masked = model.compute_step((x, y, attention_mask, winner_indices), False)

            all_losses.append(loss.item())
            all_probs.append(probs.detach().cpu())
            all_targets.append(y_masked.detach().cpu())

            # Collect race-level predictions
            batch_size = attention_mask.shape[0]
            prob_idx = 0

            for race_idx in range(batch_size):
                mask = attention_mask[race_idx].bool()
                num_horses = mask.sum().int()

                if num_horses > 1:
                    race_probs = probs[prob_idx : prob_idx + num_horses]
                    predicted_winner = race_probs.argmax().item()
                    race_predictions.append(predicted_winner)
                else:
                    race_predictions.append(0)

                prob_idx += num_horses

            race_actual.extend(winner_indices.cpu().tolist())

        # --- Calculate metrics ---
        avg_loss = sum(all_losses) / len(all_losses)
        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)

        if len(race_predictions) > 0:
            correct_races = sum(1 for pred, actual in zip(race_predictions, race_actual) if pred == actual)
            race_accuracy = correct_races / len(race_predictions)
        else:
            race_accuracy = 0.0

        accuracy = accuracy_score(all_targets, (all_probs > 0.5).float())
        auroc = roc_auc_score(all_targets, all_probs)

        return avg_loss, accuracy, auroc, race_accuracy

    def train_model(self, path_to_csv: Path, perform_eval: bool) -> None:
        print("\n--- Starting model training ---")
        print(f"CUDA Availability: {torch.cuda.is_available()}")
        print("------\n")

        config = self.config

        # --- Prepare data ---
        train_dataloader, val_dataloader, eval_dataloader, preprocessed = self._prepare_data(path_to_csv)

        # --- Prepare model and optimizer ---
        model = self._setup_model(preprocessed)
        optimizer, scheduler = self._prepare_optimizer(model, train_dataloader)

        # --- Finetuning and Evaluation Loop ---
        print("--- Starting Finetuning & Evaluation ---")
        print(f"Optimizer: \033[36m{optimizer.__class__.__name__}\033[0m")
        print(f"Scheduler: \033[36m{scheduler.__class__.__name__ if scheduler else None}\033[0m")

        # --- Training loop ---
        for epoch in range(config.max_epochs + 1):
            # --- Train epoch ---
            self._train_epoch(
                model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, train_dataloader=train_dataloader
            )

            if self.should_stop:
                print(f"Gracefully stopped at epoch {epoch}")
                break

            # --- Validation loop ---
            if config.optimizer == "prodigy-plus":
                optimizer.eval()

            epoch_avg_loss, epoch_accuracy, epoch_auroc, epoch_race_accuracy = self._validate_model(
                model, dataloader=val_dataloader
            )

            # --- Log validation metrics ---
            self.mclogger.set_context("val")

            self.mclogger.log("loss", epoch_avg_loss)
            self.mclogger.log("accuracy", epoch_accuracy)
            self.mclogger.log("auroc", epoch_auroc)
            self.mclogger.log("race_accuracy", epoch_race_accuracy)

            # Also log the epoch
            self.mclogger.set_context("meta")
            self.mclogger.log("epoch", epoch)

            # Log validation metrics to the console too
            status = "Initial" if epoch == 0 else f"Epoch: {epoch}"
            print(
                f"{status} Validation | Accuracy: {epoch_accuracy:.4f}, Avg. Loss: {epoch_avg_loss:.4f}, ROC: {epoch_auroc:.4f}, Race Acc:{epoch_race_accuracy:.4f}\n"
            )

        # --- Perform evaluations after model training ---
        self.mclogger.stop()

        if perform_eval and not self.should_stop:
            print("--- Final Evaluation on Test Set ---")
            test_avg_loss, test_accuracy, test_auroc, test_race_accuracy = self._validate_model(
                model, dataloader=eval_dataloader
            )
            print(
                f"Evaluation | Accuracy: {test_accuracy:.4f}, Avg. Loss: {test_avg_loss:.4f}, ROC: {test_auroc:.4f}, Race Acc: {test_race_accuracy:.4f}\n"
            )

        # If it reaches this point, thank fuckin god
        print("🪿 --- Training finished --- 🪿")
        return
