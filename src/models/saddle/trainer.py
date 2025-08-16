import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.models.saddle.data_processing import preprocess_df, SAINTDataset, PreProcessor
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.models.saddle.config import SADDLEConfig
from src.models.saddle.data_processing import collate_races
from prodigyplus import ProdigyPlusScheduleFree
from src.commons.logger import McLogger
from torch.optim.lr_scheduler import OneCycleLR
from schedulefree import AdamWScheduleFree
import signal
from tqdm.rich import tqdm
import warnings
from rich.console import Console
from rich.theme import Theme
from tqdm import TqdmExperimentalWarning
from src.models.saddle.saddle_model import SaddleModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import math

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

custom_theme = Theme({"info_title": "khaki1", "info_text": "bold grey85"})
console = Console(theme=custom_theme)

SCHEDULE_FREE_OPTIMIZERS = ["prodigy-plus", "adamw-schedule-free"]


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
    def __init__(self, config: SADDLEConfig):
        self.config = config

        self.should_stop: bool = False
        self.mclogger: McLogger = McLogger(config)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    @staticmethod
    def _enable_mc_dropout(model: SaddleModel):
        # Turn on dropout layers while leaving the rest in eval mode.
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
        return

    @staticmethod
    def _mc_uncertainty_entropy(race_stack: torch.Tensor) -> tuple[float, int]:
        s, h = race_stack.shape
        mean_p = race_stack.mean(dim=0)
        pred = int(mean_p.argmax().item())
        h_p = -float((mean_p * (mean_p + 1e-8).log()).sum().item())
        h_max = math.log(h) if h > 1 else 1.0
        norm_entropy = h_p / h_max if h_max > 0 else 0.0

        return norm_entropy, pred

    def _mc_race_probs(self, model: SaddleModel, batch, samples: int):
        # Returns per-race stacks: list of tensors [S, H_race]
        x, y, attention_mask, winner_indices = self._move_batch_to_device(batch)
        model.eval()

        if getattr(self.config, "mc_use_dropout", True):
            self._enable_mc_dropout(model)

        race_samples = None

        for s in range(samples):
            _, probs_flat, _ = model.compute_step((x, y, attention_mask, winner_indices), False)

            batch_size = attention_mask.shape[0]
            prob_idx = 0
            per_race = []

            for r in range(batch_size):
                mask = attention_mask[r].bool()
                num_horses = int(mask.sum().item())

                if num_horses > 1:
                    per_race.append(probs_flat[prob_idx : prob_idx + num_horses].detach().cpu())
                else:
                    per_race.append(torch.tensor([1.0]))

                prob_idx += num_horses

            if s == 0:
                race_samples = [[per_race[r]] for r in range(batch_size)]
            else:
                for r in range(batch_size):
                    race_samples[r].append(per_race[r])

        race_samples = [torch.stack(r_list, dim=0) for r_list in race_samples]

        return race_samples, winner_indices.detach().cpu()

    def _signal_handler(self, _signum, _frame):
        console.print("Received graceful shutdown signal...", style="bold magenta italic")
        self.should_stop = True

    def _prepare_data(self, path_to_csv: Path):
        """
        Prepare the different datasets to be used throughout training, validation, and evaluation.

        NOTE: There are four different sets: train, val, eval, and test (test set is a combination of val and eval).
        """

        config = self.config
        preprocessed = preprocess_df(df_path=path_to_csv)
        dataset = SAINTDataset(preprocessed)

        train_idx, test_idx = train_test_split(
            range(len(dataset)), test_size=0.1, shuffle=config.shuffle, random_state=config.random_state
        )
        validate_idx, eval_idx = train_test_split(
            test_idx, test_size=0.5, shuffle=config.shuffle, random_state=config.random_state
        )

        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, validate_idx)
        eval_dataset = torch.utils.data.Subset(dataset, eval_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

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
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_races,
        )

        # print(
        #     f"Loaded, processed, and split data: {len(train_dataset)} train, {len(val_dataset)} val, {len(eval_dataset)} test samples."
        # )
        # print(f"------\n")

        console.print(f"--- Data Preparation ---", style="info_title")
        console.print(
            f"Loaded, processed, and split data: {len(train_dataset)} train, {len(val_dataset)} val, {len(eval_dataset)} eval, {len(test_dataset)} test samples.",
            style="info_text",
        )
        console.print(f"------\n", style="info_title")

        return train_dataloader, val_dataloader, eval_dataloader, test_dataloader, preprocessed

    def _move_batch_to_device(self, batch):
        x, y, attention_mask, winner_indices = batch

        x = {k: v.to(self.device) for k, v in x.items()}
        y = y.to(self.device)
        attention_mask = attention_mask.to(self.device)
        winner_indices = winner_indices.to(self.device)

        return x, y, attention_mask, winner_indices

    def _prepare_optimizer(
        self, model: SaddleModel, train_dataloader: DataLoader
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

        elif config.optimizer == "adamw-schedule-free":
            optimizer = AdamWScheduleFree(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

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

        return optimizer, scheduler

    def _setup_model(self, preprocessed: PreProcessor) -> SaddleModel:
        config = self.config
        # pos_weight = calc_pos_weight(preprocessed)

        model = SaddleModel(
            continuous_dims=preprocessed.continuous_tensor.shape[1],
            categorical_dims=preprocessed.categorical_cardinalities,
            num_block_layers=config.num_block_layers,
            d_model=config.d_model,
            num_heads=config.num_attention_heads,
            config=config,
        )
        model = torch.compile(model, mode="default", disable=config.disable_torch_compile)
        model.to(self.device)

        return model

    def _train_epoch(
        self,
        model: SaddleModel,
        optimizer: Optimizer | ProdigyPlusScheduleFree,
        scheduler: LRScheduler | None,
        epoch: int,
        train_dataloader: DataLoader,
    ) -> None:
        """Train for one epoch"""
        config = self.config

        # --- Skip training on epoch 0 ---
        if epoch == 0:
            return

        p_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")
        for batch in p_bar:
            if self.should_stop:
                print(f"Gracefully stopped at epoch {epoch}")
                break

            # --- Move to device ---
            x, y, attention_mask, winner_indices = self._move_batch_to_device(batch)

            # --- Forward pass ---
            optimizer.zero_grad()
            loss, probs, y_masked = model.compute_step((x, y, attention_mask, winner_indices), True)

            # --- Training metrics to log ---
            if self.mclogger.should_log():
                avg_winner_confidence = torch.exp(-loss).item()
                self.mclogger.log("winner_conf", avg_winner_confidence)

            self.mclogger.log("loss", loss.item())
            self.mclogger.step()

            # --- Backward pass and step forward with optimizer or scheduler ---
            loss.backward()

            # --- Implement gradient clipping ---
            if config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)

            optimizer.step()

            # --- Scheduler step after batch completion ---
            if scheduler:
                self.mclogger.log("lr", scheduler.get_last_lr()[0])
                scheduler.step()

            p_bar.set_postfix(loss=f"{loss.item():.4f}")

    def _validate_model(self, model: SaddleModel, dataloader: DataLoader):
        all_losses = []

        # --- Collect race-level data ---
        race_predictions = []
        race_actual = []

        for batch in dataloader:
            # --- Move to device ---
            x, y, attention_mask, winner_indices = self._move_batch_to_device(batch)

            loss, probs, y_masked = model.compute_step((x, y, attention_mask, winner_indices), False)
            all_losses.append(loss.item())

            # --- Collect race-level predictions ---
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

        # === Calculate metrics ===

        avg_loss = sum(all_losses) / len(all_losses)

        if len(race_predictions) > 0:
            correct_races = sum(1 for pred, actual in zip(race_predictions, race_actual) if pred == actual)
            race_accuracy = correct_races / len(race_predictions)
        else:
            race_accuracy = 0.0

        # === Monte Carlo selective metrics (predictive entropy) ===

        # MC settings
        u_thresholds = self.config.uncertainty_thresholds
        mc_s = self.config.mc_samples
        use_mc = mc_s > 0 and self.config.mc_use_dropout

        # MC accumulators
        sel_u_correct = {tau: 0 for tau in u_thresholds}
        sel_u_covered = {tau: 0 for tau in u_thresholds}
        total_races = 0

        if use_mc:
            for batch in dataloader:
                race_stacks, winner_indices = self._mc_race_probs(model, batch, samples=mc_s)

                for r, stack in enumerate(race_stacks):
                    h = stack.shape[1]
                    total_races += 1

                    if h <= 1:
                        continue

                    u, pred = self._mc_uncertainty_entropy(stack)
                    actual = int(winner_indices[r].item())

                    for tau in u_thresholds:
                        if u <= tau:
                            sel_u_covered[tau] += 1
                            if pred == actual:
                                sel_u_correct[tau] += 1

        sel_metrics = {}
        for tau in u_thresholds:
            cov = sel_u_covered[tau] / total_races if total_races > 0 else 0.0
            acc = (sel_u_correct[tau] / sel_u_covered[tau]) if sel_u_covered[tau] > 0 else 0.0
            key_acc = f"Race Accuracy MCE"
            key_cov = f"Coverage MCE"
            sel_metrics[key_acc] = acc
            sel_metrics[key_cov] = cov

        return avg_loss, race_accuracy, sel_metrics

    def train_model(self, path_to_csv: Path) -> None:
        console.print(f"\n--- Starting Model Training ---", style="info_title")
        console.print(f"CUDA Availability: {torch.cuda.is_available()}", style="info_text")
        console.print(f"------\n", style="info_title")

        config = self.config

        # --- Prepare data ---
        train_dataloader, val_dataloader, eval_dataloader, test_dataloader, preprocessed = self._prepare_data(path_to_csv)

        # --- Prepare model and optimizer ---
        model = self._setup_model(preprocessed)
        optimizer, scheduler = self._prepare_optimizer(model, train_dataloader)

        # --- Finetuning and Evaluation Loop ---
        console.print(f"--- Starting Finetuning & Evaluation ---", style="info_title")
        console.print(f"Optimizer: {optimizer.__class__.__name__}", style="info_text")
        console.print(f"Scheduler: {scheduler.__class__.__name__ if scheduler else None}", style="info_text")

        # --- Training loop ---
        for epoch in range(config.max_epochs + 1):
            # --- Set model, optimizer, and logger to train mode ---
            self.mclogger.set_context("train")

            model.train()
            if config.optimizer in SCHEDULE_FREE_OPTIMIZERS:
                optimizer.train()

            # --- Train the current epoch iteration ---
            self._train_epoch(
                model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, train_dataloader=train_dataloader
            )

            if self.should_stop:
                print(f"Gracefully stopped at epoch {epoch}")
                break

            # --- Set model and optimizer mode to eval mode ---
            model.eval()
            if config.optimizer in SCHEDULE_FREE_OPTIMIZERS:
                optimizer.eval()

            status = "Initial" if epoch == 0 else f"Epoch: {epoch}"

            # --- Test Block ---
            test_avg_loss, test_race_accuracy, selective = self._validate_model(model, dataloader=test_dataloader)

            # --- Log test metrics ---
            self.mclogger.set_context("test")
            self.mclogger.log("loss", test_avg_loss)
            self.mclogger.log("race_accuracy", test_race_accuracy)

            for k, v in selective.items():
                self.mclogger.log(k, v)

            # --- Log test metrics to the console too ---
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in selective.items()])

            console.print(
                f"{status} Test | Race Accuracy: {test_race_accuracy:.4f}, Avg. Loss: {test_avg_loss:.4f}, {metrics_str}\n",
                style="info_text",
            )

            # --- Also log the epoch ---
            self.mclogger.set_context("meta")
            self.mclogger.log("epoch", epoch)

        # --- Perform evaluations after model training ---
        self.mclogger.stop()

        # --- If it reaches this point, thank fuckin god ---
        console.print("🪿 --- Training finished --- 🪿", style="info_title")
        return
