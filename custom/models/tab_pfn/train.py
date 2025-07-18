import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial
from pathlib import Path
from custom.models.saint_transformer.data_processing import preprocess_df
from tabpfn import TabPFNClassifier
from tabpfn.utils import meta_dataset_collator
from tabpfn.finetune_utils import clone_model_for_evaluation
import torch
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
import tabpfn
import inspect


@dataclass
class ClassifierConfig:
    ignore_pretraining_limits: bool = True
    inference_precision: torch._C = torch.float32
    n_estimators: int = 2


@dataclass
class MasterConfig:
    classifier_config: ClassifierConfig
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    test_size: float = 0.10
    random_state: int = 777
    n_inference_context_samples: int = 10000
    meta_batch_size: int = 1
    batch_size: int = 10000


def prep_data(config: MasterConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("--- 1. Data Preparation ---")

    all_data_path = Path.cwd() / "datasets" / "sample_horses_v2.csv"

    splitter = partial(train_test_split, test_size=config.test_size, random_state=config.random_state, shuffle=False)

    feature_config = preprocess_df(df_path=all_data_path, return_feature_config=True)
    base_df = feature_config.df.select(
        [*feature_config.continuous_cols, *feature_config.categorical_cols, *feature_config.target_cols]
    )

    # Impute missing values by column type
    base_df = base_df.with_columns(pl.col(feature_config.continuous_cols).fill_null(-999))
    base_df = base_df.with_columns(pl.col(feature_config.categorical_cols).fill_null("<UNK>"))

    x_data = base_df.drop("target").to_numpy()
    y_data = base_df.select("target").to_numpy().ravel()

    x_train, x_test, y_train, y_test = splitter(x_data, y_data)

    print(f"Loaded, processed, and split data: {x_train.shape[0]} train, {x_test.shape[0]} test samples.")
    print(f"------\n")

    return x_train, x_test, y_train, y_test


def setup_model_and_optimizer(config: MasterConfig) -> tuple[TabPFNClassifier, Optimizer]:
    print("--- 2. Model and Optimizer Setup ---")
    classifier_config = config.classifier_config
    classifier_config.device = config.device
    classifier_config.random_state = config.random_state

    classifier = TabPFNClassifier(
        **classifier_config.__dict__,
        fit_mode="batched",
        differentiable_input=False,
    )

    classifier._initialize_model_variables()
    optimizer = Adam(classifier.model_.parameters(), lr=1e-5)

    print(f"Using device: {"cuda" if torch.cuda.is_available() else "cpu"}")
    print(f"------\n")

    return classifier, optimizer


def evaluate_model(
    classifier: TabPFNClassifier,
    eval_config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Evaluates the model's performance on the test set."""
    eval_classifier = clone_model_for_evaluation(classifier, eval_config, TabPFNClassifier)
    eval_classifier.fit(x_train, y_train)

    try:
        probabilities = eval_classifier.predict_proba(x_test)
        roc_auc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted")
        log_loss_score = log_loss(y_test, probabilities)
    except Exception as e:
        print(f"An error occured during model evaluation: {e}")
        roc_auc, log_loss_score = np.nan, np.nan

    return roc_auc, log_loss_score


def train_model() -> None:
    classifier_config = ClassifierConfig()
    master_config = MasterConfig(classifier_config=classifier_config)

    x_train, x_test, y_train, y_test = prep_data(config=master_config)
    classifier, optimizer = setup_model_and_optimizer(config=master_config)

    splitter = partial(train_test_split, test_size=master_config.test_size)

    training_datasets = classifier.get_preprocessed_datasets(x_train, y_train, splitter, master_config.batch_size)
    finetuning_dataloader = DataLoader(training_datasets, master_config.meta_batch_size, collate_fn=meta_dataset_collator)

    loss_fn = torch.nn.CrossEntropyLoss()

    eval_config = {
        **classifier_config.__dict__,
        "inference_config": {"SUBSAMPLE_SAMPLES": master_config.n_inference_context_samples},
    }

    # --- Finetuning and Evaluation Loop ---
    print("--- 3. Starting Finetuning & Evaluation ---")

    for epoch in range(master_config.epochs + 1):
        if epoch > 0:
            p_bar = tqdm(finetuning_dataloader, f"Finetuning Epoch: {epoch}")

            for x_train_batch, x_test_batch, y_train_batch, y_test_batch, cat_ixs, confs in p_bar:
                if len(np.unique(y_train_batch)) != len(np.unique(y_test_batch)):
                    continue  # Skip batch if splits don't have all classes

                optimizer.zero_grad()
                classifier.fit_from_preprocessed(x_train_batch, y_train_batch, cat_ixs, confs)

                predictions = classifier.forward(x_test_batch, return_logits=True)
                loss = loss_fn(predictions, y_test_batch.to(master_config.device))

                loss.backward()
                optimizer.step()

                p_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_roc, epoch_log_loss = evaluate_model(classifier, eval_config, x_train, y_train, x_test, y_test)

        status = "initial" if epoch == 0 else f"Epoch: {epoch}"
        print(f"{status} Evaluation | Test ROC: {epoch_roc:.4f}, Test LogLoss: {epoch_log_loss:.4f}\n")

    print(f"--- 🪿 Finetuning Finished ---")

    return


if __name__ == "__main__":
    train_model()
