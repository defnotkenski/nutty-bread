import numpy as np
import polars as pl
from pathlib import Path
import torch
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from datasets.sample_horses_schema import COLUMN_TYPES
from custom.commons.utils import cleanup_dataframe


@dataclass
class PreProcessor:
    continuous_tensor: torch.Tensor
    categorical_tensor: torch.Tensor
    target_tensor: torch.Tensor
    categorical_cardinalities: list
    race_boundaries: list


def preprocess_df(df_path: Path):
    # Load df and cast cols to appropriate dtype
    base_df = pl.read_csv(df_path, infer_schema=False)

    base_df = base_df.with_columns(pl.col("race_date").str.to_datetime())
    base_df = base_df.with_columns(pl.col("last_pp_race_date").str.to_datetime())

    base_df = cleanup_dataframe(base_polars_df=base_df)

    base_df = base_df.cast(COLUMN_TYPES)
    base_df = base_df.sort(["race_date", "track_code", "race_number", "dollar_odds"])

    # ===== RACE AWARE: Create race boundaries to be used when batching during training =====

    race_boundaries = []
    current_start = 0

    race_groups = base_df.group_by(["race_date", "track_code", "race_number"], maintain_order=True)

    for _, group_df in race_groups:
        race_size = len(group_df)
        race_boundaries.append([current_start, current_start + race_size])

        current_start += race_size

    # ===== END RACE AWARE =====

    # Create target col
    base_df = base_df.with_columns(pl.col("official_final_position").is_in([1, 2, 3]).cast(pl.Int64).alias("target"))

    # Extract target col
    target_tensor = torch.tensor(base_df["target"].to_numpy(), dtype=torch.float32)

    # Drop target col
    base_df = base_df.drop("target")

    # Extract and scale continuous cols
    scaler = StandardScaler()

    cont_cols = base_df.select(pl.selectors.numeric()).to_numpy()
    cont_cols_scaled = scaler.fit_transform(cont_cols)

    # Impute missing data with sentinel value (temporary until solution)
    cont_cols_sentinel = np.where(np.isnan(cont_cols_scaled), -999, cont_cols_scaled)

    cont_tensor = torch.tensor(cont_cols_sentinel, dtype=torch.float32)

    # Extract categorical cols
    cat_cols_df = base_df.select(pl.selectors.string(include_categorical=True)).fill_null("<UNK>")

    encoded_cats = []
    cat_cardinalities = []

    for col in cat_cols_df.columns:
        le = LabelEncoder()
        encoded_col = le.fit_transform(cat_cols_df[col].to_numpy())

        encoded_cats.append(encoded_col)
        cat_cardinalities.append(len(le.classes_))

    assert len(encoded_cats) != 0, f"Encoded cats is 0."
    cat_tensor = torch.tensor(np.column_stack(encoded_cats), dtype=torch.long)

    return PreProcessor(
        continuous_tensor=cont_tensor,
        categorical_tensor=cat_tensor,
        target_tensor=target_tensor,
        categorical_cardinalities=cat_cardinalities,
        race_boundaries=race_boundaries,
    )


class CustomTabularDataset(Dataset):
    def __init__(self, preprocessed_data: PreProcessor):
        self.continuous = preprocessed_data.continuous_tensor
        self.categorical = preprocessed_data.categorical_tensor
        self.target = preprocessed_data.target_tensor

        # RACE AWARE
        self.race_boundaries = preprocessed_data.race_boundaries

    # def __len__(self):
    #     return len(self.continuous)

    # RACE AWARE
    def __len__(self):
        return len(self.race_boundaries)

    # def __getitem__(self, idx):
    #     return {
    #         "continuous": self.continuous[idx],
    #         "categorical": self.categorical[idx],
    #     }, self.target[
    #         idx
    #     ].unsqueeze(0)

    # RACE AWARE
    def __getitem__(self, item_idx):
        start_idx, end_idx = self.race_boundaries[item_idx]

        return {
            "continuous": self.continuous[start_idx:end_idx],
            "categorical": self.categorical[start_idx:end_idx],
        }, self.target[start_idx:end_idx]
