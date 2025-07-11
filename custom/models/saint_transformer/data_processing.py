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
from custom.commons.feature_extractor import FeatureProcessor


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

    # Feature extraction
    feature_extractor = FeatureProcessor(df=base_df, target_type="place")
    feature_config = feature_extractor.get_dataframe()

    base_df = feature_config.df

    # Create race boundaries to be used when batching during training
    race_boundaries = []
    current_start = 0

    race_groups = base_df.group_by(["race_date", "track_code", "race_number"], maintain_order=True)

    for _, group_df in race_groups:
        race_size = len(group_df)
        race_boundaries.append([current_start, current_start + race_size])

        current_start += race_size

    # Extract target col
    target_tensor = torch.tensor(base_df["target"].to_numpy(), dtype=torch.float32)

    # Drop target col
    base_df = base_df.drop("target")

    # Extract and scale continuous cols
    scaler = StandardScaler()

    cont_cols = base_df.select(feature_config.continuous_cols).to_numpy()
    cont_cols_scaled = scaler.fit_transform(cont_cols)

    # Impute missing data with sentinel value for continuous cols
    cont_cols_sentinel = np.where(np.isnan(cont_cols_scaled), -999, cont_cols_scaled)

    # Store as a torch tensor
    cont_tensor = torch.tensor(cont_cols_sentinel, dtype=torch.float32)

    # Extract and map categorical cols to integers
    cat_cols_df = base_df.select(feature_config.categorical_cols).fill_null("<UNK>")

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


def collate_races(batch):
    """
    Collate function to pad races to same length and create attention masks.
    Args:
        List of tuples (input_dict, targets) from SAINTDataset
    """

    # Separate inputs and targets
    inputs = [item[0] for item in batch]  # List of dicts
    targets = [item[1] for item in batch]  # list of target tensors

    # Find max number of horses in this batch
    max_horses = max(inp["continuous"].shape[0] for inp in inputs)
    batch_size = len(batch)

    # Get feature dimensions
    cont_features = inputs[0]["continuous"].shape[1]
    cat_features = inputs[0]["categorical"].shape[1]

    # Initialize padded tensors
    padded_continuous = torch.zeros(batch_size, max_horses, cont_features)
    padded_categorical = torch.zeros(batch_size, max_horses, cat_features, dtype=torch.long)
    padded_targets = torch.full((batch_size, max_horses), -100.0)

    attention_mask = torch.zeros(batch_size, max_horses)

    # Fill in the actual data
    for i, (inp, target) in enumerate(zip(inputs, targets)):
        num_horses = inp["continuous"].shape[0]

        padded_continuous[i, :num_horses] = inp["continuous"]
        padded_categorical[i, :num_horses] = inp["categorical"]
        padded_targets[i, :num_horses] = target
        attention_mask[i, :num_horses] = 1

    batched_input = {
        "continuous": padded_continuous,
        "categorical": padded_categorical,
    }

    return batched_input, padded_targets, attention_mask


class SAINTDataset(Dataset):
    def __init__(self, preprocessed_data: PreProcessor):
        self.continuous = preprocessed_data.continuous_tensor
        self.categorical = preprocessed_data.categorical_tensor
        self.target = preprocessed_data.target_tensor

        self.race_boundaries = preprocessed_data.race_boundaries

    def __len__(self):
        return len(self.race_boundaries)

    def __getitem__(self, item_idx):
        start_idx, end_idx = self.race_boundaries[item_idx]

        return {
            "continuous": self.continuous[start_idx:end_idx],  # (num_horses, features)
            "categorical": self.categorical[start_idx:end_idx],  # (num_horses, features)
        }, self.target[
            start_idx:end_idx
        ]  # (num_horses,)
