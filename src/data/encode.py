from dataclasses import dataclass
import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.data.metadata import build_race_metadata, RaceMetadata


@dataclass(frozen=True)
class FeatureMap:
    """
    Lists of feature columns to feed the model.
    """

    continuous: list[str]
    categorical: list[str]
    target: str


@dataclass(frozen=True)
class Preprocessed:
    """
    Tensors and metadata consumed by the trainer/dataset.
    """

    continuous_tensor: torch.Tensor
    categorical_tensor: torch.Tensor
    target_tensor: torch.Tensor
    categorical_cardinalities: list[int]
    race_boundaries: list[tuple[int, int]]
    winner_indices: torch.Tensor


def encode_to_tensors(df: pl.DataFrame, fmap: FeatureMap) -> Preprocessed:
    """
    Encode a feature DataFrame into tensors with scaling/encoding and race metadata.
    """

    # ======
    # Ensure stable race order.
    # ======

    df = df.sort(["race_date", "track_code", "race_number"])

    # ======
    # Add null indicators for both continuous and categorical inputs (before any fills)
    # ======

    missing = [c for c in [*fmap.continuous, *fmap.categorical] if c not in df.columns]
    if len(missing) != 0:
        raise ValueError(f"Missing columns for null indicators: {missing}")

    present_cont = fmap.continuous
    present_cat = fmap.categorical

    indicators = [pl.col(c).is_null().cast(pl.Int64).alias(f"{c}_is_null") for c in [*present_cont, *present_cat]]
    df = df.with_columns(indicators)

    # ======
    # Build the final continuous feature list: originals + their _is_null indicators when present
    # ======

    cont_is_null = [f"{c}_is_null" for c in present_cont]
    cat_is_null = [f"{c}_is_null" for c in present_cat]
    cont_feature_list = present_cont + cont_is_null + cat_is_null

    # ======
    # Select and scale continuous
    # ======

    cont_np = df.select(cont_feature_list).to_numpy() if cont_feature_list else np.empty((len(df), 0))

    if cont_np.size > 0:
        scaler = StandardScaler()
        cont_scaled = scaler.fit_transform(cont_np)
        cont_scaled = np.where(np.isnan(cont_scaled), -999.0, cont_scaled)
        cont_tensor = torch.tensor(cont_scaled, dtype=torch.float32)
    else:
        cont_tensor = torch.empty((len(df), 0), dtype=torch.float32)

    # ======
    # Encode categorical
    # ======

    cat_cardinalities: list[int] = []

    if present_cat:
        cat_df = df.select(present_cat).fill_null("<UNK>")
        encoded_cols = []

        for col in cat_df.columns:
            le = LabelEncoder()
            values = cat_df[col].to_numpy()
            enc = le.fit_transform(values)

            encoded_cols.append(enc)
            cat_cardinalities.append(len(le.classes_))

        cat_np = np.column_stack(encoded_cols)
        cat_tensor = torch.tensor(cat_np, dtype=torch.long)
    else:
        cat_tensor = torch.empty((len(df), 0), dtype=torch.long)

    # ======
    # Target tensor
    # ======

    if fmap.target not in df.columns:
        raise ValueError(f"Target column '{fmap.target}' not found in DataFrame")

    target_tensor = torch.tensor(df[fmap.target].to_numpy(), dtype=torch.float32)

    # ======
    # Race metadata
    # ======

    meta: RaceMetadata = build_race_metadata(df, target_col=fmap.target)

    return Preprocessed(
        continuous_tensor=cont_tensor,
        categorical_tensor=cat_tensor,
        target_tensor=target_tensor,
        categorical_cardinalities=cat_cardinalities,
        race_boundaries=meta.race_boundaries,
        winner_indices=torch.tensor(meta.winner_indices, dtype=torch.long),
    )
