from dataclasses import dataclass
import numpy as np
import polars as pl
import torch
from src.data.metadata import build_race_metadata, RaceMetadata
from sklearn.preprocessing import OrdinalEncoder
import json
from pathlib import Path


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
    top2_indices: list[list[int]]
    top3_indices: list[list[int]]


def encode_to_tensors(
    df: pl.DataFrame, fmap: FeatureMap, train_race_indices: list[int] | None = None, save_artifacts_dir: Path | None = None
) -> Preprocessed:
    """
    Encode a feature DataFrame into tensors with scaling/encoding and race metadata.
    """

    # Ensure stable race order.
    df = df.sort(["race_date", "track_code", "race_number"])

    # Race metadata (needed whether we fit on all or on train)
    meta: RaceMetadata = build_race_metadata(df, target_col=fmap.target)
    race_boundaries = meta.race_boundaries

    # Decide which rows to fit on
    if train_race_indices is None:
        fit_row_idx = np.arange(len(df), dtype=np.int64)
    else:
        fit_rows: list[int] = []
        for r in train_race_indices:
            s, e = race_boundaries[r]
            fit_rows.extend(range(s, e))
        fit_row_idx = np.array(fit_rows, dtype=np.int64)

    # Add null indicators for both continuous and categorical inputs (before any fills)
    missing = [c for c in [*fmap.continuous, *fmap.categorical] if c not in df.columns]
    if len(missing) != 0:
        raise ValueError(f"Missing columns for null indicators: {missing}")

    present_cont = fmap.continuous
    present_cat = fmap.categorical

    indicators = [pl.col(c).is_null().cast(pl.Int64).alias(f"{c}_is_null") for c in [*present_cont, *present_cat]]
    df = df.with_columns(indicators)

    # ======
    # Scale only the true continuous features with explicit masked standardization
    # ======

    # Build separate lists
    # - cont_core: true continuous features to scale
    # - indicator_cols: 0/1 null flags to append unscaled

    fit_mean = None
    fit_std = None
    fit_cat_categories = None

    indicator_cols = [f"{c}_is_null" for c in present_cont] + [f"{c}_is_null" for c in present_cat]
    cont_core = fmap.continuous

    # Select arrays
    cont_core_np = df.select(cont_core).to_numpy() if cont_core else np.empty((len(df), 0))
    ind_np = df.select(indicator_cols).to_numpy().astype(np.float32) if indicator_cols else np.empty((len(df), 0))

    sentinel_z = -5.0

    if cont_core_np.size > 0:
        x = cont_core_np.astype(np.float32)
        mask = np.isnan(x)

        # Fit on train rows only (or all rows if train_race_indices is None)
        x_fit = x[fit_row_idx, :]
        mask_fit = mask[fit_row_idx, :]

        n_obs = (~mask_fit).sum(axis=0)
        missing_cols = [cont_core[i] for i in np.where(n_obs == 0)[0]]
        assert len(missing_cols) == 0, f"Continuous columns have all values missing: {missing_cols}"

        # Compute stats on observed values only
        mean = np.nanmean(x_fit, axis=0)
        std = np.nanstd(x_fit, axis=0)

        fit_mean = mean
        fit_std = std

        # Assert: no zero-variance columns
        const_cols = [cont_core[i] for i in np.where(std == 0.0)[0]]
        assert len(const_cols) == 0, f"Continuous columns have zero variance (constant): {const_cols}"

        # Standardize observed valies; assign fixed z-score for missing
        x_filled = np.where(mask, mean, x)
        z = (x_filled - mean) / std
        z[mask] = sentinel_z
        cont_scaled = z.astype(np.float32)

        # scaler = StandardScaler()
        # cont_scaled = scaler.fit_transform(cont_core_np)
        # cont_scaled = np.where(np.isnan(cont_scaled), -999.0, cont_scaled).astype(np.float32)
    else:
        cont_scaled = np.empty((len(df), 0), dtype=np.float32)

    # Concat indicators (unscaled) to the right
    if cont_scaled.size and ind_np.size:
        cont_full = np.concatenate([cont_scaled, ind_np], axis=1)
    elif cont_scaled.size:
        cont_full = cont_scaled
    else:
        cont_full = ind_np

    cont_tensor = torch.tensor(cont_full, dtype=torch.float32)

    # ======
    # Encode categorical (unknown-safe for ordinal encoding).
    # ======

    cat_cardinalities: list[int] = []

    if present_cat:
        # Select categorical columns and replace nulls with a literal token.
        # This keeps "missingness" explicit and avoids NaN-handling quirks.
        cat_df = df.select(present_cat).fill_null("<UNK>")

        x_cat = cat_df.to_numpy()
        x_cat_fit = x_cat[fit_row_idx, :]

        # Fit OrdinalEncoder with unknown handling.
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
        enc.fit(x_cat_fit)

        fit_cat_categories = enc.categories_

        x_enc = enc.transform(x_cat)

        # Shift codes so 0 is reserved for "unknown."
        x_enc = (x_enc + 1).astype(np.int64)

        # Cardinalities: number of learned categories + 1.
        cat_cardinalities = [len(cats) + 1 for cats in enc.categories_]
        cat_tensor = torch.tensor(x_enc, dtype=torch.long)
    else:
        cat_tensor = torch.empty((len(df), 0), dtype=torch.long)

    # ======
    # Target tensor.
    # ======

    if fmap.target not in df.columns:
        raise ValueError(f"Target column '{fmap.target}' not found in DataFrame")

    target_tensor = torch.tensor(df[fmap.target].to_numpy(), dtype=torch.float32)

    # ======
    # Race metadata.
    # ======

    # Optionally persist train-fitted preprocessors.
    if save_artifacts_dir is not None:
        # Convert Numpy/None -> plain python lists for JSON + saver signature.
        mean_list = fit_mean.tolist() if isinstance(fit_mean, np.ndarray) else []
        std_list = fit_std.tolist() if isinstance(fit_std, np.ndarray) else []
        cat_categories_list = (
            [arr.astype(str).tolist() for arr in fit_cat_categories] if isinstance(fit_cat_categories, (list, tuple)) else []
        )

        save_preprocessors(
            out_dir=save_artifacts_dir,
            cont_core=cont_core,
            mean=mean_list,
            std=std_list,
            sentinel_z=sentinel_z,
            cat_cols=present_cat,
            cat_categories=cat_categories_list,
            fmap=fmap,
        )

    return Preprocessed(
        continuous_tensor=cont_tensor,
        categorical_tensor=cat_tensor,
        target_tensor=target_tensor,
        categorical_cardinalities=cat_cardinalities,
        race_boundaries=meta.race_boundaries,
        winner_indices=torch.tensor(meta.winner_indices, dtype=torch.long),
        top2_indices=meta.top2_indices,
        top3_indices=meta.top3_indices,
    )


def save_preprocessors(
    out_dir: Path,
    cont_core: list[str],
    mean: list[float],
    std: list[float],
    sentinel_z: float,
    cat_cols: list[str],
    cat_categories: list[list[str]],
    fmap: FeatureMap,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Continuous stats
    cont_payload = {
        "columns": cont_core,
        "sentinel_z": float(sentinel_z),
        "mean": mean,
        "std": std,
    }
    (out_dir / "continuous_stats.json").write_text(json.dumps(cont_payload, indent=2))

    # Categorical vocab
    cat_payload = {
        "columns": cat_cols,
        "categories": cat_categories,
        "unknown_index": 0,  # we shift by +1, so 0=unknown
    }
    (out_dir / "categorical_vocab.json").write_text(json.dumps(cat_payload, indent=2))

    # Feature map snapshot (for sanity)
    fmap_payload = {
        "continuous": fmap.continuous,
        "categorical": fmap.categorical,
        "target": fmap.target,
    }
    (out_dir / "feature_map.json").write_text(json.dumps(fmap_payload, indent=2))
