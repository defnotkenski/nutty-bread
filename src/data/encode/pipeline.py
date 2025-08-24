from pathlib import Path
import numpy as np
import polars as pl
import torch
from src.data.metadata import build_race_metadata
from .feature_types import FeatureMap, Preprocessed
from .continuous import fit_stats, transform as transform_cont
from .categorical import fit_encoder, transform as transform_cat
from .artifacts import save_preprocessors


def encode_to_tensors(
    df: pl.DataFrame, fmap: FeatureMap, train_race_indices: list[int], save_artifacts_dir: Path | None = None
) -> Preprocessed:
    """
    Encode a feature DataFrame into tensors with scaling/encoding and race metadata.
    """

    # Stable race order.
    df = df.sort(["race_date", "track_code", "race_number"])

    # Race metadata.
    meta = build_race_metadata(df, target_col=fmap.target)
    race_boundaries = meta.race_boundaries

    # Fit rows (by races if provided).
    rows: list[int] = []
    for r in train_race_indices:
        s, e = race_boundaries[r]
        rows.extend(range(s, e))

    fit_row_idx = np.array(rows, dtype=np.int64)
    assert fit_row_idx.size != 0, f"train_race_indices produced no rows to fit on."

    # Column presence stack.
    required = [*fmap.continuous, *fmap.categorical, fmap.target]

    missing = [c for c in required if c not in df.columns]
    assert len(missing) == 0, f"Missing columns for null indicators: {missing}"

    # Null indicators (for cat + cont)
    indicator_cols = [f"{c}_is_null" for c in fmap.continuous] + [f"{c}_is_null" for c in fmap.categorical]
    df = df.with_columns(
        [pl.col(c).is_null().cast(pl.Int64).alias(f"{c}_is_null") for c in [*fmap.continuous, *fmap.categorical]]
    )

    # Continuous: masked standardization + appen indicators.
    cont_core_np = df.select(fmap.continuous).to_numpy()
    ind_np = df.select(indicator_cols).to_numpy().astype(np.float32)

    sentinel_z = -5.0

    x_fit = cont_core_np[fit_row_idx, :]
    mean, std = fit_stats(x_fit)

    cont_scaled = transform_cont(cont_core_np, mean=mean, std=std, sentinel_z=sentinel_z)

    fit_mean = mean.tolist()
    fit_std = std.tolist()

    cont_full = np.concatenate([cont_scaled, ind_np], axis=1)
    cont_tensor = torch.tensor(cont_full, dtype=torch.float32)

    # Categorical: ordinal with unknown -> 0.
    assert fmap.categorical, f"Issue with initializing present_cat variable."

    cat_df = df.select(fmap.categorical).fill_null("<UNK>")
    x_cat = cat_df.to_numpy()
    x_cat_fit = x_cat[fit_row_idx, :]

    enc, categories = fit_encoder(x_cat_fit)
    x_enc, cat_cardinalities = transform_cat(x_cat, enc=enc)

    cat_tensor = torch.tensor(x_enc, dtype=torch.long)

    # Target.
    target_tensor = torch.tensor(df[fmap.target].to_numpy(), dtype=torch.float32)

    # Optional: persist preprocessors
    if save_artifacts_dir is not None:
        save_preprocessors(
            out_dir=save_artifacts_dir,
            cont_core=fmap.continuous,
            mean=fit_mean,
            std=fit_std,
            sentinel_z=sentinel_z,
            cat_cols=fmap.categorical,
            cat_categories=categories,
            fmap=fmap,
        )

    return Preprocessed(
        continuous_tensor=cont_tensor,
        categorical_tensor=cat_tensor,
        target_tensor=target_tensor,
        categorical_cardinalities=cat_cardinalities,
        race_boundaries=race_boundaries,
        winner_indices=torch.tensor(meta.winner_indices, dtype=torch.long),
        top2_indices=meta.top2_indices,
        top3_indices=meta.top3_indices,
    )
