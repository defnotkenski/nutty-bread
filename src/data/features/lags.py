from dataclasses import dataclass
import polars as pl


@dataclass(frozen=True)
class AugmentedFrame:
    """
    DataFrame with lagged (recent_0_*) features joined from the previous race.
    """

    df: pl.DataFrame


def add_lags(feature_df: pl.DataFrame, lookup_df: pl.DataFrame | None, base_cols: list[str]) -> AugmentedFrame:
    """
    Join previous-race features for each horse using last_pp_* keys and add lagged normalized rank.
    """

    source_df = lookup_df if lookup_df is not None else feature_df

    left_keys = ["last_pp_race_date", "last_pp_track_code", "last_pp_race_number", "horse_name"]
    right_keys = ["race_date", "track_code", "race_number", "horse_name"]

    non_key_bases = [c for c in base_cols if c not in right_keys]

    # === Build expressions to avoid duplicate names when a desired col is also a key. ===

    right_select = right_keys + non_key_bases

    df = feature_df.join(
        source_df.select(right_select),
        left_on=left_keys,
        right_on=right_keys,
        how="left",
        suffix="_recent_0",
        coalesce=False,
    )

    # === Rename newly added columns from *_recent_0 to recent_0_*. ===

    original_cols = set(feature_df.columns)
    new_cols = [col for col in df.columns if col not in original_cols]

    rename_map = {}
    for c in new_cols:
        base = c[: -len("_recent_0")] if c.endswith("_recent_0") else c
        rename_map[c] = f"recent_0_{base}"

    df = df.rename(rename_map)

    # === Compute lagged normalized rank if inputs are present. ===

    df = df.with_columns(
        pl.when(pl.col("recent_0_field_size") > 1)
        .then(1 - (pl.col("recent_0_rank_in_odds") - 1) / (pl.col("recent_0_field_size") - 1))
        .otherwise(pl.lit(1.0))
        .alias("recent_0_rank_in_odds_frac")
    )

    return AugmentedFrame(df=df)
