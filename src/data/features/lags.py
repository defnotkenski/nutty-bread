from dataclasses import dataclass
import polars as pl


@dataclass(frozen=True)
class AugmentedFrame:
    """
    DataFrame with lagged (recent_0_*) features joined from the previous race.
    """

    df: pl.DataFrame


def add_lags(feature_df: pl.DataFrame, lookup_df: pl.DataFrame | None) -> AugmentedFrame:
    """
    Join previous-race features for each horse using last_pp_* keys.
    """

    source_df = lookup_df if lookup_df is not None else feature_df

    left_keys = ["last_pp_race_date", "last_pp_track_code", "last_pp_race_number", "horse_name"]
    right_keys = ["race_date", "track_code", "race_number", "horse_name"]

    # non_key_bases = [c for c in base_cols if c not in right_keys]

    # === Build expressions to avoid duplicate names when a desired col is also a key. ===

    # right_select = right_keys + non_key_bases

    df = feature_df.join(
        # source_df.select(right_select),
        source_df,
        left_on=left_keys,
        right_on=right_keys,
        how="left",
        suffix="_recent_0",
        coalesce=False,
        validate="m:1",
    )

    # === Rename newly added columns from *_recent_0 to recent_0_*. ===

    original_cols = set(feature_df.columns)
    new_cols = [col for col in df.columns if col not in original_cols]

    rename_map = {}
    for c in new_cols:
        base = c[: -len("_recent_0")] if c.endswith("_recent_0") else c
        rename_map[c] = f"recent_0_{base}"

    df = df.rename(rename_map)

    return AugmentedFrame(df=df)
