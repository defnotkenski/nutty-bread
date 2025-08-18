from dataclasses import dataclass
import polars as pl


@dataclass(frozen=True)
class AugmentedFrame:
    """
    DataFrame with lagged (recent_0_*) features joined from the previous race.
    """

    df: pl.DataFrame


def add_lags(feature_df: pl.DataFrame, lookup_df: pl.DataFrame | None = None) -> AugmentedFrame:
    """
    Join previous-race features for each horse using last_pp_* keys and add lagged normalized rank.
    """

    source_df = lookup_df if lookup_df is not None else feature_df

    # ======
    # Right-side columns we want to fetch from the previous race (if present).
    # ======

    desired_cols = [
        "race_type",
        "distance_furlongs",
        "race_purse",
        "field_size",
        "course_surface",
        "class_rating",
        "track_conditions",
        "runup_distance",
        "rail_distance",
        "sealed",
        "rank_in_odds",
        "rank_in_odds_frac",
        "days_since_last_race",
        "trainer_win_pct",
        "start_position",
        "point_of_call_1_position",
        "point_of_call_1_lengths",
        "point_of_call_5_position",
        "point_of_call_5_lengths",
        "point_of_call_final_position",
        "point_of_call_final_lengths",
        "speed_rating",
        "race_speed_vs_par",
        "horse_speed_vs_par",
        "horse_time_vs_winner",
        "speed_rating_vs_field_avg",
        "speed_rating_vs_winner",
    ]

    # ======
    # Only select columns that exist on the right side to prevent missing-column errors
    # ======

    available_cols = [c for c in desired_cols if c in source_df.columns]

    right_keys = ["race_date", "track_code", "race_number", "horse_name"]
    right_select = right_keys + available_cols

    df = feature_df.join(
        source_df.select(right_select),
        left_on=["last_pp_race_date", "last_pp_track_code", "last_pp_race_number", "horse_name"],
        right_on=right_keys,
        how="left",
        suffix="_recent_0",
        coalesce=False,
    )

    # ======
    # Rename newly added columns from *_recent_0 to recent_0_*
    # ======

    original_cols = set(feature_df.columns)
    new_cols = [col for col in df.columns if col not in original_cols]

    # df = df.rename({c: f"{c}_recent_0" for c in new_cols if c.replace("_recent_0", "") in right_select})
    # df = df.rename({c: f"recent_0_{c.replace('_recent_0', '')}" for c in df.columns if c.endswith("_recent_0")})

    rename_map = {}
    for c in new_cols:
        if c.endswith("_recent_0"):
            base = c[: -len("_recent_0")]
        else:
            base = c

        rename_map[c] = f"recent_0_{base}"

    df = df.rename(rename_map)

    # ======
    # Compute lagged normalized rank if inputs are present.
    # ======

    df = df.with_columns(
        pl.when(pl.col("recent_0_field_size") > 1)
        .then(1 - (pl.col("recent_0_rank_in_odds") - 1) / (pl.col("recent_0_field_size") - 1))
        .otherwise(pl.lit(1.0))
        .alias("recent_0_rank_in_odds_frac")
    )

    return AugmentedFrame(df=df)
