from dataclasses import dataclass
import polars as pl


@dataclass(frozen=True)
class FeatureFrame:
    """
    DataFrame with prediction-safe or results-derived engineered features.
    """

    df: pl.DataFrame


def build_results_derived_features(df: pl.DataFrame) -> FeatureFrame:
    """
    Add results-derived features
    """

    # Trainer cumulative stats
    df = df.with_columns(
        (
            pl.col("official_final_position")
            .is_in([1, 2, 3])  # define "win%" as top-3; change to [1] for strict wins
            .cum_sum()
            .over("trainer_full_name", order_by=["race_date", "track_code", "race_number"])
            .cast(pl.Int64)
            - 1  # remove current race contribution
        )
        .clip(lower_bound=0)
        .alias("trainer_wins")
    )

    df = df.with_columns(
        pl.int_range(pl.len())
        .over("trainer_full_name", order_by=["race_date", "track_code", "race_number"])
        .cast(pl.Int64)
        .alias("trainer_entries")
    )

    df = df.with_columns(
        pl.when(pl.col("trainer_entries") == 0)
        .then(0.0)
        .otherwise(pl.col("trainer_wins") / pl.col("trainer_entries"))
        .alias("trainer_win_pct")
    )

    # Results-derived feature calculations (leakage-safe)
    # race_speed_vs_par = win_time - par_time (when par_time != 0.0)
    df = df.with_columns(
        pl.when(pl.col("par_time") != 0.00)
        .then(pl.col("win_time") - pl.col("par_time"))
        .otherwise(pl.lit(None))
        .alias("race_speed_vs_par")
    )

    # Compute horse_finish_time from final beaten lengths (1 length ~ 0.2s)
    length_seconds = 0.2
    df = df.with_columns(
        (pl.col("point_of_call_final_lengths") * length_seconds + pl.col("win_time")).alias("horse_finish_time")
    )

    # horse_speed_vs_par = horse_finish_time - par_time
    df = df.with_columns((pl.col("horse_finish_time") - pl.col("par_time")).alias("horse_speed_vs_par"))

    # horse_time_vs_winner = horse_finish_time - win_time
    df = df.with_columns((pl.col("horse_finish_time") - pl.col("win_time")).alias("horse_time_vs_winner"))

    # field_avg_speed_rating (average of others in the race)
    df = df.with_columns(
        ((pl.col("speed_rating").sum() - pl.col("speed_rating")) / (pl.len() - 1))
        .over(["race_date", "track_code", "race_number"])
        .alias("field_avg_speed_rating")
    )

    # speed_rating_vs_field_avg
    df = df.with_columns((pl.col("speed_rating") - pl.col("field_avg_speed_rating")).alias("speed_rating_vs_field_avg"))

    # speed_rating_vs_winner
    df = df.with_columns(
        pl.col("speed_rating")
        .get(pl.col("official_final_position").arg_min())
        .over(["race_date", "track_code", "race_number"])
        .alias("speed_rating_winner")
    )
    df = df.with_columns((pl.col("speed_rating") - pl.col("speed_rating_winner")).alias("speed_rating_vs_winner"))

    return FeatureFrame(df=df)


def build_prediction_safe_features(raw_df: pl.DataFrame) -> FeatureFrame:
    """
    Add prediction-safe features to the raw, typed, cleaned DataFrame
    """

    working_df = raw_df

    # === Distance in furlongs ===

    working_df = working_df.with_columns((pl.col("distance").cast(pl.Float64) / 100).alias("distance_furlongs"))

    # === Field size per race ===

    working_df = working_df.with_columns(pl.count().over(["race_date", "track_code", "race_number"]).alias("field_size"))

    # === Ordinal rank by odds within race (1=favorite). Keeps order, not magnitude. ===

    working_df = working_df.with_columns(
        pl.col("dollar_odds").rank(method="ordinal").over(["race_date", "track_code", "race_number"]).alias("rank_in_odds")
    )

    # === Rank fraction that normalizes rank based on race size. ===

    working_df = working_df.with_columns(
        pl.when(pl.col("field_size") > 1)
        .then((1 - (pl.col("rank_in_odds") - 1) / (pl.col("field_size") - 1)))
        .otherwise(pl.lit(1.0))
        .alias("rank_in_odds_frac")
    )

    # ===== Days since the last race. =====

    working_df = working_df.with_columns(
        (pl.col("race_date") - pl.col("last_pp_race_date")).dt.total_days().alias("days_since_last_race")
    )

    # ===== Trainer key for downstream jobs. =====

    working_df = working_df.with_columns(
        pl.concat_str([pl.col("trainer_first_name"), pl.col("trainer_last_name")], separator=" ").alias("trainer_full_name")
    )

    return FeatureFrame(df=working_df)
