"""
Utility functions and helper methods.

This module contains reusable utility functions that support various operations
throughout the application.
"""

import polars as pl


def cleanup_dataframe(base_polars_df: pl.DataFrame) -> pl.DataFrame:
    # Convert all instances of " " to None across all columns.
    cols_to_replace = ["pace_call_1", "pace_call_2", "trainer_first_name", "owner_full_name"]

    base_polars_df = base_polars_df.with_columns(
        [
            pl.when(pl.col(col) == " ").then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
            for col in cols_to_replace
            if col in base_polars_df.columns
        ]
    )

    # Check for single space " " in all columns.
    # _df_check_1_post = base_polars_df.select([(pl.col("*").exclude("race_date", "last_pp_race_date") == " ").sum()])

    return base_polars_df
