from dataclasses import dataclass
from typing import Literal
import polars as pl


@dataclass(frozen=True)
class LabeledFrame:
    """DataFrame carrying the training label/target column."""

    df: pl.DataFrame
    target_col: str


def add_target(df: pl.DataFrame, target_type: Literal["win", "show", "place"]) -> LabeledFrame:
    """
    Add a binary target column according to the task definition.
    """

    target_col = "target"

    if target_type == "win":
        df = df.with_columns((pl.col("official_final_position") == 1).cast(pl.Int64).alias(target_col))
    elif target_type == "show":
        df = df.with_columns((pl.col("official_final_position") <= 2).cast(pl.Int64).alias(target_col))
    else:
        df = df.with_columns((pl.col("official_final_position") <= 3).cast(pl.Int64).alias(target_col))

    return LabeledFrame(df=df, target_col="target")
