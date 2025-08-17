from dataclasses import dataclass
from pathlib import Path
import polars as pl
from datasets.schemas.sample_horses_schema import COLUMN_TYPES
from src.commons.utils import cleanup_dataframe


@dataclass(frozen=True)
class RawFrame:
    """
    Container for the raw, typed, cleaned Polars Dataframe from disk.
    """

    df: pl.DataFrame


def load_raw_frame(path: Path) -> RawFrame:
    """
    Load and normalize the raw races CSV into a typed, cleaned Polars Dataframe.
    """

    df = pl.read_csv(path, infer_schema=False)
    df = df.with_columns(pl.col("race_date").str.to_datetime())
    df = df.with_columns(pl.col("last_pp_race_date").str.to_datetime())
    df = cleanup_dataframe(base_polars_df=df)
    df = df.cast(COLUMN_TYPES)

    # Keep ordering stable for downstream grouping
    # Last key ensures predictable horse order

    df = df.sort(["race_date", "track_code", "race_number", "dollar_odds"])

    return RawFrame(df=df)
