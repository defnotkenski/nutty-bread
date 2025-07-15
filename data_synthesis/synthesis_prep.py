import polars as pl
from pathlib import Path
from datasets.sample_horses_schema import COLUMN_TYPES


def train() -> None:
    script_dir = Path(__file__).parents[1] / "datasets" / "sample_horses_v2.csv"
    df = pl.read_csv(script_dir, infer_schema=False)
    df = df.cast(COLUMN_TYPES)

    cardinality = df.select(pl.selectors.string()).select(
        [pl.col(col).n_unique().alias(f"{col}_cardinality") for col in df.select(pl.selectors.string()).columns]
    )

    return cardinality


if __name__ == "__main__":
    train()
