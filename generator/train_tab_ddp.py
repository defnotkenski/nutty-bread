import polars as pl
from pathlib import Path


def train() -> None:
    script_dir = Path(__file__).parents[1] / "datasets" / "sample_horses_v2.csv"
    polars_csv = pl.read_csv(script_dir, infer_schema=False)

    return


if __name__ == "__main__":
    train()
