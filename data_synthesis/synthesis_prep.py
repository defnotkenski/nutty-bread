import polars as pl
from pathlib import Path


def train() -> None:
    script_dir = Path(__file__).parent / "samples.csv"
    _df = pl.read_csv(script_dir, infer_schema=True)

    return


if __name__ == "__main__":
    train()
