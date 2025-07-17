import polars as pl
from pathlib import Path


def train() -> None:
    script_dir_samples = Path(__file__).parent / "samples.csv"
    script_dir_ground = Path(
        "/Users/kennylao/PycharmProjects/neural-learning/custom/models/saint_transformer/sample_horses_v2_post.csv"
    )

    _sample_df = pl.read_csv(script_dir_samples, infer_schema=True)
    _sample_ground_df = pl.read_csv(script_dir_ground, infer_schema=True)

    return


if __name__ == "__main__":
    train()
