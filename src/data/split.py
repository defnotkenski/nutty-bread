from dataclasses import dataclass
import polars as pl


@dataclass(frozen=True)
class RaceMetadata:
    """Race-wise indexing metadata for batching and evaluation."""

    race_boundaries: list[tuple[int, int]]  # [(start, end), ...] in dataframe order
    winner_indices: list[int]  # argmax(target) per race in same order


def build_race_metadata(df: pl.DataFrame, target_col: str = "target") -> RaceMetadata:
    """
    Compute race boundaries and winner indices from a labeled frame.
    """

    df = df.sort(["race_date", "track_code", "race_number"])
    groups = df.group_by(["race_date", "track_code", "race_number"], maintain_order=True)

    # ======
    # Boundaries
    # ======

    race_boundaries: list[tuple[int, int]] = []
    start = 0

    for _, g in groups:
        n = len(g)
        race_boundaries.append((start, start + n))
        start += n

    # ======
    # Winner indices
    # ======

    winner_indices: list[int] = []

    for _, g in groups:
        arr = g[target_col].to_numpy()
        winner_indices.append(int(arr.argmax()) if len(arr) > 0 else 0)

    return RaceMetadata(race_boundaries=race_boundaries, winner_indices=winner_indices)
