from dataclasses import dataclass
import polars as pl


@dataclass(frozen=True)
class RaceMetadata:
    """Race-wise indexing metadata for batching and evaluation."""

    race_boundaries: list[tuple[int, int]]  # [(start, end), ...] in dataframe order
    winner_indices: list[int]
    top2_indices: list[list[int]]
    top3_indices: list[list[int]]


def build_race_metadata(df: pl.DataFrame, target_col: str = "target") -> RaceMetadata:
    """
    Compute race boundaries, winner, and top-k indices from a labeled frame.
    """

    df = df.sort(["race_date", "track_code", "race_number"])
    groups = df.group_by(["race_date", "track_code", "race_number"], maintain_order=True)

    # === Boundaries ===

    race_boundaries: list[tuple[int, int]] = []

    start = 0
    for _, g in groups:
        n = len(g)
        race_boundaries.append((start, start + n))
        start += n

    # === Winner & top-k indices ===

    winner_indices: list[int] = []
    top2_indices: list[list[int]] = []
    top3_indices: list[list[int]] = []

    for _, g in groups:
        assert len(g) != 0, f"Length of group is 0."

        # Choose the horse with the best (lowest) official_final_position.
        # Fill nulls with a large sentinel to avoid selecting missing values.
        pos = g["official_final_position"].fill_null(1_000_000)
        idx = int(pos.arg_min())
        winner_indices.append(idx)

        # All horses within top-2 and top-3
        arr = pos.to_numpy()
        top2_indices.append([int(i) for i, v in enumerate(arr) if v <= 2])
        top3_indices.append([int(i) for i, v in enumerate(arr) if v <= 3])

    return RaceMetadata(
        race_boundaries=race_boundaries, winner_indices=winner_indices, top2_indices=top2_indices, top3_indices=top3_indices
    )
