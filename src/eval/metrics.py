import numpy as np
from typing import Literal

from src.data.metadata import RaceMetadata


def top1_hit_from_metadata(preds: np.ndarray, meta: RaceMetadata, target_type: Literal["win", "place", "show"]) -> float:
    k = 1 if target_type == "win" else 2 if target_type == "place" else 3
    hits = 0
    total = len(meta.race_boundaries)

    for i, (start, end) in enumerate(meta.race_boundaries):
        race_preds = preds[start:end]
        pick = race_preds.argmax()

        if k == 1:
            ok = pick == meta.winner_indices[i]
        elif k == 2:
            ok = pick in meta.top2_indices[i]
        else:
            ok = pick in meta.top3_indices[i]

        hits += int(ok)

    return hits / total if total else 0.0
