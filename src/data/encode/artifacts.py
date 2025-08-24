from pathlib import Path
import json
from .feature_types import FeatureMap


def save_preprocessors(
    out_dir: Path,
    cont_core: list[str],
    mean: list[float],
    std: list[float],
    sentinel_z: float,
    cat_cols: list[str],
    cat_categories: list[list[str]],
    fmap: FeatureMap,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Continuous stats
    cont_payload = {
        "columns": cont_core,
        "sentinel_z": float(sentinel_z),
        "mean": mean,
        "std": std,
    }
    (out_dir / "continuous_stats.json").write_text(json.dumps(cont_payload, indent=2))

    # Categorical vocab
    cat_payload = {
        "columns": cat_cols,
        "categories": cat_categories,
        "unknown_index": 0,  # 0 reserved since we shift +1
    }
    (out_dir / "categorical_vocab.json").write_text(json.dumps(cat_payload, indent=2))

    # Feature map snapshot
    fmap_payload = {
        "continuous": fmap.continuous,
        "categorical": fmap.categorical,
        "target": fmap.target,
    }

    (out_dir / "feature_map.json").write_text(json.dumps(fmap_payload, indent=2))
