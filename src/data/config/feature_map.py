from pathlib import Path
import polars as pl
import yaml
from src.data.encode import FeatureMap

DEFAULT_SECTIONS = ("current_race", "current_horse", "current_horse_lags")
OPPONENT_SECTIONS = ("other_horse", "other_horse_lags")


def _load_yaml_features(yaml_path: Path) -> dict:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Feature YAML at {yaml_path} must be a mapping of sections.")

    return data


def build_feature_map_from_yaml(
    df: pl.DataFrame,
    yaml_path: Path,
    include_opponents: bool,
    target: str = "target",
) -> FeatureMap:
    """
    Build FeatureMap by intersecting YAML-allowed columns with the DataFrame.
    """
    df = df.sort(["race_date", "track_code", "race_number"])

    features_dict = _load_yaml_features(yaml_path)
    sections = list(DEFAULT_SECTIONS)
    if include_opponents:
        sections += list(OPPONENT_SECTIONS)

    allowed: list[str] = []
    for sec in sections:
        block = features_dict.get(sec, [])
        allowed.extend(block)

    # Exclude target and duplicates
    allowed = [c for c in dict.fromkeys(allowed) if c != target]

    # Check presence
    missing = [c for c in allowed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing YAML features not in DataFrame: {missing}")

    present = [c for c in allowed if c in df.columns]

    # Split by dtype based on df
    numeric_cols = set(df.select(pl.selectors.numeric()).columns)
    string_cols = set(df.select(pl.selectors.string()).columns)

    continuous = [c for c in present if c in numeric_cols]
    categorical = [c for c in present if c in string_cols]

    return FeatureMap(continuous=continuous, categorical=categorical, target=target)
