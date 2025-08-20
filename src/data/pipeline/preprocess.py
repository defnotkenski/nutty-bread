from pathlib import Path
from typing import Literal
from src.data.ingest import load_raw_frame
from src.data.features.enrich import build_prediction_safe_features, build_results_derived_features
from src.data.features.lags import add_lags
from src.data.features.targets import add_target
from src.data.encode import encode_to_tensors, Preprocessed
from src.data.config.feature_map import build_feature_map_from_yaml


def preprocess_df(path: Path, target_type: Literal["win", "show", "place"]) -> Preprocessed:
    """Full preprocessing pipeline: CSV → features → label → tensors + race metadata."""
    raw = load_raw_frame(path).df

    predict_safe_features = build_prediction_safe_features(raw).df
    results_derived_features = build_results_derived_features(predict_safe_features).df

    # yaml_path = Path.cwd() / "src" / "data" / "config" / "model_features.yaml"

    # yaml_features = load_yaml_features(yaml_path)
    # lag_feature_names = yaml_features.get("current_horse_lags", []) or []
    # required_lag_bases = [name[len("recent_0_") :] for name in lag_feature_names if name.startswith("recent_0_")]

    augmented = add_lags(results_derived_features, lookup_df=None).df
    labeled = add_target(augmented, target_type=target_type).df

    yaml_path = Path.cwd() / "src" / "data" / "config" / "model_features.yaml"
    fmap = build_feature_map_from_yaml(labeled, yaml_path=yaml_path, include_opponents=False)

    return encode_to_tensors(labeled, fmap)
