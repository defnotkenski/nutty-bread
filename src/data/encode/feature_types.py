from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class FeatureMap:
    """
    Lists of feature columns to feed the model.
    """

    continuous: list[str]
    categorical: list[str]
    target: str


@dataclass(frozen=True)
class Preprocessed:
    """
    Tensors and metadata consumed by the trainer/dataset.
    """

    continuous_tensor: torch.Tensor
    categorical_tensor: torch.Tensor
    target_tensor: torch.Tensor
    categorical_cardinalities: list[int]
    race_boundaries: list[tuple[int, int]]
    winner_indices: torch.Tensor
    top2_indices: list[list[int]]
    top3_indices: list[list[int]]
