from torch.utils.data import Dataset
from src.data.encode import Preprocessed


class SaddleDataset(Dataset):
    def __init__(self, preprocessed_data: Preprocessed):
        self.continuous = preprocessed_data.continuous_tensor
        self.categorical = preprocessed_data.categorical_tensor
        self.target = preprocessed_data.target_tensor
        self.winner_indices = preprocessed_data.winner_indices

        self.race_boundaries = preprocessed_data.race_boundaries

    def __len__(self):
        return len(self.race_boundaries)

    def __getitem__(self, item_idx):
        start_idx, end_idx = self.race_boundaries[item_idx]

        return (
            {
                "continuous": self.continuous[start_idx:end_idx],  # (num_horses, features)
                "categorical": self.categorical[start_idx:end_idx],  # (num_horses, features)
            },
            self.target[start_idx:end_idx],
            self.winner_indices[item_idx],
        )
