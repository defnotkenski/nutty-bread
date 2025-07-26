import torch
from collections import deque
import random


class EBTReplayBuffer:
    """
    Replay buffer for storing and sampling previous predictions during EBT training.
    """

    def __init__(self, max_size: int, sample_percent: float):
        self.max_size = max_size
        self.sample_percent = sample_percent
        self.buffer = deque(maxlen=max_size)

    def add(self, predictions: torch.Tensor):
        """Add predictions to buffer. predictions: [batch_size, horse_len]"""
        # Store each race prediction separately
        for race_pred in predictions:
            self.buffer.append(race_pred.detach().clone())

    def sample(self, batch_size: int, horse_len: int, device: torch.device):
        """Sample predictions from buffer to replace part of current batch."""
        if len(self.buffer) == 0:
            return None

        sample_size = int(batch_size * self.sample_percent)
        if sample_size == 0:
            return None

        sampled_predictions = random.sample(list(self.buffer), min(sample_size, len(self.buffer)))

        # Pad/truncate to match horse_len and stack
        processed_samples = []
        for pred in sampled_predictions:
            if len(pred) > horse_len:
                processed_samples.append(pred[:horse_len])
            else:
                padded = torch.zeros(horse_len, device=device)
                padded[: len(pred)] = pred
                processed_samples.append(padded)

        return torch.stack(processed_samples)

    def __len__(self):
        return len(self.buffer)
