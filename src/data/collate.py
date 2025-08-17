from torch import Tensor
import torch


def collate_races(batch) -> tuple[dict, Tensor, Tensor, Tensor]:
    """
    Collate function to pad races to same length and create attention masks.
    Args:
        List of tuples (input_dict, targets, winner_idx) from SAINTDataset
    """

    # Separate inputs and targets
    inputs = [item[0] for item in batch]  # List of dicts
    targets = [item[1] for item in batch]  # list of target tensors
    winner_indices = [item[2] for item in batch]  # List of winner indices

    # Find max number of horses in this batch
    max_horses = max(inp["continuous"].shape[0] for inp in inputs)
    batch_size = len(batch)

    # Get feature dimensions
    cont_features = inputs[0]["continuous"].shape[1]
    cat_features = inputs[0]["categorical"].shape[1]

    # Initialize padded tensors
    padded_continuous = torch.zeros(batch_size, max_horses, cont_features)
    padded_categorical = torch.zeros(batch_size, max_horses, cat_features, dtype=torch.long)
    padded_targets = torch.full((batch_size, max_horses), -100.0)

    attention_mask = torch.zeros(batch_size, max_horses)

    # Fill in the actual data
    for i, (inp, target) in enumerate(zip(inputs, targets)):
        num_horses = inp["continuous"].shape[0]

        padded_continuous[i, :num_horses] = inp["continuous"]
        padded_categorical[i, :num_horses] = inp["categorical"]
        padded_targets[i, :num_horses] = target
        attention_mask[i, :num_horses] = 1

    batched_input = {
        "continuous": padded_continuous,
        "categorical": padded_categorical,
    }

    winner_indices_tensor = torch.tensor(winner_indices, dtype=torch.long)

    return batched_input, padded_targets, attention_mask, winner_indices_tensor
