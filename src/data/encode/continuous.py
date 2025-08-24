import numpy as np


def fit_stats(x_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute column-wise mean/std on x_fit ignoring NaNs.
    Raises if any column has all-missing or zero variance.
    """
    if x_fit.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    mask_fit = np.isnan(x_fit)
    n_obs = (~mask_fit).sum(axis=0)
    if np.any(n_obs == 0):
        idx = np.where(n_obs == 0)[0]
        raise ValueError(f"Continuous columns have all values missing at indices: {idx.tolist()}")

    mean = np.nanmean(x_fit, axis=0).astype(np.float32)
    std = np.nanstd(x_fit, axis=0).astype(np.float32)

    if np.any(std == 0.0):
        idx = np.where(std == 0.0)[0]
        raise ValueError(f"Continuous columns have zero variance (constant) at indices: {idx.tolist()}")

    return mean, std


def transform(x_all: np.ndarray, mean: np.ndarray, std: np.ndarray, sentinel_z: float) -> np.ndarray:
    """
    Standardize observed values and set NaNs to sentinel z-score.
    """
    if x_all.size == 0:
        return x_all.astype(np.float32)

    x = x_all.astype(np.float32)
    mask = np.isnan(x)

    x_filled = np.where(mask, mean, x)
    z = (x_filled - mean) / std
    z[mask] = sentinel_z

    return z.astype(np.float32)
