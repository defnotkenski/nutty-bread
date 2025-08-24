import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from typing import List, Tuple


def fit_encoder(x_fit: np.ndarray) -> Tuple[OrdinalEncoder, List[List[str]]]:
    """
    Fit an OrdinalEncoder on training rows.
    Assumes nulls already replaced (e.g., '<UNK>').
    Returns the fitted encoder and learned category lists (as strings).
    """
    if x_fit.size == 0:
        # No categorical features scenario; return a trivially fitted encoder
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
        enc.fit(np.empty((1, 0), dtype=object))
        return enc, []

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    enc.fit(x_fit)

    categories = [list(map(str, cats)) for cats in enc.categories_]

    return enc, categories


def transform(x_all: np.ndarray, enc: OrdinalEncoder) -> Tuple[np.ndarray, List[int]]:
    """
    Transform all rows with a fitted encoder; unknown -> 0 via +1 shift.
    Returns encoded array (int64) and per-column cardinalities.
    Assumes x_all nulls already replaced (e.g., '<UNK>').
    """
    if x_all.size == 0:
        return x_all.astype(np.int64), []

    x_enc = enc.transform(x_all)
    x_enc = (x_enc + 1).astype(np.int64)  # shift so 0 is reserved for unknown

    cardinalities = [len(cats) + 1 for cats in enc.categories_]

    return x_enc, cardinalities
