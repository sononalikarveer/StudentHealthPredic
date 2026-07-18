import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Iterator, Tuple
from src.config import N_FOLDS, SEED

def get_cv_splits(X: pd.DataFrame, y: pd.Series, n_splits: int = N_FOLDS, random_state: int = SEED) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns an iterator of (train_idx, val_idx) using StratifiedKFold.
    Ensures that class distributions are maintained across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(X, y)
