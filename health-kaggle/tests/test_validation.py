import pytest
import numpy as np
import pandas as pd
from src.validation import get_cv_splits
from src.data import DataLoader

def test_oof_coverage():
    loader = DataLoader()
    X, y, ids = loader.load_train()
    
    n_samples = len(X)
    oof_predictions = np.zeros(n_samples)
    oof_coverage = np.zeros(n_samples)
    
    splits = list(get_cv_splits(X, y))
    
    # Check that we got N_FOLDS splits
    from src.config import N_FOLDS
    assert len(splits) == N_FOLDS
    
    for train_idx, val_idx in splits:
        # Simulate making predictions
        oof_predictions[val_idx] = 1
        oof_coverage[val_idx] += 1
        
        # Ensure no overlap between train and val
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        
    # Check that every row was validated exactly once
    assert np.all(oof_coverage == 1)
    
    # Row order matches implicitly because we use val_idx to assign predictions
    # to a zeros array of size n_samples initialized in the original order.
