import pytest
import pandas as pd
import numpy as np
from src.data import DataLoader

def test_data_loader_initialization():
    loader = DataLoader()
    assert loader.id_col == 'id'
    assert loader.target_col == 'health_condition'

def test_load_train_and_schema():
    loader = DataLoader()
    X, y, ids = loader.load_train()
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(ids, pd.Series)
    
    # ID and Target should not be in features
    assert loader.id_col not in X.columns
    assert loader.target_col not in X.columns
    
    # Schema checks
    assert len(loader.cat_cols) > 0
    assert len(loader.num_cols) > 0
    assert len(X.columns) == len(loader.cat_cols) + len(loader.num_cols)
    
    # Target encoding checks
    assert y.dtype == np.int64 or y.dtype == np.int32
    classes = loader.get_classes()
    assert len(classes) > 1

def test_load_test():
    loader = DataLoader()
    loader.load_train() # Need to load train first to discover schema and fit encoder
    X_test, ids_test = loader.load_test()
    
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(ids_test, pd.Series)
    assert loader.id_col not in X_test.columns
    assert loader.target_col not in X_test.columns
    
    assert list(X_test.columns) == loader.num_cols + loader.cat_cols or set(X_test.columns) == set(loader.num_cols + loader.cat_cols)

def test_decode_predictions():
    loader = DataLoader()
    loader.load_train()
    
    # Check that decoding gives back original classes format
    sample_preds = np.array([0, 1])
    decoded = loader.decode_predictions(sample_preds)
    assert len(decoded) == 2
    assert isinstance(decoded[0], str)
