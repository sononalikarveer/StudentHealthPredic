import pandas as pd
import numpy as np

def create_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced feature engineering for Kaggle Top 10.
    Returns a new dataframe with engineered features.
    """
    X_new = X.copy()
    
    # 1. Missing indicators for features with >5% missingness
    cols_with_missing = ['sleep_duration', 'calorie_expenditure', 'water_intake', 'stress_level', 'sleep_quality', 'physical_activity_level']
    for col in cols_with_missing:
        if col in X_new.columns:
            X_new[f'{col}_is_missing'] = X_new[col].isnull().astype(int)
            
    return X_new
