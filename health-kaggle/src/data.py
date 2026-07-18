import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict
from src.config import TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH

class DataLoader:
    def __init__(self):
        self.id_col = 'id'
        self.target_col = 'health_condition'
        self.cat_cols = []
        self.num_cols = []
        self.label_encoder = LabelEncoder()
        
    def load_train(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Loads train data and splits into X, y, and id."""
        df = pd.read_csv(TRAIN_PATH)
        
        y_str = df[self.target_col]
        # Fit label encoder
        self.label_encoder.fit(y_str)
        y = pd.Series(self.label_encoder.transform(y_str), name=self.target_col)
        
        X = df.drop(columns=[self.id_col, self.target_col])
        self._discover_schema(X)
        
        # Cast categorical columns to string explicitly (keeps NaN as NaN but usually handles it safely)
        for col in self.cat_cols:
            X[col] = X[col].astype(str)
            X.loc[X[col] == 'nan', col] = np.nan
            
        return X, y, df[self.id_col]

    def load_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads test data, returns features and IDs."""
        df = pd.read_csv(TEST_PATH)
        X = df.drop(columns=[self.id_col])
        
        for col in self.cat_cols:
            X[col] = X[col].astype(str)
            X.loc[X[col] == 'nan', col] = np.nan
            
        return X, df[self.id_col]
        
    def _discover_schema(self, X: pd.DataFrame):
        """Dynamically find numerical and categorical columns."""
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    def get_classes(self) -> np.ndarray:
        return self.label_encoder.classes_

    def decode_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y_pred)

    def get_sample_submission(self) -> pd.DataFrame:
        return pd.read_csv(SAMPLE_SUB_PATH)
