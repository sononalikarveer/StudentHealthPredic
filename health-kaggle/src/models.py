import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from src.data import DataLoader

class CatBoostWrapper:
    def __init__(self, cat_features, num_features=None, random_state=42, **kwargs):
        self.cat_features = cat_features
        
        cb_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'random_seed': random_state,
            'auto_class_weights': 'Balanced',
            'verbose': False,
            'early_stopping_rounds': 100
        }
        cb_params.update(kwargs)
        
        self.model = CatBoostClassifier(**cb_params)
        
    def fit(self, X_train, y_train, X_val, y_val):
        # Explicitly fill NaNs with a string for CatBoost categorical features
        X_train_cb = X_train.copy()
        X_val_cb = X_val.copy()
        for col in self.cat_features:
            X_train_cb[col] = X_train_cb[col].fillna('Missing')
            X_val_cb[col] = X_val_cb[col].fillna('Missing')
            
        self.model.fit(
            X_train_cb, y_train,
            cat_features=self.cat_features,
            eval_set=(X_val_cb, y_val)
        )
        return self
        
    def predict_proba(self, X):
        X_cb = X.copy()
        for col in self.cat_features:
            X_cb[col] = X_cb[col].fillna('Missing')
        return self.model.predict_proba(X_cb)
        
    def get_feature_importance(self, X):
        return self.model.get_feature_importance()

class LGBMWrapper:
    def __init__(self, num_features, cat_features, random_state=42, **kwargs):
        self.num_features = num_features
        self.cat_features = cat_features
        
        numeric_transformer = SimpleImputer(strategy='median')
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_features),
                ('cat', categorical_transformer, self.cat_features)
            ],
            remainder='passthrough'
        )
        
        # Set defaults that can be overridden by kwargs
        lgbm_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'class_weight': 'balanced',
            'random_state': random_state,
            'verbose': -1,
            'early_stopping_rounds': 100
        }
        lgbm_params.update(kwargs)
        
        self.model = LGBMClassifier(**lgbm_params)
        
    def fit(self, X_train, y_train, X_val, y_val):
        X_train_trans = self.preprocessor.fit_transform(X_train)
        X_val_trans = self.preprocessor.transform(X_val)
        
        self.model.fit(
            X_train_trans, y_train,
            eval_set=[(X_val_trans, y_val)],
            callbacks=[] # LightGBM early stopping is configured in init params for recent versions
        )
        return self
        
    def predict_proba(self, X):
        X_trans = self.preprocessor.transform(X)
        return self.model.predict_proba(X_trans)
        
    def get_feature_importance(self, X):
        return self.model.feature_importances_

class XGBWrapper:
    def __init__(self, num_features, cat_features, random_state=42):
        self.num_features = num_features
        self.cat_features = cat_features
        
        numeric_transformer = SimpleImputer(strategy='median')
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ]
        )
        
        self.model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=random_state,
            early_stopping_rounds=100,
            eval_metric='mlogloss',
            # XGB doesn't have class_weight='balanced' natively for multi-class; we will rely on LightGBM/CatBoost primarily 
            # or we could manually compute sample_weight. We'll leave it without weights for diversity unless necessary.
        )
        
    def fit(self, X_train, y_train, X_val, y_val):
        X_train_trans = self.preprocessor.fit_transform(X_train)
        X_val_trans = self.preprocessor.transform(X_val)
        
        self.model.fit(
            X_train_trans, y_train,
            eval_set=[(X_val_trans, y_val)],
            verbose=False
        )
        return self
        
    def predict_proba(self, X):
        X_trans = self.preprocessor.transform(X)
        return self.model.predict_proba(X_trans)
        
    def get_feature_importance(self, X):
        return self.model.feature_importances_
