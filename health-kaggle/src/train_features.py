import pandas as pd
import numpy as np
from src.data import DataLoader
from src.models import LGBMWrapper
from src.validation import get_cv_splits
from src.features import create_features
from src.config import SEED

def main():
    print("Evaluating Feature Engineering with LightGBM...")
    loader = DataLoader()
    X_train, y_train, ids_train = loader.load_train()
    
    # Apply feature engineering
    X_train_fe = create_features(X_train)
    
    # Re-evaluate features using the NEW columns to identify changes
    new_cat_cols = loader.cat_cols + [c for c in X_train_fe.columns if c not in X_train.columns and X_train_fe[c].dtype == 'object']
    new_num_cols = loader.num_cols + [c for c in X_train_fe.columns if c not in X_train.columns and X_train_fe[c].dtype != 'object']
    
    splits = list(get_cv_splits(X_train_fe, y_train))
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"--- Fold {fold+1}/{len(splits)} ---")
        X_tr, y_tr = X_train_fe.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train_fe.iloc[val_idx], y_train.iloc[val_idx]
        
        model = LGBMWrapper(num_features=new_num_cols, cat_features=new_cat_cols, random_state=SEED+fold)
        model.fit(X_tr, y_tr, X_va, y_va)
        
        from sklearn.metrics import balanced_accuracy_score
        preds_proba = model.predict_proba(X_va)
        preds = np.argmax(preds_proba, axis=1)
        score = balanced_accuracy_score(y_va, preds)
        print(f"Fold {fold+1} Balanced Accuracy: {score:.5f}")
        scores.append(score)
        
    print(f"\nLightGBM with FE Mean Balanced Accuracy: {np.mean(scores):.5f} ± {np.std(scores):.5f}")

if __name__ == "__main__":
    main()
