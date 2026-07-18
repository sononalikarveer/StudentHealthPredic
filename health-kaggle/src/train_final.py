import pandas as pd
import numpy as np
from src.data import DataLoader
from src.features import create_features
from src.models import LGBMWrapper, CatBoostWrapper
from src.validation import get_cv_splits
from src.config import SEED, MODELS_DIR, OOF_DIR, SUBMISSIONS_DIR

def train_and_predict(model_class, model_name, X_train, y_train, X_test, ids_train, ids_test, loader, **params):
    print(f"\n========================================")
    print(f"Final Training {model_name}")
    print(f"========================================")
    
    splits = list(get_cv_splits(X_train, y_train))
    num_classes = len(loader.label_encoder.classes_)
    oof_preds = np.zeros((len(X_train), num_classes))
    test_preds = np.zeros((len(X_test), num_classes))
    
    cat_cols = loader.cat_cols + [c for c in X_train.columns if c not in loader.cat_cols+loader.num_cols and X_train[c].dtype == 'object']
    num_cols = loader.num_cols + [c for c in X_train.columns if c not in loader.cat_cols+loader.num_cols and X_train[c].dtype != 'object']
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"--- Fold {fold+1}/{len(splits)} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model = model_class(num_features=num_cols, cat_features=cat_cols, random_state=SEED+fold, **params)
        model.fit(X_tr, y_tr, X_va, y_va)
        
        oof_preds[val_idx] = model.predict_proba(X_va)
        test_preds += model.predict_proba(X_test) / len(splits)
        
    oof_df = pd.DataFrame(oof_preds, columns=[f"prob_{c}" for c in loader.label_encoder.classes_])
    oof_df['id'] = ids_train.values
    oof_df['target'] = y_train.values
    oof_df.to_csv(OOF_DIR / f'final_oof_{model_name.lower()}.csv', index=False)
    
    test_df = pd.DataFrame(test_preds, columns=[f"prob_{c}" for c in loader.label_encoder.classes_])
    test_df['id'] = ids_test.values
    test_df.to_csv(SUBMISSIONS_DIR / f'final_test_probs_{model_name.lower()}.csv', index=False)
    
    return oof_preds, test_preds

def main():
    loader = DataLoader()
    X_train, y_train, ids_train = loader.load_train()
    X_test, ids_test = loader.load_test()
    
    # Feature Engineering
    X_train_fe = create_features(X_train)
    X_test_fe = create_features(X_test)
    
    # Load tuned parameters if they exist
    lgbm_params = {}
    try:
        lgbm_params = pd.read_csv(MODELS_DIR / 'best_lgbm_params.csv', index_col=0).iloc[:, 0].to_dict()
        print("Loaded Optuna tuned hyperparameters for LightGBM.")
        if 'num_leaves' in lgbm_params: lgbm_params['num_leaves'] = int(lgbm_params['num_leaves'])
        if 'max_depth' in lgbm_params: lgbm_params['max_depth'] = int(lgbm_params['max_depth'])
        if 'min_child_samples' in lgbm_params: lgbm_params['min_child_samples'] = int(lgbm_params['min_child_samples'])
        if 'n_estimators' in lgbm_params: lgbm_params['n_estimators'] = int(lgbm_params['n_estimators'])
    except Exception:
        print("Optuna tuned hyperparameters not found, using defaults for LightGBM.")
        
    train_and_predict(LGBMWrapper, 'LightGBM', X_train_fe, y_train, X_test_fe, ids_train, ids_test, loader, **lgbm_params)
    
    # CatBoost is already super strong, we just use default params but with the new features
    train_and_predict(CatBoostWrapper, 'CatBoost', X_train_fe, y_train, X_test_fe, ids_train, ids_test, loader)
    
    print("\nFinal training completed. Run ensemble.py next.")

if __name__ == "__main__":
    main()
