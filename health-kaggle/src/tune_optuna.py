import argparse
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from src.data import DataLoader
from src.features import create_features
from src.models import LGBMWrapper, CatBoostWrapper
from src.config import SEED, MODELS_DIR

def objective_lgbm(trial, X, y, cat_cols, num_cols):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(skf.split(X, y))
    
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = LGBMWrapper(num_features=num_cols, cat_features=cat_cols, random_state=SEED, **params)
    model.fit(X_tr, y_tr, X_va, y_va)
    
    preds_proba = model.predict_proba(X_va)
    preds = np.argmax(preds_proba, axis=1)
    
    return balanced_accuracy_score(y_va, preds)

def objective_catboost(trial, X, y, cat_cols):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'iterations': trial.suggest_int('iterations', 200, 500)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(skf.split(X, y))
    
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = CatBoostWrapper(cat_features=cat_cols, random_state=SEED, **params)
    model.fit(X_tr, y_tr, X_va, y_va)
    
    preds_proba = model.predict_proba(X_va)
    preds = np.argmax(preds_proba, axis=1)
    
    return balanced_accuracy_score(y_va, preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lgbm', 'catboost'], default='lgbm')
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    
    print(f"Starting Optuna tuning for {args.model.upper()} with Engineered Features...")
    loader = DataLoader()
    X_train, y_train, ids_train = loader.load_train()
    
    X_train_fe = create_features(X_train)
    
    new_cat_cols = loader.cat_cols + [c for c in X_train_fe.columns if c not in X_train.columns and X_train_fe[c].dtype == 'object']
    new_num_cols = loader.num_cols + [c for c in X_train_fe.columns if c not in X_train.columns and X_train_fe[c].dtype != 'object']
    
    study = optuna.create_study(direction='maximize', study_name=f'{args.model}_tuning')
    
    if args.model == 'lgbm':
        study.optimize(lambda trial: objective_lgbm(trial, X_train_fe, y_train, new_cat_cols, new_num_cols), n_trials=args.trials)
        save_path = MODELS_DIR / 'best_lgbm_params.csv'
    else:
        study.optimize(lambda trial: objective_catboost(trial, X_train_fe, y_train, new_cat_cols), n_trials=args.trials)
        save_path = MODELS_DIR / 'best_catboost_params.csv'
        
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Balanced Accuracy on 1 Fold): {trial.value:.5f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    pd.Series(trial.params).to_csv(save_path)
    print(f"Saved params to {save_path}")

if __name__ == "__main__":
    main()
