import os
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
import pickle

from src.data import DataLoader
from src.validation import get_cv_splits
from src.models import CatBoostWrapper, LGBMWrapper, XGBWrapper
from src.config import SEED, MODELS_DIR, OOF_DIR, SUBMISSIONS_DIR, REPORTS_DIR

def train_model(model_class, model_name, X_train, y_train, X_test, ids_train, ids_test, loader):
    print(f"\n{'='*40}\nTraining {model_name}\n{'='*40}")
    
    oof_preds = np.zeros((len(X_train), len(loader.get_classes())))
    test_preds = np.zeros((len(X_test), len(loader.get_classes())))
    
    splits = list(get_cv_splits(X_train, y_train))
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"--- Fold {fold+1}/{len(splits)} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        if model_name == 'CatBoost':
            model = model_class(cat_features=loader.cat_cols, random_state=SEED+fold)
        else:
            model = model_class(num_features=loader.num_cols, cat_features=loader.cat_cols, random_state=SEED+fold)
            
        model.fit(X_tr, y_tr, X_va, y_va)
        
        # OOF predict
        val_preds_proba = model.predict_proba(X_va)
        oof_preds[val_idx] = val_preds_proba
        
        # Test predict
        test_preds += model.predict_proba(X_test) / len(splits)
        
        # Fold metrics
        val_preds_labels = np.argmax(val_preds_proba, axis=1)
        score = balanced_accuracy_score(y_va, val_preds_labels)
        scores.append(score)
        print(f"Fold {fold+1} Balanced Accuracy: {score:.5f}")
        
    print(f"\n{model_name} CV Mean Balanced Accuracy: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    
    # Global OOF metrics
    oof_labels = np.argmax(oof_preds, axis=1)
    global_score = balanced_accuracy_score(y_train, oof_labels)
    print(f"{model_name} Global OOF Balanced Accuracy: {global_score:.5f}")
    
    print("\nClassification Report (OOF):")
    target_names = [str(c) for c in loader.get_classes()]
    print(classification_report(y_train, oof_labels, target_names=target_names))
    
    print("\nConfusion Matrix (OOF):")
    cm = confusion_matrix(y_train, oof_labels)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in target_names], columns=[f"Pred {c}" for c in target_names])
    print(cm_df)
    
    # Save OOF
    oof_df = pd.DataFrame(oof_preds, columns=[f"prob_{c}" for c in target_names])
    oof_df['id'] = ids_train.values
    oof_df.to_csv(OOF_DIR / f'oof_{model_name.lower()}.csv', index=False)
    
    # Save Test probabilities
    test_df = pd.DataFrame(test_preds, columns=[f"prob_{c}" for c in target_names])
    test_df['id'] = ids_test.values
    test_df.to_csv(SUBMISSIONS_DIR / f'test_probs_{model_name.lower()}.csv', index=False)
    
    return oof_preds, test_preds, np.mean(scores), np.std(scores)

def main():
    loader = DataLoader()
    X_train, y_train, ids_train = loader.load_train()
    X_test, ids_test = loader.load_test()
    
    results = []
    
    # Train CatBoost
    oof_cb, test_cb, mean_cb, std_cb = train_model(
        CatBoostWrapper, 'CatBoost', 
        X_train, y_train, X_test, ids_train, ids_test, loader
    )
    results.append({'Model': 'CatBoost', 'CV Mean': mean_cb, 'CV Std': std_cb})
    
    # Generate valid submission for CatBoost
    sub = loader.get_sample_submission()
    test_preds_labels = np.argmax(test_cb, axis=1)
    sub[loader.target_col] = loader.decode_predictions(test_preds_labels)
    sub.to_csv(SUBMISSIONS_DIR / 'submission_catboost.csv', index=False)
    print("\nSaved initial CatBoost submission to submission_catboost.csv")
    
    # Train LightGBM
    oof_lgb, test_lgb, mean_lgb, std_lgb = train_model(
        LGBMWrapper, 'LightGBM', 
        X_train, y_train, X_test, ids_train, ids_test, loader
    )
    results.append({'Model': 'LightGBM', 'CV Mean': mean_lgb, 'CV Std': std_lgb})
    
    # Train XGBoost
    oof_xgb, test_xgb, mean_xgb, std_xgb = train_model(
        XGBWrapper, 'XGBoost', 
        X_train, y_train, X_test, ids_train, ids_test, loader
    )
    results.append({'Model': 'XGBoost', 'CV Mean': mean_xgb, 'CV Std': std_xgb})
    
    # Save experiment report
    res_df = pd.DataFrame(results)
    res_df.to_markdown(REPORTS_DIR / 'baseline_experiments.md', index=False)
    print("\nBaseline experiments finished. Summary saved to outputs/reports/baseline_experiments.md")

if __name__ == "__main__":
    main()
