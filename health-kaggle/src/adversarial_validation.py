import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.data import DataLoader
from src.models import LGBMWrapper
from src.config import SEED

def run_adversarial_validation():
    print("Running Adversarial Validation...")
    loader = DataLoader()
    X_train, _, _ = loader.load_train()
    X_test, _ = loader.load_test()
    
    # Label train as 0, test as 1
    X_train['adv_target'] = 0
    X_test['adv_target'] = 1
    
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_adv = X_combined['adv_target']
    X_adv = X_combined.drop(columns=['adv_target'])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    scores = []
    feature_importances = np.zeros(len(X_adv.columns))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_adv, y_adv)):
        X_tr, y_tr = X_adv.iloc[train_idx], y_adv.iloc[train_idx]
        X_va, y_va = X_adv.iloc[val_idx], y_adv.iloc[val_idx]
        
        # Use our LGBMWrapper
        model = LGBMWrapper(num_features=loader.num_cols, cat_features=loader.cat_cols, random_state=SEED+fold)
        model.fit(X_tr, y_tr, X_va, y_va)
        
        preds = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, preds)
        scores.append(score)
        
        # Aggregate importance
        feature_importances += model.get_feature_importance(X_va) / 5
        
    mean_auc = np.mean(scores)
    print(f"\nAdversarial Validation ROC-AUC: {mean_auc:.5f} ± {np.std(scores):.5f}")
    
    if mean_auc > 0.6:
        print("Warning: Train and Test distributions show material differences.")
    else:
        print("Train and Test distributions are similar (ROC-AUC ~ 0.5).")
        
    print("\nTop 5 Features driving train/test separation:")
    fi_df = pd.DataFrame({
        'Feature': X_adv.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    print(fi_df.head())

if __name__ == "__main__":
    run_adversarial_validation()
