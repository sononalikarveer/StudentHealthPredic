import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from src.data import DataLoader
from src.features import create_features
from src.models import CatBoostWrapper, LGBMWrapper
from src.config import SEED, MODELS_DIR, OOF_DIR, SUBMISSIONS_DIR, DATA_DIR

def main():
    print("========================================")
    print("Training Final Models with Pseudo-Labels")
    print("========================================")
    
    loader = DataLoader()
    X_train, y_train, ids_train = loader.load_train()
    X_test, ids_test = loader.load_test()
    
    pseudo_df = pd.read_csv(DATA_DIR / 'pseudo_labels.csv')
    X_pseudo = pseudo_df.drop(columns=['target'])
    y_pseudo = pseudo_df['target']
    
    # Feature engineering
    X_train_fe = create_features(X_train)
    X_test_fe = create_features(X_test)
    X_pseudo_fe = create_features(X_pseudo)
    
    # Find new categoricals
    new_cat_cols = loader.cat_cols + [c for c in X_train_fe.columns if c not in X_train.columns and X_train_fe[c].dtype == 'object']
    new_num_cols = loader.num_cols + [c for c in X_train_fe.columns if c not in X_train.columns and X_train_fe[c].dtype != 'object']
    
    # Tuned LightGBM parameters
    lgb_params = pd.read_csv(MODELS_DIR / 'best_lgbm_params.csv', index_col=0).iloc[:, 0].to_dict()
    int_cols = ['num_leaves', 'max_depth', 'min_child_samples', 'n_estimators']
    for c in int_cols:
        if c in lgb_params:
            lgb_params[c] = int(lgb_params[c])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    oof_preds_cb = np.zeros((len(X_train), 3))
    oof_preds_lgb = np.zeros((len(X_train), 3))
    
    test_preds_cb = np.zeros((len(X_test), 3))
    test_preds_lgb = np.zeros((len(X_test), 3))
    
    print("\nTraining LightGBM with Pseudo-Labels...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_fe, y_train)):
        print(f"--- Fold {fold+1}/5 ---")
        
        # Original train subset
        X_tr_orig, y_tr_orig = X_train_fe.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train_fe.iloc[val_idx], y_train.iloc[val_idx]
        
        # Augment training set with pseudo-labels
        X_tr = pd.concat([X_tr_orig, X_pseudo_fe], axis=0, ignore_index=True)
        y_tr = pd.concat([y_tr_orig, pd.Series(y_pseudo.values)], axis=0, ignore_index=True)
        
        # Train LightGBM
        model_lgb = LGBMWrapper(num_features=new_num_cols, cat_features=new_cat_cols, random_state=SEED+fold, **lgb_params)
        model_lgb.fit(X_tr, y_tr, X_va, y_va)
        
        oof_preds_lgb[val_idx] = model_lgb.predict_proba(X_va)
        test_preds_lgb += model_lgb.predict_proba(X_test_fe) / 5
        
    print("\nTraining CatBoost with Pseudo-Labels...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_fe, y_train)):
        print(f"--- Fold {fold+1}/5 ---")
        
        X_tr_orig, y_tr_orig = X_train_fe.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train_fe.iloc[val_idx], y_train.iloc[val_idx]
        
        X_tr = pd.concat([X_tr_orig, X_pseudo_fe], axis=0, ignore_index=True)
        y_tr = pd.concat([y_tr_orig, pd.Series(y_pseudo.values)], axis=0, ignore_index=True)
        
        # Train CatBoost (Default params)
        model_cb = CatBoostWrapper(cat_features=new_cat_cols, random_state=SEED+fold)
        model_cb.fit(X_tr, y_tr, X_va, y_va)
        
        oof_preds_cb[val_idx] = model_cb.predict_proba(X_va)
        test_preds_cb += model_cb.predict_proba(X_test_fe) / 5
        
    # Save OOF
    class_names = loader.label_encoder.classes_
    prob_cols = [f'prob_{c}' for c in class_names]
    
    oof_cb_df = pd.DataFrame(oof_preds_cb, columns=prob_cols)
    oof_cb_df['id'] = ids_train
    oof_cb_df['target'] = y_train.values
    oof_cb_df.to_csv(OOF_DIR / 'pseudo_oof_catboost.csv', index=False)
    
    oof_lgb_df = pd.DataFrame(oof_preds_lgb, columns=prob_cols)
    oof_lgb_df['id'] = ids_train
    oof_lgb_df['target'] = y_train.values
    oof_lgb_df.to_csv(OOF_DIR / 'pseudo_oof_lightgbm.csv', index=False)
    
    # Save Test
    test_cb_df = pd.DataFrame(test_preds_cb, columns=prob_cols)
    test_cb_df['id'] = ids_test
    test_cb_df.to_csv(SUBMISSIONS_DIR / 'pseudo_test_probs_catboost.csv', index=False)
    
    test_lgb_df = pd.DataFrame(test_preds_lgb, columns=prob_cols)
    test_lgb_df['id'] = ids_test
    test_lgb_df.to_csv(SUBMISSIONS_DIR / 'pseudo_test_probs_lightgbm.csv', index=False)
    
    print("\nPseudo-label training completed. Run ensemble_pseudo.py next.")

if __name__ == "__main__":
    main()
