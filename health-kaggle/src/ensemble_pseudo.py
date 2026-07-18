import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src.config import OOF_DIR, SUBMISSIONS_DIR, SEED
def main():
    print("Ensembling final pseudo-labeled models using Stacking (Logistic Regression)...")
    oof_cb = pd.read_csv(OOF_DIR / 'pseudo_oof_catboost.csv')
    oof_lgb = pd.read_csv(OOF_DIR / 'pseudo_oof_lightgbm.csv')
    test_cb = pd.read_csv(SUBMISSIONS_DIR / 'pseudo_test_probs_catboost.csv')
    test_lgb = pd.read_csv(SUBMISSIONS_DIR / 'pseudo_test_probs_lightgbm.csv')
    y_true = oof_cb['target'].values
    prob_cols = [c for c in oof_cb.columns if c.startswith('prob_')]
    classes = [c.replace('prob_', '') for c in prob_cols]
    # Concatenate OOF probabilities
    cb_probs = oof_cb[prob_cols].values
    lgb_probs = oof_lgb[prob_cols].values
    X_oof = np.hstack((cb_probs, lgb_probs))
    # Concatenate Test probabilities
    cb_test_probs = test_cb[prob_cols].values
    lgb_test_probs = test_lgb[prob_cols].values
    X_test = np.hstack((cb_test_probs, lgb_test_probs))
    # Meta-model training (Logistic Regression)
    meta_model = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
    # Evaluate meta-model using CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_oof))
    print("Evaluating Meta-Model (CV):")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_oof, y_true)):
        X_tr, y_tr = X_oof[train_idx], y_true[train_idx]
        X_va, y_va = X_oof[val_idx], y_true[val_idx]
        meta_model.fit(X_tr, y_tr)
        oof_preds[val_idx] = meta_model.predict(X_va)
    score = balanced_accuracy_score(y_true, oof_preds)
    print(f"Blended OOF Balanced Accuracy (Stacking): {score:.5f}")
    # Train final meta-model on all data
    meta_model.fit(X_oof, y_true)
    # Predict on test data
    final_preds = meta_model.predict(X_test)
    final_pred_labels = [classes[int(p)] for p in final_preds]
    submission = pd.DataFrame({
        'id': test_cb['id'],
        'health_condition': final_pred_labels
    })
    submission.to_csv(SUBMISSIONS_DIR / 'submission_pseudo.csv', index=False)
    print("Saved final pseudo submission to outputs/submissions/submission_pseudo.csv")
if __name__ == "__main__":
    main()
