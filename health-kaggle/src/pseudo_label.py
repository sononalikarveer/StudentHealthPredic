import pandas as pd
import numpy as np
from src.config import DATA_DIR, SUBMISSIONS_DIR
from src.data import DataLoader

def main():
    print("Generating Pseudo-Labels from Test Set...")
    
    loader = DataLoader()
    
    test_cb = pd.read_csv(SUBMISSIONS_DIR / 'final_test_probs_catboost.csv')
    test_lgb = pd.read_csv(SUBMISSIONS_DIR / 'final_test_probs_lightgbm.csv')
    
    prob_cols = [c for c in test_cb.columns if c.startswith('prob_')]
    
    cb_test_probs = test_cb[prob_cols].values
    lgb_test_probs = test_lgb[prob_cols].values
    
    # Use our optimal blend weight
    blend_weight = 0.39
    final_test_probs = blend_weight * cb_test_probs + (1 - blend_weight) * lgb_test_probs
    
    # Get max probability for each row
    max_probs = np.max(final_test_probs, axis=1)
    preds = np.argmax(final_test_probs, axis=1)
    
    # Threshold for pseudo-labeling (99% confidence)
    CONFIDENCE_THRESHOLD = 0.99
    
    confident_mask = max_probs > CONFIDENCE_THRESHOLD
    
    # Load raw test data
    test_df, test_ids = loader.load_test()
    
    pseudo_df = test_df[confident_mask].copy()
    pseudo_df['target'] = preds[confident_mask]
    
    # Drop rows that have NaN in features because we don't want to learn from imputed data too much
    # Wait, we want to keep them to teach the model how to handle NaNs? 
    # Let's keep them, tree models handle NaNs natively or via indicators.
    
    pseudo_labels_path = DATA_DIR / 'pseudo_labels.csv'
    pseudo_df.to_csv(pseudo_labels_path, index=False)
    
    print(f"Total Test Rows: {len(test_df)}")
    print(f"Highly Confident Rows (> {CONFIDENCE_THRESHOLD}): {len(pseudo_df)}")
    print(f"Saved Pseudo-Labels to {pseudo_labels_path}")

if __name__ == "__main__":
    main()
