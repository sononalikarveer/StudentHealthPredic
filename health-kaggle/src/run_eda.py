import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import DataLoader
from src.config import REPORTS_DIR

def run_eda():
    os.makedirs(REPORTS_DIR / 'images', exist_ok=True)
    loader = DataLoader()
    
    print("Loading data...")
    X_train, y_train, ids_train = loader.load_train()
    X_test, ids_test = loader.load_test()
    
    # Target distribution
    target_counts = y_train.value_counts().sort_index()
    target_props = y_train.value_counts(normalize=True).sort_index()
    
    # Original target names
    classes = loader.get_classes()
    
    report = ["# Data Audit and EDA Summary\n"]
    
    # Basic info
    report.append("## Dataset Shapes")
    report.append(f"- **Train**: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    report.append(f"- **Test**: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    report.append(f"- **ID Column**: `{loader.id_col}`")
    report.append(f"- **Target Column**: `{loader.target_col}`\n")
    
    report.append("## Dtypes")
    report.append("```\n" + str(X_train.dtypes) + "\n```\n")
    
    # Duplicates
    dup_train = X_train.duplicated().sum()
    dup_test = X_test.duplicated().sum()
    report.append(f"## Duplicates\n- Train duplicates (excluding ID/Target): {dup_train}\n- Test duplicates: {dup_test}\n")
    
    # Missing values
    missing = X_train.isnull().sum()
    missing_pct = (missing / len(X_train)) * 100
    missing_df = pd.DataFrame({'Missing': missing, 'Pct': missing_pct}).query("Missing > 0")
    report.append("## Missing Values (Train)")
    if len(missing_df) > 0:
        report.append(missing_df.to_markdown() + "\n")
    else:
        report.append("No missing values found in train set.\n")
        
    missing_test = X_test.isnull().sum()
    missing_test_pct = (missing_test / len(X_test)) * 100
    missing_test_df = pd.DataFrame({'Missing': missing_test, 'Pct': missing_test_pct}).query("Missing > 0")
    report.append("## Missing Values (Test)")
    if len(missing_test_df) > 0:
        report.append(missing_test_df.to_markdown() + "\n")
    else:
        report.append("No missing values found in test set.\n")
        
    # Unique values
    report.append("## Unique Values per Feature (Train)")
    nunique = X_train.nunique()
    report.append("```\n" + str(nunique) + "\n```\n")
    
    # Target
    report.append("## Target Distribution")
    target_df = pd.DataFrame({'Class ID': target_counts.index, 'Class Name': classes, 'Count': target_counts.values, 'Proportion': target_props.values})
    report.append(target_df.to_markdown() + "\n")
    
    # PLOTS
    print("Generating plots...")
    
    # 1. Target distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x=loader.decode_predictions(y_train.values))
    plt.title('Target Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'images' / 'target_dist.png')
    plt.close()
    
    # 2. Missingness plot (if any missing values)
    if len(missing_df) > 0:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_df.index, y=missing_df['Pct'])
        plt.title('Missingness Percentage (Train)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'images' / 'missingness.png')
        plt.close()
    
    # 3. Correlation heatmap
    if loader.num_cols:
        plt.figure(figsize=(10, 8))
        corr = X_train[loader.num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title('Correlation Heatmap (Numerical)')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'images' / 'corr_heatmap.png')
        plt.close()
        
    # Write report
    report_text = "\n".join(report)
    report_text += "\n## EDA Plots\n"
    report_text += "![Target Dist](images/target_dist.png)\n"
    if len(missing_df) > 0:
        report_text += "![Missingness](images/missingness.png)\n"
    if loader.num_cols:
        report_text += "![Correlation](images/corr_heatmap.png)\n"
        
    with open(REPORTS_DIR / 'eda_summary.md', 'w') as f:
        f.write(report_text)
        
    print("EDA Complete. Report saved to outputs/reports/eda_summary.md")

if __name__ == "__main__":
    run_eda()
