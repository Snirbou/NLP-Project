import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import sys
import os

# Define input file path
INPUT_FILE = '../../results/all_sentences_features.csv'

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)

def prepare_data(df):
    # Separate Training (Biblical/Rabbinic) and Modern data
    train_df_raw = df[df['corpus'].isin(['Biblical', 'Rabbinic'])].copy()
    modern_df = df[df['corpus'] == 'Modern'].copy()
    
    # Balance the training data
    # User requested to downsample Rabbinic data to avoid bias (approx 1/5th or matching Biblical size)
    biblical_df = train_df_raw[train_df_raw['corpus'] == 'Biblical']
    rabbinic_df = train_df_raw[train_df_raw['corpus'] == 'Rabbinic']
    
    n_biblical = len(biblical_df)
    n_rabbinic = len(rabbinic_df)
    
    print(f"Original Training Counts: Biblical={n_biblical}, Rabbinic={n_rabbinic}")
    
    # Downsample Rabbinic to match Biblical size
    # We use random_state for reproducibility
    rabbinic_sampled = rabbinic_df.sample(n=n_biblical, random_state=42)
    
    train_df = pd.concat([biblical_df, rabbinic_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced Training Counts: Biblical={len(train_df[train_df['corpus']=='Biblical'])}, Rabbinic={len(train_df[train_df['corpus']=='Rabbinic'])}")
    
    # Define features (all numeric columns except metadata)
    # Metadata columns: corpus, sub_corpus, filename
    # We also exclude target 'corpus'
    metadata_cols = ['corpus', 'sub_corpus', 'filename']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"Features used: {feature_cols}")
    
    # Handle NaN - fill with 0
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    modern_df[feature_cols] = modern_df[feature_cols].fillna(0)
    
    return train_df, modern_df, feature_cols

def train_and_evaluate(train_df, feature_cols):
    X = train_df[feature_cols]
    y = train_df['corpus']
    
    # Initialize Classifier
    # Using RandomForest as a robust default
    # Note: removed class_weight='balanced' since we manually balanced the dataset
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Define metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, pos_label='Biblical', average='binary'), # Treating Biblical as "positive" for metric calculation or just specific label
        'recall': make_scorer(recall_score, pos_label='Biblical', average='binary'),
        'f1': make_scorer(f1_score, pos_label='Biblical', average='binary')
    }
    
    # Note: Precision/Recall/F1 need a specific pos_label if binary. 
    # Let's check distribution first.
    print("\nClass distribution in Training Set:")
    print(y.value_counts())
    
    # Since we have two classes, we can pick one as 'positive'. Let's pick 'Biblical'.
    # Adjust scoring to handle labels correctly.
    # We can also use 'weighted' or 'macro' average if we treat it symmetrically, 
    # but the prompt asks for precision/recall/f-measure generically.
    # Usually in this context (Biblical vs Rabbinic), we want to know how well we distinguish them.
    # I will report for 'Biblical' class (and maybe 'Rabbinic' implicitly).
    
    print("\nRunning 10-fold Cross Validation...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # We need to map labels to ensure pos_label works or pass pos_label to scorer
    # Scikit-learn string labels work if pos_label is specified.
    
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
    
    print("\n--- Cross Validation Results (Target: Biblical) ---")
    print(f"Accuracy:  {scores['test_accuracy'].mean():.4f} (+/- {scores['test_accuracy'].std() * 2:.4f})")
    print(f"Precision: {scores['test_precision'].mean():.4f} (+/- {scores['test_precision'].std() * 2:.4f})")
    print(f"Recall:    {scores['test_recall'].mean():.4f} (+/- {scores['test_recall'].std() * 2:.4f})")
    print(f"F1 Score:  {scores['test_f1'].mean():.4f} (+/- {scores['test_f1'].std() * 2:.4f})")
    
    # Fit on full training data
    print("\nRetraining on full training set...")
    clf.fit(X, y)
    
    return clf

def classify_modern(clf, modern_df, feature_cols):
    print("\nClassifying Modern Hebrew sentences...")
    X_modern = modern_df[feature_cols]
    
    predictions = clf.predict(X_modern)
    modern_df['predicted_class'] = predictions
    
    # Aggregate results by sub_corpus
    results = modern_df.groupby('sub_corpus')['predicted_class'].value_counts(normalize=True).unstack().fillna(0)
    
    print("\n--- Modern Hebrew Classification Results (Distribution) ---")
    print(results)
    
    # Also raw counts
    results_counts = modern_df.groupby('sub_corpus')['predicted_class'].value_counts().unstack().fillna(0)
    print("\n--- Modern Hebrew Classification Results (Counts) ---")
    print(results_counts)
    
    return modern_df

def main():
    # Adjust path if running from src/analysis
    # Script is in src/analysis/classify_sentences.py
    # CSV is in ../../results/all_sentences_features.csv
    
    # Check if file exists relative to script execution
    # Assuming script is run from project root or src/analysis
    # We will try absolute path or relative from known root.
    
    # Use absolute path based on workspace
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Assuming structure: .../NLP-Project/src/analysis
    # Data: .../NLP-Project/results/
    # Go up 3 levels to reach project root containing results folder
    data_path = os.path.join(base_path, '../../../results/all_sentences_features.csv')
    
    df = load_data(data_path)
    
    train_df, modern_df, feature_cols = prepare_data(df)
    
    clf = train_and_evaluate(train_df, feature_cols)
    
    classified_modern = classify_modern(clf, modern_df, feature_cols)
    
    # Optional: Save predictions
    output_path = os.path.join(base_path, '../../../results/modern_classification_predictions.csv')
    classified_modern.to_csv(output_path, index=False)
    print(f"\nDetailed predictions saved to {output_path}")

if __name__ == "__main__":
    main()

