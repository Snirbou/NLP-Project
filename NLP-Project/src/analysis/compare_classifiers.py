import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import sys
import os

# Define input file path
INPUT_FILE = '../../results/all_sentences_features.csv'

# Define feature groups based on project description
# Lexical: based on words/lemmas presence (we have counts of specific pairs, unique lemmas count, etc)
# Ideally we'd have bag-of-words or TF-IDF, but we work with extracted features.
# Based on available features in all_sentences_features.csv:
# 'cnt_biblical_pairs', 'cnt_mishnaic_pairs', 'num_unique_lemmas' seem lexical.
LEXICAL_FEATURES = [
    'num_unique_lemmas', 
    'cnt_biblical_pairs', 
    'cnt_mishnaic_pairs'
]

# Morphological: based on form, POS counts, tense counts, suffixes
# 'cnt_noun', 'cnt_verb', 'cnt_adj', 'cnt_pron', 'cnt_adv', 'cnt_past', 'cnt_pres', 'cnt_fut', 'cnt_gerund', 'cnt_infinitive', 'cnt_poss_construct', 'cnt_poss_shel'
MORPHOLOGICAL_FEATURES = [
    'cnt_noun', 'cnt_verb', 'cnt_adj', 'cnt_pron', 'cnt_adv',
    'cnt_past', 'cnt_pres', 'cnt_fut',
    'cnt_poss_construct', 'cnt_poss_shel',
    'cnt_gerund', 'cnt_infinitive'
]

# Syntactic: based on sentence structure, tree depth, word order, subordination
# 'tree_depth', 'len_tokens', 'cnt_subordination', 'is_v1', 'is_v2'
SYNTACTIC_FEATURES = [
    'tree_depth', 
    'len_tokens', 
    'cnt_subordination', 
    'is_v1', 
    'is_v2'
]

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)

def get_balanced_train_data(df):
    # Separate Training (Biblical/Rabbinic)
    train_df_raw = df[df['corpus'].isin(['Biblical', 'Rabbinic'])].copy()
    
    biblical_df = train_df_raw[train_df_raw['corpus'] == 'Biblical']
    rabbinic_df = train_df_raw[train_df_raw['corpus'] == 'Rabbinic']
    
    n_biblical = len(biblical_df)
    
    # Downsample Rabbinic to match Biblical size
    rabbinic_sampled = rabbinic_df.sample(n=n_biblical, random_state=42)
    
    train_df = pd.concat([biblical_df, rabbinic_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    return train_df

def train_evaluate_classifier(name, features, train_df, modern_df):
    print(f"\n[{name}] Starting evaluation...")
    print(f"[{name}] Features ({len(features)}): {features}")
    
    X = train_df[features].fillna(0)
    y = train_df['corpus']
    
    # Use n_jobs=-1 to use all CPU cores
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Cross Validation
    print(f"[{name}] Running 10-fold Cross Validation...")
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, pos_label='Biblical'),
        'recall': make_scorer(recall_score, pos_label='Biblical'),
        'f1': make_scorer(f1_score, pos_label='Biblical')
    }
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # n_jobs=-1 here might conflict with RF n_jobs, but usually scikit-learn handles it. 
    # Safest is to let RF parallelize internal trees or CV parallelize folds. 
    # Given small dataset (11k) and RF (100 trees), parallelizing folds (CV) might be better or equal.
    # Let's parallelize CV and keep RF sequential or vice versa. 
    # RF parallel (n_jobs=-1) is usually good enough. 
    # Let's enable verbose in cross_validate if possible, but manual prints are better.
    
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    print(f"[{name}] CV Completed.")
    print(f"--- {name} CV Results (Target: Biblical) ---")
    print(f"Accuracy:  {scores['test_accuracy'].mean():.4f}")
    print(f"Precision: {scores['test_precision'].mean():.4f}")
    print(f"Recall:    {scores['test_recall'].mean():.4f}")
    print(f"F1 Score:  {scores['test_f1'].mean():.4f}")
    
    # Train on full balanced set
    print(f"[{name}] Retraining on full balanced dataset...")
    clf.fit(X, y)
    
    # Classify Modern
    print(f"[{name}] Classifying {len(modern_df)} Modern Hebrew sentences...")
    X_modern = modern_df[features].fillna(0)
    predictions = clf.predict(X_modern)
    print(f"[{name}] Classification done.")
    
    # Return raw predictions to be aggregated later
    return predictions, scores

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../../../results/all_sentences_features.csv')
    
    df = load_data(data_path)
    
    train_df = get_balanced_train_data(df)
    modern_df = df[df['corpus'] == 'Modern'].copy()
    
    # Dictionary to store results
    classifiers = {
        'Lexical': LEXICAL_FEATURES,
        'Morphological': MORPHOLOGICAL_FEATURES,
        'Syntactic': SYNTACTIC_FEATURES,
        'Combined': LEXICAL_FEATURES + MORPHOLOGICAL_FEATURES + SYNTACTIC_FEATURES
    }
    
    results_summary = []
    
    for name, features in classifiers.items():
        preds, scores = train_evaluate_classifier(name, features, train_df, modern_df)
        
        # Store Modern Hebrew stats
        modern_df[f'pred_{name}'] = preds
        
        # Aggregate stats
        stats = modern_df.groupby('sub_corpus')[f'pred_{name}'].value_counts(normalize=True).unstack().fillna(0)
        biblical_pct = stats.get('Biblical', 0)
        if isinstance(biblical_pct, float): # Handle case where only one class exists
             biblical_pct = pd.Series(biblical_pct, index=stats.index)
        
        results_summary.append({
            'Classifier': name,
            'CV_Accuracy': scores['test_accuracy'].mean(),
            'CV_F1': scores['test_f1'].mean(),
            'Modern_Biblical_Pct_News': biblical_pct.get('news', 0),
            'Modern_Biblical_Pct_Medical': biblical_pct.get('medical', 0),
            'Modern_Biblical_Pct_Tapuz': biblical_pct.get('tapuz', 0),
            'Modern_Biblical_Pct_Blogs': biblical_pct.get('blogs', 0)
        })

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    print("\n\n=== Comparative Summary ===")
    print(summary_df)
    
    output_path = os.path.join(base_path, '../../../results/classifiers_comparison.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"Comparison saved to {output_path}")
    
    # Save full predictions
    preds_output_path = os.path.join(base_path, '../../../results/modern_predictions_by_classifier.csv')
    modern_df[['corpus', 'sub_corpus', 'filename'] + [f'pred_{name}' for name in classifiers]].to_csv(preds_output_path, index=False)
    print(f"Detailed predictions saved to {preds_output_path}")

if __name__ == "__main__":
    main()

