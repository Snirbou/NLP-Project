import json
import pathlib
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import sys
import os
import random
from tqdm import tqdm

# Configuration
MODEL_NAME = "dicta-il/dictabert"
BATCH_SIZE = 32
MAX_LENGTH = 128 # Limit sequence length for speed
SEED = 42
MODERN_SAMPLE_SIZE = 2000 # Limit Modern Hebrew classification for reasonable runtime on CPU

def get_corpus_label(file_path):
    parts = file_path.parts
    if 'mikraOutput' in parts:
        return 'Biblical'
    elif 'hazalOutput' in parts:
        return 'Rabbinic'
    elif 'modernOutput' in parts:
        return 'Modern'
    return 'Unknown'

def get_sub_corpus(file_path):
    parts = file_path.parts
    if 'modernOutput' in parts:
        idx = parts.index('modernOutput')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return 'General'

def load_sentences_from_json(root_path):
    print(f"Scanning files in {root_path}...")
    root = pathlib.Path(root_path)
    files = list(root.rglob('*.json'))
    
    corpus_data = {
        'Biblical': [],
        'Rabbinic': [],
        'Modern': []
    }
    
    print(f"Found {len(files)} files. Extracting sentences...")
    
    for file_path in tqdm(files):
        try:
            label = get_corpus_label(file_path)
            if label == 'Unknown':
                continue
                
            sub_corpus = get_sub_corpus(file_path) if label == 'Modern' else 'General'
            
            with open(file_path, 'r', encoding='utf-8') as f:
                sentences = json.load(f)
                
            for sent_obj in sentences:
                text = sent_obj.get('sentence', '').strip()
                if text:
                    corpus_data[label].append({
                        'text': text,
                        'sub_corpus': sub_corpus
                    })
                    
        except Exception as e:
            continue
            
    return corpus_data

def get_embeddings(texts, tokenizer, model, device):
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating Embeddings"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        # Tokenize
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**encoded_input)
            
        # Get CLS token (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(cls_embeddings)
        
    return np.array(embeddings)

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_path, '../../data/output')
    
    # 1. Load Data
    corpus_data = load_sentences_from_json(input_dir)
    
    print("\nTotal Sentences Found:")
    for k, v in corpus_data.items():
        print(f"{k}: {len(v)}")
        
    # 2. Balance Training Data
    biblical_sentences = corpus_data['Biblical']
    rabbinic_sentences = corpus_data['Rabbinic']
    
    n_biblical = len(biblical_sentences)
    print(f"\nBalancing Training Data to {n_biblical} samples per class...")
    
    random.seed(SEED)
    rabbinic_sample = random.sample(rabbinic_sentences, n_biblical)
    
    train_data = []
    train_labels = []
    
    for item in biblical_sentences:
        train_data.append(item['text'])
        train_labels.append('Biblical')
        
    for item in rabbinic_sample:
        train_data.append(item['text'])
        train_labels.append('Rabbinic')
        
    # Shuffle
    combined = list(zip(train_data, train_labels))
    random.shuffle(combined)
    train_data, train_labels = zip(*combined)
    train_labels = np.array(train_labels)
    
    # 3. Prepare Modern Data (Sampled)
    print(f"\nSampling Modern Hebrew Data (Limit: {MODERN_SAMPLE_SIZE} per sub-corpus)...")
    modern_data_map = {} # sub_corpus -> list of texts
    
    for item in corpus_data['Modern']:
        sub = item['sub_corpus']
        if sub not in modern_data_map:
            modern_data_map[sub] = []
        modern_data_map[sub].append(item['text'])
        
    modern_texts = []
    modern_metadata = [] # stores sub_corpus
    
    for sub, items in modern_data_map.items():
        # Take sample
        sample_size = min(len(items), MODERN_SAMPLE_SIZE)
        sample = random.sample(items, sample_size)
        modern_texts.extend(sample)
        modern_metadata.extend([sub] * sample_size)
        print(f"  {sub}: {sample_size} sentences")
        
    # 4. Load Model
    print(f"\nLoading DictaBERT ({MODEL_NAME})...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    
    # 5. Generate Embeddings
    print("\nGenerating Embeddings for Training Set...")
    X_train = get_embeddings(train_data, tokenizer, model, device)
    
    print("\nGenerating Embeddings for Modern Hebrew Set...")
    X_modern = get_embeddings(modern_texts, tokenizer, model, device)
    
    # 6. Train & Evaluate Classifier
    print("\nTraining Classifier (Random Forest on Embeddings)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, pos_label='Biblical'),
        'recall': make_scorer(recall_score, pos_label='Biblical'),
        'f1': make_scorer(f1_score, pos_label='Biblical')
    }
    
    cv_results = cross_validate(clf, X_train, train_labels, cv=10, scoring=scoring)
    
    print("\n--- DictaBERT Embeddings CV Results (Target: Biblical) ---")
    print(f"Accuracy:  {cv_results['test_accuracy'].mean():.4f}")
    print(f"Precision: {cv_results['test_precision'].mean():.4f}")
    print(f"Recall:    {cv_results['test_recall'].mean():.4f}")
    print(f"F1 Score:  {cv_results['test_f1'].mean():.4f}")
    
    # 7. Final Training and Prediction
    clf.fit(X_train, train_labels)
    modern_preds = clf.predict(X_modern)
    
    # 8. Aggregate Results
    results_df = pd.DataFrame({
        'sub_corpus': modern_metadata,
        'predicted_class': modern_preds
    })
    
    summary = results_df.groupby('sub_corpus')['predicted_class'].value_counts(normalize=True).unstack().fillna(0)
    print("\n--- Modern Hebrew Classification (DictaBERT) ---")
    print(summary)
    
    # Save Report
    output_path = os.path.join(base_path, '../../../results/dictabert_classification_report.txt')
    with open(output_path, 'w') as f:
        f.write("DictaBERT Embedding Classification Report\n")
        f.write("=======================================\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training Size: {len(train_data)} (Balanced)\n\n")
        f.write("CV Results (Biblical vs Rabbinic):\n")
        f.write(f"Accuracy:  {cv_results['test_accuracy'].mean():.4f}\n")
        f.write(f"Precision: {cv_results['test_precision'].mean():.4f}\n")
        f.write(f"Recall:    {cv_results['test_recall'].mean():.4f}\n")
        f.write(f"F1 Score:  {cv_results['test_f1'].mean():.4f}\n\n")
        f.write("Modern Hebrew Classification Distribution:\n")
        f.write(summary.to_string())
        
    print(f"\nReport saved to {output_path}")

if __name__ == "__main__":
    main()

