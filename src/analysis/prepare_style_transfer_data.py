import pandas as pd
import json
import random
import os

def extract_news_sentences():
    # Load features to find News sentences
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_path, '../../../results/all_sentences_features.csv')
    
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Filter for Modern News
    news_df = df[(df['corpus'] == 'Modern') & (df['sub_corpus'] == 'news')]
    
    print(f"Found {len(news_df)} News sentences.")
    
    # Select 50 random files/sentences
    # We need the actual text, not just features. Features CSV has filename.
    # We will pick 50 filenames, then load the JSONs to get the text.
    
    sampled_indices = random.sample(range(len(news_df)), 50)
    sampled_rows = news_df.iloc[sampled_indices]
    
    sentences_data = []
    
    data_root = os.path.join(base_path, '../../data/output')
    
    for _, row in sampled_rows.iterrows():
        # Reconstruct path: data/output/modernOutput/news/.../filename
        # The filename in CSV is just the basename (e.g., '1.json').
        # We need to search for it or assume structure.
        # Let's search in modernOutput/news
        
        fname = row['filename']
        # We'll use glob to find it to be safe about directory structure
        found = list(pathlib.Path(data_root).rglob(fname))
        
        if found:
            fpath = found[0]
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Find the sentence corresponding to this row?
                # The CSV row corresponds to a specific sentence in the file.
                # But the CSV doesn't store sentence index.
                # However, usually we can just take the first sentence or any sentence from that file 
                # if the file is small (often 1 sentence per file in some exports, or we just pick one).
                # Actually, looking at `extract_features.py`, it processes all sentences in a file.
                # Let's just pick a random sentence from the file text.
                
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get('sentence', '')
                    if text:
                        sentences_data.append(text)
        
        if len(sentences_data) >= 50:
            break
            
    # Save to file
    output_path = os.path.join(base_path, '../../../data/style_transfer_input_50.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sentences_data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(sentences_data)} sentences to {output_path}")

import pathlib
if __name__ == "__main__":
    extract_news_sentences()

