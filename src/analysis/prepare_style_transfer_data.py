import pandas as pd
import json
import random
import os
import pathlib

# Configuration: Samples per genre for statistical robustness
# Total: 400 sentences (sufficient for ~5% margin of error)
# Distribution strategy: Prioritize genres where Biblical affinity is expected
GENRE_SAMPLES = {
    'news': 150,       # Formal register - strong Biblical affinity expected
    'literary': 100,   # Poetic/narrative - potentially strongest Biblical affinity
    'blogs': 75,       # Informal written register
    'tapuz': 50,       # Conversational/forum style
    'medical': 25      # Technical/specialized register
}

def extract_sentences_by_genre(df, genre, n_samples, data_root):
    """Extract n random sentences from a specific Modern Hebrew genre."""
    
    # Map genre names to sub_corpus values in the CSV
    genre_map = {
        'news': 'news',
        'literary': 'literary',
        'blogs': 'blogs',
        'tapuz': 'tapuz',
        'medical': 'medical'
    }
    
    sub_corpus_name = genre_map.get(genre, genre)
    
    # Filter for this genre
    genre_df = df[(df['corpus'] == 'Modern') & (df['sub_corpus'] == sub_corpus_name)]
    
    print(f"  Genre '{genre}': Found {len(genre_df)} sentences, sampling {n_samples}...")
    
    if len(genre_df) == 0:
        print(f"  WARNING: No sentences found for genre '{genre}'")
        return []
    
    # Sample randomly (with replacement if needed)
    actual_samples = min(n_samples, len(genre_df))
    sampled_indices = random.sample(range(len(genre_df)), actual_samples)
    sampled_rows = genre_df.iloc[sampled_indices]
    
    sentences_data = []
    
    for _, row in sampled_rows.iterrows():
        fname = row['filename']
        
        # Search for the file in the data output directory
        found = list(pathlib.Path(data_root).rglob(fname))
        
        if found:
            fpath = found[0]
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract sentence text from JSON structure
                    if isinstance(data, list) and len(data) > 0:
                        text = data[0].get('sentence', '')
                        if text:
                            sentences_data.append({
                                'text': text,
                                'genre': genre,
                                'filename': fname
                            })
            except Exception as e:
                print(f"    Error reading {fname}: {e}")
                continue
        
        if len(sentences_data) >= actual_samples:
            break
    
    print(f"  Successfully extracted {len(sentences_data)} sentences from '{genre}'")
    return sentences_data

def prepare_multi_genre_dataset():
    """
    Prepare a statistically robust dataset of 400 Modern Hebrew sentences
    distributed across 5 genres for style transfer analysis.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_path, '../../results/raw_data/all_sentences_features.csv')
    
    print("=" * 60)
    print("Style Transfer Data Preparation (Expanded)")
    print("=" * 60)
    print(f"Loading features from: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"Total sentences in dataset: {len(df)}")
    
    data_root = os.path.join(base_path, '../../data/output')
    
    all_sentences = []
    
    print("\nSampling by genre:")
    for genre, n_samples in GENRE_SAMPLES.items():
        genre_sentences = extract_sentences_by_genre(df, genre, n_samples, data_root)
        all_sentences.extend(genre_sentences)
    
    print("\n" + "=" * 60)
    print(f"Total sentences collected: {len(all_sentences)}")
    print(f"Target: {sum(GENRE_SAMPLES.values())} sentences")
    
    # Shuffle to mix genres (prevents batch effects)
    random.shuffle(all_sentences)
    
    # Save to file with new naming convention
    output_path = os.path.join(base_path, '../../data/style_transfer_input_400.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_sentences, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to: {output_path}")
    print("=" * 60)
    
    # Print genre distribution for verification
    print("\nGenre Distribution:")
    for genre in GENRE_SAMPLES.keys():
        count = sum(1 for s in all_sentences if s['genre'] == genre)
        print(f"  {genre}: {count} sentences")

if __name__ == "__main__":
    prepare_multi_genre_dataset()

