import pandas as pd
import pathlib
import sys
import numpy as np
from scipy.spatial.distance import euclidean

# Configuration
INPUT_FILE = "results/all_sentences_features.csv"
OUTPUT_DIR = pathlib.Path("results/layer2_stats")

def load_data():
    path = pathlib.Path(INPUT_FILE)
    if path.exists():
        print(f"Loading data from: {path}")
        return pd.read_csv(path)
    print(f"Error: No input CSV found at {path}. Please run extract_features.py first.")
    sys.exit(1)

def ensure_output_dir():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_stats(df):
    ensure_output_dir()
    created_files = []

    # 1. corpus_overview_stats.csv
    overview_cols = ['len_tokens', 'tree_depth', 'num_unique_lemmas']
    if all(c in df.columns for c in overview_cols):
        overview = df.groupby('corpus')[overview_cols].mean().round(2)
        path = OUTPUT_DIR / "corpus_overview_stats.csv"
        overview.to_csv(path)
        created_files.append(path.name)

    # 2. word_order_v1_v2_stats.csv
    if 'is_v1' in df.columns and 'is_v2' in df.columns:
        word_order = df.groupby('corpus')[['is_v1', 'is_v2']].sum()
        total = word_order['is_v1'] + word_order['is_v2']
        word_order['V1_percent'] = (word_order['is_v1'] / total * 100).fillna(0).round(2)
        word_order['V2_percent'] = (word_order['is_v2'] / total * 100).fillna(0).round(2)
        path = OUTPUT_DIR / "word_order_v1_v2_stats.csv"
        word_order.to_csv(path)
        created_files.append(path.name)

    # 3. subordination_words_stats.csv
    # Using 'cnt_subordination' as extracted by extract_features.py
    if 'cnt_subordination' in df.columns and 'len_tokens' in df.columns:
        subord = df.groupby('corpus')[['cnt_subordination', 'len_tokens']].sum()
        subord['Subordination_per_1k'] = (subord['cnt_subordination'] / subord['len_tokens'] * 1000).fillna(0).round(2)
        path = OUTPUT_DIR / "subordination_words_stats.csv"
        subord[['Subordination_per_1k']].to_csv(path)
        created_files.append(path.name)

    # 4. possession_constructions_stats.csv
    if 'cnt_poss_construct' in df.columns and 'cnt_poss_shel' in df.columns:
        poss = df.groupby('corpus')[['cnt_poss_construct', 'cnt_poss_shel']].sum()
        poss['Construct_Shel_Ratio'] = poss.apply(
            lambda row: row['cnt_poss_construct'] / row['cnt_poss_shel'] if row['cnt_poss_shel'] > 0 else (np.inf if row['cnt_poss_construct'] > 0 else 0), axis=1
        ).round(2)
        path = OUTPUT_DIR / "possession_constructions_stats.csv"
        poss.to_csv(path)
        created_files.append(path.name)

    # 5. gerund_infinitive_stats.csv
    if 'cnt_gerund' in df.columns and 'cnt_infinitive' in df.columns and 'len_tokens' in df.columns:
        forms = df.groupby('corpus')[['cnt_gerund', 'cnt_infinitive', 'len_tokens']].sum()
        forms['Gerund_per_1k'] = (forms['cnt_gerund'] / forms['len_tokens'] * 1000).fillna(0).round(2)
        forms['Infinitive_per_1k'] = (forms['cnt_infinitive'] / forms['len_tokens'] * 1000).fillna(0).round(2)
        path = OUTPUT_DIR / "gerund_infinitive_stats.csv"
        forms[['Gerund_per_1k', 'Infinitive_per_1k']].to_csv(path)
        created_files.append(path.name)

    # 6. pos_distribution_stats.csv
    pos_cols = ['cnt_noun', 'cnt_verb', 'cnt_adj', 'cnt_pron', 'cnt_adv']
    available_pos = [c for c in pos_cols if c in df.columns]
    if available_pos:
        pos_dist = df.groupby('corpus')[available_pos].sum()
        row_sums = pos_dist.sum(axis=1)
        for col in available_pos:
            pos_dist[f'{col}_pct'] = (pos_dist[col] / row_sums * 100).fillna(0).round(2)
            
        result_cols = [c for c in pos_dist.columns if '_pct' in c]
        path = OUTPUT_DIR / "pos_distribution_stats.csv"
        pos_dist[result_cols].to_csv(path)
        created_files.append(path.name)

    # 7. doron_lexical_pairs_stats.csv
    pair_cols = ['cnt_biblical_pairs', 'cnt_mishnaic_pairs']
    if all(c in df.columns for c in pair_cols):
        pairs = df.groupby('corpus')[pair_cols].sum()
        path = OUTPUT_DIR / "doron_lexical_pairs_stats.csv"
        pairs.to_csv(path)
        created_files.append(path.name)

    # 8. corpus_distance_matrix.csv
    numeric_df = df.select_dtypes(include=[np.number])
    if 'corpus' in df.columns:
        numeric_df['corpus'] = df['corpus']
        corpus_vectors = numeric_df.groupby('corpus').mean()
        
        corpora = corpus_vectors.index.tolist()
        n = len(corpora)
        dist_matrix = pd.DataFrame(index=corpora, columns=corpora, dtype=float)
        
        for i in range(n):
            for j in range(n):
                c1 = corpora[i]
                c2 = corpora[j]
                dist = euclidean(corpus_vectors.loc[c1], corpus_vectors.loc[c2])
                dist_matrix.loc[c1, c2] = dist
                
        path = OUTPUT_DIR / "corpus_distance_matrix.csv"
        dist_matrix.round(4).to_csv(path)
        created_files.append(path.name)

    print(f"Successfully created {len(created_files)} files in {OUTPUT_DIR}:")
    for f in created_files:
        print(f"- {f}")

if __name__ == "__main__":
    try:
        data = load_data()
        generate_stats(data)
    except Exception as e:
        print(f"Error during execution: {e}")

