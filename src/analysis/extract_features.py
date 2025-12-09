import json
import pathlib
import pandas as pd
import sys
import traceback

# Constants
SUBORDINATION_WORDS = {'ש', 'אשר', 'כי', 'כאשר', 'מאשר', 'אם', 'פן'}
BIBLICAL_PAIRS = {'עץ', 'אף', 'שמש', 'ירח', 'גר', 'פחד'}
MISHNAIC_PAIRS = {'אילן', 'חוטם', 'חמה', 'לבנה', 'דר', 'ירא'}

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
    # Try to identify sub-corpus for Modern (e.g., news, blogs)
    parts = file_path.parts
    if 'modernOutput' in parts:
        idx = parts.index('modernOutput')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return 'General'

def calculate_tree_depth(tokens, current_idx, children_map):
    def get_depth(idx):
        if idx not in children_map:
            return 1
        return 1 + max((get_depth(child) for child in children_map[idx]), default=0)
    return get_depth(current_idx)

def extract_features_from_sentence(sentence_obj, file_label, sub_corpus, file_name):
    dicta = sentence_obj.get('dicta', {})
    tokens = dicta.get('tokens', [])
    if not tokens:
        return None

    # 1. General
    len_tokens = len(tokens)
    lemmas = set()
    
    # Pre-calculations for dependencies
    children_map = {}
    root_idx = -1
    
    for i, t in enumerate(tokens):
        lex = t.get('lex', '')
        lemmas.add(lex)
        
        syntax = t.get('syntax', {})
        head = syntax.get('dep_head_idx')
        
        # Identify Root
        # Some Dicta outputs might rely on dep_func='root' explicitly
        if syntax.get('dep_func') == 'root' or head == -1:
            root_idx = i
            
        if head is not None and head != -1:
            if head not in children_map:
                children_map[head] = []
            children_map[head].append(i)
    
    # If explicit root not found, try heuristics (first node with head -1)
    if root_idx == -1:
         for i, t in enumerate(tokens):
            if t.get('syntax', {}).get('dep_head_idx') == -1:
                root_idx = i
                break
    
    tree_depth = calculate_tree_depth(tokens, root_idx, children_map) if root_idx != -1 else 0

    # Initialize counters
    subordination_count = 0
    biblical_pair_count = 0
    mishnaic_pair_count = 0
    
    pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'PRON': 0, 'ADV': 0}
    # Future/Past/Pres logic might need adjustment based on specific Dicta tags
    tense_counts = {'Past': 0, 'Pres': 0, 'Fut': 0} 
    
    possession_construct = 0
    possession_shel = 0
    
    # Word Order
    is_v1 = 0
    is_v2 = 0
    nsubj_indices = []
    
    # Gerund/Infinitive logic
    gerund_count = 0
    infinitive_count = 0

    for i, token in enumerate(tokens):
        lex = token.get('lex', '')
        morph = token.get('morph', {})
        pos = morph.get('pos')
        feats = morph.get('feats', {})
        syntax = token.get('syntax', {})
        dep_func = syntax.get('dep_func')
        dep_head_idx = syntax.get('dep_head_idx')
        
        # 2. Subordination (Lemma check)
        if lex in SUBORDINATION_WORDS:
            subordination_count += 1
            
        # 3. Pairs
        if lex in BIBLICAL_PAIRS:
            biblical_pair_count += 1
        if lex in MISHNAIC_PAIRS:
            mishnaic_pair_count += 1
            
        # 4. Morphology
        if pos in pos_counts:
            pos_counts[pos] += 1
            
        if pos == 'VERB':
            tense = feats.get('Tense')
            if tense in tense_counts:
                tense_counts[tense] += 1
                
            # 5. Gerund vs Infinitive
            has_finite_features = any(k in feats for k in ['Person', 'Gender', 'Number', 'Tense'])
            is_non_finite = not has_finite_features
            
            # Additional heuristic: if pos is VERB but VerbForm is Inf?
            # Dicta sometimes puts VerbForm in feats.
            if feats.get('VerbForm') == 'Inf':
                is_non_finite = True
            
            if is_non_finite:
                # Check for subject dependency
                has_nsubj = False
                if i in children_map:
                    for child_idx in children_map[i]:
                        if tokens[child_idx]['syntax'].get('dep_func') == 'nsubj':
                            has_nsubj = True
                            break
                
                has_suffix = morph.get('suffix', False)
                
                if has_nsubj or has_suffix:
                    gerund_count += 1
                else:
                    infinitive_count += 1

        # 6. Possession
        if dep_func == 'compound:smixut':
            possession_construct += 1
        if lex == 'של':
            possession_shel += 1
            
        # Collect nsubj for Word Order check
        if dep_head_idx == root_idx and dep_func == 'nsubj':
            nsubj_indices.append(i)

    # 5. Word Order (V1 vs V2)
    # Check relation between root (verb) and nsubj
    if root_idx != -1 and nsubj_indices:
        # Take the first nsubj found
        nsubj_idx = nsubj_indices[0]
        if root_idx < nsubj_idx:
            is_v1 = 1
        elif root_idx > nsubj_idx:
            is_v2 = 1

    return {
        'corpus': file_label,
        'sub_corpus': sub_corpus,
        'filename': file_name,
        'len_tokens': len_tokens,
        'tree_depth': tree_depth,
        'num_unique_lemmas': len(lemmas),
        'cnt_subordination': subordination_count,
        'cnt_biblical_pairs': biblical_pair_count,
        'cnt_mishnaic_pairs': mishnaic_pair_count,
        'cnt_noun': pos_counts['NOUN'],
        'cnt_verb': pos_counts['VERB'],
        'cnt_adj': pos_counts['ADJ'],
        'cnt_pron': pos_counts['PRON'],
        'cnt_adv': pos_counts['ADV'],
        'cnt_past': tense_counts['Past'],
        'cnt_pres': tense_counts['Pres'],
        'cnt_fut': tense_counts['Fut'],
        'is_v1': is_v1,
        'is_v2': is_v2,
        'cnt_poss_construct': possession_construct,
        'cnt_poss_shel': possession_shel,
        'cnt_gerund': gerund_count,
        'cnt_infinitive': infinitive_count
    }

def process_files(root_path, output_csv):
    root = pathlib.Path(root_path)
    all_features = []
    processed_count = 0
    
    print(f"Starting production traversal from: {root}")
    files = list(root.rglob('*.json'))
    total_files = len(files)
    print(f"Found {total_files} files to process.")
    
    try:
        for i, file_path in enumerate(files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sentences = json.load(f)
                    
                label = get_corpus_label(file_path)
                sub = get_sub_corpus(file_path)
                
                for sent_obj in sentences:
                    features = extract_features_from_sentence(sent_obj, label, sub, file_path.name)
                    if features:
                        all_features.append(features)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count}/{total_files} files...")
                    
            except Exception as e:
                # Log error but continue
                # print(f"Error processing {file_path}: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress so far...")
    except Exception as e:
        print(f"\nCritical error: {e}")
        traceback.print_exc()
    finally:
        if all_features:
            print(f"Saving {len(all_features)} sentences to {output_csv}...")
            df = pd.DataFrame(all_features)
            output_path = pathlib.Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            print("Done.")
        else:
            print("No features extracted.")

if __name__ == "__main__":
    input_dir = "NLP-Project/data/output"
    output_file = "results/all_sentences_features.csv"
    process_files(input_dir, output_file)

