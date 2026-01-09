import json
import os
import glob

def extract_features_from_folder(folder_path, feature='lex'):
    all_corpus_sentences = []
    
    search_pattern = os.path.join(folder_path, '**', '*.json')
    json_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(json_files)} files in {folder_path}")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for entry in data:
                    sentence_features = []
                    for token in entry['dicta']['tokens']:
                        if feature == 'lex':
                            val = token.get('lex')
                        else:
                            val = token['morph'].get('pos')
                            
                        if val and val not in [',', '.', ';', '?', '!', '[BLANK]']:
                            sentence_features.append(val.replace("##", ""))
                    
                    if sentence_features:
                        all_corpus_sentences.append(sentence_features)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return all_corpus_sentences