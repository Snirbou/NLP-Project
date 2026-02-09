import os
import matplotlib.pyplot as plt
import numpy as np
from data_loader import extract_features_from_folder
from calculate_perplexity import train_and_test_perplexity

def main():
    # Configuration: change this to 'lex' for lexical analysis or 'pos' for syntactic analysis
    feature = 'lex'
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.abspath(os.path.join(current_dir, "../../../data/output"))
    bible_folder = os.path.join(base_data_path, 'mikraOutput')
    mishna_folder = os.path.join(base_data_path, 'hazalOutput')
    modern_folder = os.path.join(base_data_path, 'modernOutput')

    print("Loading Biblical corpus...")
    bible_lemmas = extract_features_from_folder(bible_folder, feature=feature)
    
    print("Loading Mishnaic corpus...")
    mishna_lemmas = extract_features_from_folder(mishna_folder, feature=feature)
    
    print("Loading Modern corpus...")
    modern_lemmas = extract_features_from_folder(modern_folder, feature=feature)

    n_values = [1, 2, 3]
    results_bible = []
    results_mishna = []

    for n in n_values:
        print(f"\n--- Calculating Perplexity for n={n} ---")
        
        ppl_bible = train_and_test_perplexity(bible_lemmas, modern_lemmas, n_order=n)
        ppl_mishna = train_and_test_perplexity(mishna_lemmas, modern_lemmas, n_order=n)
        
        results_bible.append(ppl_bible)
        results_mishna.append(ppl_mishna)
        
        print(f"n={n} | Bible PPL: {ppl_bible:.2f} | Mishna PPL: {ppl_mishna:.2f}")

    x = np.arange(len(n_values))
    width = 0.35 

    plt.figure(figsize=(10, 6))
    
    rects1 = plt.bar(x - width/2, results_bible, width, label='Biblical Model', color='skyblue', edgecolor='black')
    rects2 = plt.bar(x + width/2, results_mishna, width, label='Mishnaic Model', color='salmon', edgecolor='black')

    # Dynamic title based on feature type
    feature_type = 'Syntactic' if feature == 'pos' else 'Lexical'
    plt.title(f'{feature_type} Surprise (Log Perplexity) in Modern Hebrew')
    plt.xlabel('N-gram Order (n)')
    plt.ylabel('Perplexity Score (Log Scale)')
    
    plt.yscale('log') 
    
    plt.xticks(x, [f'n={n}' for n in n_values]) 
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7, which="both")

    # Save with feature-specific filename
    save_path = os.path.join(current_dir, f"perplexity_{feature}_log_bars.png")
    plt.savefig(save_path)
    print(f"\nGraph saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
