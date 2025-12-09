import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Configuration
INPUT_DIR = pathlib.Path("results/layer2_stats")
OUTPUT_DIR = pathlib.Path("results/plots")

def ensure_output_dir():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_word_order():
    """Bar chart for Word Order (V1/V2) per corpus."""
    file_path = INPUT_DIR / "word_order_v1_v2_stats.csv"
    if not file_path.exists():
        return
    
    df = pd.read_csv(file_path, index_col=0)
    if 'V1_percent' not in df.columns or 'V2_percent' not in df.columns:
        return

    ax = df[['V1_percent', 'V2_percent']].plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Word Order Distribution (V1 vs V2)")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Corpus")
    plt.legend(["V1 (Verb-Subject)", "V2 (Subject-Verb)"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "word_order_v1_v2.png")
    plt.close()
    print("Generated word_order_v1_v2.png")

def plot_infinite_forms():
    """Bar chart for Infinite Forms (Gerund/Inf) per corpus."""
    file_path = INPUT_DIR / "gerund_infinitive_stats.csv"
    if not file_path.exists():
        return
    
    df = pd.read_csv(file_path, index_col=0)
    ax = df.plot(kind='bar', figsize=(10, 6))
    plt.title("Infinite Forms Frequency (per 1000 tokens)")
    plt.ylabel("Frequency")
    plt.xlabel("Corpus")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "infinite_forms_freq.png")
    plt.close()
    print("Generated infinite_forms_freq.png")

def plot_possession():
    """Stacked bar for Possession (Construct vs Shel)."""
    file_path = INPUT_DIR / "possession_constructions_stats.csv"
    if not file_path.exists():
        return
        
    df = pd.read_csv(file_path, index_col=0)
    cols = ['cnt_poss_construct', 'cnt_poss_shel']
    if not all(c in df.columns for c in cols):
        return
        
    # Normalize to percentage for stacked bar? Or raw counts?
    # Prompt asks for "Stacked bar", usually implies counts or composition.
    # Let's do 100% stacked bar to show preference style
    totals = df[cols].sum(axis=1)
    df_pct = df[cols].div(totals, axis=0) * 100
    
    ax = df_pct.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Possession Style Preference (Construct vs 'Shel')")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Corpus")
    plt.legend(["Construct State", "Shel Preposition"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "possession_style.png")
    plt.close()
    print("Generated possession_style.png")

def plot_pos_distribution():
    """Heatmap for POS distribution."""
    file_path = INPUT_DIR / "pos_distribution_stats.csv"
    if not file_path.exists():
        return
        
    df = pd.read_csv(file_path, index_col=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Part-of-Speech Distribution (%)")
    plt.ylabel("Corpus")
    plt.xlabel("POS Tag")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pos_distribution_heatmap.png")
    plt.close()
    print("Generated pos_distribution_heatmap.png")

def main():
    if not INPUT_DIR.exists():
        print(f"Error: Stats directory {INPUT_DIR} not found.")
        sys.exit(1)
        
    ensure_output_dir()
    
    try:
        plot_word_order()
        plot_infinite_forms()
        plot_possession()
        plot_pos_distribution()
        print(f"All plots saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()

