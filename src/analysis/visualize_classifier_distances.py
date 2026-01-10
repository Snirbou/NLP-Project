import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# --- הגדרות נתיבים ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'results', 'classification_data')
OUTPUT_IMG = os.path.join(DATA_DIR, 'classifiers_feature_distance_heatmap.png')

def find_input_file():
    """מוצא את קובץ המטריצה הנכון (cosine)."""
    # מנסה למצוא את הקובץ המדויק
    expected_file = os.path.join(DATA_DIR, 'classifiers_feature_distance_matrix_cosine.csv')
    if os.path.exists(expected_file):
        return expected_file
    
    # אם לא, מנסה את השם הכללי
    generic_file = os.path.join(DATA_DIR, 'classifiers_feature_distance_matrix.csv')
    if os.path.exists(generic_file):
        return generic_file

    # אם לא, מחפש כל קובץ CSV שיש לו 'matrix' בשם
    candidates = glob.glob(os.path.join(DATA_DIR, '*matrix*.csv'))
    if candidates:
        # מעדיף את cosine אם יש
        cosine_files = [f for f in candidates if 'cosine' in f]
        if cosine_files:
            return cosine_files[0]
        return candidates[0]
    
    return None

def main():
    input_csv = find_input_file()
    
    if not input_csv:
        print(f"Error: Could not find distance matrix CSV in {DATA_DIR}")
        print("Please check if 'classifier_feature_distance.py' ran successfully.")
        return

    print(f"Generating heatmap from: {input_csv}")

    # טעינת המטריצה
    try:
        df = pd.read_csv(input_csv, index_col=0)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # הגדרות הגרף
    plt.figure(figsize=(9, 7))
    
    # יצירת Heatmap
    # cmap='Blues' -> כחול כהה = מרחק גדול (שונה), כחול בהיר = מרחק קטן (דומה)
    sns.heatmap(df, annot=True, cmap='Blues', fmt=".3f", vmin=0, vmax=0.6,
                cbar_kws={'label': 'Cosine Distance (Lower = More Similar)'})
    
    plt.title('Classifier Behavioral Similarity\n(Feature Distribution Distance)', fontsize=14)
    plt.tight_layout()
    
    # שמירה
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"Heatmap image saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()