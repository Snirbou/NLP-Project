import pandas as pd
import pathlib
import datetime
import sys

# Configuration
INPUT_DIR = pathlib.Path("results/layer2_stats")
PLOTS_DIR = pathlib.Path("results/plots")
OUTPUT_FILE = pathlib.Path("results/FINAL_RESEARCH_REPORT.md")

# Ensure plots directory exists for relative linking
REL_PLOTS_DIR = pathlib.Path("plots")

def load_csv(filename):
    path = INPUT_DIR / filename
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None

def generate_header():
    date_str = datetime.date.today().strftime("%d/%m/%Y")
    return f"""# דוח מחקר סופי: השוואת עברית מודרנית למקורות
**תאריך:** {date_str}
**צוות:** אושר כהן, יותם צור, שיר בן אדרת, עומרי הירש, שניר בוקריס

---

## תקציר
דוח זה מציג את ממצאי הניתוח הממוחשב שנערך על קורפוסים של עברית מקראית, חז"לית ומודרנית.
המטרה: לבחון את השערת פרופ' עידית דורון בדבר "הדילוג ההיסטורי" של העברית המודרנית אל המקרא.

---
"""

def section_distance_analysis():
    df = load_csv("corpus_distance_matrix.csv")
    if df is None:
        return "\n\n## 1. ניתוח מרחקים (Distance Analysis)\n\n*הנתונים חסרים.*\n"

    markdown_table = df.to_markdown()
    
    # Logic for interpretation
    # Assume we have 'Modern', 'Biblical', 'Rabbinic' in index/columns
    interpretation = ""
    if 'Modern' in df.index and 'Biblical' in df.columns and 'Rabbinic' in df.columns:
        dist_biblical = df.loc['Modern', 'Biblical']
        dist_rabbinic = df.loc['Modern', 'Rabbinic']
        
        comparison = "קרובה יותר למקרא" if dist_biblical < dist_rabbinic else "קרובה יותר לחז\"ל"
        interpretation = f"""
### פרשנות המרחקים
המרחק האוקלידי בין וקטורי הפיצ'רים הממוצעים מראה כי העברית המודרנית **{comparison}**.
* מרחק מהמקרא: **{dist_biblical:.4f}**
* מרחק מחז"ל: **{dist_rabbinic:.4f}**

משמעות הדבר היא שבשקלול כלל המאפיינים שנבדקו (תחביר, מורפולוגיה, לקסיקון), הדמיון הווקטורי נוטה לכיוון המצוין לעיל.
"""

    return f"""
## 1. השורה התחתונה: ניתוח דמיון וקטורי
טבלה זו מציגה את המרחק האוקלידי בין הקורפוסים. ערך נמוך יותר מצביע על דמיון רב יותר.

{markdown_table}
{interpretation}
"""

def section_word_order():
    df = load_csv("word_order_v1_v2_stats.csv")
    content = "\n## 2. מבנה תחבירי: סדר מילים (V1 vs V2)\n"
    
    if df is not None:
        content += "\n### נתונים גולמיים ואחוזים\n"
        content += df.to_markdown()
        content += "\n"
    else:
        content += "\n*הנתונים חסרים.*\n"
        
    # Check for plot
    # Filename in plot_results.py was "word_order_v1_v2.png"
    plot_filename = "word_order_v1_v2.png"
    if (PLOTS_DIR / plot_filename).exists():
        content += f"\n![התפלגות סדר מילים]({REL_PLOTS_DIR}/{plot_filename})\n"
        
    content += """
### פרשנות
במחקר הבלשני, המקרא מאופיין במבנה V1 (פועל לפני נושא - VSO), בעוד לשון חז"ל והעברית המודרנית נוטות למבנה V2 (נושא לפני פועל - SVO).
הגרף והטבלה לעיל מאפשרים לראות האם המודרנית אכן מציגה דפוס מובהק של V2 כמו חז"ל, או שישנה חזרה מסוימת למבני V1 המקראיים (במיוחד במשלבים גבוהים או ספרותיים).
"""
    return content

def section_infinite_forms():
    df = load_csv("gerund_infinitive_stats.csv")
    content = "\n## 3. צורות מקור (Gerund vs Infinitive)\n"
    
    if df is not None:
        content += "\n### שכיחות מנורמלת (ל-1000 מילים)\n"
        content += df.to_markdown()
        content += "\n"
    else:
        content += "\n*הנתונים חסרים.*\n"
        
    plot_filename = "infinite_forms_freq.png"
    if (PLOTS_DIR / plot_filename).exists():
        content += f"\n![שכיחות צורות מקור]({REL_PLOTS_DIR}/{plot_filename})\n"
        
    content += """
### פרשנות
פרופ' דורון טוענת כי העברית המודרנית החזירה לשימוש את ה-Gerund (שם פועל נטוי או עם נושא) שהיה נפוץ במקרא ונעלם בלשון חז"ל.
* **Infinitive:** שם פועל רגיל ("ללכת").
* **Gerund:** שם פועל המתפקד כפועל עם נושא ("בליכתו", "בהיות המלך").

עלייה בשכיחות ה-Gerund במודרנית לעומת חז"ל תהווה תמיכה בטענת "הדילוג ההיסטורי".
"""
    return content

def section_possession():
    df = load_csv("possession_constructions_stats.csv")
    content = "\n## 4. הבעת שייכות (Possession)\n"
    
    if df is not None:
        content += "\n### יחס סמיכות מול 'של'\n"
        content += df.to_markdown()
        content += "\n"
    else:
        content += "\n*הנתונים חסרים.*\n"
        
    plot_filename = "possession_style.png"
    if (PLOTS_DIR / plot_filename).exists():
        content += f"\n![סגנון שייכות]({REL_PLOTS_DIR}/{plot_filename})\n"
        
    content += """
### פרשנות
* **מקרא:** שימוש כמעט בלעדי בסמיכות (Construct State).
* **חז"ל:** מעבר נרחב לשימוש במלית היחס "של".
* **מודרנית:** שימוש מעורב.

היחס (Ratio) בטבלה מציג פי כמה נפוצה הסמיכות מאשר "של". ערך גבוה מעיד על סגנון מקראי/גבוה יותר.
"""
    return content

def section_lexical():
    df = load_csv("doron_lexical_pairs_stats.csv")
    content = "\n## 5. העדפות לקסיקליות (זוגות דורון)\n"
    
    if df is not None:
        content += "\n### ספירת מופעים לזוגות מייצגים\n"
        content += df.to_markdown()
        content += "\n"
    else:
        content += "\n*הנתונים חסרים.*\n"
        
    content += """
### פרשנות
נבדקו זוגות מילים נרדפות שבהן אחת מזוהה עם המקרא (כגון: עץ, שמש, אף) והשנייה עם חז"ל (אילן, חמה, חוטם).
דומיננטיות של הטור המקראי בקורפוס המודרני מעידה על בחירה לקסיקלית המדלגת על רובד חז"ל.
"""
    return content

def main():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR} not found.")
        sys.exit(1)

    print("Generating report...")
    
    report_content = generate_header()
    report_content += section_distance_analysis()
    report_content += section_word_order()
    report_content += section_infinite_forms()
    report_content += section_possession()
    report_content += section_lexical()
    
    # Save report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Report generated successfully at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

