Project Workflow and Results Summary
====================================
Date: [Current Date]
Task: Classification of Modern Hebrew Sentences (Biblical vs. Rabbinic)

1. Workflow Overview
--------------------
This document summarizes the computational workflow executed to build, train, and analyze classifiers for distinguishing between Biblical and Rabbinic Hebrew, and subsequently classifying Modern Hebrew sentences.

### Step 1: Environment Setup
*   Initialized a Python virtual environment.
*   Installed necessary libraries: `pandas` for data manipulation, `scikit-learn` for machine learning models.

### Step 2: Data Exploration & Preprocessing
*   **Source Data**: `all_sentences_features.csv` containing extracted features for Biblical, Rabbinic, and Modern Hebrew sentences.
*   **Feature Selection**: Used 20 extracted features covering:
    *   **Lexical**: Unique lemmas count, specific Biblical/Mishnaic word pairs.
    *   **Morphological**: Part-of-Speech counts (Noun, Verb, etc.), Tenses (Past, Present, Future), Suffixes, Infinitive/Gerund forms.
    *   **Syntactic**: Tree depth, Sentence length, Subordination, Word order (V1/V2).
*   **Class Imbalance Handling**:
    *   Initial analysis revealed a severe imbalance: Rabbinic (~26k sentences) vs Biblical (~5.8k sentences).
    *   **Solution**: Applied random undersampling to the Rabbinic dataset to create a perfectly balanced training set (~5,823 sentences per class).

### Step 3: Classifier Implementation
*   **Algorithm**: Random Forest Classifier (100 trees).
*   **Validation**: Used 10-Fold Cross-Validation (Stratified) to ensure robust performance metrics (Accuracy, Precision, Recall, F1).
*   **Experiments**:
    1.  **Main Classifier**: Trained on all features combined.
    2.  **Feature Ablation**: Created separate classifiers for specific feature groups (Lexical, Morphological, Syntactic) to isolate stylistic drivers.

2. Main Results
---------------

### A. Classifier Performance (Biblical vs. Rabbinic)
The models were evaluated on their ability to correctly identify Biblical Hebrew against Rabbinic Hebrew.

| Classifier Type | Accuracy | F1 Score (Biblical) | Key Insight |
|-----------------|----------|---------------------|-------------|
| **Combined**    | **89.4%**| **0.90**            | Best overall performance. |
| Morphological   | 85.3%    | 0.86                | Strongest single indicator. |
| Syntactic       | 80.0%    | 0.81                | Good, but less discriminative. |
| Lexical         | 75.5%    | 0.79                | Weakest; vocabulary alone is insufficient. |

### B. Modern Hebrew Classification
We classified ~700,000 Modern Hebrew sentences from various genres (News, Medical, Blogs, Forums).

*   **Overall Trend**: Modern Hebrew is structurally and morphologically overwhelmingly **Rabbinic** (80-90%).
*   **The "Lexical Illusion"**: The Lexical classifier labeled a much higher percentage of Modern Hebrew as "Biblical" (43-70%). This suggests that while Modern Hebrew vocabulary shares many roots with the Bible, the grammatical structure in which these words are used is fundamentally Rabbinic.

### C. Genre Analysis
Different genres showed varying degrees of affinity to Biblical style (according to the best-performing "Combined" model):

1.  **Formal Registers (News & Medical)**:
    *   Showed the highest affinity to Biblical Hebrew (~20-21% classified as Biblical).
    *   Likely due to higher usage of V1 structures (verb-first), higher register vocabulary, and complex syntax.
2.  **Informal Registers (Blogs & Forums)**:
    *   Showed the strongest affinity to Rabbinic Hebrew (~89-97% classified as Rabbinic).
    *   Characterized by SVO (Subject-Verb-Object) order, use of present tense (Benoni), and "She-" subordination—all hallmarks of Rabbinic/Modern syntax.

3. Conclusion
-------------
The classification experiment confirms the linguistic hypothesis: **Modern Hebrew syntax and morphology are direct descendants of Rabbinic Hebrew**, despite a lexical layer that often draws heavily from Biblical Hebrew. The "Combined" classifier successfully integrated these conflicting signals to provide the most accurate assessment of linguistic proximity.

