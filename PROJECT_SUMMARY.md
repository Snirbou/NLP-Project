# Project Summary & Roadmap

## 1. Project Workflow Overview

This project implements a computational linguistic pipeline to test Prof. Edit Doron's hypothesis: that Modern Hebrew (MH) syntax has performed a "Historical Leap," skipping over Mishnaic/Rabbinic Hebrew (RH) traits to readopt Biblical Hebrew (BH) characteristics.

The workflow was executed in the following stages:

### Phase 1: Data Processing & Parsing
*   **Input:** Raw text corpora were curated for three periods: Biblical (Tanakh), Rabbinic (Mishna, Rambam), and Modern (News, Literature, Blogs, Medical).
*   **Parsing:** The **DictaBERT** model was deployed to generate rich morphological and syntactic data for every sentence.
*   **Output:** A mirror of the input corpus in JSON format, containing Lemmas, POS tags, and Dependency Trees.

### Phase 2: Feature Extraction (Layer 1)
*   **Logic:** A Python extraction engine (`src/analysis/extract_features.py`) processed the JSONs to quantify specific linguistic markers defined in the research proposal.
*   **Features Extracted:**
    *   **Word Order:** V1 (Verb-Subject) vs. V2 (Subject-Verb) structures.
    *   **Infinite Forms:** Distinction between Gerunds (inflected/nominative infinitives) and standard Infinitives.
    *   **Possession:** Construct state (Smixut) vs. Analytical possession ("Shel").
    *   **Lexicon:** Occurrences of specific "Biblical" vs. "Rabbinic" synonyms (Doron's Pairs).
    *   **Subordination:** Usage of specific subordinating conjunctions (e.g., "asher" vs. "she-").

### Phase 3: Aggregation & Statistics (Layer 2 & 3)
*   **Aggregation:** Feature vectors were aggregated by corpus to produce high-level statistics.
*   **Visualization:** Plots were generated to compare distributions across the three distinct time periods.
*   **Distance Metrics:** Euclidean distances were calculated between the feature vectors of the corpora.

### Phase 4: Advanced Modeling (Layer 4)
*   **Classification:** Machine learning models (Random Forest, 100 trees) were trained on 20 features (Lexical, Morphological, Syntactic). To handle class imbalance, the Rabbinic dataset was undersampled to match the Biblical size (~5,800 sentences).
*   **DictaBERT Integration:** A state-of-the-art Hebrew LLM was used to generate embeddings, achieving near-perfect separation (99%) between Biblical and Rabbinic texts.
*   **Style Transfer Experiment:** A Generative AI (DictaLM 2.0) was tasked with rewriting Modern sentences into Biblical and Rabbinic styles to measure "edit distance."
*   **Perplexity Analysis:** An N-gram language model (N=1,2,3) measured how "surprised" a Biblical vs. Rabbinic model is when encountering Modern Hebrew text.

### Phase 5: Human Evaluation (Poll)
*   **Methodology:** A survey of 50 Hebrew speakers comparing the understandability of 50 paired Biblical and Rabbinic sentences.
*   **Analysis:** Respondents flagged difficulties related to syntax, grammar, and vocabulary to identify which historical layer feels more "native" or understandable.

---

## 2. Key Research Conclusions

### A. The "Paradox" of Modern Hebrew
Our multi-layered analysis revealed a striking contradiction between different metrics:
*   **Statistical Vectors (Layer 2):** Point to **Biblical Hebrew** (Distance ~11.3 vs 35.1).
*   **Classifiers (Layer 4):** Point to **Rabbinic Hebrew** (95-99% confidence).
*   **Human Intuition:** Is split right down the middle (51% Rabbinic vs 49% Biblical).

### B. Resolution: Lexicon vs. Structure
The discrepancy is resolved by distinguishing between *which* words are used and *how* they are used.

1.  **Lexical Affinity (Biblical):**
    *   Modern Hebrew speakers consistently prefer Biblical roots over Mishnaic ones (e.g., "Etz" vs "Ilan").
    *   **Evidence:** The Modern corpus contains **8,249 "Biblical" word choices** versus only **1,829 "Mishnaic" ones**.
    *   Generative AI (Style Transfer) found it easier to rewrite Modern text into Biblical style because the **vocabulary overlap** is massive.
    *   This explains why the statistical vectors (heavily influenced by word counts) pointed to the Bible.

2.  **Structural Skeleton (Rabbinic):**
    *   **Word Order:** Modern Hebrew is **V2 (SVO)** (~80%), identical to Rabbinic (71%) but opposite to Biblical V1 (VSO, 73%).
    *   **Possession:** Modern Hebrew uses the analytic "Shel" (Ratio 2.8:1), aligning with Rabbinic usage (6.7:1), while Biblical Hebrew relies almost exclusively on the Construct State (363:1).
    *   **Morphology:** The Tense system (Past/Present/Future) and Suffix usage are overwhelmingly Rabbinic.
    *   **Classifiers:** Because ML models prioritize predictive structure, they correctly identified the "skeleton" as Rabbinic.

3.  **The Human Experience:**
    *   Speakers sit exactly in the middle of this tug-of-war. The **Biblical Lexicon** makes the text feel familiar, while the **Rabbinic Syntax** makes the structure intuitive.
    *   **Main Difficulty:** Syntax was cited as the primary barrier (41%) for both historical languages, proving that structure is the defining hurdle.

### C. Final Verdict
Modern Hebrew wears a **Biblical Lexical Mask** over a **Rabbinic Grammatical Skeleton**.
While Doron's hypothesis of a "Historical Leap" is validated in terms of vocabulary and style (Gerunds, poetic register), the fundamental syntactic machinery remains linear to the Mishnaic tradition.

---

## 3. Detailed Technical Results (Appendix)

### A. Classifier Performance (Biblical vs. Rabbinic)
Validation results (10-Fold CV) for the Random Forest model:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **0.8940** (+/- 0.0151) |
| **Precision** | 0.8786 |
| **Recall** | 0.9146 |
| **F1 Score** | 0.8961 |

### B. Modern Hebrew Classification by Genre
Percentage of sentences classified as **Rabbinic** vs. **Biblical** by the Random Forest model:

| Genre (Sub-Corpus) | Biblical (%) | Rabbinic (%) |
| :--- | :--- | :--- |
| **Blogs** | 11.04% | **88.96%** |
| **Medical** | 19.74% | **80.26%** |
| **News** | 21.35% | **78.65%** |
| **Tapuz (Forums)** | 14.34% | **85.66%** |

*Note: The DictaBERT model (Layer 4) showed even stronger Rabbinic dominance (>95%) across all genres.*

---

## 4. Missing Components & Next Steps

### Missing / Incomplete Sections
Based on the original `project_description.txt`, the following requirements appear to be unimplemented or not fully documented:

1.  **Subordination Analysis (Specific Words):**
    *   *Requirement:* Detailed stats for words like "ya'an", "ekev", etc. (Doron's examples).
    *   *Status:* Basic stats exist in `subordination_words_stats.csv`. While a dedicated "deep dive" report isn't separate, the raw data is available in `results/aggregated_stats/`.
2.  **Detailed Classification Report:**
    *   *Status:* Completed. See `MANUAL_CLASSIFIER_REPORT.md` and `DICTABERT_ANALYSIS.md` in `results/reports/english/`. Detailed CSVs are in `results/classification_data/`.

### Completed Requirements (Previously Missing)
*   **Human Judgment Experiment (Section 5):** Completed. See `HUMAN_EVALUATION_REPORT.md`. Findings integrated into Section 2 of this summary.

### Suggested Fixes & Improvements
1.  **Refine Word Order Analysis:** The current V1/V2 detector is heuristic. Improve accuracy by handling complex sentences (e.g., sentences starting with temporal clauses) to ensure the V1 count isn't under-reported in Modern Hebrew.
2.  **Code Cleanup:**
    *   Add docstrings to `extract_features.py` to explain the linguistic logic for each feature.
    *   Create a `requirements.txt` file (currently missing) to ensure reproducibility of the Python environment.
3.  **Expand Style Transfer:** Run the style transfer experiment on other Modern genres (e.g., Literature, Blogs) to see if the "Biblical affinity" holds true outside of formal News text.
