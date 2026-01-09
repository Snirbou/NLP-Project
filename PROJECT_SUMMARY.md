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
*   **Classification:** Machine learning models were trained to distinguish between BH and RH, then applied to MH to see which class it "naturally" falls into.
*   **Style Transfer Experiment:** A Generative AI (DictaLM 2.0) was tasked with rewriting Modern sentences into Biblical and Rabbinic styles to measure "edit distance."

---

## 2. Key Research Conclusions

Based on the `FINAL_RESEARCH_REPORT.md` and experimental results, the findings are:

1.  **Vector Similarity (Support for Hypothesis):**
    *   The overall Euclidean distance analysis places Modern Hebrew significantly closer to **Biblical Hebrew** (Distance ~11.3) than to Rabbinic Hebrew (Distance ~35.1). This is the strongest evidence supporting the "Historical Leap" hypothesis.

2.  **Style Transfer (Support for Hypothesis):**
    *   Generative AI required fewer edits (smaller Levenshtein distance) to transform Modern News text into Biblical style (0.67) compared to Rabbinic style (0.73), suggesting the "native" style of formal Modern Hebrew is inherently more "Biblical."

3.  **Specific Syntactic Features (Nuanced/Mixed):**
    *   **Gerunds:** Modern Hebrew shows a resurgence of Gerund usage (7.22 per 1k words), surpassing even Rabbinic levels, which aligns with Doron's claim of re-adopting Biblical forms.
    *   **Word Order:** Modern Hebrew is overwhelmingly **V2 (SVO)** (~80%), similar to Rabbinic Hebrew. Biblical Hebrew is distinctively **V1 (VSO)** (~73%). Here, Modern Hebrew *does not* mimic the Biblical structure.
    *   **Possession:** Modern Hebrew heavily utilizes the analytic "Shel" (ratio 2.8), even more so than Rabbinic Hebrew. The Biblical exclusive preference for the Construct State is not reflected in Modern usage.

**Overall Verdict:** While the rigid syntactic skeleton (Word Order) of Modern Hebrew remains closer to the Rabbinic/European SVO model, the broader stylistic, lexical, and morphological profile (Gerunds, Vector Distance) strongly drifts back towards Biblical norms, validating the core of Doron's observation.

---

## 3. Missing Components & Next Steps

### Missing / Incomplete Sections
Based on the original `project_description.txt`, the following requirements appear to be unimplemented or not fully documented:

1.  **Human Judgment Experiment (Section 5):**
    *   *Requirement:* A study involving human participants rating sentences for comprehensibility.
    *   *Status:* **Missing.** No results data or analysis found in `results/`.
2.  **Subordination Analysis (Specific Words):**
    *   *Requirement:* Detailed stats for words like "ya'an", "ekev", etc. (Doron's examples).
    *   *Status:* Basic subordination stats exist (`subordination_words_stats.csv`), but a deep dive analysis of these specific connective words is not highlighted in the final report.
3.  **Detailed Classification Report:**
    *   *Status:* While classification was performed, the detailed breakdown of *which* specific Modern sub-genres (e.g., Sports vs. Literature) lean more towards Biblical/Rabbinic is not fully detailed in the summary text, though CSVs exist.

### Suggested Fixes & Improvements
1.  **Implement Human Evaluation:** Design a simple Google Form with 50 sentence pairs (BH/RH) and distribute to 10-20 subjects to fulfill the project requirement.
2.  **Refine Word Order Analysis:** The current V1/V2 detector is heuristic. Improve accuracy by handling complex sentences (e.g., sentences starting with temporal clauses) to ensure the V1 count isn't under-reported in Modern Hebrew.
3.  **Code Cleanup:**
    *   Add docstrings to `extract_features.py` to explain the linguistic logic for each feature.
    *   Create a `requirements.txt` file (currently missing) to ensure reproducibility of the Python environment.
4.  **Expand Style Transfer:** Run the style transfer experiment on other Modern genres (e.g., Literature, Blogs) to see if the "Biblical affinity" holds true outside of formal News text.

