# Work Plan and Output Architecture – Comparing Modern Hebrew to Ancient Sources

## 1. Rationale and Goals (Research Goals)
The project statistically examines Prof. Edit Doron's hypothesis, which claims that the syntax of Modern Hebrew (MH) performed a "Historical Leap" and re-adopted Biblical Hebrew (BH) characteristics, while abandoning Mishnaic Hebrew (RH) traits.

### Research Questions and Metrics:
1.  **Complexity and General Characteristics:** Does Modern Hebrew exhibit complexity ("depth") similar to Biblical Hebrew?
    *   *Metrics:* Sentence length, lexical richness, average syntactic tree depth.
2.  **Subordination:** Checking for "Convergence" – use of Biblical structures with Mishnaic words.
    *   *Metrics:* Frequency of subordinating conjunctions (she-, asher, ki, im, pen).
3.  **Word Order (V1 vs V2):** Checking the dominance of Subject-before-Verb (SVO/V2) versus Verb-before-Subject (VSO/V1).
    *   *Metrics:* Identifying the position of the `root` relative to the `nsubj` in the dependency tree.
4.  **Possession:** Biblical Construct State (Smixut) vs. the Mishnaic analytic "Shel".
    *   *Metrics:* Ratio between `compound:smixut` tag and the word "shel".
5.  **Infinite Forms (Gerund vs Infinitive):** Doron's strong evidence – the return of the Gerund (inflected infinitive/with subject) in Modern Hebrew.
    *   *Metrics:* Identifying non-inflected verbs with subjects (Gerund) versus non-inflected verbs without subjects (Infinitive).
6.  **Lexical Profile:** Use of "marked" (Mishnaic) words vs. "unmarked" (Biblical) words.
    *   *Metrics:* Counting word pairs (etz/ilan, shemesh/hama) and POS distribution.

## 2. Workflow (Pipeline)
1.  **Data Loading:** Loading parsed JSON files from the `data/output/` directory.
    *   Identifying the Corpus (`Biblical`, `Rabbinic`, `Modern`) based on the folder name.
2.  **Feature Extraction:** Iterating over every sentence and extracting a flat feature vector (based on the questions above).
3.  **Aggregation:** Creating summary tables for each corpus and sub-corpus.
4.  **Modeling:** Training a Classifier to distinguish between BH and RH and testing it on Modern Hebrew.

## 3. Output Architecture
The system generates outputs in 4 layers. All files will be saved in the `results/` directory (or another defined location).

### Layer 1: The Master Table
*   **Filename:** `all_sentences_features.csv`
*   **Description:** Raw table, one row per sentence. Contains all counts, lengths, and binary indicators (is_v1, has_gerund).

### Layer 2: Aggregated Statistics (CSV)
These files are used to generate the graphs in the report:
1.  `corpus_overview_stats.csv` – Average length, tree depth, and linguistic richness for each corpus.
2.  `word_order_v1_v2_stats.csv` – Percentage of V1 vs V2.
3.  `subordination_words_stats.csv` – Normalized frequency of subordinating conjunctions.
4.  `possession_constructions_stats.csv` – Ratio of Construct State vs. "Shel".
5.  `gerund_infinitive_stats.csv` – Frequency of infinite forms (per 1000 words).
6.  `pos_distribution_stats.csv` – Part of Speech distribution.
7.  `doron_lexical_pairs_stats.csv` – Comparison table for specific word pairs.
8.  `corpus_distance_matrix.csv` – Vector distance matrix between corpora.

### Layer 3: Qualitative Examples (TXT)
*   `example_sentences_v1_v2.txt` – Examples of sentences identified as V1/V2.
*   `example_sentences_possessive.txt` – Examples of different possession structures.

### Layer 4: Classification Results (CSV)
*   `classifier_metrics_historical.csv` – Model performance on historical data.
*   `classifier_predictions_modern.csv` – How the model classified Modern texts (percentage of closeness to Biblical/Rabbinic).

## 4. Technical Implementation Guidelines
*   The Python code will rely on Dicta's JSON fields: `tokens`, `lex`, `morph`, `syntax`.
*   Gerund identification requires a combined check of `morph` (lack of inflection) and `syntax` (existence of `nsubj`).
*   V1/V2 identification requires comparing the indices of `root_idx` and `nsubj`.

