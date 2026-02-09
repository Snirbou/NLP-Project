# Repository Structure

This document provides an overview of the directory structure and key files in the `NLP-Project` repository.

## Directory Layout

```
NLP-Project/
├── data/                       # Input texts and experiment data
├── results/                    # Analysis outputs and visualizations
├── src/                        # Source code for analysis pipeline
├── README.md                   # Project overview
├── REPO_STRUCTURE.md           # This file
├── final_report.docx           # Final research report (Hebrew)
└── requirements.txt            # Python dependencies
```

## Detailed Description

### 1. `data/`
Contains input data and prepared datasets.

*   **`input/`**: Raw Hebrew text files organized by corpus
    *   `hazal/`: Mishnaic Hebrew texts (Mishna, Rambam)
    *   `mikra/`: Biblical Hebrew texts (Torah, Tanach)
    *   `modern/`: Modern Hebrew texts across multiple genres
*   **`style_transfer_input_400.json`**: Stratified sample of 400 Modern Hebrew sentences for LLM style transfer experiment (150 news, 100 literary, 75 blogs, 50 forums, 25 medical)

### 2. `results/`
All analysis outputs, organized by function:

#### `results/raw_data/`
*   **`all_sentences_features.csv`**: Complete feature matrix extracted from all parsed sentences (54MB)
    *   Contains 20 linguistic features per sentence: lexical, morphological, and syntactic markers

#### `results/aggregated_stats/`
Statistical summaries aggregated by corpus (8 files, ~1KB each):
*   **`corpus_overview_stats.csv`**: Sentence length, tree depth, unique lemmas
*   **`word_order_v1_v2_stats.csv`**: Verb-Subject (V1) vs Subject-Verb (V2) word order distribution
*   **`possession_constructions_stats.csv`**: Construct state (smixut) vs "של" (shel) usage
*   **`gerund_infinitive_stats.csv`**: Gerund and infinitive frequency per 1000 tokens
*   **`doron_lexical_pairs_stats.csv`**: Biblical vs Mishnaic lexical pair counts
*   **`subordination_words_stats.csv`**: Subordination conjunction frequency
*   **`pos_distribution_stats.csv`**: Part-of-speech tag distributions
*   **`corpus_distance_matrix.csv`**: Euclidean distance between corpus feature vectors

#### `results/classification_data/`
Machine learning classification results:
*   **`classifiers_comparison.csv`**: Performance metrics for Lexical, Morphological, Syntactic, and Combined classifiers
*   **`classifiers_feature_distance_matrix_euclidean.csv`**: Euclidean feature-space distances between corpora
*   **`classifiers_feature_distance_matrix_cosine.csv`**: Cosine similarity between corpus feature vectors
*   **`classifiers_feature_distance_heatmap.png`**: Visualization of corpus distances

#### `results/style_transfer_data/`
LLM style transfer experiment output:
*   **`style_transfer_results_400.json`**: Results from 400 Modern Hebrew sentences rewritten into Biblical and Mishnaic styles using DictaLM 2.0, including normalized Levenshtein distances and per-genre analysis

#### `results/plots/`
Visualization outputs (6 files):
*   **`word_order_v1_v2.png`**: V1/V2 distribution across corpora
*   **`possession_style.png`**: Construct state vs "shel" preference
*   **`infinite_forms_freq.png`**: Gerund and infinitive usage
*   **`pos_distribution_heatmap.png`**: Part-of-speech distribution heatmap
*   **`perplexity_lex_log_bars.png`**: Lexical N-gram perplexity comparison
*   **`perplexity_pos_log_bars.png`**: Syntactic (POS) N-gram perplexity comparison

### 3. `src/`
Python source code for the complete analysis pipeline:

#### `src/dictaParsing/`
DictaBERT parsing pipeline for Hebrew text processing:
*   **`dicta_batch_parser.py`**: Interface with DictaBERT-Joint model for morphological and syntactic analysis
*   **`batch_run_recursive.py`**: Recursive batch processor for directories of `.txt` files

#### `src/analysis/`
Feature extraction, statistical analysis, and machine learning (9 files):
*   **`extract_features.py`**: Extracts 20 linguistic features from parsed JSON into tabular CSV format
*   **`generate_stats.py`**: Aggregates features and computes corpus-level statistics
*   **`classify_sentences.py`**: Trains Random Forest classifier on all features (89.51% accuracy)
*   **`compare_classifiers.py`**: Compares Lexical, Morphological, Syntactic, and Combined classifiers
*   **`dictabert_embedding_classifier.py`**: Deep learning classifier using DictaBERT embeddings (99.87% accuracy)
*   **`plot_results.py`**: Generates all visualization plots from aggregated statistics
*   **`prepare_style_transfer_data.py`**: Stratified sampling of 400 Modern Hebrew sentences across 5 genres
*   **`style_transfer_analysis.py`**: LLM-based style transfer using DictaLM 2.0 (7B parameters)
*   **`perplexity/`**: N-gram perplexity analysis module
    *   `main.py`: Orchestrates perplexity pipeline and generates plots
    *   `calculate_perplexity.py`: Trains Lidstone-smoothed N-gram models
    *   `data_loader.py`: Extracts lexical/syntactic features from parsed corpus

### 4. Root-Level Files

*   **`README.md`**: High-level project overview, folder structure summary, and team information
*   **`REPO_STRUCTURE.md`**: This file - detailed documentation of repository organization
*   **`final_report.docx`**: Complete Hebrew research report (formatted for academic submission)
*   **`requirements.txt`**: Python package dependencies
    *   Core: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
    *   NLP: nltk, python-Levenshtein
    *   Deep Learning: torch, transformers, accelerate
    *   Progress: tqdm

---

## Data Flow

1. **Parsing**: Raw `.txt` files → DictaBERT → Parsed JSON (morphology + syntax)
2. **Feature Extraction**: JSON → 20 linguistic features → `all_sentences_features.csv`
3. **Aggregation**: Feature CSV → Statistical summaries → `aggregated_stats/`
4. **Classification**: Features → ML models → `classification_data/`
5. **Visualization**: Statistics → Plots → `plots/`
6. **Style Transfer**: Sample sentences → LLM rewriting → Distance analysis

