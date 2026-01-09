# Perplexity Analysis Tool - Hebrew Syntax Project

## 1. Overview
This module calculates the Perplexity (PPL) score of Modern Hebrew text using language models trained on either Biblical or Mishnaic corpora. Perplexity is a measurement of how "surprised" a model is by a new text.
Lower Perplexity = Higher similarity/predictability.
Higher Perplexity = Lower similarity.

## 2. Directory Structure
PlaintextNLP-Project/

NLP-Project/
├── data/
│   └── output/
│       ├── hazalOutput/      # JSON files from Dicta (Mishnaic corpus)
│       ├── mikraOutout/      # JSON files from Dicta (Biblical corpus)
│       └── modernOutput/     # JSON files from Dicta (Modern news/blogs)
└── src/
    └── perplexity/
        ├── main.py                # Pipeline orchestrator
        ├── data_loader.py         # Recursive JSON parser
        └── calculate_perplexity.py # N-gram model and PPL logic

## 3. Methodology
To ensure the analysis focuses on syntax and structure rather than surface-level differences (like vowel pointing or spelling variants), the models are trained on:
1. Lemmas (lex): The dictionary root of each word provided by the Dicta analyzer
2. POS Tags (pos): Abstract syntactic categories (Verb, Noun, ADP, etc.) to test pure structural similarity.

The Algorithm:
We use an N-gram Language Model with Lidstone Smoothing ($\gamma=0.1$). Smoothing is essential to handle "Out-of-Vocabulary" (OOV) words—modern terms that do not appear in ancient corpora.

## 4. How to Run
Install dependencies: pip install nltk matplotlib pandas
Execute the analysis: python src/perplexity/main.py

## 5. Analyzing the N-gram Orders
We evaluate the results across three levels of granularity to separate vocabulary overlap from syntactic flow:
* n=1 (Unigram - Vocabulary Depth):
    * This measure reflects simple word frequency and the overall "word bank".
    * Interpretation: The model with the lower bar at n=1 indicates which historical period shares the most individual words and lemmas with the modern corpus.

* n=2 (Bigram - Local Syntax):
    * This measures the probability of word pairs appearing together, capturing immediate structural relationships (e.g., how specific verbs typically pair with specific nouns or prepositions).
    * Interpretation: The model with the lower bar at n=2 suggests that the local, short-range syntactic patterns of the modern text are more predictable based on that specific historical layer.

* n=3 (Trigram - Sentence Structure):
    * This represents more complex syntactic sequences and the "rhythm" of the language. It is the most robust test for underlying sentence structure and flow.
    * Interpretation: The model with the lower bar at n=3 indicates which historical period’s overarching structural "DNA" is most prevalent in the modern language. This provides the strongest evidence for claims regarding long-term syntactic continuity.