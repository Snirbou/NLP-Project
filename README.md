# Hebrew Syntax Parsing – Mini Project (2026)

This repository contains the infrastructure and results for our NLP project, whose goal is to examine computationally whether Modern Hebrew is syntactically closer to Biblical Hebrew or to Mishnaic Hebrew - following the hypothesis proposed by Edit Doron.

For a detailed summary of the results and conclusions, see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

---

## 📌 Project Overview

The project includes:

-   A recursive text-processing pipeline that scans folders of Hebrew `.txt` files.
-   Integration with **DictaBERT-Joint** (Dicta) for morphological and syntactic analysis.
-   Feature extraction scripts to identify linguistic markers (Word Order, Gerunds, etc.).
-   Statistical analysis and visualization of corpora differences.
-   Machine learning classifiers and Style Transfer experiments using LLMs.

---

## 📁 Folder Structure

For a detailed explanation of every file and folder, see [REPO_STRUCTURE.md](REPO_STRUCTURE.md).

```
NLP-Project/
├── data/                       # Raw input texts and experiment data
│   ├── input/                  # Hebrew corpus files (Biblical, Mishnaic, Modern)
│   └── style_transfer_input_400.json
├── results/                    # All analysis outputs
│   ├── aggregated_stats/       # Statistical summaries by corpus
│   ├── classification_data/    # ML classifier results
│   ├── plots/                  # Visualizations
│   ├── raw_data/               # Full feature matrix CSV
│   ├── reports/                # Analysis reports (English/Hebrew)
│   └── style_transfer_data/    # LLM experiment results
├── src/                        # Python source code
│   ├── analysis/               # Feature extraction, statistics, ML classifiers
│   │   └── perplexity/         # N-gram perplexity analysis
│   └── dictaParsing/           # DictaBERT text processing pipeline
├── final_report.docx           # Final Hebrew research report
├── PROJECT_SUMMARY.md          # Complete project workflow & conclusions
├── REPO_STRUCTURE.md           # Detailed repository documentation
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## 👥 Project Team

-   Osher Cohen
-   Yotam Tsur
-   Shir Ben Aderet
-   Omri Hirsch
-   Snir Boukris
