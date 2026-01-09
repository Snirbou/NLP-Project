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

For a detailed explanation of every file and folder, see [docs/english/REPO_STRUCTURE.md](docs/english/REPO_STRUCTURE.md).

```
NLP-Project/
├── data/                  # Input texts and parsed JSON outputs
├── docs/                  # Documentation (Requirements, Architecture, etc.)
├── results/               # Analysis results, plots, and reports
├── src/                   # Python source code
│   ├── analysis/          # Feature extraction and statistics scripts
│   └── dictaParsing/      # Text processing pipeline
└── README.md              # Project description
```

---

## 👥 Project Team

-   Osher Cohen
-   Yotam Tsur
-   Shir Ben Aderet
-   Omri Hirsch
-   Snir Boukris
