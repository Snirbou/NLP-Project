# Hebrew Syntax Parsing – Mini Project (2026)

This repository contains the initial infrastructure for our  NLP project, whose goal is to examine computationally whether Modern Hebrew is syntactically closer to Biblical Hebrew or to Mishnaic Hebrew - following the hypothesis proposed by Edit Doron.  

---

## 📌 Project Overview

The project currently includes:

- A recursive text-processing script that scans folders of Hebrew `.txt` files.
- A sentence-splitting module for Hebrew (., ?, !, ׃).
- Integration with **DictaBERT-Joint** (Dicta) for:
  - Lemmas  
  - POS tags  
  - Morphological features  
  - Dependency relations  
  - Head–dependent structure per token
- Automatic JSON output for each parsed sentence.

This forms the foundation for the later stages of the project, where we will compute syntactic and morphological statistics across multiple Hebrew corpora.

---

## 📁 Folder Structure

NLP-Project/
│
├── batch_run_recursive.py # Recursively parses all .txt files in input_texts/
├── dicta_batch_parser.py # Wraps the DictaBERT model
└── README.md # Project description

---

## 👥 Project Team

- Osher Cohen
- Yotam Tsur
- Shir Ben Aderet
- Omri Hirsch
- Snir Boukris
