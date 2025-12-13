DictaBERT Embedding Classification Report
=======================================
Date: [Current Date]
Model: dicta-il/dictabert
Training Size: 11,646 (Balanced: 5,823 Biblical / 5,823 Rabbinic)

1. Cross-Validation Results (Biblical vs Rabbinic)
--------------------------------------------------
The DictaBERT model achieved near-perfect accuracy in distinguishing between the two ancient dialects. This vastly outperforms the manual feature-based model (which achieved ~89%).

| Metric    | Score    |
|-----------|----------|
| Accuracy  | 0.9987   |
| Precision | 0.9990   |
| Recall    | 0.9985   |
| F1 Score  | 0.9987   |

> **Insight:** DictaBERT captures the full semantic and stylistic context of the sentence, allowing it to identify the dialect with almost 100% certainty, unlike the manual model which relied on counting specific isolated features (like specific words or constructs).

2. Modern Hebrew Classification Distribution
--------------------------------------------
Percentage of sentences classified as **Biblical** vs **Rabbinic** by DictaBERT.

| Genre (Sub-Corpus) | Biblical (%) | Rabbinic (%) |
|--------------------|--------------|--------------|
| Blogs              | 1.30%        | 98.70%       |
| Medical            | 0.55%        | 99.45%       |
| News               | 5.20%        | 94.80%       |
| Tapuz (Forums)     | 4.75%        | 95.25%       |

3. Comparison & Analysis
------------------------
When comparing these results to the Manual Feature Model (Random Forest on counts):

| Genre    | % Biblical (Manual Model) | % Biblical (DictaBERT) |
|----------|---------------------------|------------------------|
| News     | 21.0%                     | 5.2%                   |
| Medical  | 19.6%                     | 0.6%                   |
| Tapuz    | 14.4%                     | 4.8%                   |
| Blogs    | 11.2%                     | 1.3%                   |

### **Key Conclusions**
1.  **Overwhelming Rabbinic Dominance:** When looking at the full context of a sentence (as DictaBERT does), Modern Hebrew is overwhelmingly classified as **Rabbinic** (95-99% across all genres).
2.  **The "Biblical Illusion" Corrected:** The Manual Model previously classified ~20% of News/Medical sentences as Biblical. DictaBERT reduces this to ~0.6-5%. This suggests that the "Biblical" feel of formal Hebrew is likely superficial (e.g., specific vocabulary choices or high register), but the deep structure remains firmly Rabbinic.
3.  **Genre Consistency:** Unlike the manual model which showed large variance between genres (11% vs 21%), DictaBERT sees all Modern Hebrew genres as fundamentally the same dialect (Rabbinic), with only very minor stylistic differences.
