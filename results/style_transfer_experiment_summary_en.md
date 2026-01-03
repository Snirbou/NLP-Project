# Style Transfer Experiment Report: Historical Edit Distance
**Date:** [Current Date]

---

## 1. Rationale
Instead of classifying sentences (as done in Layer 4), this experiment measures the "distance" between Modern Hebrew and Biblical/Rabbinic Hebrew through a **Rewriting Task**.
**Hypothesis:** If Modern Hebrew is linguistically closer to Biblical Hebrew (as Doron suggests), a Large Language Model (LLM) should require fewer changes (edits) to transform a Modern sentence into a Biblical one, compared to transforming it into a Rabbinic one.

## 2. Methodology & Code
The script (`src/analysis/style_transfer_analysis.py`) performs the following:

1.  **Data Sampling:** Loads 50 random sentences from the Modern Hebrew News corpus ("Haaretz").
2.  **Language Model:** Uses a generative LLM. (In this feasibility test, we used a small model `hebrew-gpt_neo-small`. For final research, `DictaLM-2.0-Instruct` is recommended).
3.  **Rewriting Task:** The model receives two prompts for each sentence:
    *   "Rewrite the following sentence into **Biblical Hebrew**..."
    *   "Rewrite the following sentence into **Mishnaic (Rabbinic) Hebrew**..."
4.  **Distance Measurement:** Calculates the **Normalized Levenshtein Distance** between the original Modern sentence and the rewritten output.
    *   0.0 = Identical sentences.
    *   1.0 = Completely different sentences.

## 3. Test Run Results (Preliminary)
The following results were obtained from a feasibility run (Test Mode) on 5 sentences using a small model:

| Target Style | Average Edit Distance | Interpretation |
|:---|:---:|:---|
| **Biblical Hebrew** | **0.8777** | Requires massive changes (88% of text edited) |
| **Rabbinic Hebrew** | **0.7815** | Requires fewer changes (78% of text edited) |

## 4. Analysis & Conclusions (Proof of Concept)
Even though a weak "dummy" model was used, the results are consistent with our classifier findings:
*   The distance to **Rabbinic Hebrew** (0.78) is **smaller** than the distance to **Biblical Hebrew** (0.88).
*   **Implication:** It is "easier" to translate Modern Hebrew into Rabbinic Hebrew than into Biblical Hebrew. This supports the structural analysis that the Modern syntactic base is Rabbinic, thus requiring less "effort" to rewrite.

## 5. Recommendations
To obtain statistically significant research results, run the existing code using the full `dicta-il/dictalm2.0-instruct` model on a GPU-enabled machine. The full model can perform subtle syntactic transformations (e.g., VSO/SVO flipping, tense conversion) with much higher accuracy, likely sharpening the edit distance gap further.

