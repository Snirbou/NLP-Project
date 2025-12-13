DictaBERT Analysis and Integration Plan
=======================================
Source: "DictaBERT: A State-of-the-Art BERT Suite for Modern Hebrew" (Shmidman et al., 2023)
Hugging Face Collection: https://huggingface.co/collections/dicta-il/dictabert

1. What is DictaBERT?
---------------------
DictaBERT is a BERT-based language model specifically pre-trained on large-scale Hebrew corpora (Modern, Rabbinic, and historical texts). It addresses the unique challenges of Hebrew (complex morphology, root system, high ambiguity) better than multilingual models like mBERT.

### Key Capabilities Relevant to Our Project:
*   **DictaBERT-Morph (`dicta-il/dictabert-morph`)**: A fine-tuned model for morphological analysis. It can accurately segment prefixes (מ, ש, ה, ו, כ, ל, ב) and tag POS, gender, number, person, and tense with higher accuracy than older tools.
*   **DictaBERT-Seg (`dicta-il/dictabert-seg`)**: specialized for segmentation, critical for correctly splitting "וכשביתו" into "ו-כ-ש-בית-ו".
*   **DictaBERT-Lex (`dicta-il/dictabert-lex`)**: Likely useful for lexical embeddings, capturing semantic relationships between Biblical and Rabbinic vocabulary better than standard embeddings.
*   **DictaBERT-Syntax (`dicta-il/dictabert-syntax`)**: Can assist in generating dependency trees or syntactic features if our current parser (likely YAP or generic Dicta parser) is insufficient.
*   **Masked Language Modeling (Fill-Mask)**: The base model can predict missing words. This can be used to score "perplexity" or "likelihood" of a sentence appearing in a Biblical vs Rabbinic context if fine-tuned on those specific corpora.

2. Integration Strategies for Our Classification Task
-----------------------------------------------------

We can integrate DictaBERT at three different levels of complexity:

### Level 1: Better Feature Extraction (Replacement for Current Parser)
*   **Current State**: We currently rely on a JSON output from an existing tool (likely Dicta's older web parser or YAP) to get features like `cnt_noun`, `cnt_poss_construct`.
*   **DictaBERT Integration**: Use `dicta-il/dictabert-morph` to re-process the raw text.
    *   **Benefit**: Higher accuracy in identifying difficult forms (e.g., distinguishing Benoni as Noun vs Verb), leading to more reliable morphological features for our classifier.

### Level 2: Embeddings as Features (Hybrid Classifier)
*   **Strategy**: Instead of just counting "nouns" or "verbs", we pass the sentence through the base `dicta-il/dictabert` model to get a **sentence embedding** (a vector of numbers representing the semantic/stylistic meaning of the sentence).
*   **Action**:
    1.  Pass every Biblical/Rabbinic/Modern sentence through DictaBERT.
    2.  Extract the `[CLS]` token vector (768 dimensions).
    3.  Train a Logistic Regression or Random Forest on these 768 features (plus our existing manual features).
*   **Benefit**: Captures subtle stylistic nuances (word choice, phrasing patterns) that manual counts might miss.

### Level 3: Fine-Tuning for Classification (End-to-End Deep Learning)
*   **Strategy**: Take the pre-trained `dicta-il/dictabert` and "fine-tune" it specifically for our binary classification task (Biblical vs. Rabbinic).
*   **Action**:
    1.  Add a classification head on top of DictaBERT.
    2.  Train it on our 11,646 balanced sentences.
    3.  Use this fine-tuned model to predict the class of Modern Hebrew sentences directly.
*   **Benefit**: Likely to yield the highest possible accuracy (potentially >95%) because the model "reads" the whole sentence contextually rather than just looking at isolated features.

3. Recommendation
-----------------
Given that we have already built a feature-based workflow (Layer 4) and achieved good results (~89%), **Level 2 (Embeddings)** or **Level 3 (Fine-Tuning)** would be the most impactful additions.

*   **If we want to explain *why* (linguistic insights):** Stick to our current feature-based Random Forest, perhaps using DictaBERT (Level 1) only to clean up the data if we suspect errors.
*   **If we want the *best possible classifier*:** Implement Level 3 (Fine-Tuning DictaBERT). This would essentially replace the manual feature extraction step with a state-of-the-art neural approach.

4. Feasibility Note
-------------------
Running DictaBERT requires `torch` and `transformers` libraries. It is computationally heavier than our current script. For 700k Modern Hebrew sentences, inference might take several hours on a standard CPU (vs minutes for our current Random Forest).

