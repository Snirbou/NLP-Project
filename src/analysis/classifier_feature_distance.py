import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise


CLASSIFIER_COLUMNS = [
    "pred_Lexical",
    "pred_Morphological",
    "pred_Syntactic",
    "pred_Combined",
]

IDENTIFIER_COLUMNS = ["corpus", "sub_corpus", "filename"]

LABELS = ["Biblical", "Rabbinic"]


def _build_feature_column_list(df: pd.DataFrame) -> List[str]:
    """
    Infer which columns to treat as numeric feature inputs.

    We exclude identifier and prediction columns and keep the remaining
    numeric columns for downstream normalization and aggregation.
    """
    excluded = set(IDENTIFIER_COLUMNS + CLASSIFIER_COLUMNS + ["predicted_class"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in excluded]
    return feature_cols


def normalize_features(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct normalized feature set for distance computations.

    Steps:
    1. Create per-token rates for all `cnt_*` columns using `len_tokens`
       as the length measure.
    2. Keep a small set of structural features (`len_tokens`,
       `tree_depth`, `is_v1`, `is_v2`) as they are.
    3. Apply z-score normalization over the Modern subset for all
       resulting features, with safeguards for zero variance.
    """
    df = df.copy()

    if "len_tokens" not in df.columns:
        raise ValueError("Expected column 'len_tokens' not found in DataFrame.")

    # Step 1: build per-token rates from count columns.
    rate_cols: List[str] = []
    len_tokens = df["len_tokens"].replace(0, np.nan)

    for col in feature_cols:
        if col.startswith("cnt_"):
            rate_col = f"rate_{col[4:]}"
            df[rate_col] = (df[col] / len_tokens).fillna(0.0)
            rate_cols.append(rate_col)

    # Step 2: structural/base features to keep.
    base_features = []
    for col in ["len_tokens", "tree_depth", "is_v1", "is_v2", "num_unique_lemmas"]:
        if col in df.columns:
            base_features.append(col)

    # Final feature list for distances: base features + rate features.
    dist_feature_cols = base_features + rate_cols

    # Step 3: z-score normalization across Modern sentences.
    df_norm = pd.DataFrame(index=df.index)
    for col in dist_feature_cols:
        series = df[col].astype(float)
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            # If the feature is constant, set normalized value to 0.0.
            df_norm[col] = 0.0
        else:
            df_norm[col] = (series - mean) / std

    return df_norm, dist_feature_cols


def build_classifier_vectors(
    df: pd.DataFrame, df_norm: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.Series]]]:
    """
    Build one feature distribution vector per classifier.

    For each classifier C in CLASSIFIER_COLUMNS and each label L in
    LABELS, compute the mean normalized feature vector over all Modern
    sentences where pred_C == L. Concatenate the two label-specific
    means into a single vector V_C in the order [Biblical, Rabbinic].

    Returns:
        - vectors_df: DataFrame with classifiers as index and
          concatenated feature components as columns.
        - per_label_means: nested dict mapping classifier -> label ->
          Series of per-feature means (for interpretation/reporting).
    """
    vectors: Dict[str, List[float]] = {}
    per_label_means: Dict[str, Dict[str, pd.Series]] = {}

    for clf_col in CLASSIFIER_COLUMNS:
        if clf_col not in df.columns:
            raise ValueError(f"Classifier prediction column '{clf_col}' missing from merged DataFrame.")

        clf_name = clf_col.replace("pred_", "")
        label_means: Dict[str, pd.Series] = {}
        concatenated: List[float] = []

        for label in LABELS:
            mask = df[clf_col] == label
            subset = df_norm.loc[mask, feature_cols]

            if subset.empty:
                # If a classifier does not predict a given label at all,
                # fall back to a zero vector for that segment, and note
                # it in the per-label means structure.
                mean_vec = pd.Series(0.0, index=feature_cols)
            else:
                mean_vec = subset.mean(axis=0)

            label_means[label] = mean_vec
            concatenated.extend(mean_vec.values.tolist())

        per_label_means[clf_name] = label_means

        # Build column names encoding label and feature for transparency.
        vec_index: List[str] = []
        for label in LABELS:
            vec_index.extend([f"{label}__{feat}" for feat in feature_cols])

        vectors[clf_name] = concatenated

    # Assemble into a DataFrame for downstream distance computation.
    # All classifiers share the same concatenated feature ordering.
    if vectors:
        example_name = next(iter(vectors))
        num_components = len(vectors[example_name])
        vector_cols: List[str] = []
        for label in LABELS:
            vector_cols.extend([f"{label}__{feat}" for feat in feature_cols])

        data = np.vstack([vectors[name] for name in vectors.keys()])
        vectors_df = pd.DataFrame(data=data, index=list(vectors.keys()), columns=vector_cols)
    else:
        vectors_df = pd.DataFrame()

    return vectors_df, per_label_means


def compute_distance_matrices(
    vectors_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise cosine and Euclidean distance matrices between
    classifier feature distribution vectors.
    """
    if vectors_df.empty:
        raise ValueError("No classifier vectors available for distance computation.")

    names = vectors_df.index.tolist()
    data = vectors_df.values

    cosine_dist = pairwise.cosine_distances(data)
    euclidean_dist = pairwise.pairwise_distances(data, metric="euclidean")

    cosine_df = pd.DataFrame(cosine_dist, index=names, columns=names)
    euclidean_df = pd.DataFrame(euclidean_dist, index=names, columns=names)

    # Zero out the diagonal explicitly for readability.
    for df in (cosine_df, euclidean_df):
        np.fill_diagonal(df.values, 0.0)

    return cosine_df, euclidean_df


def summarize_distance_matrix(cosine_df: pd.DataFrame) -> str:
    """
    Produce a human-readable text summary of the cosine distance matrix.

    The summary includes:
      - a small table of pairwise distances,
      - nearest and farthest neighbors per classifier,
      - global minimum and maximum off-diagonal distances.
    """
    classifiers = cosine_df.index.tolist()

    lines: List[str] = []
    lines.append("Feature-Based Classifier Distance Analysis")
    lines.append("========================================")
    lines.append("")
    lines.append("Cosine distance matrix between classifier feature-distribution vectors:")
    lines.append("")

    # Render a simple markdown-style table.
    header = "| Classifier | " + " | ".join(classifiers) + " |"
    sep = "|-----------|" + "|".join(["-----------" for _ in classifiers]) + "|"
    lines.append(header)
    lines.append(sep)
    for clf in classifiers:
        row_vals = [f"{cosine_df.loc[clf, other]:.3f}" for other in classifiers]
        row = "| " + clf + " | " + " | ".join(row_vals) + " |"
        lines.append(row)

    lines.append("")

    # Global min/max off-diagonal distances.
    mask_offdiag = ~np.eye(len(classifiers), dtype=bool)
    offdiag_values = cosine_df.values[mask_offdiag]
    if offdiag_values.size > 0:
        min_val = offdiag_values.min()
        max_val = offdiag_values.max()
        min_idx = np.where(cosine_df.values == min_val)
        max_idx = np.where(cosine_df.values == max_val)

        def _pair_from_indices(indices: Tuple[np.ndarray, np.ndarray]) -> List[Tuple[str, str]]:
            pairs: List[Tuple[str, str]] = []
            rows, cols = indices
            for r, c in zip(rows, cols):
                if r != c:
                    pairs.append((classifiers[r], classifiers[c]))
            return pairs

        closest_pairs = _pair_from_indices(min_idx)
        farthest_pairs = _pair_from_indices(max_idx)

        lines.append(f"Smallest off-diagonal cosine distance: {min_val:.3f}")
        for a, b in closest_pairs:
            lines.append(f"  - Closest pair: {a} vs {b}")

        lines.append(f"Largest off-diagonal cosine distance: {max_val:.3f}")
        for a, b in farthest_pairs:
            lines.append(f"  - Farthest pair: {a} vs {b}")

    lines.append("")
    lines.append("Per-classifier nearest and farthest neighbors (by cosine distance):")
    lines.append("")

    for clf in classifiers:
        row = cosine_df.loc[clf].copy()
        row_no_diag = row.drop(index=clf)
        nearest_name = row_no_diag.idxmin()
        nearest_val = row_no_diag.min()
        farthest_name = row_no_diag.idxmax()
        farthest_val = row_no_diag.max()
        lines.append(
            f"- {clf}: closest to {nearest_name} (d = {nearest_val:.3f}), "
            f"farthest from {farthest_name} (d = {farthest_val:.3f})."
        )

    lines.append("")
    lines.append(
        "Interpretation: smaller distances indicate that two classifiers rely on "
        "more similar patterns of linguistic features (for both Biblical and Rabbinic "
        "predictions) when classifying Modern Hebrew sentences."
    )

    return "\n".join(lines)


def write_outputs(
    cosine_df: pd.DataFrame,
    euclidean_df: pd.DataFrame,
    out_dir: str,
) -> str:
    """
    Persist distance matrices to CSV and generate a textual report.

    Returns the path to the report file for convenience.
    """
    os.makedirs(out_dir, exist_ok=True)

    cosine_path = os.path.join(out_dir, "classifiers_feature_distance_matrix_cosine.csv")
    euclidean_path = os.path.join(
        out_dir, "classifiers_feature_distance_matrix_euclidean.csv"
    )
    report_path = os.path.join(
        out_dir, "classifiers_feature_distance_report.txt"
    )

    cosine_df.to_csv(cosine_path)
    euclidean_df.to_csv(euclidean_path)

    summary_text = summarize_distance_matrix(cosine_df)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return report_path


def load_modern_features(path: str) -> pd.DataFrame:
    """
    Load Modern Hebrew sentence features and (combined) predictions.

    The CSV is expected to be the output of `classify_sentences.py`
    restricted to the Modern corpus.
    """
    df = pd.read_csv(path)

    # In case the file contains multiple corpora, filter explicitly.
    if "corpus" in df.columns:
        df = df[df["corpus"] == "Modern"].copy()

    return df.reset_index(drop=True)


def load_classifier_predictions(path: str) -> pd.DataFrame:
    """
    Load per-classifier predictions for Modern Hebrew sentences.

    The file is expected to have the same number and order of rows
    as `modern_classification_predictions.csv`, with identifier
    columns followed by `pred_*` columns for each classifier.
    """
    df = pd.read_csv(path)

    # Filter to Modern if the file contains additional corpora.
    if "corpus" in df.columns:
        df = df[df["corpus"] == "Modern"].copy()

    return df.reset_index(drop=True)


def merge_features_and_predictions(
    df_feat: pd.DataFrame, df_pred: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge Modern features with per-classifier predictions.

    We first check that identifier columns match row-wise and, if so,
    simply join on the existing index. This is the expected case since
    both CSVs are generated from the same underlying Modern subset in
    a fixed order.
    """
    if len(df_feat) != len(df_pred):
        raise ValueError(
            f"Row count mismatch between features ({len(df_feat)}) and predictions ({len(df_pred)})."
        )

    missing_cols = [c for c in IDENTIFIER_COLUMNS if c not in df_feat.columns or c not in df_pred.columns]
    if missing_cols:
        raise ValueError(f"Missing identifier columns in inputs: {missing_cols}")

    # Check that identifiers align row-wise; this should hold given the pipeline.
    id_equal = (df_feat[IDENTIFIER_COLUMNS].values == df_pred[IDENTIFIER_COLUMNS].values).all()
    if not id_equal:
        # As a safeguard, try a merge on the identifier columns and require 1:1 matches.
        merged = pd.merge(
            df_feat,
            df_pred[IDENTIFIER_COLUMNS + CLASSIFIER_COLUMNS],
            on=IDENTIFIER_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(df_feat):
            raise ValueError(
                "Identifier-based merge between features and predictions did not preserve all rows; "
                "please verify that the CSVs were generated from the same Modern subset."
            )
        return merged

    # Happy path: identifiers are aligned row-wise.
    for col in CLASSIFIER_COLUMNS:
        if col not in df_pred.columns:
            raise ValueError(f"Expected classifier prediction column '{col}' not found in predictions CSV.")

    merged = df_feat.copy()
    for col in CLASSIFIER_COLUMNS:
        merged[col] = df_pred[col].values

    return merged


def main() -> None:
    """
    Entry point for manual execution.

    This function currently just wires data loading and merging.
    Further steps (feature normalization, distance computation, and
    report generation) are implemented in subsequent iterations.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Project structure: NLP-Project/src/analysis -> go up two levels to reach project root.
    project_root = os.path.abspath(os.path.join(base_path, "..", ".."))
    results_dir = os.path.join(project_root, "results")
    classification_dir = os.path.join(results_dir, "classification_data")

    modern_features_path = os.path.join(
        classification_dir, "modern_classification_predictions.csv"
    )
    modern_preds_by_clf_path = os.path.join(
        classification_dir, "modern_predictions_by_classifier.csv"
    )

    print(f"Loading Modern features from: {modern_features_path}")
    df_feat = load_modern_features(modern_features_path)

    print(f"Loading per-classifier predictions from: {modern_preds_by_clf_path}")
    df_pred = load_classifier_predictions(modern_preds_by_clf_path)

    print("Merging features with classifier predictions...")
    df_merged = merge_features_and_predictions(df_feat, df_pred)

    print(f"Merged DataFrame shape: {df_merged.shape}")

    # Infer feature columns and construct normalized feature set.
    feature_cols = _build_feature_column_list(df_merged)
    print(f"Number of raw feature columns considered: {len(feature_cols)}")

    df_norm, dist_feature_cols = normalize_features(df_merged, feature_cols)
    print(f"Number of normalized feature columns for distances: {len(dist_feature_cols)}")

    # Build per-classifier feature distribution vectors.
    vectors_df, _ = build_classifier_vectors(df_merged, df_norm, dist_feature_cols)
    print("Per-classifier feature distribution vectors constructed.")

    # Compute distance matrices.
    cosine_df, euclidean_df = compute_distance_matrices(vectors_df)
    print("Cosine distance matrix:")
    print(cosine_df)
    print("Euclidean distance matrix:")
    print(euclidean_df)

    # Write outputs to the classification_data directory.
    report_path = write_outputs(cosine_df, euclidean_df, classification_dir)
    print(f"Distance matrices and report written to '{classification_dir}'.")
    print(f"Textual summary report: {report_path}")


if __name__ == "__main__":
    main()


