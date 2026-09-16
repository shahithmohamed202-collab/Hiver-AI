import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "human_annotations_clean.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "ec_model_predictions.csv"
)


FEATURES = [
    "D1_context_specificity",
    "D2_action_specificity",
    "D3_dm_redirect",
]


def encode_d4(value):
    """
    Encode categorical D4 outcome.

    We intentionally do not assign arbitrary numerical
    weights such as 1.0 / 0.7 / 0.4.

    Instead, use one-hot categorical indicators.
    """

    return {
        "positive_reaction": 1,
        "negative_reaction": 2,
        "neutral_reaction": 3,
        "silence_after_action": 4,
        "no_action_observed": 5,
    }.get(
        str(value),
        5
    )


def main():

    print("=" * 70)
    print("EVIDENCE COMPLETENESS MODEL")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Keep only fully annotated rows
    # --------------------------------------------------------

    df["evidence_complete"] = (
        df["evidence_complete"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[
        df["evidence_complete"].isin(
            ["YES", "NO"]
        )
    ].copy()

    if len(df) < 50:
        raise ValueError(
            "Not enough annotated examples. "
            "At least 50 are recommended."
        )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    X = pd.DataFrame(
        {
            "D1_context_specificity":
                pd.to_numeric(
                    df["D1_context_specificity"]
                ),

            "D2_action_specificity":
                pd.to_numeric(
                    df["D2_action_specificity"]
                ),

            "D3_dm_redirect":
                pd.to_numeric(
                    df["D3_dm_redirect"]
                ),

            "D4_positive":
                (
                    df["D4_observable_outcome"]
                    == "positive_reaction"
                ).astype(int),

            "D4_negative":
                (
                    df["D4_observable_outcome"]
                    == "negative_reaction"
                ).astype(int),

            "D4_neutral":
                (
                    df["D4_observable_outcome"]
                    == "neutral_reaction"
                ).astype(int),

            "D4_silence":
                (
                    df["D4_observable_outcome"]
                    == "silence_after_action"
                ).astype(int),

            "D4_no_action":
                (
                    df["D4_observable_outcome"]
                    == "no_action_observed"
                ).astype(int),
        }
    )

    y = (
        df["evidence_complete"]
        == "YES"
    ).astype(int)

    print(
        f"\nTraining examples: {len(df):,}"
    )

    print(
        f"Complete: {y.sum():,}"
    )

    print(
        f"Incomplete: {(1 - y).sum():,}"
    )

    # --------------------------------------------------------
    # Check both classes exist
    # --------------------------------------------------------

    if y.nunique() < 2:
        raise ValueError(
            "Human annotations contain only one class. "
            "We need both YES and NO examples."
        )

    # --------------------------------------------------------
    # Cross-validated predictions
    # --------------------------------------------------------

    class_counts = y.value_counts()

    min_class = class_counts.min()

    n_splits = min(
        5,
        int(min_class)
    )

    if n_splits < 2:
        raise ValueError(
            "Not enough examples in one class "
            "for cross-validation."
        )

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        penalty="l2",
        solver="liblinear",
        random_state=42,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CROSS-VALIDATED RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    print("\nConfusion matrix:")

    print(cm)

    # --------------------------------------------------------
    # Train final model on all annotations
    # --------------------------------------------------------

    model.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # Feature coefficients
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL COEFFICIENTS")
    print("=" * 70)

    coefficients = pd.DataFrame(
        {
            "feature": X.columns,
            "coefficient":
                model.coef_[0],
        }
    )

    coefficients["abs_coefficient"] = (
        coefficients["coefficient"]
        .abs()
    )

    coefficients = coefficients.sort_values(
        "abs_coefficient",
        ascending=False
    )

    print(
        coefficients[
            ["feature", "coefficient"]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    result = df[
        [
            "annotation_id",
            "conversation_id",
            "evidence_complete",
        ]
    ].copy()

    result["ec_probability"] = probabilities

    result["ec_prediction"] = np.where(
        predictions == 1,
        "YES",
        "NO",
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nPredictions saved to:\n{OUTPUT_FILE}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()