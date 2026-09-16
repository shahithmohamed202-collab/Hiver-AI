import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "human_annotation_sample.csv"
)


REQUIRED_COLUMNS = [
    "annotation_id",
    "conversation_id",
    "D1_context_specificity",
    "D2_action_specificity",
    "D3_dm_redirect",
    "D4_observable_outcome",
    "evidence_complete",
    "action_observed",
    "customer_reaction_observed",
    "dm_redirect",
]


def clean_value(value):
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def main():

    print("=" * 70)
    print("HUMAN ANNOTATION VALIDATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"\nRows in annotation file: {len(df):,}"
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Missing columns:")

        for column in missing_columns:
            print(f"  - {column}")

        return

    # --------------------------------------------------------
    # Normalize annotation values
    # --------------------------------------------------------

    annotation_columns = [
        "evidence_complete",
        "action_observed",
        "customer_reaction_observed",
        "dm_redirect",
    ]

    for column in annotation_columns:
        df[column] = df[column].apply(clean_value)

    # --------------------------------------------------------
    # Validate YES / NO values
    # --------------------------------------------------------

    print("\nAnnotation validation:")

    all_valid = True

    for column in annotation_columns:

        values = set(
            value
            for value in df[column]
            if value
        )

        invalid = values - {"YES", "NO"}

        if invalid:

            all_valid = False

            print(
                f"  {column}: INVALID VALUES "
                f"{sorted(invalid)}"
            )

        else:

            print(
                f"  {column}: OK"
            )

    # --------------------------------------------------------
    # Count completed annotations
    # --------------------------------------------------------

    complete_mask = (
        df[annotation_columns]
        .notna()
        .all(axis=1)
        & (
            df[annotation_columns]
            != ""
        ).all(axis=1)
    )

    completed = int(
        complete_mask.sum()
    )

    incomplete = len(df) - completed

    print(
        f"\nCompleted annotations : {completed:,}"
    )

    print(
        f"Incomplete annotations: {incomplete:,}"
    )

    # --------------------------------------------------------
    # Evidence completeness distribution
    # --------------------------------------------------------

    print(
        "\nEvidence completeness:"
    )

    print(
        df["evidence_complete"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # Other annotation distributions
    # --------------------------------------------------------

    for column in [
        "action_observed",
        "customer_reaction_observed",
        "dm_redirect",
    ]:

        print(
            f"\n{column}:"
        )

        print(
            df[column]
            .value_counts(dropna=False)
            .to_string()
        )

    # --------------------------------------------------------
    # Basic consistency checks
    # --------------------------------------------------------

    print(
        "\nConsistency checks:"
    )

    # If evidence is complete, there should normally
    # be either an action or useful observable evidence.
    suspicious_complete = df[
        (df["evidence_complete"] == "YES")
        & (df["action_observed"] == "NO")
        & (df["customer_reaction_observed"] == "NO")
    ]

    print(
        "  Complete but no action/reaction: "
        f"{len(suspicious_complete):,}"
    )

    # DM redirect itself does not mean incomplete,
    # so we only flag it for manual inspection.
    complete_with_dm = df[
        (df["evidence_complete"] == "YES")
        & (df["dm_redirect"] == "YES")
    ]

    print(
        "  Complete + DM redirect: "
        f"{len(complete_with_dm):,}"
    )

    # --------------------------------------------------------
    # Duplicate conversation IDs
    # --------------------------------------------------------

    duplicate_ids = df[
        df["conversation_id"].duplicated(
            keep=False
        )
    ]

    print(
        "  Duplicate conversation IDs: "
        f"{len(duplicate_ids):,}"
    )

    # --------------------------------------------------------
    # Save cleaned annotations
    # --------------------------------------------------------

    cleaned_file = (
        BASE_DIR
        / "output"
        / "human_annotations_clean.csv"
    )

    df.to_csv(
        cleaned_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if all_valid and incomplete == 0:
        print(
            "STATUS: READY FOR MODEL TRAINING"
        )
    else:
        print(
            "STATUS: FIX ANNOTATIONS BEFORE MODEL TRAINING"
        )

    print("=" * 70)

    print(
        f"\nCleaned file:\n{cleaned_file}"
    )


if __name__ == "__main__":
    main()