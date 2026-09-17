from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "output" / "human_verification_150.csv"
OUTPUT_FILE = PROJECT_ROOT / "output" / "human_annotations_clean.csv"

LABEL_COLUMNS = [
    "evidence_complete",
    "action_observed",
    "customer_reaction_observed",
    "dm_redirect",
]

EXPECTED_ROWS = 150
VALID_LABELS = {"YES", "NO"}


def normalize_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value in VALID_LABELS:
        return value

    return value


def main():
    print("=" * 70)
    print("HUMAN ANNOTATION VALIDATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"\nERROR: Annotation file not found:")
        print(INPUT_FILE)
        return

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput file:")
    print(INPUT_FILE)

    print(f"\nRows in annotation file: {len(df)}")

    if len(df) != EXPECTED_ROWS:
        print(
            f"WARNING: Expected {EXPECTED_ROWS} rows, "
            f"but found {len(df)} rows."
        )

    # Normalize labels
    for column in LABEL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(normalize_label)

    print("\nAnnotation validation:")

    all_valid = True

    for column in LABEL_COLUMNS:
        if column not in df.columns:
            print(f"  {column}: MISSING COLUMN")
            all_valid = False
            continue

        invalid = df[column].notna() & ~df[column].isin(VALID_LABELS)

        missing = df[column].isna().sum()

        if invalid.sum() == 0 and missing == 0:
            print(f"  {column}: OK")
        else:
            all_valid = False

            if missing > 0:
                print(f"  {column}: {missing} missing")

            if invalid.sum() > 0:
                print(f"  {column}: {invalid.sum()} invalid values")

    # Completed rows
    completed_mask = df[LABEL_COLUMNS].notna().all(axis=1)

    completed = int(completed_mask.sum())
    incomplete = int((~completed_mask).sum())

    print(f"\nCompleted annotations : {completed}")
    print(f"Incomplete annotations: {incomplete}")

    # Distribution
    print("\nFinal label distributions:")

    for column in LABEL_COLUMNS:
        print(f"\n{column}:")

        if column in df.columns:
            counts = df[column].value_counts(dropna=False)

            yes_count = int((df[column] == "YES").sum())
            no_count = int((df[column] == "NO").sum())

            print(f"  YES: {yes_count}")
            print(f"  NO : {no_count}")

            if df[column].isna().sum() > 0:
                print(f"  MISSING: {int(df[column].isna().sum())}")

    # Consistency checks
    print("\nConsistency checks:")

    # Evidence complete but neither action nor reaction
    complete_no_support = (
        (df["evidence_complete"] == "YES")
        & (df["action_observed"] == "NO")
        & (df["customer_reaction_observed"] == "NO")
    ).sum()

    print(
        f"  Complete but no action/reaction: "
        f"{int(complete_no_support)}"
    )

    # Complete + DM redirect is allowed, so this is informational only
    complete_dm = (
        (df["evidence_complete"] == "YES")
        & (df["dm_redirect"] == "YES")
    ).sum()

    print(f"  Complete + DM redirect: {int(complete_dm)}")

    # Duplicate conversation IDs
    duplicate_ids = 0

    if "conversation_id" in df.columns:
        duplicate_ids = int(
            df["conversation_id"].duplicated().sum()
        )

    print(f"  Duplicate conversation IDs: {duplicate_ids}")

    # Save only complete annotations
    clean_df = df.loc[completed_mask].copy()

    clean_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)

    if (
        len(df) == EXPECTED_ROWS
        and completed == EXPECTED_ROWS
        and all_valid
        and duplicate_ids == 0
    ):
        print("STATUS: VALIDATED — 150/150 annotations complete")
    else:
        print("STATUS: FIX ANNOTATIONS BEFORE MODEL TRAINING")

    print("=" * 70)

    print("\nCleaned file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()