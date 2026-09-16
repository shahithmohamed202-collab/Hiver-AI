import json
import random
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "amazonhelp_evidence_features.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "human_annotation_sample.csv"
)

SAMPLE_SIZE = 200
RANDOM_SEED = 42


def main():

    print("=" * 70)
    print("CREATING HUMAN ANNOTATION SAMPLE")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    random.seed(RANDOM_SEED)

    records = []

    print("\nReading evidence-feature dataset...")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            records.append(json.loads(line))

    print(
        f"Total conversations available: "
        f"{len(records):,}"
    )

    if len(records) <= SAMPLE_SIZE:
        sample = records

    else:
        # ----------------------------------------------------
        # Stratify by D2 x D3 x D4 so the sample contains
        # different evidence patterns.
        # ----------------------------------------------------

        groups = {}

        for record in records:

            key = (
                record["D2_action_specificity"],
                record["D3_dm_redirect"],
                record["D4_observable_outcome"],
            )

            groups.setdefault(key, []).append(record)

        print(
            f"Evidence strata discovered: "
            f"{len(groups)}"
        )

        sample = []

        # First take approximately equal numbers
        # from each evidence stratum.
        per_group = max(
            1,
            SAMPLE_SIZE // len(groups)
        )

        for group_records in groups.values():

            take = min(
                per_group,
                len(group_records)
            )

            sample.extend(
                random.sample(
                    group_records,
                    take
                )
            )

        # Fill remaining slots randomly.
        if len(sample) < SAMPLE_SIZE:

            selected_ids = {
                record["conversation_id"]
                for record in sample
            }

            remaining = [
                record
                for record in records
                if record["conversation_id"]
                not in selected_ids
            ]

            remaining_needed = (
                SAMPLE_SIZE - len(sample)
            )

            if remaining:

                sample.extend(
                    random.sample(
                        remaining,
                        min(
                            remaining_needed,
                            len(remaining)
                        )
                    )
                )

    random.shuffle(sample)

    rows = []

    for index, record in enumerate(
        sample,
        start=1
    ):

        customer_messages = [
            tweet["text"]
            for tweet in record["tweets"]
            if tweet["inbound"]
        ]

        agent_messages = [
            tweet["text"]
            for tweet in record["tweets"]
            if (
                not tweet["inbound"]
                and tweet["author_id"]
                == "AmazonHelp"
            )
        ]

        rows.append(
            {
                "annotation_id": index,

                "conversation_id":
                    record["conversation_id"],

                "customer_id":
                    record["customer_id"],

                "turn_count":
                    record["turn_count"],

                "D1_context_specificity":
                    record["D1_context_specificity"],

                "D2_action_specificity":
                    record["D2_action_specificity"],

                "D3_dm_redirect":
                    record["D3_dm_redirect"],

                "D4_observable_outcome":
                    record["D4_observable_outcome"],

                "customer_messages":
                    " || ".join(
                        customer_messages
                    ),

                "agent_messages":
                    " || ".join(
                        agent_messages
                    ),

                # Human annotation columns
                "evidence_complete":
                    "",

                "action_observed":
                    "",

                "customer_reaction_observed":
                    "",

                "dm_redirect":
                    "",

                "notes":
                    "",
            }
        )

    df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("ANNOTATION SAMPLE CREATED")
    print("=" * 70)

    print(
        f"Sample size: {len(df):,}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("\nHuman annotation columns:")
    print("  evidence_complete")
    print("  action_observed")
    print("  customer_reaction_observed")
    print("  dm_redirect")
    print("  notes")

    print("\nDone.")


if __name__ == "__main__":
    main()