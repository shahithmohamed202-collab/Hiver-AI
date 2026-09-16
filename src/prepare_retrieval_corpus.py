import json
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "ec_retrieval_dataset.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "retrieval_corpus.jsonl"
)


# ============================================================
# ACTION SIGNALS
# ============================================================

ACTION_PATTERNS = [
    r"\bchecked\b",
    r"\breviewed\b",
    r"\bupdated\b",
    r"\brefunded\b",
    r"\bissued\b",
    r"\bprocessed\b",
    r"\breplaced\b",
    r"\bcancelled\b",
    r"\bcanceled\b",
    r"\bverified\b",
    r"\breset\b",
    r"\bresent\b",
    r"\bescalated\b",
    r"\binvestigated\b",
    r"\bconfirmed\b",
    r"\bcontacted\b",
    r"\bresolved\b",
    r"\bfixed\b",
    r"\bactivated\b",
    r"\bremoved\b",
    r"\bchanged\b",
    r"\brestored\b",
    r"\bcredited\b",
    r"\bforwarded\b",
    r"\bsent\b",
    r"\breceived\b",
    r"\btried\b",
]

ACTION_REGEX = [
    re.compile(
        pattern,
        re.IGNORECASE
    )
    for pattern in ACTION_PATTERNS
]


# ============================================================
# DM REDIRECT SIGNALS
# ============================================================

DM_PATTERNS = [
    r"\bdm\b",
    r"\bdirect message\b",
    r"\bprivate message\b",
    r"\bsend us a message\b",
    r"\bmessage us\b",
]

DM_REGEX = [
    re.compile(
        pattern,
        re.IGNORECASE
    )
    for pattern in DM_PATTERNS
]


# ============================================================
# GENERIC RESPONSE SIGNALS
# ============================================================

GENERIC_PATTERNS = [
    r"\bsorry\b",
    r"\bapolog",
    r"\bthank you\b",
    r"\bthanks\b",
    r"\bwelcome\b",
    r"\bplease let us know\b",
    r"\bwe'd be happy to help\b",
    r"\bwe are happy to help\b",
]

GENERIC_REGEX = [
    re.compile(
        pattern,
        re.IGNORECASE
    )
    for pattern in GENERIC_PATTERNS
]


# ============================================================
# HELPERS
# ============================================================

def action_count(text):

    return sum(
        1
        for pattern in ACTION_REGEX
        if pattern.search(text)
    )


def has_dm_redirect(text):

    return any(
        pattern.search(text)
        for pattern in DM_REGEX
    )


def is_generic(text):

    return any(
        pattern.search(text)
        for pattern in GENERIC_REGEX
    )


# ============================================================
# SELECT BEST AGENT RESPONSE
# ============================================================

def select_best_agent_response(
    conversation_text
):
    """
    Select the strongest grounding response from the
    conversation text.

    The EC score itself comes from the separately computed
    evidence features. This function is only for selecting
    a useful textual response for retrieval.
    """

    candidates = []

    lines = conversation_text.splitlines()

    for index, line in enumerate(lines):

        if not line.startswith("AGENT:"):
            continue

        text = line[
            len("AGENT:"):
        ].strip()

        if not text:
            continue

        actions = action_count(text)
        dm = has_dm_redirect(text)
        generic = is_generic(text)

        score = 0.0

        # Explicit action language is strongest.
        score += actions * 10.0

        # Give longer informative responses some weight.
        score += min(
            len(text),
            500
        ) / 100.0

        # DM-only messages are weak grounding evidence.
        if dm and actions == 0:
            score -= 15.0

        # Generic-only responses are weak.
        if (
            generic
            and actions == 0
        ):
            score -= 5.0

        candidates.append({
            "index": index,
            "text": text,
            "score": score,
            "action_count": actions,
            "dm_redirect": dm,
        })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            item["score"],
            item["action_count"],
            len(item["text"])
        )
    )


# ============================================================
# SELECT CUSTOMER QUERY
# ============================================================

def select_customer_query(
    conversation_text
):

    for line in conversation_text.splitlines():

        if not line.startswith(
            "CUSTOMER:"
        ):
            continue

        text = line[
            len("CUSTOMER:"):
        ].strip()

        if text:
            return text

    return None


# ============================================================
# EVIDENCE TIER
# ============================================================

def evidence_tier(score):

    if score >= 0.75:
        return "HIGH"

    if score >= 0.50:
        return "MEDIUM"

    return "LOW"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PREPARING RETRIEVAL CORPUS"
    )
    print("=" * 70)

    print(
        f"\nInput : {INPUT_FILE}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    processed = 0
    saved = 0
    skipped = 0

    action_response_count = 0
    dm_only_count = 0
    generic_only_count = 0

    tier_counts = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
    }

    total_ec = 0.0

    with (
        INPUT_FILE.open(
            "r",
            encoding="utf-8"
        ) as infile,

        OUTPUT_FILE.open(
            "w",
            encoding="utf-8"
        ) as outfile
    ):

        for line in infile:

            line = line.strip()

            if not line:
                continue

            processed += 1

            record = json.loads(line)

            conversation_text = str(
                record.get(
                    "conversation_text",
                    ""
                )
            ).strip()

            customer_query = (
                select_customer_query(
                    conversation_text
                )
            )

            best_agent = (
                select_best_agent_response(
                    conversation_text
                )
            )

            if (
                not customer_query
                or not best_agent
            ):
                skipped += 1
                continue

            # ------------------------------------------------
            # Existing EC score
            # ------------------------------------------------

            ec_score = float(
                record.get(
                    "evidence_completeness_score",
                    0.0
                )
            )

            tier = record.get(
                "evidence_tier"
            )

            if tier not in {
                "HIGH",
                "MEDIUM",
                "LOW"
            }:
                tier = evidence_tier(
                    ec_score
                )

            total_ec += ec_score
            tier_counts[tier] += 1

            # ------------------------------------------------
            # Response statistics
            # ------------------------------------------------

            if (
                best_agent["action_count"]
                > 0
            ):
                action_response_count += 1

            if (
                best_agent["dm_redirect"]
                and best_agent["action_count"]
                == 0
            ):
                dm_only_count += 1

            if (
                is_generic(
                    best_agent["text"]
                )
                and best_agent[
                    "action_count"
                ] == 0
            ):
                generic_only_count += 1

            # ------------------------------------------------
            # Retrieval record
            # ------------------------------------------------

            retrieval_record = {
                "conversation_id":
                    record.get(
                        "conversation_id"
                    ),

                "customer_id":
                    record.get(
                        "customer_id"
                    ),

                "customer_query":
                    customer_query,

                "agent_response":
                    best_agent["text"],

                "conversation_text":
                    conversation_text,

                "turn_count":
                    record.get(
                        "turn_count"
                    ),

                "D1_context_specificity":
                    record.get(
                        "D1_context_specificity",
                        0.0
                    ),

                "D2_action_specificity":
                    record.get(
                        "D2_action_specificity",
                        0.0
                    ),

                "D3_dm_redirect":
                    record.get(
                        "D3_dm_redirect",
                        0
                    ),

                "D4_observable_outcome":
                    record.get(
                        "D4_observable_outcome",
                        "no_action_observed"
                    ),

                "evidence_completeness_score":
                    ec_score,

                "evidence_tier":
                    tier,

                "selected_agent_action_count":
                    best_agent[
                        "action_count"
                    ],

                "selected_agent_dm_redirect":
                    best_agent[
                        "dm_redirect"
                    ],
            }

            outfile.write(
                json.dumps(
                    retrieval_record,
                    ensure_ascii=False
                ) + "\n"
            )

            saved += 1

    average_ec = (
        total_ec / saved
        if saved
        else 0.0
    )

    print("\n" + "=" * 70)
    print(
        "RETRIEVAL CORPUS CREATED"
    )
    print("=" * 70)

    print(
        f"\nRecords processed : "
        f"{processed:,}"
    )

    print(
        f"Records saved     : "
        f"{saved:,}"
    )

    print(
        f"Records skipped   : "
        f"{skipped:,}"
    )

    print(
        f"\nAverage EC score : "
        f"{average_ec:.4f}"
    )

    print(
        "\nEvidence tier distribution:"
    )

    for tier in (
        "HIGH",
        "MEDIUM",
        "LOW"
    ):

        count = tier_counts[tier]

        percentage = (
            count / saved * 100
            if saved
            else 0.0
        )

        print(
            f"  {tier:<8}"
            f"{count:>8,}"
            f" ({percentage:6.2f}%)"
        )

    print(
        "\nSelected responses with explicit action : "
        f"{action_response_count:,}"
    )

    print(
        "Selected responses that are DM-only     : "
        f"{dm_only_count:,}"
    )

    print(
        "Selected responses that are generic-only: "
        f"{generic_only_count:,}"
    )

    print(
        "\nOutput saved to:"
    )

    print(OUTPUT_FILE)

    print("\nDone.")


if __name__ == "__main__":
    main()