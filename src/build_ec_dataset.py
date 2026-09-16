import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EVIDENCE_FILE = (
    BASE_DIR
    / "output"
    / "amazonhelp_evidence_features.jsonl"
)

CONVERSATIONS_FILE = (
    BASE_DIR
    / "output"
    / "amazonhelp_conversations.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "ec_retrieval_dataset.jsonl"
)


# ============================================================
# D4 -> EVIDENCE VALUE
# ============================================================

D4_VALUES = {
    "positive_reaction": 1.00,
    "silence_after_action": 0.70,
    "neutral_reaction": 0.55,
    "negative_reaction": 0.20,
    "no_action_observed": 0.00,
}


# ============================================================
# EVIDENCE COMPLETENESS SCORE
# ============================================================

def calculate_ec_score(d1, d2, d3, d4):
    """
    Deterministic Evidence Completeness heuristic.

    This is NOT a learned probability.

    D1 = context specificity
    D2 = action specificity
    D3 = DM redirect before action
    D4 = observable outcome
    """

    d4_value = D4_VALUES.get(
        str(d4),
        0.0
    )

    score = (
        0.30 * float(d1)
        + 0.30 * float(d2)
        + 0.30 * d4_value
        + 0.10 * (1.0 - int(d3))
    )

    return round(
        max(0.0, min(1.0, score)),
        4
    )


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
# TEXT HELPERS
# ============================================================

def is_inbound(tweet):
    """
    Handles both boolean and numeric representations
    of the inbound field.
    """

    value = tweet.get("inbound")

    if value is True:
        return True

    if value == 1:
        return True

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes"
        }

    return False


def extract_messages(tweets):

    customer_messages = []
    agent_messages = []

    for tweet in tweets:

        text = str(
            tweet.get("text", "")
        ).strip()

        if not text:
            continue

        if is_inbound(tweet):
            customer_messages.append(text)
        else:
            agent_messages.append(text)

    if not customer_messages or not agent_messages:
        return None, None

    customer_query = customer_messages[0]

    # Prefer the longest agent response as the initial
    # grounding response. This is a preprocessing heuristic.
    agent_response = max(
        agent_messages,
        key=len
    )

    return customer_query, agent_response


def build_conversation_text(tweets):

    parts = []

    for tweet in tweets:

        text = str(
            tweet.get("text", "")
        ).strip()

        if not text:
            continue

        role = (
            "CUSTOMER"
            if is_inbound(tweet)
            else "AGENT"
        )

        parts.append(
            f"{role}: {text}"
        )

    return "\n".join(parts)


# ============================================================
# LOAD RECONSTRUCTED CONVERSATIONS
# ============================================================

def load_conversations():

    conversations = {}

    print(
        f"\nLoading conversations from:\n"
        f"{CONVERSATIONS_FILE}"
    )

    with CONVERSATIONS_FILE.open(
        "r",
        encoding="utf-8"
    ) as infile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            conversation = json.loads(line)

            conversation_id = str(
                conversation.get(
                    "conversation_id"
                )
            )

            conversations[
                conversation_id
            ] = conversation

    print(
        f"Conversations loaded : "
        f"{len(conversations):,}"
    )

    return conversations


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BUILDING EC RETRIEVAL DATASET"
    )
    print("=" * 70)

    print(
        f"\nEvidence file      : "
        f"{EVIDENCE_FILE}"
    )

    print(
        f"Conversation file  : "
        f"{CONVERSATIONS_FILE}"
    )

    print(
        f"Output             : "
        f"{OUTPUT_FILE}"
    )

    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"Evidence file not found:\n"
            f"{EVIDENCE_FILE}"
        )

    if not CONVERSATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Conversation file not found:\n"
            f"{CONVERSATIONS_FILE}"
        )

    conversations = load_conversations()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    processed = 0
    saved = 0
    skipped_missing_conversation = 0
    skipped_no_messages = 0

    tier_counts = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
    }

    score_sum = 0.0

    with (
        EVIDENCE_FILE.open(
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

            evidence = json.loads(line)

            conversation_id = str(
                evidence.get(
                    "conversation_id"
                )
            )

            conversation = conversations.get(
                conversation_id
            )

            if conversation is None:
                skipped_missing_conversation += 1
                continue

            tweets = conversation.get(
                "tweets",
                []
            )

            customer_query, agent_response = (
                extract_messages(tweets)
            )

            if (
                not customer_query
                or not agent_response
            ):
                skipped_no_messages += 1
                continue

            # ------------------------------------------------
            # Evidence features
            # ------------------------------------------------

            d1 = float(
                evidence.get(
                    "d1_context_specificity",
                    0.0
                )
            )

            d2 = float(
                evidence.get(
                    "d2_action_specificity",
                    0.0
                )
            )

            d3 = int(
                evidence.get(
                    "d3_dm_redirect_before_action",
                    0
                )
            )

            d4 = evidence.get(
                "d4_observable_outcome",
                "no_action_observed"
            )

            # ------------------------------------------------
            # EC score
            # ------------------------------------------------

            score = calculate_ec_score(
                d1,
                d2,
                d3,
                d4
            )

            tier = evidence_tier(
                score
            )

            # ------------------------------------------------
            # Full conversation text
            # ------------------------------------------------

            conversation_text = (
                build_conversation_text(
                    tweets
                )
            )

            # ------------------------------------------------
            # Output
            # ------------------------------------------------

            record = {
                "conversation_id":
                    evidence.get(
                        "conversation_id"
                    ),

                "customer_id":
                    evidence.get(
                        "customer_id"
                    ),

                "customer_query":
                    customer_query,

                "agent_response":
                    agent_response,

                "conversation_text":
                    conversation_text,

                "turn_count":
                    evidence.get(
                        "turn_count"
                    ),

                "D1_context_specificity":
                    d1,

                "D2_action_specificity":
                    d2,

                "D3_dm_redirect":
                    d3,

                "D4_observable_outcome":
                    d4,

                "evidence_completeness_score":
                    score,

                "evidence_tier":
                    tier,
            }

            outfile.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            saved += 1
            tier_counts[tier] += 1
            score_sum += score

    average_score = (
        score_sum / saved
        if saved
        else 0.0
    )

    print("\n" + "=" * 70)
    print(
        "EC RETRIEVAL DATASET CREATED"
    )
    print("=" * 70)

    print(
        f"\nEvidence records processed : "
        f"{processed:,}"
    )

    print(
        f"Records saved              : "
        f"{saved:,}"
    )

    print(
        f"Missing conversations      : "
        f"{skipped_missing_conversation:,}"
    )

    print(
        f"No usable messages         : "
        f"{skipped_no_messages:,}"
    )

    print(
        f"\nAverage EC score           : "
        f"{average_score:.4f}"
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
        "\nOutput saved to:"
    )

    print(OUTPUT_FILE)

    print("\nDone.")


if __name__ == "__main__":
    main()