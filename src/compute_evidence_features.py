import json
import re
from pathlib import Path
from collections import Counter


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "output" / "amazonhelp_conversations.jsonl"
OUTPUT_PATH = BASE_DIR / "output" / "amazonhelp_evidence_features.jsonl"

AGENT_NAME = "AmazonHelp"


# =============================================================================
# REGEX / TEXT HELPERS
# =============================================================================

def normalize(text):
    if text is None:
        return ""

    text = str(text)

    # Normalize HTML entities commonly present in TWCS.
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    return re.sub(r"\s+", " ", text).strip()


def words(text):
    return re.findall(r"\b[\w'-]+\b", normalize(text).lower())


def contains_pattern(text, patterns):
    text = normalize(text).lower()

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def has_digit(text):
    return bool(re.search(r"\d", normalize(text)))


# =============================================================================
# D1 — CONTEXT SPECIFICITY
# =============================================================================

DOMAIN_KEYWORDS = {
    "order",
    "refund",
    "return",
    "delivery",
    "delivered",
    "package",
    "parcel",
    "shipment",
    "shipping",
    "payment",
    "charged",
    "charge",
    "account",
    "password",
    "login",
    "replacement",
    "product",
    "item",
    "damaged",
    "broken",
    "tracking",
    "address",
    "cancel",
    "cancelled",
    "canceled",
    "invoice",
    "billing",
    "credit",
    "debit",
}


def has_domain_keyword(text):
    token_set = set(words(text))

    return bool(token_set & DOMAIN_KEYWORDS)


def compute_d1(conversation):
    customer_messages = [
        normalize(tweet.get("text", ""))
        for tweet in conversation.get("tweets", [])
        if tweet.get("inbound") is True
        and str(tweet.get("author_id")) != AGENT_NAME
    ]

    if not customer_messages:
        return 0.0

    opening = customer_messages[0]

    token_count = len(words(opening))

    score = 0.0

    # Context length.
    if token_count >= 5:
        score += 0.25

    if token_count >= 10:
        score += 0.20

    if token_count >= 20:
        score += 0.15

    # Domain-specific context.
    if has_domain_keyword(opening):
        score += 0.25

    # Concrete identifiers.
    if has_digit(opening):
        score += 0.15

    return min(score, 1.0)


# =============================================================================
# D2 — ACTION / EVIDENCE SPECIFICITY
# =============================================================================

# DM redirects are deliberately separated from useful evidence.
DM_PATTERNS = [
    r"\bdm\b",
    r"\bdirect message\b",
    r"\bprivate message\b",
    r"\bmessage us privately\b",
    r"\bsend us a dm\b",
    r"\bsend me a dm\b",
    r"\bplease dm\b",
    r"\bcontact us via dm\b",
    r"\bconnect with us via dm\b",
    r"\breach out to us by phone\b",
    r"\bcontact us by phone\b",
]


# Future / conditional / request language.
# These patterns prevent promises and requests from being treated as
# completed observable actions.
FUTURE_OR_REQUEST_PATTERNS = [
    r"\bwe['’]ll\b",
    r"\bi['’]ll\b",
    r"\bwe will\b",
    r"\bi will\b",
    r"\bwe can\b",
    r"\bi can\b",
    r"\bwe could\b",
    r"\bi could\b",
    r"\bwe may\b",
    r"\bi may\b",
    r"\bwe would\b",
    r"\bi would\b",
    r"\bwe['’]d like to\b",
    r"\bi['’]d like to\b",
    r"\bwe would like to\b",
    r"\bi would like to\b",
    r"\blet us\b",
    r"\blet['’]s\b",
    r"\bplease\b",
    r"\bkindly\b",
    r"\bwould you\b",
    r"\bcould you\b",
    r"\bcan you\b",
    r"\bhave you\b",
    r"\bdid you\b",
    r"\bif you\b",
    r"\byou can\b",
    r"\byou may\b",
]


# Strong evidence of an action that has actually occurred.
OBSERVED_ACTION_PATTERNS = [
    # Present perfect / completed action.
    r"\bwe['’]ve\s+(?:sent|responded|received|forwarded|verified|confirmed|updated|processed|refunded|credited|replaced|reset|resent|escalated|contacted)\b",
    r"\bi['’]ve\s+(?:sent|responded|received|forwarded|verified|confirmed|updated|processed|refunded|credited|replaced|reset|resent|escalated|contacted)\b",

    r"\bwe have\s+(?:sent|responded|received|forwarded|verified|confirmed|updated|processed|refunded|credited|replaced|reset|resent|escalated|contacted)\b",
    r"\bi have\s+(?:sent|responded|received|forwarded|verified|confirmed|updated|processed|refunded|credited|replaced|reset|resent|escalated|contacted)\b",

    # Explicit checking / investigation that has already happened.
    r"\bi['’]ve\s+(?:checked|reviewed|investigated|looked into|tried)\b",
    r"\bwe['’]ve\s+(?:checked|reviewed|investigated|looked into|tried)\b",

    r"\bi have\s+(?:checked|reviewed|investigated|looked into|tried)\b",
    r"\bwe have\s+(?:checked|reviewed|investigated|looked into|tried)\b",

    # Simple past.
    r"\b(?:i|we)\s+(?:checked|reviewed|investigated|verified|confirmed|processed|refunded|replaced|reset|resent|escalated|contacted|resolved|fixed|restored|activated|removed|cancelled|canceled|credited|forwarded|sent|received|tried)\b",

    # Passive completed actions.
    r"\b(?:has|have|was|were)\s+been\s+(?:sent|processed|refunded|credited|updated|replaced|cancelled|canceled|verified|confirmed|received|forwarded|escalated)\b",

    # Ongoing support work.
    r"\bwe['’]?re\s+working on\b",
    r"\bwe are\s+working on\b",
    r"\bi['’]?m\s+working on\b",
    r"\bi am\s+working on\b",

    r"\bwe['’]?re\s+following up\b",
    r"\bwe are\s+following up\b",
    r"\bi['’]?m\s+following up\b",
    r"\bi am\s+following up\b",

    r"\bwe['’]?re\s+looking into\b",
    r"\bwe are\s+looking into\b",

    # Explicit completed attempts.
    r"\b(?:i|we)\s+tried to\b",
    r"\b(?:i|we)\s+attempted to\b",

    # Correspondence / response evidence.
    r"\bwe['’]ve\s+sent\s+the\s+correspondence\b",
    r"\bwe have\s+sent\s+the\s+correspondence\b",
    r"\bwe['’]ve\s+responded\s+to\s+you\b",
    r"\bwe have\s+responded\s+to\s+you\b",
    r"\bwe['’]ve\s+received\s+your\s+details\b",
    r"\bwe have\s+received\s+your\s+details\b",
]


# Specific informational or actionable guidance.
# This is useful evidence but is deliberately scored below a confirmed
# completed action.
SPECIFIC_GUIDANCE_PATTERNS = [
    # Time / policy information.
    r"\b\d+\s*(?:-|–|to)\s*\d+\s*(?:business\s+)?days?\b",
    r"\bwithin\s+\d+\s+days?\b",
    r"\btakes\s+\d+\s+(?:business\s+)?days?\b",
    r"\bmay take\s+\d+\s+(?:business\s+)?days?\b",

    # Tracking / status guidance.
    r"\btrack\s+(?:the\s+)?status\b",
    r"\btrack\s+your\s+(?:order|package|shipment|refund|delivery)\b",
    r"\bcheck\s+(?:the\s+)?status\b",
    r"\bcheck\s+your\s+(?:order|package|shipment|refund|delivery)\b",

    # Specific support options.
    r"\brefund\s*/\s*replace\b",
    r"\brefund\s+or\s+replace\b",
    r"\breturn\s*/\s*replacement\b",
    r"\breturn\s+process\b",
    r"\breturn\s+options\b",
    r"\breplacement\s+options\b",
    r"\bavailable\s+options\b",

    # Concrete links / forms / channels.
    r"\bfill\s+(?:this|the)\s+form\b",
    r"\buse\s+(?:this|the)\s+form\b",
    r"\bvisit\s+us\s+here\b",
    r"\bstart\s+the\s+return\s+process\b",
    r"\bcontact\s+us\b",
    r"\breach\s+out\s+to\s+us\b",
    r"\bget\s+in\s+touch\s+with\s+us\b",

    # Account / order specific guidance.
    r"\bcheck\s+your\s+(?:email|account|billing\s+statement)\b",
    r"\bnext\s+billing\s+statement\b",
    r"\bregistered\s+email\b",
]


# Generic support language.
GENERIC_ACK_PATTERNS = [
    r"\bi['’]?m sorry\b",
    r"\bwe['’]?re sorry\b",
    r"\bsorry about that\b",
    r"\bwe want to help\b",
    r"\bi want to help\b",
    r"\bwe can help\b",
    r"\bi can help\b",
    r"\bi understand\b",
    r"\bwe understand\b",
    r"\bi get your concern\b",
    r"\bthanks for contacting\b",
    r"\bthanks for reaching out\b",
    r"\bwe['’]?re here to help\b",
]


def classify_agent_message(text):
    """
    Return a graded evidence class.

    0.00 = generic / redirect / future promise
    0.45 = specific instruction or request
    0.65 = specific informational/actionable guidance
    1.00 = observable completed action

    This avoids treating every useful sentence as a completed resolution.
    """

    text = normalize(text)

    if not text:
        return "generic_ack", 0.0

    # DM redirects have the lowest evidence value because the actual account
    # action is not observable in the public transcript.
    if contains_pattern(text, DM_PATTERNS):
        return "dm_redirect", 0.0

    # Completed/observable action gets highest priority.
    if contains_pattern(text, OBSERVED_ACTION_PATTERNS):
        return "specific_action", 1.0

    # Specific guidance can still be useful grounding evidence.
    if contains_pattern(text, SPECIFIC_GUIDANCE_PATTERNS):
        # Requests embedded in otherwise specific guidance are weaker than
        # a completed action but stronger than a generic acknowledgement.
        if contains_pattern(text, FUTURE_OR_REQUEST_PATTERNS):
            return "specific_guidance", 0.60

        return "specific_guidance", 0.70

    # A request/instruction alone is useful but does not prove an action
    # happened.
    if contains_pattern(text, FUTURE_OR_REQUEST_PATTERNS):
        return "specific_instruction", 0.40

    # Generic acknowledgement.
    if contains_pattern(text, GENERIC_ACK_PATTERNS):
        return "generic_ack", 0.10

    return "generic_ack", 0.10


def compute_d2(conversation):
    agent_messages = [
        normalize(tweet.get("text", ""))
        for tweet in conversation.get("tweets", [])
        if tweet.get("inbound") is False
        and str(tweet.get("author_id")) == AGENT_NAME
    ]

    if not agent_messages:
        return 0.0

    scores = []

    for message in agent_messages:
        _, score = classify_agent_message(message)
        scores.append(score)

    return max(scores) if scores else 0.0


# =============================================================================
# D3 — DM REDIRECT BEFORE ACTION
# =============================================================================

def compute_d3(conversation):
    tweets = conversation.get("tweets", [])

    first_specific_action_index = None
    first_dm_index = None

    for index, tweet in enumerate(tweets):

        if (
            tweet.get("inbound") is False
            and str(tweet.get("author_id")) == AGENT_NAME
        ):
            text = normalize(tweet.get("text", ""))

            if contains_pattern(text, DM_PATTERNS):
                if first_dm_index is None:
                    first_dm_index = index

            classification, score = classify_agent_message(text)

            if score >= 0.60:
                if first_specific_action_index is None:
                    first_specific_action_index = index

    # No DM redirect.
    if first_dm_index is None:
        return 0

    # DM happened before useful evidence.
    if (
        first_specific_action_index is None
        or first_dm_index < first_specific_action_index
    ):
        return 1

    return 0


# =============================================================================
# D4 — OBSERVABLE OUTCOME
# =============================================================================

POSITIVE_PATTERNS = [
    r"\bthanks\b",
    r"\bthank you\b",
    r"\bthx\b",
    r"\bgot it\b",
    r"\bgot this\b",
    r"\bthat worked\b",
    r"\bworks now\b",
    r"\bproblem solved\b",
    r"\bsolved\b",
    r"\bresolved\b",
    r"\breceived it\b",
    r"\breceived my\b",
    r"\bgot my\b",
    r"\bappreciate\b",
    r"\bhelped\b",
]

NEGATIVE_PATTERNS = [
    r"\bstill not\b",
    r"\bstill hasn't\b",
    r"\bstill have not\b",
    r"\bstill waiting\b",
    r"\bdoesn't work\b",
    r"\bdoes not work\b",
    r"\bnot working\b",
    r"\bdidn't work\b",
    r"\bdid not work\b",
    r"\bno update\b",
    r"\bno response\b",
    r"\bnot resolved\b",
    r"\bnot fixed\b",
    r"\bstill unresolved\b",
    r"\bworst\b",
    r"\bterrible\b",
    r"\bdisappointed\b",
    r"\bfrustrated\b",
    r"\bangry\b",
    r"\bunacceptable\b",
]


def compute_d4(conversation):
    tweets = conversation.get("tweets", [])

    # Track whether the agent has provided useful evidence.
    useful_agent_action_seen = False

    for index, tweet in enumerate(tweets):

        if (
            tweet.get("inbound") is False
            and str(tweet.get("author_id")) == AGENT_NAME
        ):
            text = normalize(tweet.get("text", ""))

            _, score = classify_agent_message(text)

            if score >= 0.60:
                useful_agent_action_seen = True

            continue

        # Customer reaction after an agent response.
        if (
            tweet.get("inbound") is True
            and str(tweet.get("author_id")) != AGENT_NAME
        ):
            text = normalize(tweet.get("text", ""))

            if not text:
                continue

            has_negative = contains_pattern(
                text,
                NEGATIVE_PATTERNS,
            )

            has_positive = contains_pattern(
                text,
                POSITIVE_PATTERNS,
            )

            # Negative takes precedence when mixed.
            if has_negative:
                return "negative_reaction"

            if has_positive:
                return "positive_reaction"

    # No explicit customer reaction.
    if useful_agent_action_seen:
        return "silence_after_action"

    return "no_action_observed"


# =============================================================================
# EC SCORE
# =============================================================================

D4_VALUES = {
    "positive_reaction": 1.00,
    "silence_after_action": 0.70,
    "neutral_reaction": 0.55,
    "negative_reaction": 0.20,
    "no_action_observed": 0.00,
}


def compute_ec(d1, d2, d3, d4):
    """
    Rule-based MVP Evidence Completeness score.

    This is intentionally NOT described as a calibrated probability.
    It is a transparent heuristic composite.
    """

    d4_value = D4_VALUES.get(d4, 0.0)

    score = (
        0.30 * d1
        + 0.30 * d2
        + 0.30 * d4_value
        + 0.10 * (1.0 - d3)
    )

    return round(min(max(score, 0.0), 1.0), 4)


def evidence_tier(ec_score):
    if ec_score >= 0.75:
        return "HIGH"

    if ec_score >= 0.50:
        return "MEDIUM"

    return "LOW"


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_conversation(conversation):
    conversation_id = conversation.get("conversation_id")
    customer_id = conversation.get("customer_id")

    d1 = compute_d1(conversation)
    d2 = compute_d2(conversation)
    d3 = compute_d3(conversation)
    d4 = compute_d4(conversation)

    ec = compute_ec(
        d1=d1,
        d2=d2,
        d3=d3,
        d4=d4,
    )

    turn_count = len(
        conversation.get("tweets", [])
    )

    return {
        "conversation_id": conversation_id,
        "customer_id": customer_id,

        "d1_context_specificity": round(d1, 4),
        "d2_action_specificity": round(d2, 4),
        "d3_dm_redirect_before_action": int(d3),
        "d4_observable_outcome": d4,

        "ec_score": ec,
        "evidence_tier": evidence_tier(ec),

        "turn_count": turn_count,
    }


def main():

    print("=" * 80)
    print("EVIDENCE COMPLETENESS FEATURE EXTRACTION")
    print("=" * 80)

    print(f"\nInput : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0

    d1_values = []
    d2_values = []
    d3_values = []

    d2_class_counts = Counter()
    d4_counts = Counter()
    tier_counts = Counter()

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as infile, open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as outfile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            conversation = json.loads(line)

            result = process_conversation(
                conversation
            )

            outfile.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            total += 1

            d1_values.append(
                result["d1_context_specificity"]
            )

            d2_values.append(
                result["d2_action_specificity"]
            )

            d3_values.append(
                result["d3_dm_redirect_before_action"]
            )

            d4_counts[
                result["d4_observable_outcome"]
            ] += 1

            tier_counts[
                result["evidence_tier"]
            ] += 1

            # Reconstruct broad D2 category for reporting.
            d2 = result["d2_action_specificity"]

            if d2 >= 0.99:
                d2_class_counts["specific_action"] += 1
            elif d2 >= 0.60:
                d2_class_counts["specific_guidance"] += 1
            elif d2 >= 0.30:
                d2_class_counts["specific_instruction"] += 1
            else:
                d2_class_counts["generic_or_redirect"] += 1

    average_d1 = (
        sum(d1_values) / len(d1_values)
        if d1_values
        else 0.0
    )

    average_d2 = (
        sum(d2_values) / len(d2_values)
        if d2_values
        else 0.0
    )

    average_d3 = (
        sum(d3_values) / len(d3_values)
        if d3_values
        else 0.0
    )

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"\nConversations processed : {total:,}"
    )

    print(
        f"Average D1               : {average_d1:.4f}"
    )

    print(
        f"Average D2               : {average_d2:.4f}"
    )

    print(
        f"Average D3               : {average_d3:.4f}"
    )

    print("\nD2 evidence classification:")

    total_d2_messages = sum(
        d2_class_counts.values()
    )

    for category, count in d2_class_counts.items():

        percentage = (
            100.0 * count / total_d2_messages
            if total_d2_messages
            else 0.0
        )

        print(
            f"  {category:24s} "
            f"{count:8,d} "
            f"({percentage:6.2f}%)"
        )

    print("\nD4 outcome distribution:")

    for outcome, count in d4_counts.most_common():

        percentage = (
            100.0 * count / total
            if total
            else 0.0
        )

        print(
            f"  {outcome:28s} "
            f"{count:8,d} "
            f"({percentage:6.2f}%)"
        )

    print("\nEvidence tier distribution:")

    for tier in ["HIGH", "MEDIUM", "LOW"]:

        count = tier_counts.get(
            tier,
            0
        )

        percentage = (
            100.0 * count / total
            if total
            else 0.0
        )

        print(
            f"  {tier:10s} "
            f"{count:8,d} "
            f"({percentage:6.2f}%)"
        )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()