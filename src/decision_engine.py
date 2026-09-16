import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

INDEX_PATH = OUTPUT_DIR / "semantic_index.faiss"
METADATA_PATH = OUTPUT_DIR / "semantic_index_metadata.pkl"
CORPUS_PATH = OUTPUT_DIR / "retrieval_corpus.jsonl"


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

# Initial experimental thresholds.
AUTO_MIN_SEMANTIC = 0.72
AUTO_MIN_INTENT = 0.75
AUTO_MIN_EC = 0.75

CLARIFY_MIN_SEMANTIC = 0.65
CLARIFY_MIN_INTENT = 0.50
CLARIFY_MIN_EC = 0.50

# Decision score weights.
SEMANTIC_WEIGHT = 0.55
INTENT_WEIGHT = 0.15
STATE_WEIGHT = 0.10
EC_WEIGHT = 0.20


# =============================================================================
# INTENT DEFINITIONS
# =============================================================================

INTENT_KEYWORDS = {
    "refund": [
        "refund",
        "refunded",
        "refunds",
        "money back",
        "moneyback",
        "reimburse",
        "reimbursement",
        "credited back",
        "credit back",
    ],

    "wrong_item": [
        "wrong item",
        "wrong product",
        "incorrect item",
        "incorrect product",
        "different item",
        "sent the wrong",
        "received the wrong",
        "received incorrect",
    ],

    "damaged": [
        "damaged",
        "damage",
        "broken",
        "cracked",
        "defective",
        "destroyed",
    ],

    "replacement": [
        "replacement",
        "replace",
        "exchange",
        "swap",
    ],

    "return": [
        "return",
        "send back",
        "returning",
    ],

    "payment": [
        "payment",
        "paid",
        "pay",
        "card",
        "debit",
        "credit card",
        "payment failed",
        "payment declined",
        "charged",
        "charge",
    ],

    "delivery": [
        "delivery",
        "delivered",
        "package",
        "parcel",
        "shipment",
        "shipping",
        "arrive",
        "arrived",
        "not delivered",
    ],

    "order": [
        "order",
        "ordered",
        "ordering",
        "purchase",
        "bought",
    ],

    "account": [
        "account",
        "login",
        "log in",
        "sign in",
        "signin",
        "password",
        "username",
    ],
}


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

STATE_KEYWORDS = {
    "marked_delivered_not_received": [
        "marked delivered",
        "says delivered",
        "shows delivered",
        "says it was delivered",
        "package was delivered",
        "parcel was delivered",
        "delivered but",
        "delivered and",
    ],

    "received_wrong": [
        "wrong item",
        "wrong product",
        "incorrect item",
        "incorrect product",
        "different item",
        "received the wrong",
        "sent the wrong",
    ],

    "damaged": [
        "damaged",
        "damage",
        "broken",
        "cracked",
        "defective",
    ],

    "cancelled": [
        "cancelled",
        "canceled",
        "cancel my order",
        "order was cancelled",
        "order was canceled",
    ],

    "declined": [
        "declined",
        "rejected",
        "failed",
        "payment failed",
        "payment declined",
    ],

    "not_received": [
        "not received",
        "haven't received",
        "have not received",
        "didn't receive",
        "did not receive",
        "never received",
        "not arrived",
        "hasn't arrived",
        "hasnt arrived",
        "has not arrived",
    ],

    "delayed": [
        "not arrived",
        "hasn't arrived",
        "hasnt arrived",
        "have not arrived",
        "haven't received",
        "have not received",
        "still waiting",
        "waiting",
        "delayed",
        "late",
        "past due",
        "more than a week",
        "more than a month",
    ],
}


# =============================================================================
# LOADING
# =============================================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def load_system():
    print("=" * 80)
    print("EVIDENCE-COMPLETENESS-AWARE DECISION ENGINE")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # FAISS
    # -------------------------------------------------------------------------

    print("Loading FAISS index...")

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found:\n{INDEX_PATH}"
        )

    index = faiss.read_index(str(INDEX_PATH))

    print(f"FAISS vectors: {index.ntotal:,}")
    print(f"Vector dimension: {index.d}")
    print()

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    print("Loading metadata...")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_PATH}"
        )

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    metadata_records = metadata["records"]

    print(f"Metadata records: {len(metadata_records):,}")
    print()

    # -------------------------------------------------------------------------
    # RETRIEVAL CORPUS
    # -------------------------------------------------------------------------

    print("Loading retrieval corpus...")

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval corpus not found:\n{CORPUS_PATH}"
        )

    corpus = load_jsonl(CORPUS_PATH)

    print(f"Corpus records: {len(corpus):,}")
    print()

    # -------------------------------------------------------------------------
    # CONSISTENCY VALIDATION
    # -------------------------------------------------------------------------

    if len(metadata_records) != len(corpus):
        raise RuntimeError(
            f"Metadata/corpus mismatch: "
            f"{len(metadata_records)} vs {len(corpus)}"
        )

    if index.ntotal != len(corpus):
        raise RuntimeError(
            f"FAISS/corpus mismatch: "
            f"{index.ntotal} vs {len(corpus)}"
        )

    # -------------------------------------------------------------------------
    # EMBEDDING MODEL
    # -------------------------------------------------------------------------

    print(f"Loading embedding model: {MODEL_NAME}")

    try:
        model = SentenceTransformer(MODEL_NAME)
        model_mode = "LOCAL / OFFLINE"
    except Exception as exc:
        raise RuntimeError(
            f"Could not load embedding model '{MODEL_NAME}'. "
            f"The model should already exist in the local cache.\n"
            f"Original error: {exc}"
        )

    print("Embedding model loaded from local cache.")
    print()

    # -------------------------------------------------------------------------
    # SYSTEM VALIDATION
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("SYSTEM VALIDATION")
    print("=" * 80)
    print()

    print(f"FAISS index       : {index.ntotal:,} vectors")
    print(f"Corpus            : {len(corpus):,} records")
    print(f"Metadata          : {len(metadata_records):,} records")
    print(f"Embedding model   : {MODEL_NAME}")
    print(f"Model mode        : {model_mode}")
    print()

    if not corpus:
        raise RuntimeError("Retrieval corpus is empty.")

    required_fields = [
        "conversation_id",
        "customer_query",
        "agent_response",
        "D1_context_specificity",
        "D2_action_specificity",
        "D3_dm_redirect",
        "D4_observable_outcome",
        "evidence_completeness_score",
        "evidence_tier",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in corpus[0]
    ]

    if missing_fields:
        raise RuntimeError(
            f"Retrieval corpus is missing required fields: {missing_fields}"
        )

    print("Evidence schema    : VALID")
    print("EC field           : evidence_completeness_score")
    print("Tier field         : evidence_tier")
    print()

    print("All retrieval components loaded successfully.")
    print()

    return index, metadata_records, corpus, model


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(text):
    text = str(text or "").lower()

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    # Normalize common apostrophe variants.
    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
    )

    return text.strip()


# =============================================================================
# INTENT DETECTION
# =============================================================================

def detect_intent(query):
    """
    Detect the primary customer-support intent.

    Important design decision:
    Specific intents are checked before broad delivery/order intents.

    This prevents a query such as:

        "My refund hasn't arrived yet"

    from being incorrectly classified as "delivery" simply because
    the word "arrived" appears in the query.
    """

    text = normalize_text(query)

    # -------------------------------------------------------------------------
    # 1. REFUND
    # -------------------------------------------------------------------------
    #
    # Refund gets explicit priority because refund complaints often contain
    # delivery-like words such as "arrived", "received", or "came".
    #

    refund_patterns = [
        "refund",
        "refunded",
        "refunds",
        "money back",
        "moneyback",
        "reimburse",
        "reimbursement",
        "credited back",
        "credit back",
        "refund pending",
        "refund missing",
        "refund delayed",
        "refund hasn't arrived",
        "refund has not arrived",
        "refund hasnt arrived",
        "refund not received",
        "refund hasn't come",
        "refund has not come",
        "refund hasnt come",
        "haven't received my refund",
        "have not received my refund",
        "havent received my refund",
        "waiting for refund",
    ]

    if any(pattern in text for pattern in refund_patterns):
        return "refund"

    # -------------------------------------------------------------------------
    # 2. WRONG ITEM
    # -------------------------------------------------------------------------

    wrong_item_patterns = [
        "wrong item",
        "wrong product",
        "incorrect item",
        "incorrect product",
        "different item",
        "sent the wrong",
        "received the wrong",
        "received incorrect",
    ]

    if any(pattern in text for pattern in wrong_item_patterns):
        return "wrong_item"

    # -------------------------------------------------------------------------
    # 3. DAMAGED
    # -------------------------------------------------------------------------

    damaged_patterns = [
        "damaged",
        "damage",
        "broken",
        "cracked",
        "defective",
        "destroyed",
    ]

    if any(pattern in text for pattern in damaged_patterns):
        return "damaged"

    # -------------------------------------------------------------------------
    # 4. REPLACEMENT
    # -------------------------------------------------------------------------

    replacement_patterns = [
        "replacement",
        "replace",
        "exchange",
        "swap",
    ]

    if any(pattern in text for pattern in replacement_patterns):
        return "replacement"

    # -------------------------------------------------------------------------
    # 5. RETURN
    # -------------------------------------------------------------------------

    return_patterns = [
        "return",
        "send back",
        "returning",
    ]

    if any(pattern in text for pattern in return_patterns):
        return "return"

    # -------------------------------------------------------------------------
    # 6. PAYMENT
    # -------------------------------------------------------------------------

    payment_patterns = [
        "payment",
        "paid",
        "pay",
        "card",
        "debit",
        "credit card",
        "payment failed",
        "payment declined",
        "charged",
        "charge",
    ]

    if any(pattern in text for pattern in payment_patterns):
        return "payment"

    # -------------------------------------------------------------------------
    # 7. DELIVERY
    # -------------------------------------------------------------------------

    delivery_patterns = [
        "delivery",
        "delivered",
        "package",
        "parcel",
        "shipment",
        "shipping",
        "arrive",
        "arrived",
        "not delivered",
    ]

    if any(pattern in text for pattern in delivery_patterns):
        return "delivery"

    # -------------------------------------------------------------------------
    # 8. ORDER
    # -------------------------------------------------------------------------

    order_patterns = [
        "order",
        "ordered",
        "ordering",
        "purchase",
        "bought",
    ]

    if any(pattern in text for pattern in order_patterns):
        return "order"

    # -------------------------------------------------------------------------
    # 9. ACCOUNT
    # -------------------------------------------------------------------------

    account_patterns = [
        "account",
        "login",
        "log in",
        "sign in",
        "signin",
        "password",
        "username",
    ]

    if any(pattern in text for pattern in account_patterns):
        return "account"

    return "unknown"


# =============================================================================
# STATE DETECTION
# =============================================================================

def detect_state(query):
    """
    Detect the current issue state.

    More specific states are checked first.
    """

    text = normalize_text(query)

    ordered_states = [
        "marked_delivered_not_received",
        "received_wrong",
        "damaged",
        "cancelled",
        "declined",
        "not_received",
        "delayed",
    ]

    for state in ordered_states:
        for keyword in STATE_KEYWORDS[state]:
            if keyword in text:
                return state

    return "unknown"


# =============================================================================
# COMPATIBILITY
# =============================================================================

def intent_compatibility(query_intent, record_intent):
    if query_intent == "unknown" or not record_intent:
        return 0.0

    record_intent = str(record_intent)

    # Exact match.
    if query_intent == record_intent:
        return 1.0

    related = {
        "refund": {
            "payment",
            "return",
        },

        "return": {
            "refund",
            "wrong_item",
            "damaged",
        },

        "wrong_item": {
            "delivery",
            "replacement",
            "return",
        },

        "damaged": {
            "replacement",
            "return",
            "wrong_item",
        },

        "replacement": {
            "wrong_item",
            "damaged",
            "return",
        },

        "delivery": {
            "order",
        },

        "order": {
            "delivery",
            "payment",
        },

        "payment": {
            "order",
            "refund",
        },

        "account": set(),
    }

    if record_intent in related.get(query_intent, set()):
        return 0.5

    return 0.0


def state_compatibility(query_state, record_state):
    if query_state == "unknown":
        return 0.0

    if not record_state:
        return 0.5

    record_state = str(record_state)

    # Exact match.
    if query_state == record_state:
        return 1.0

    related = {
        "delayed": {
            "not_received",
            "marked_delivered_not_received",
        },

        "not_received": {
            "delayed",
            "marked_delivered_not_received",
        },

        "marked_delivered_not_received": {
            "not_received",
            "delayed",
        },

        "received_wrong": {
            "damaged",
        },

        "damaged": {
            "received_wrong",
            "not_received",
        },

        "cancelled": {
            "declined",
        },

        "declined": {
            "cancelled",
        },
    }

    if record_state in related.get(query_state, set()):
        return 0.5

    return 0.0


# =============================================================================
# EVIDENCE EXTRACTION
# =============================================================================

def get_evidence_fields(record):
    """
    These field names exactly match retrieval_corpus.jsonl.
    """

    d1 = float(
        record.get(
            "D1_context_specificity",
            0.0,
        )
        or 0.0
    )

    d2 = float(
        record.get(
            "D2_action_specificity",
            0.0,
        )
        or 0.0
    )

    d3 = float(
        record.get(
            "D3_dm_redirect",
            0,
        )
        or 0
    )

    d4 = record.get(
        "D4_observable_outcome",
        "no_action_observed",
    )

    ec = float(
        record.get(
            "evidence_completeness_score",
            0.0,
        )
        or 0.0
    )

    tier = str(
        record.get(
            "evidence_tier",
            "LOW",
        )
        or "LOW"
    )

    return d1, d2, d3, d4, ec, tier


# =============================================================================
# DECISION SCORE
# =============================================================================

def calculate_decision_score(
    semantic_similarity,
    intent_score,
    state_score,
    ec_score,
):
    return (
        SEMANTIC_WEIGHT * semantic_similarity
        + INTENT_WEIGHT * intent_score
        + STATE_WEIGHT * state_score
        + EC_WEIGHT * ec_score
    )


# =============================================================================
# DECISION
# =============================================================================

def make_decision(
    semantic_similarity,
    intent_score,
    state_score,
    ec_score,
):
    # -------------------------------------------------------------------------
    # AUTO RESPOND
    # -------------------------------------------------------------------------

    if (
        semantic_similarity >= AUTO_MIN_SEMANTIC
        and intent_score >= AUTO_MIN_INTENT
        and ec_score >= AUTO_MIN_EC
    ):
        return (
            "AUTO_RESPOND",
            "strong semantic match with compatible intent and sufficiently complete evidence",
        )

    # -------------------------------------------------------------------------
    # ASK CLARIFICATION
    # -------------------------------------------------------------------------

    if (
        semantic_similarity >= CLARIFY_MIN_SEMANTIC
        and intent_score >= CLARIFY_MIN_INTENT
        and ec_score >= CLARIFY_MIN_EC
    ):
        return (
            "ASK_CLARIFICATION",
            "retrieved evidence is relevant but requires additional customer context",
        )

    # -------------------------------------------------------------------------
    # ESCALATE: EVIDENCE
    # -------------------------------------------------------------------------

    if ec_score < CLARIFY_MIN_EC:
        return (
            "ESCALATE",
            "insufficient evidence completeness",
        )

    # -------------------------------------------------------------------------
    # ESCALATE: INTENT
    # -------------------------------------------------------------------------

    if intent_score < CLARIFY_MIN_INTENT:
        return (
            "ESCALATE",
            "intent compatibility is insufficient",
        )

    # -------------------------------------------------------------------------
    # ESCALATE: STATE
    # -------------------------------------------------------------------------

    if state_score < 0.50:
        return (
            "ESCALATE",
            "issue state compatibility is insufficient",
        )

    # -------------------------------------------------------------------------
    # DEFAULT ESCALATION
    # -------------------------------------------------------------------------

    return (
        "ESCALATE",
        "retrieval confidence is insufficient for a grounded response",
    )


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_candidates(
    query,
    index,
    corpus,
    model,
    query_intent,
    query_state,
):
    """
    Retrieve semantically similar historical conversations.

    IMPORTANT:
    The retrieval corpus does not contain explicit intent/state fields.

    Therefore historical intent and state are detected dynamically from
    each historical customer query rather than reading nonexistent fields.
    """

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embedding = embedding.astype("float32")

    similarities, indices = index.search(
        embedding,
        TOP_K,
    )

    candidates = []

    for similarity, idx in zip(
        similarities[0],
        indices[0],
    ):
        if idx < 0 or idx >= len(corpus):
            continue

        record = corpus[int(idx)]

        # ---------------------------------------------------------------------
        # FIX:
        # Historical corpus does not contain "intent" or "state".
        # Detect them from the historical customer query.
        # ---------------------------------------------------------------------

        historical_query = record.get(
            "customer_query",
            "",
        )

        record_intent = detect_intent(historical_query)
        record_state = detect_state(historical_query)

        intent_score = intent_compatibility(
            query_intent,
            record_intent,
        )

        state_score = state_compatibility(
            query_state,
            record_state,
        )

        (
            d1,
            d2,
            d3,
            d4,
            ec_score,
            evidence_tier,
        ) = get_evidence_fields(record)

        semantic_similarity = float(similarity)

        decision_score = calculate_decision_score(
            semantic_similarity,
            intent_score,
            state_score,
            ec_score,
        )

        candidates.append(
            {
                "rank": 0,

                "semantic_similarity": semantic_similarity,

                "intent_score": intent_score,

                "state_score": state_score,

                "ec_score": ec_score,

                "evidence_tier": evidence_tier,

                "d1": d1,
                "d2": d2,
                "d3": d3,
                "d4": d4,

                "decision_score": decision_score,

                "record_intent": record_intent,
                "record_state": record_state,

                "record": record,
            }
        )

    # -------------------------------------------------------------------------
    # Rank using the complete decision score.
    # -------------------------------------------------------------------------

    candidates.sort(
        key=lambda x: x["decision_score"],
        reverse=True,
    )

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        candidate["rank"] = rank

    return candidates


# =============================================================================
# DISPLAY
# =============================================================================

def display_candidate(candidate):
    record = candidate["record"]

    print("-" * 80)

    print(
        f"Rank {candidate['rank']} | "
        f"Semantic={candidate['semantic_similarity']:.4f} | "
        f"Intent={candidate['intent_score']:.2f} | "
        f"State={candidate['state_score']:.2f} | "
        f"EC={candidate['ec_score']:.4f} | "
        f"Decision={candidate['decision_score']:.4f}"
    )

    print(
        f"Detected historical intent: "
        f"{candidate['record_intent']} | "
        f"Detected historical state: "
        f"{candidate['record_state']} | "
        f"Evidence tier: "
        f"{candidate['evidence_tier']}"
    )

    print()

    print("Evidence signals:")

    print(
        f"  D1 Context specificity : "
        f"{candidate['d1']:.4f}"
    )

    print(
        f"  D2 Action specificity  : "
        f"{candidate['d2']:.4f}"
    )

    print(
        f"  D3 DM redirect         : "
        f"{candidate['d3']}"
    )

    print(
        f"  D4 Observable outcome  : "
        f"{candidate['d4']}"
    )

    print()

    print("Historical customer:")

    print(
        f"  {record.get('customer_query', '')}"
    )

    print()

    print("Historical agent:")

    print(
        f"  {record.get('agent_response', '')}"
    )

    print()


# =============================================================================
# LIVE QUERY
# =============================================================================

def process_query(
    query,
    index,
    corpus,
    model,
):
    print()
    print("-" * 80)
    print("LIVE QUERY ANALYSIS")
    print("-" * 80)

    query_intent = detect_intent(query)
    query_state = detect_state(query)

    print(
        f"Query : {query}"
    )

    print(
        f"Detected intent : {query_intent}"
    )

    print(
        f"Detected state  : {query_state}"
    )

    candidates = retrieve_candidates(
        query,
        index,
        corpus,
        model,
        query_intent,
        query_state,
    )

    if not candidates:
        print()
        print("No retrieval candidates found.")
        return

    best = candidates[0]

    decision, reason = make_decision(
        best["semantic_similarity"],
        best["intent_score"],
        best["state_score"],
        best["ec_score"],
    )

    print()

    print("=" * 80)
    print("DECISION")
    print("=" * 80)

    print()

    print(decision)

    print()

    print("Reason:")
    print(reason)

    print()

    print("Best evidence candidate:")

    print(
        f"  Semantic similarity : "
        f"{best['semantic_similarity']:.4f}"
    )

    print(
        f"  Intent compatibility: "
        f"{best['intent_score']:.4f}"
    )

    print(
        f"  State compatibility : "
        f"{best['state_score']:.4f}"
    )

    print(
        f"  Evidence completeness: "
        f"{best['ec_score']:.4f}"
    )

    print(
        f"  Evidence tier       : "
        f"{best['evidence_tier']}"
    )

    print(
        f"  Decision score      : "
        f"{best['decision_score']:.4f}"
    )

    print()

    print("=" * 80)
    print("TOP RETRIEVED EVIDENCE")
    print("=" * 80)

    print()

    for candidate in candidates:
        display_candidate(candidate)


# =============================================================================
# MAIN
# =============================================================================

def main():
    index, metadata, corpus, model = load_system()

    print("=" * 80)
    print("READY FOR LIVE QUERIES")
    print("=" * 80)

    print()

    print("Try these test queries:")

    print(
        "  1. My refund hasn't arrived yet"
    )

    print(
        "  2. I received the wrong item"
    )

    print(
        "  3. My package says delivered but I never got it"
    )

    print(
        "  4. I received a damaged product"
    )

    print()

    print("Type 'exit' to stop.")

    print()

    while True:
        try:
            query = input(
                "-" * 80
                + "\nCustomer query > "
            ).strip()

        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not query:
            continue

        if query.lower() == "exit":
            break

        try:
            process_query(
                query,
                index,
                corpus,
                model,
            )

        except Exception as exc:
            print()
            print("ERROR:")
            print(exc)
            print()


if __name__ == "__main__":
    main()