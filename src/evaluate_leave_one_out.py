"""
Hiver — Leave-One-Out Human Alignment Evaluation

Purpose:
Evaluate whether evidence-aware retrieval selects historical conversations
with stronger human-verified evidence characteristics than baseline retrieval.

Method:
- 150 human-verified conversations are used as evaluation queries.
- The exact source conversation is excluded from retrieval candidates.
- All policies use the same top-20 semantic candidate pool.
- Policies:
    1. semantic_only
    2. semantic_intent
    3. semantic_ec
    4. full
- Production TOP_K remains unchanged at 5.
"""

from pathlib import Path
import json
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS = ROOT / "output" / "human_annotations_clean.csv"
EC_PREDICTIONS = ROOT / "output" / "ec_model_predictions.csv"

INDEX_PATH = ROOT / "output" / "semantic_index.faiss"
METADATA_PATH = ROOT / "output" / "semantic_index_metadata.pkl"
CORPUS_PATH = ROOT / "output" / "retrieval_corpus.jsonl"

OUTPUT_JSON = ROOT / "output" / "human_alignment_leave_one_out.json"
OUTPUT_CSV = ROOT / "output" / "human_alignment_leave_one_out_details.csv"


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

# Evaluation candidate pool.
# Production TOP_K=5 remains unchanged.
EVAL_CANDIDATES = 20

# Reranking weights
SEMANTIC_EC_SEMANTIC_WEIGHT = 0.70
SEMANTIC_EC_EC_WEIGHT = 0.30

FULL_SEMANTIC_WEIGHT = 0.55
FULL_INTENT_WEIGHT = 0.15
FULL_STATE_WEIGHT = 0.10
FULL_EC_WEIGHT = 0.20


# ============================================================
# INTENT / STATE DETECTION
# ============================================================

INTENT_KEYWORDS = {
    "refund": [
        "refund",
        "money back",
        "refund not received",
        "refunded",
        "reimbursement",
    ],
    "delivery": [
        "delivery",
        "delivered",
        "late",
        "delay",
        "delayed",
        "shipping",
        "shipment",
        "package",
        "parcel",
        "where is my order",
    ],
    "wrong_item": [
        "wrong item",
        "wrong product",
        "incorrect item",
        "different item",
        "received the wrong",
    ],
    "return": [
        "return",
        "send back",
        "returning",
        "return item",
    ],
    "payment": [
        "payment",
        "charged",
        "charge",
        "billing",
        "paid",
        "card",
        "transaction",
    ],
    "account": [
        "account",
        "login",
        "log in",
        "sign in",
        "password",
        "locked out",
    ],
    "order": [
        "order",
        "cancel",
        "cancelled",
        "canceled",
        "purchase",
    ],
    "product": [
        "product",
        "item",
        "device",
        "replacement",
        "damaged",
        "broken",
    ],
    "technical": [
        "app",
        "website",
        "error",
        "not working",
        "doesn't work",
        "cannot",
        "can't",
        "issue",
        "problem",
    ],
}


STATE_KEYWORDS = {
    "damaged": [
        "damaged",
        "broken",
        "cracked",
        "defective",
    ],
    "wrong_item": [
        "wrong item",
        "wrong product",
        "incorrect item",
    ],
    "late": [
        "late",
        "delay",
        "delayed",
        "overdue",
    ],
    "not_received": [
        "not received",
        "never received",
        "didn't receive",
        "did not receive",
        "missing",
    ],
    "cancelled": [
        "cancelled",
        "canceled",
        "cancel",
    ],
    "payment_failed": [
        "payment failed",
        "payment declined",
        "declined",
        "failed payment",
    ],
    "refund": [
        "refund",
        "refunded",
        "money back",
    ],
    "replacement": [
        "replacement",
        "replace",
    ],
}


RELATED_INTENTS = {
    ("wrong_item", "product"),
    ("product", "wrong_item"),
    ("delivery", "order"),
    ("order", "delivery"),
    ("payment", "order"),
    ("order", "payment"),
    ("technical", "account"),
    ("account", "technical"),
    ("return", "refund"),
    ("refund", "return"),
}


RELATED_STATES = {
    ("late", "not_received"),
    ("not_received", "late"),
    ("refund", "cancelled"),
    ("cancelled", "refund"),
}


def detect_intent(text):
    """
    Detect the most specific intent.

    Refund is checked first because many refund messages also contain
    order/payment vocabulary.
    """
    text = str(text).lower()

    priority = [
        "refund",
        "wrong_item",
        "return",
        "payment",
        "account",
        "delivery",
        "product",
        "technical",
        "order",
    ]

    for intent in priority:
        keywords = INTENT_KEYWORDS[intent]
        if any(keyword in text for keyword in keywords):
            return intent

    return "unknown"


def detect_state(text):
    text = str(text).lower()

    priority = [
        "damaged",
        "wrong_item",
        "not_received",
        "late",
        "payment_failed",
        "cancelled",
        "refund",
        "replacement",
    ]

    for state in priority:
        keywords = STATE_KEYWORDS[state]
        if any(keyword in text for keyword in keywords):
            return state

    return "unknown"


def intent_compatibility(query_intent, candidate_intent):
    if query_intent == "unknown" or candidate_intent == "unknown":
        return 0.0

    if query_intent == candidate_intent:
        return 1.0

    if (query_intent, candidate_intent) in RELATED_INTENTS:
        return 0.5

    return 0.0


def state_compatibility(query_state, candidate_state):
    if query_state == "unknown" or candidate_state == "unknown":
        return 0.0

    if query_state == candidate_state:
        return 1.0

    if (query_state, candidate_state) in RELATED_STATES:
        return 0.5

    return 0.0


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_text(value):
    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HIVER — LEAVE-ONE-OUT HUMAN ALIGNMENT EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    if not ANNOTATIONS.exists():
        raise FileNotFoundError(
            f"Missing annotations file:\n{ANNOTATIONS}"
        )

    annotations = pd.read_csv(ANNOTATIONS)

    print(f"\nHuman annotations : {len(annotations):,}")

    required_annotation_columns = [
        "annotation_id",
        "conversation_id",
        "evidence_complete",
        "customer_messages",
        "agent_messages",
    ]

    missing = [
        col
        for col in required_annotation_columns
        if col not in annotations.columns
    ]

    if missing:
        raise ValueError(
            f"Missing annotation columns: {missing}"
        )

    annotations = annotations.dropna(
        subset=[
            "conversation_id",
            "evidence_complete",
        ]
    ).copy()

    annotations["conversation_id"] = (
        annotations["conversation_id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Load EC predictions
    # --------------------------------------------------------

    if not EC_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Missing EC predictions:\n{EC_PREDICTIONS}"
        )

    ec_predictions = pd.read_csv(EC_PREDICTIONS)

    print(f"EC predictions    : {len(ec_predictions):,}")

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing FAISS index:\n{INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing FAISS metadata:\n{METADATA_PATH}"
        )

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Missing retrieval corpus:\n{CORPUS_PATH}"
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    corpus = load_jsonl(CORPUS_PATH)

    print(f"FAISS vectors     : {index.ntotal:,}")
    print(f"Corpus records    : {len(corpus):,}")

    if index.ntotal != len(corpus):
        raise ValueError(
            "FAISS index size does not match retrieval corpus size."
        )

    # --------------------------------------------------------
    # Validate index metadata order
    # --------------------------------------------------------

    metadata_records = metadata.get("records", [])

    if len(metadata_records) != len(corpus):
        raise ValueError(
            "Semantic metadata count does not match corpus count."
        )

    for i in range(min(10, len(corpus))):
        metadata_query = normalize_text(
            metadata_records[i].get("customer_query", "")
        )

        corpus_query = normalize_text(
            corpus[i].get("customer_query", "")
        )

        if metadata_query != corpus_query:
            raise ValueError(
                "Semantic metadata and retrieval corpus ordering differ."
            )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # Prepare corpus lookup
    # --------------------------------------------------------

    corpus_by_id = {}

    for idx, record in enumerate(corpus):

        conversation_id = normalize_text(
            record.get("conversation_id", "")
        )

        if conversation_id:
            corpus_by_id[conversation_id] = idx

    # --------------------------------------------------------
    # Prepare EC model lookup
    # --------------------------------------------------------

    ec_prediction_by_annotation = {}

    for _, row in ec_predictions.iterrows():

        annotation_id = normalize_text(
            row.get("annotation_id", "")
        )

        if annotation_id:
            ec_prediction_by_annotation[annotation_id] = {
                "ec_probability": safe_float(
                    row.get("ec_probability", 0.0)
                ),
                "ec_prediction": normalize_text(
                    row.get("ec_prediction", "")
                ),
            }

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        f"\nEvaluating {len(annotations):,} "
        f"human-verified examples..."
    )

    print(
        f"Semantic candidate pool: top {EVAL_CANDIDATES}"
    )

    print(
        "Exact source conversation: EXCLUDED"
    )

    print()

    details = []

    policies = [
        "semantic_only",
        "semantic_intent",
        "semantic_ec",
        "full",
    ]

    skipped = 0

    # --------------------------------------------------------
    # Query each human-verified example
    # --------------------------------------------------------

    for row_number, (_, human_row) in enumerate(
        annotations.iterrows(),
        start=1,
    ):

        conversation_id = normalize_text(
            human_row["conversation_id"]
        )

        annotation_id = normalize_text(
            human_row["annotation_id"]
        )

        customer_messages = normalize_text(
            human_row["customer_messages"]
        )

        human_label = normalize_text(
            human_row["evidence_complete"]
        ).upper()

        if conversation_id not in corpus_by_id:
            skipped += 1
            continue

        # Query text.
        query_text = customer_messages

        if not query_text:
            skipped += 1
            continue

        # ----------------------------------------------------
        # Encode query
        # ----------------------------------------------------

        embedding = model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Search top 20 semantic candidates
        # ----------------------------------------------------

        similarities, indices = index.search(
            embedding,
            EVAL_CANDIDATES + 1,
        )

        query_intent = detect_intent(query_text)
        query_state = detect_state(query_text)

        candidate_rows = []

        for semantic_similarity, candidate_index in zip(
            similarities[0],
            indices[0],
        ):

            candidate_index = int(candidate_index)

            if candidate_index < 0:
                continue

            candidate = corpus[candidate_index]

            candidate_conversation_id = normalize_text(
                candidate.get("conversation_id", "")
            )

            # ------------------------------------------------
            # CRITICAL:
            # Exclude exact source conversation.
            # ------------------------------------------------

            if candidate_conversation_id == conversation_id:
                continue

            candidate_query = normalize_text(
                candidate.get("customer_query", "")
            )

            candidate_intent = detect_intent(candidate_query)
            candidate_state = detect_state(candidate_query)

            semantic_similarity = safe_float(
                semantic_similarity
            )

            ec_score = safe_float(
                candidate.get(
                    "evidence_completeness_score",
                    0.0,
                )
            )

            intent_score = intent_compatibility(
                query_intent,
                candidate_intent,
            )

            state_score = state_compatibility(
                query_state,
                candidate_state,
            )

            semantic_ec_score = (
                SEMANTIC_EC_SEMANTIC_WEIGHT
                * semantic_similarity
                +
                SEMANTIC_EC_EC_WEIGHT
                * ec_score
            )

            full_score = (
                FULL_SEMANTIC_WEIGHT
                * semantic_similarity
                +
                FULL_INTENT_WEIGHT
                * intent_score
                +
                FULL_STATE_WEIGHT
                * state_score
                +
                FULL_EC_WEIGHT
                * ec_score
            )

            candidate_rows.append(
                {
                    "candidate_index": candidate_index,
                    "conversation_id": candidate_conversation_id,
                    "semantic_similarity": semantic_similarity,
                    "ec_score": ec_score,
                    "intent_score": intent_score,
                    "state_score": state_score,
                    "semantic_ec_score": semantic_ec_score,
                    "full_score": full_score,
                    "candidate_intent": candidate_intent,
                    "candidate_state": candidate_state,
                    "evidence_tier": normalize_text(
                        candidate.get(
                            "evidence_tier",
                            "",
                        )
                    ),
                }
            )

            if len(candidate_rows) >= EVAL_CANDIDATES:
                break

        if not candidate_rows:
            skipped += 1
            continue

        # ----------------------------------------------------
        # Select candidate for each policy
        # ----------------------------------------------------

        semantic_only = max(
            candidate_rows,
            key=lambda x: x["semantic_similarity"],
        )

        semantic_intent = max(
            candidate_rows,
            key=lambda x: (
                x["intent_score"],
                x["semantic_similarity"],
            ),
        )

        semantic_ec = max(
            candidate_rows,
            key=lambda x: x["semantic_ec_score"],
        )

        full = max(
            candidate_rows,
            key=lambda x: x["full_score"],
        )

        selected = {
            "semantic_only": semantic_only,
            "semantic_intent": semantic_intent,
            "semantic_ec": semantic_ec,
            "full": full,
        }

        # ----------------------------------------------------
        # Save details
        # ----------------------------------------------------

        for policy_name, candidate in selected.items():

            details.append(
                {
                    "annotation_id": annotation_id,
                    "conversation_id": conversation_id,
                    "human_evidence_complete": human_label,
                    "query_intent": query_intent,
                    "query_state": query_state,
                    "policy": policy_name,
                    "selected_conversation_id": candidate[
                        "conversation_id"
                    ],
                    "selected_semantic_similarity": candidate[
                        "semantic_similarity"
                    ],
                    "selected_ec_score": candidate[
                        "ec_score"
                    ],
                    "selected_intent_score": candidate[
                        "intent_score"
                    ],
                    "selected_state_score": candidate[
                        "state_score"
                    ],
                    "selected_evidence_tier": candidate[
                        "evidence_tier"
                    ],
                    "candidate_pool_size": len(
                        candidate_rows
                    ),
                    "source_excluded": True,
                }
            )

        if row_number % 25 == 0:
            print(
                f"Processed {row_number:,}/{len(annotations):,}"
            )

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    details_df = pd.DataFrame(details)

    if details_df.empty:
        raise RuntimeError(
            "No evaluation results were produced."
        )

    # --------------------------------------------------------
    # Calculate summary
    # --------------------------------------------------------

    summary = {}

    for policy in policies:

        policy_df = details_df[
            details_df["policy"] == policy
        ]

        mean_ec = policy_df[
            "selected_ec_score"
        ].mean()

        medium_high_rate = (
            policy_df[
                "selected_evidence_tier"
            ]
            .isin(["MEDIUM", "HIGH"])
            .mean()
        )

        high_rate = (
            policy_df[
                "selected_evidence_tier"
            ]
            .eq("HIGH")
            .mean()
        )

        mean_similarity = policy_df[
            "selected_semantic_similarity"
        ].mean()

        summary[policy] = {
            "n": int(len(policy_df)),
            "mean_selected_ec": float(mean_ec),
            "medium_high_rate": float(
                medium_high_rate
            ),
            "high_rate": float(high_rate),
            "mean_selected_semantic_similarity": float(
                mean_similarity
            ),
        }

    # --------------------------------------------------------
    # Calculate EC improvement over semantic-only
    # --------------------------------------------------------

    baseline = summary["semantic_only"]

    for policy in policies:

        summary[policy][
            "delta_mean_ec_vs_semantic_only"
        ] = (
            summary[policy]["mean_selected_ec"]
            -
            baseline["mean_selected_ec"]
        )

        summary[policy][
            "delta_medium_high_rate_vs_semantic_only"
        ] = (
            summary[policy]["medium_high_rate"]
            -
            baseline["medium_high_rate"]
        )

        summary[policy][
            "delta_high_rate_vs_semantic_only"
        ] = (
            summary[policy]["high_rate"]
            -
            baseline["high_rate"]
        )

    # --------------------------------------------------------
    # Human-label distribution
    # --------------------------------------------------------

    human_distribution = (
        annotations[
            "evidence_complete"
        ]
        .astype(str)
        .str.upper()
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Save detailed CSV
    # --------------------------------------------------------

    details_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Save JSON report
    # --------------------------------------------------------

    report = {
        "evaluation": {
            "name": "leave_one_out_human_alignment",
            "human_annotations": int(
                len(annotations)
            ),
            "evaluated_examples": int(
                len(
                    details_df[
                        details_df["policy"]
                        == "semantic_only"
                    ]
                )
            ),
            "skipped_examples": int(skipped),
            "candidate_pool": EVAL_CANDIDATES,
            "source_conversation_excluded": True,
            "model": MODEL_NAME,
            "faiss_vectors": int(index.ntotal),
            "corpus_records": int(len(corpus)),
        },
        "human_label_distribution": human_distribution,
        "policies": summary,
        "interpretation": {
            "metric_meaning": (
                "Selected evidence completeness measures whether "
                "each retrieval policy preferentially selects historical "
                "conversations with stronger evidence characteristics."
            ),
            "not_answer_quality": (
                "These metrics do not directly measure answer correctness, "
                "resolution accuracy, or customer satisfaction."
            ),
            "human_alignment_note": (
                "The human labels describe evidence completeness of the "
                "150 sampled historical conversations. They are not "
                "ground-truth relevance labels for every retrieved candidate."
            ),
            "methodology_note": (
                "The evaluation uses leave-one-out retrieval so the exact "
                "source conversation associated with each human example "
                "cannot be selected as its own retrieval result."
            ),
        },
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT RESULTS")
    print("=" * 70)

    print(
        f"\nHuman examples       : "
        f"{len(annotations):,}"
    )

    print(
        f"Evaluated examples   : "
        f"{report['evaluation']['evaluated_examples']:,}"
    )

    print(
        f"Skipped examples     : "
        f"{skipped:,}"
    )

    print(
        f"Candidate pool       : "
        f"top {EVAL_CANDIDATES}"
    )

    print(
        "Source conversation  : EXCLUDED"
    )

    print("\nRETRIEVAL EVIDENCE SUMMARY")

    for policy in policies:

        result = summary[policy]

        print(f"\n{policy}")

        print(
            f"  Mean selected EC : "
            f"{result['mean_selected_ec']:.4f}"
        )

        print(
            f"  MEDIUM/HIGH rate : "
            f"{result['medium_high_rate']:.4f} "
            f"({result['medium_high_rate'] * 100:.2f}%)"
        )

        print(
            f"  HIGH rate        : "
            f"{result['high_rate']:.4f} "
            f"({result['high_rate'] * 100:.2f}%)"
        )

        print(
            f"  Mean similarity  : "
            f"{result['mean_selected_semantic_similarity']:.4f}"
        )

        print(
            f"  Δ mean EC vs baseline : "
            f"{result['delta_mean_ec_vs_semantic_only']:+.4f}"
        )

        print(
            f"  Δ MED/HIGH vs baseline: "
            f"{result['delta_medium_high_rate_vs_semantic_only'] * 100:+.2f} pp"
        )

        print(
            f"  Δ HIGH vs baseline    : "
            f"{result['delta_high_rate_vs_semantic_only'] * 100:+.2f} pp"
        )

    print("\nOUTPUTS")

    print(
        f"  JSON : {OUTPUT_JSON}"
    )

    print(
        f"  CSV  : {OUTPUT_CSV}"
    )

    print("\nSTATUS: COMPLETE")


if __name__ == "__main__":
    main()