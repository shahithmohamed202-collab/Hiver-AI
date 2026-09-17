import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

ANNOTATIONS_FILE = OUTPUT_DIR / "human_annotations_clean.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "ec_model_predictions.csv"

INDEX_FILE = OUTPUT_DIR / "semantic_index.faiss"
METADATA_FILE = OUTPUT_DIR / "semantic_index_metadata.pkl"
CORPUS_FILE = OUTPUT_DIR / "retrieval_corpus.jsonl"

RESULT_FILE = OUTPUT_DIR / "human_alignment_evaluation.json"
DETAIL_FILE = OUTPUT_DIR / "human_alignment_details.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


# =============================================================================
# LOADERS
# =============================================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def load_system():
    print("=" * 80)
    print("HUMAN ALIGNMENT EVALUATION")
    print("=" * 80)

    print("\nLoading human annotations...")

    annotations = pd.read_csv(
        ANNOTATIONS_FILE,
        encoding="utf-8-sig",
    )

    print(f"Human annotations: {len(annotations):,}")

    print("\nLoading EC model predictions...")

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        encoding="utf-8-sig",
    )

    print(f"EC predictions: {len(predictions):,}")

    print("\nLoading FAISS index...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"FAISS vectors: {index.ntotal:,}"
    )

    print("\nLoading metadata...")

    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    records = metadata["records"]

    print(
        f"Metadata records: {len(records):,}"
    )

    print("\nLoading retrieval corpus...")

    corpus = load_jsonl(
        CORPUS_FILE
    )

    print(
        f"Corpus records: {len(corpus):,}"
    )

    if len(annotations) != 150:
        raise RuntimeError(
            f"Expected 150 human annotations, "
            f"found {len(annotations)}."
        )

    if len(predictions) != 150:
        raise RuntimeError(
            f"Expected 150 EC predictions, "
            f"found {len(predictions)}."
        )

    if index.ntotal != len(records):
        raise RuntimeError(
            "FAISS/metadata mismatch."
        )

    if index.ntotal != len(corpus):
        raise RuntimeError(
            "FAISS/corpus mismatch."
        )

    return (
        annotations,
        predictions,
        index,
        records,
        corpus,
    )


# =============================================================================
# HUMAN LABEL PREPARATION
# =============================================================================

def prepare_human_labels(
    annotations,
    predictions,
):
    annotations = annotations.copy()
    predictions = predictions.copy()

    annotations["conversation_id"] = (
        annotations["conversation_id"]
        .astype(str)
    )

    predictions["conversation_id"] = (
        predictions["conversation_id"]
        .astype(str)
    )

    annotations["human_complete"] = (
        annotations["evidence_complete"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    predictions["ec_prediction"] = (
        predictions["ec_prediction"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    predictions["ec_probability"] = pd.to_numeric(
        predictions["ec_probability"],
        errors="coerce",
    )

    merged = annotations.merge(
        predictions[
            [
                "conversation_id",
                "ec_probability",
                "ec_prediction",
            ]
        ],
        on="conversation_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 150:
        raise RuntimeError(
            f"Expected 150 matched annotations, "
            f"found {len(merged)}."
        )

    merged["human_binary"] = (
        merged["human_complete"] == "YES"
    ).astype(int)

    merged["ec_binary"] = (
        merged["ec_prediction"] == "YES"
    ).astype(int)

    return merged


# =============================================================================
# EC MODEL EVALUATION
# =============================================================================

def evaluate_ec_model(df):

    y_true = df["human_binary"].to_numpy()

    y_pred = df["ec_binary"].to_numpy()

    probabilities = df[
        "ec_probability"
    ].to_numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }


# =============================================================================
# RETRIEVAL HELPERS
# =============================================================================

def get_ec(record):

    try:
        return float(
            record.get(
                "evidence_completeness_score",
                0.0,
            )
            or 0.0
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def detect_intent(query):

    text = str(
        query or ""
    ).lower()

    patterns = [
        (
            "refund",
            [
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
        ),
        (
            "wrong_item",
            [
                "wrong item",
                "wrong product",
                "incorrect item",
                "incorrect product",
                "different item",
                "sent the wrong",
                "received the wrong",
                "received incorrect",
            ],
        ),
        (
            "damaged",
            [
                "damaged",
                "damage",
                "broken",
                "cracked",
                "defective",
                "destroyed",
            ],
        ),
        (
            "replacement",
            [
                "replacement",
                "replace",
                "exchange",
                "swap",
            ],
        ),
        (
            "return",
            [
                "return",
                "send back",
                "returning",
            ],
        ),
        (
            "payment",
            [
                "payment",
                "paid",
                "pay",
                "card",
                "debit",
                "credit card",
                "charged",
                "charge",
            ],
        ),
        (
            "delivery",
            [
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
        ),
        (
            "order",
            [
                "order",
                "ordered",
                "ordering",
                "purchase",
                "bought",
            ],
        ),
        (
            "account",
            [
                "account",
                "login",
                "log in",
                "sign in",
                "signin",
                "password",
                "username",
            ],
        ),
    ]

    for intent, keywords in patterns:
        if any(
            keyword in text
            for keyword in keywords
        ):
            return intent

    return "unknown"


def intent_compatibility(
    query_intent,
    record_intent,
):

    if (
        query_intent == "unknown"
        or not record_intent
    ):
        return 0.0

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

    if record_intent in related.get(
        query_intent,
        set(),
    ):
        return 0.5

    return 0.0


def detect_state(query):

    text = str(
        query or ""
    ).lower()

    states = [
        (
            "marked_delivered_not_received",
            [
                "marked delivered",
                "says delivered",
                "shows delivered",
                "says it was delivered",
                "package was delivered",
                "parcel was delivered",
                "delivered but",
                "delivered and",
            ],
        ),
        (
            "received_wrong",
            [
                "wrong item",
                "wrong product",
                "incorrect item",
                "incorrect product",
                "different item",
                "received the wrong",
                "sent the wrong",
            ],
        ),
        (
            "damaged",
            [
                "damaged",
                "damage",
                "broken",
                "cracked",
                "defective",
            ],
        ),
        (
            "cancelled",
            [
                "cancelled",
                "canceled",
                "cancel my order",
                "order was cancelled",
                "order was canceled",
            ],
        ),
        (
            "declined",
            [
                "declined",
                "rejected",
                "failed",
                "payment failed",
                "payment declined",
            ],
        ),
        (
            "not_received",
            [
                "not received",
                "haven't received",
                "have not received",
                "didn't receive",
                "did not receive",
                "never received",
                "not arrived",
                "hasn't arrived",
                "has not arrived",
            ],
        ),
        (
            "delayed",
            [
                "not arrived",
                "hasn't arrived",
                "have not arrived",
                "haven't received",
                "still waiting",
                "waiting",
                "delayed",
                "late",
                "past due",
                "more than a week",
                "more than a month",
            ],
        ),
    ]

    for state, keywords in states:
        if any(
            keyword in text
            for keyword in keywords
        ):
            return state

    return "unknown"


def state_compatibility(
    query_state,
    record_state,
):

    if query_state == "unknown":
        return 0.0

    if not record_state:
        return 0.5

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

    if record_state in related.get(
        query_state,
        set(),
    ):
        return 0.5

    return 0.0


# =============================================================================
# RETRIEVAL POLICIES
# =============================================================================

def retrieve_top_candidates(
    query,
    index,
    corpus,
    model,
):

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    similarities, indices = index.search(
        embedding,
        TOP_K,
    )

    candidates = []

    query_intent = detect_intent(query)
    query_state = detect_state(query)

    for similarity, idx in zip(
        similarities[0],
        indices[0],
    ):

        if idx < 0:
            continue

        record = corpus[int(idx)]

        record_intent = detect_intent(
            record.get(
                "customer_query",
                "",
            )
        )

        record_state = detect_state(
            record.get(
                "customer_query",
                "",
            )
        )

        intent_score = intent_compatibility(
            query_intent,
            record_intent,
        )

        state_score = state_compatibility(
            query_state,
            record_state,
        )

        ec_score = get_ec(record)

        semantic = float(similarity)

        candidates.append(
            {
                "record": record,
                "semantic": semantic,
                "intent": intent_score,
                "state": state_score,
                "ec": ec_score,
                "conversation_id": str(
                    record.get(
                        "conversation_id"
                    )
                ),
            }
        )

    return candidates


def select_semantic(candidates):

    return max(
        candidates,
        key=lambda x: x["semantic"],
    )


def select_intent(candidates):

    return max(
        candidates,
        key=lambda x: (
            x["intent"],
            x["semantic"],
        ),
    )


def select_ec(candidates):

    return max(
        candidates,
        key=lambda x: (
            0.70 * x["semantic"]
            + 0.30 * x["ec"],
        ),
    )


def select_full(candidates):

    return max(
        candidates,
        key=lambda x: (
            0.55 * x["semantic"]
            + 0.15 * x["intent"]
            + 0.10 * x["state"]
            + 0.20 * x["ec"],
        ),
    )


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():

    (
        annotations,
        predictions,
        index,
        metadata_records,
        corpus,
    ) = load_system()

    human_df = prepare_human_labels(
        annotations,
        predictions,
    )

    print("\n" + "=" * 80)
    print("EC MODEL vs HUMAN LABELS")
    print("=" * 80)

    ec_metrics = evaluate_ec_model(
        human_df
    )

    print(
        f"\nAccuracy : {ec_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {ec_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {ec_metrics['recall']:.4f}"
    )

    print(
        f"F1       : {ec_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {ec_metrics['roc_auc']:.4f}"
    )

    print("\nConfusion matrix:")

    print(
        np.array(
            ec_metrics["confusion_matrix"]
        )
    )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Embedding model loaded.")

    # -------------------------------------------------------------------------
    # Retrieval evaluation
    # -------------------------------------------------------------------------

    rows = []

    print(
        "\nEvaluating retrieval policies "
        "on 150 human-verified examples..."
    )

    for position, human_row in human_df.iterrows():

        query = str(
            human_row.get(
                "customer_messages",
                "",
            )
        )

        # Use the first customer message when available.
        if not query.strip():
            continue

        candidates = retrieve_top_candidates(
            query,
            index,
            corpus,
            model,
        )

        if not candidates:
            continue

        semantic = select_semantic(
            candidates
        )

        intent = select_intent(
            candidates
        )

        ec = select_ec(
            candidates
        )

        full = select_full(
            candidates
        )

        human_label = human_row[
            "human_complete"
        ]

        rows.append(
            {
                "annotation_id":
                    int(
                        human_row[
                            "annotation_id"
                        ]
                    ),

                "human_conversation_id":
                    str(
                        human_row[
                            "conversation_id"
                        ]
                    ),

                "human_evidence_complete":
                    human_label,

                "semantic_conversation_id":
                    semantic[
                        "conversation_id"
                    ],

                "semantic_ec":
                    semantic["ec"],

                "intent_conversation_id":
                    intent[
                        "conversation_id"
                    ],

                "intent_ec":
                    intent["ec"],

                "ec_conversation_id":
                    ec[
                        "conversation_id"
                    ],

                "ec_selected_score":
                    (
                        0.70
                        * ec["semantic"]
                        + 0.30
                        * ec["ec"]
                    ),

                "ec_selected_ec":
                    ec["ec"],

                "full_conversation_id":
                    full[
                        "conversation_id"
                    ],

                "full_selected_score":
                    (
                        0.55
                        * full["semantic"]
                        + 0.15
                        * full["intent"]
                        + 0.10
                        * full["state"]
                        + 0.20
                        * full["ec"]
                    ),

                "full_selected_ec":
                    full["ec"],
            }
        )

    details = pd.DataFrame(
        rows
    )

    if len(details) == 0:
        raise RuntimeError(
            "No retrieval evaluations were produced."
        )

    # -------------------------------------------------------------------------
    # Summary metrics
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RETRIEVAL EVIDENCE SUMMARY")
    print("=" * 80)

    retrieval_summary = {}

    for policy, ec_column in [
        (
            "semantic_only",
            "semantic_ec",
        ),
        (
            "semantic_intent",
            "intent_ec",
        ),
        (
            "semantic_ec",
            "ec_selected_ec",
        ),
        (
            "full",
            "full_selected_ec",
        ),
    ]:

        values = details[
            ec_column
        ].astype(float)

        mean_ec = float(
            values.mean()
        )

        high_medium = float(
            (
                values >= 0.50
            ).mean()
        )

        high = float(
            (
                values >= 0.75
            ).mean()
        )

        retrieval_summary[
            policy
        ] = {
            "mean_selected_ec":
                mean_ec,

            "high_or_medium_rate":
                high_medium,

            "high_rate":
                high,
        }

        print(
            f"\n{policy}"
        )

        print(
            f"  Mean selected EC : "
            f"{mean_ec:.4f}"
        )

        print(
            f"  MEDIUM/HIGH rate : "
            f"{high_medium:.4f}"
        )

        print(
            f"  HIGH rate        : "
            f"{high:.4f}"
        )

    # -------------------------------------------------------------------------
    # Save detailed evaluation
    # -------------------------------------------------------------------------

    details.to_csv(
        DETAIL_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Save JSON summary
    # -------------------------------------------------------------------------

    output = {
        "evaluation": {
            "human_examples": int(
                len(human_df)
            ),

            "retrieval_examples": int(
                len(details)
            ),

            "note":
                "Human labels are verification labels. "
                "Retrieval metrics measure the evidence characteristics "
                "of selected historical candidates and are not direct "
                "answer-quality or resolution-accuracy metrics.",
        },

        "ec_model_vs_human":
            ec_metrics,

        "retrieval_policy_summary":
            retrieval_summary,
    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        f"\nDetailed results:\n{DETAIL_FILE}"
    )

    print(
        f"\nSummary:\n{RESULT_FILE}"
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()