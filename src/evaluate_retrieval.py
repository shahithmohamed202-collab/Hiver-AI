import json
import pickle
from pathlib import Path
from collections import Counter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_FILE = BASE_DIR / "output" / "retrieval_corpus.jsonl"
INDEX_FILE = BASE_DIR / "output" / "semantic_index.faiss"
METADATA_FILE = BASE_DIR / "output" / "semantic_index_metadata.pkl"

RESULTS_FILE = BASE_DIR / "output" / "retrieval_evaluation.json"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

# Initial experimental weighting.
# This is NOT claimed to be optimal.
SEMANTIC_WEIGHT = 0.70
EC_WEIGHT = 0.30


# ============================================================
# CONTROLLED TEST QUERIES
# ============================================================

TEST_QUERIES = [
    {
        "id": "Q01",
        "query": "My refund hasn't arrived yet",
        "intent": "refund",
        "state": "not_received",
    },
    {
        "id": "Q02",
        "query": "I received the wrong item",
        "intent": "wrong_item",
        "state": "received_wrong",
    },
    {
        "id": "Q03",
        "query": "My package has not been delivered",
        "intent": "delivery",
        "state": "not_received",
    },
    {
        "id": "Q04",
        "query": "My package says delivered but I never got it",
        "intent": "delivery",
        "state": "marked_delivered_not_received",
    },
    {
        "id": "Q05",
        "query": "I want to return my order",
        "intent": "return",
        "state": "return_requested",
    },
    {
        "id": "Q06",
        "query": "I was charged twice for the same order",
        "intent": "payment",
        "state": "duplicate_charge",
    },
    {
        "id": "Q07",
        "query": "My order was cancelled unexpectedly",
        "intent": "order",
        "state": "cancelled",
    },
    {
        "id": "Q08",
        "query": "I need help tracking my order",
        "intent": "delivery",
        "state": "tracking",
    },
    {
        "id": "Q09",
        "query": "I cannot sign into my account",
        "intent": "account",
        "state": "login_problem",
    },
    {
        "id": "Q10",
        "query": "My payment was declined",
        "intent": "payment",
        "state": "payment_declined",
    },
    {
        "id": "Q11",
        "query": "I received a damaged product",
        "intent": "product",
        "state": "damaged",
    },
    {
        "id": "Q12",
        "query": "My order is taking too long to arrive",
        "intent": "delivery",
        "state": "delayed",
    },
    {
        "id": "Q13",
        "query": "I have not received my replacement item",
        "intent": "replacement",
        "state": "not_received",
    },
    {
        "id": "Q14",
        "query": "I need to update my delivery address",
        "intent": "delivery",
        "state": "address_change",
    },
    {
        "id": "Q15",
        "query": "I was charged but my order was not placed",
        "intent": "payment",
        "state": "charged_no_order",
    },
]


# ============================================================
# INTENT KEYWORDS
# ============================================================

INTENT_KEYWORDS = {
    "refund": [
        "refund",
        "refunded",
        "reimbursement",
        "money back",
        "cash back",
    ],
    "wrong_item": [
        "wrong item",
        "wrong product",
        "incorrect item",
        "incorrect product",
        "different item",
    ],
    "delivery": [
        "package",
        "parcel",
        "delivered",
        "delivery",
        "arrive",
        "arrived",
        "shipping",
        "shipment",
        "tracking",
        "track",
        "courier",
        "address",
    ],
    "return": [
        "return",
        "send back",
        "returning",
    ],
    "payment": [
        "payment",
        "charged",
        "charge",
        "billing",
        "declined",
        "duplicate",
        "twice",
        "transaction",
    ],
    "account": [
        "account",
        "login",
        "log in",
        "sign in",
        "password",
        "username",
    ],
    "product": [
        "product",
        "item",
        "damaged",
        "broken",
        "defective",
    ],
    "replacement": [
        "replacement",
        "replace",
        "replaced",
    ],
    "order": [
        "order",
        "cancelled",
        "canceled",
    ],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    return " ".join(str(text).lower().split())


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(text):
    text = normalize_text(text)

    scores = Counter()

    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                scores[intent] += 1

    if not scores:
        return "unknown"

    return scores.most_common(1)[0][0]


# ============================================================
# INTENT COMPATIBILITY
# ============================================================

def intent_compatibility(query_intent, record_intent):
    if query_intent == "unknown" or record_intent == "unknown":
        return 0.0

    if query_intent == record_intent:
        return 1.0

    # Closely related support intents.
    related = {
        "wrong_item": {"product", "return"},
        "product": {"wrong_item", "return"},
        "delivery": {"order"},
        "order": {"delivery"},
        "refund": {"return", "payment"},
        "return": {"refund", "wrong_item", "product"},
        "payment": {"refund", "order"},
        "replacement": {"product", "delivery"},
    }

    if record_intent in related.get(query_intent, set()):
        return 0.5

    return 0.0


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus():
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"Retrieval corpus not found:\n{CORPUS_FILE}"
        )

    records = []

    print("Loading retrieval corpus...")

    with CORPUS_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    print(f"Corpus records: {len(records):,}")

    return records


# ============================================================
# LOAD FAISS
# ============================================================

def load_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found:\n{INDEX_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    print("\nLoading FAISS index...")

    index = faiss.read_index(str(INDEX_FILE))

    print(f"FAISS vectors: {index.ntotal:,}")
    print(f"Vector dimension: {index.d}")

    print("\nLoading metadata...")

    with METADATA_FILE.open("rb") as f:
        metadata = pickle.load(f)

    metadata_records = metadata["records"]

    print(f"Metadata records: {len(metadata_records):,}")

    if len(metadata_records) != index.ntotal:
        raise RuntimeError(
            "FAISS and metadata are inconsistent."
        )

    return index, metadata_records


# ============================================================
# MODEL
# ============================================================

def load_model():
    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    return model


# ============================================================
# EMBEDDING
# ============================================================

def embed_query(model, query):
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.astype(np.float32)


# ============================================================
# EC SCORE
# ============================================================

def get_ec_score(record):
    value = record.get(
        "evidence_completeness_score",
        0.0
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# RECORD INTENT
# ============================================================

def get_record_intent(record):
    text = record.get("customer_query", "")

    return detect_intent(text)


# ============================================================
# RETRIEVE
# ============================================================

def retrieve_candidates(
    index,
    records,
    model,
    query,
    top_k=TOP_K,
):
    query_vector = embed_query(model, query)

    similarities, indices = index.search(
        query_vector,
        top_k,
    )

    candidates = []

    for rank, (similarity, index_position) in enumerate(
        zip(similarities[0], indices[0]),
        start=1,
    ):
        if index_position < 0:
            continue

        record = records[int(index_position)]

        semantic_similarity = float(similarity)

        ec_score = get_ec_score(record)

        record_intent = get_record_intent(record)

        candidates.append(
            {
                "semantic_rank": rank,
                "index_position": int(index_position),
                "semantic_similarity": semantic_similarity,
                "ec_score": ec_score,
                "record_intent": record_intent,
                "evidence_tier": record.get(
                    "evidence_tier",
                    "UNKNOWN",
                ),
                "conversation_id": record.get(
                    "conversation_id"
                ),
                "customer_query": record.get(
                    "customer_query",
                    "",
                ),
                "agent_response": record.get(
                    "agent_response",
                    "",
                ),
                "D1": record.get(
                    "D1_context_specificity",
                    0.0,
                ),
                "D2": record.get(
                    "D2_action_specificity",
                    0.0,
                ),
                "D3": record.get(
                    "D3_dm_redirect",
                    0,
                ),
                "D4": record.get(
                    "D4_observable_outcome",
                    "unknown",
                ),
            }
        )

    return candidates


# ============================================================
# SCORE VARIANTS
# ============================================================

def score_semantic_only(candidate):
    return candidate["semantic_similarity"]


def score_semantic_intent(
    candidate,
    query_intent,
):
    compatibility = intent_compatibility(
        query_intent,
        candidate["record_intent"],
    )

    return (
        0.85 * candidate["semantic_similarity"]
        + 0.15 * compatibility
    )


def score_semantic_ec(candidate):
    return (
        SEMANTIC_WEIGHT
        * candidate["semantic_similarity"]
        + EC_WEIGHT
        * candidate["ec_score"]
    )


def score_full(
    candidate,
    query_intent,
):
    compatibility = intent_compatibility(
        query_intent,
        candidate["record_intent"],
    )

    semantic_intent_score = (
        0.85 * candidate["semantic_similarity"]
        + 0.15 * compatibility
    )

    return (
        0.70 * semantic_intent_score
        + 0.30 * candidate["ec_score"]
    )


# ============================================================
# RANK VARIANT
# ============================================================

def rank_candidates(
    candidates,
    query_intent,
    variant,
):
    scored = []

    for candidate in candidates:
        if variant == "semantic_only":
            score = score_semantic_only(candidate)

        elif variant == "semantic_intent":
            score = score_semantic_intent(
                candidate,
                query_intent,
            )

        elif variant == "semantic_ec":
            score = score_semantic_ec(candidate)

        elif variant == "full":
            score = score_full(
                candidate,
                query_intent,
            )

        else:
            raise ValueError(
                f"Unknown variant: {variant}"
            )

        item = dict(candidate)

        item["variant_score"] = float(score)

        scored.append(item)

    scored.sort(
        key=lambda x: x["variant_score"],
        reverse=True,
    )

    for rank, item in enumerate(
        scored,
        start=1,
    ):
        item["final_rank"] = rank

    return scored


# ============================================================
# QUALITY PROXIES
# ============================================================

def calculate_top1_ec(ranked):
    if not ranked:
        return 0.0

    return ranked[0]["ec_score"]


def calculate_top1_similarity(ranked):
    if not ranked:
        return 0.0

    return ranked[0]["semantic_similarity"]


def calculate_top1_intent_match(
    ranked,
    query_intent,
):
    if not ranked:
        return 0.0

    compatibility = intent_compatibility(
        query_intent,
        ranked[0]["record_intent"],
    )

    return compatibility


def calculate_high_medium_rate(ranked):
    if not ranked:
        return 0.0

    good = sum(
        1
        for item in ranked
        if item["evidence_tier"] in {
            "HIGH",
            "MEDIUM",
        }
    )

    return good / len(ranked)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print("=" * 80)
    print("RETRIEVAL VARIANT EVALUATION")
    print("=" * 80)

    records = load_corpus()

    index, metadata_records = load_index()

    # The FAISS metadata contains the exact record ordering
    # used when the index was built.
    if len(metadata_records) != len(records):
        raise RuntimeError(
            "Corpus and FAISS metadata have different lengths."
        )

    model = load_model()

    variants = [
        "semantic_only",
        "semantic_intent",
        "semantic_ec",
        "full",
    ]

    all_results = []

    aggregate = {
        variant: {
            "top1_similarity": [],
            "top1_ec": [],
            "top1_intent_match": [],
            "top5_high_medium_rate": [],
        }
        for variant in variants
    }

    print("\n")
    print("=" * 80)
    print("RUNNING CONTROLLED EVALUATION")
    print("=" * 80)

    for test in TEST_QUERIES:
        query_id = test["id"]
        query = test["query"]
        expected_intent = test["intent"]

        print("\n" + "-" * 80)
        print(
            f"{query_id}: {query}"
        )
        print(
            f"Expected intent: {expected_intent}"
        )

        candidates = retrieve_candidates(
            index,
            metadata_records,
            model,
            query,
            TOP_K,
        )

        query_result = {
            "id": query_id,
            "query": query,
            "expected_intent": expected_intent,
            "state": test["state"],
            "variants": {},
        }

        for variant in variants:
            ranked = rank_candidates(
                candidates,
                expected_intent,
                variant,
            )

            top1 = ranked[0] if ranked else None

            aggregate[variant][
                "top1_similarity"
            ].append(
                calculate_top1_similarity(ranked)
            )

            aggregate[variant][
                "top1_ec"
            ].append(
                calculate_top1_ec(ranked)
            )

            aggregate[variant][
                "top1_intent_match"
            ].append(
                calculate_top1_intent_match(
                    ranked,
                    expected_intent,
                )
            )

            aggregate[variant][
                "top5_high_medium_rate"
            ].append(
                calculate_high_medium_rate(
                    ranked
                )
            )

            query_result["variants"][variant] = {
                "top5": [
                    {
                        "rank": item["final_rank"],
                        "conversation_id": item[
                            "conversation_id"
                        ],
                        "score": round(
                            item["variant_score"],
                            4,
                        ),
                        "semantic_similarity": round(
                            item[
                                "semantic_similarity"
                            ],
                            4,
                        ),
                        "ec_score": round(
                            item["ec_score"],
                            4,
                        ),
                        "evidence_tier": item[
                            "evidence_tier"
                        ],
                        "record_intent": item[
                            "record_intent"
                        ],
                        "D1": item["D1"],
                        "D2": item["D2"],
                        "D3": item["D3"],
                        "D4": item["D4"],
                        "customer_query": item[
                            "customer_query"
                        ],
                        "agent_response": item[
                            "agent_response"
                        ],
                    }
                    for item in ranked
                ]
            }

            if top1:
                print(
                    f"\n{variant}"
                )
                print(
                    f"  Top-1 conversation : "
                    f"{top1['conversation_id']}"
                )
                print(
                    f"  Score              : "
                    f"{top1['variant_score']:.4f}"
                )
                print(
                    f"  Semantic similarity: "
                    f"{top1['semantic_similarity']:.4f}"
                )
                print(
                    f"  EC score           : "
                    f"{top1['ec_score']:.4f}"
                )
                print(
                    f"  Evidence tier      : "
                    f"{top1['evidence_tier']}"
                )
                print(
                    f"  Record intent      : "
                    f"{top1['record_intent']}"
                )

        all_results.append(query_result)

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    summary = {}

    for variant in variants:
        summary[variant] = {
            "mean_top1_similarity": round(
                float(
                    np.mean(
                        aggregate[variant][
                            "top1_similarity"
                        ]
                    )
                ),
                4,
            ),
            "mean_top1_ec": round(
                float(
                    np.mean(
                        aggregate[variant][
                            "top1_ec"
                        ]
                    )
                ),
                4,
            ),
            "mean_top1_intent_compatibility": round(
                float(
                    np.mean(
                        aggregate[variant][
                            "top1_intent_match"
                        ]
                    )
                ),
                4,
            ),
            "mean_top5_high_medium_rate": round(
                float(
                    np.mean(
                        aggregate[variant][
                            "top5_high_medium_rate"
                        ]
                    )
                ),
                4,
            ),
        }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {
        "project": (
            "Evidence-Completeness-Aware "
            "Customer Support RAG"
        ),
        "evaluation_note": (
            "Controlled retrieval comparison using "
            "15 manually designed support queries. "
            "These are evaluation queries, not human "
            "ground-truth annotations."
        ),
        "ec_note": (
            "Evidence Completeness is a deterministic "
            "heuristic feature and is not a calibrated "
            "probability."
        ),
        "weight_note": (
            "The 70/30 semantic/EC weighting is an "
            "initial experimental baseline, not an "
            "optimized value."
        ),
        "model": MODEL_NAME,
        "top_k": TOP_K,
        "variants": variants,
        "summary": summary,
        "queries": all_results,
    }

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"\nQueries evaluated: "
        f"{len(TEST_QUERIES)}"
    )

    print("\nVariant comparison:")

    for variant in variants:
        result = summary[variant]

        print("\n" + variant)

        print(
            "  Mean top-1 semantic similarity : "
            f"{result['mean_top1_similarity']:.4f}"
        )

        print(
            "  Mean top-1 EC score            : "
            f"{result['mean_top1_ec']:.4f}"
        )

        print(
            "  Mean top-1 intent compatibility: "
            f"{result['mean_top1_intent_compatibility']:.4f}"
        )

        print(
            "  Mean top-5 HIGH/MEDIUM rate     : "
            f"{result['mean_top5_high_medium_rate']:.4f}"
        )

    print("\nResults saved to:")
    print(RESULTS_FILE)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()