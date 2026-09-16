import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_FILE = (
    BASE_DIR
    / "output"
    / "semantic_index.faiss"
)

METADATA_FILE = (
    BASE_DIR
    / "output"
    / "semantic_index_metadata.pkl"
)

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

SEMANTIC_WEIGHT = 0.70
EC_WEIGHT = 0.30


# ============================================================
# LOAD INDEX
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

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"FAISS vectors: {index.ntotal:,}"
    )

    print(
        f"Vector dimension: {index.d}"
    )

    print("\nLoading metadata...")

    with METADATA_FILE.open(
        "rb"
    ) as f:

        metadata = pickle.load(f)

    records = metadata["records"]

    if len(records) != index.ntotal:
        raise RuntimeError(
            "FAISS/metadata mismatch:\n"
            f"FAISS vectors = {index.ntotal:,}\n"
            f"Metadata records = {len(records):,}"
        )

    print(
        f"Metadata records: {len(records):,}"
    )

    return index, metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():

    print(
        f"\nLoading embedding model:\n"
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Embedding model loaded.")

    return model


# ============================================================
# EMBED QUERY
# ============================================================

def embed_query(
    model,
    query
):

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.astype(
        np.float32
    )


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
    except (
        TypeError,
        ValueError
    ):
        return 0.0


# ============================================================
# COMBINED SCORE
# ============================================================

def combined_score(
    semantic_similarity,
    ec_score
):

    score = (
        SEMANTIC_WEIGHT
        * semantic_similarity
        + EC_WEIGHT
        * ec_score
    )

    return round(
        score,
        4
    )


# ============================================================
# SEARCH
# ============================================================

def search(
    index,
    records,
    model,
    query,
    top_k=TOP_K
):

    query_vector = embed_query(
        model,
        query
    )

    similarities, indices = (
        index.search(
            query_vector,
            top_k
        )
    )

    results = []

    for rank, (
        similarity,
        index_position
    ) in enumerate(
        zip(
            similarities[0],
            indices[0]
        ),
        start=1
    ):

        if index_position < 0:
            continue

        record = records[
            int(index_position)
        ]

        semantic_similarity = float(
            similarity
        )

        ec = get_ec_score(
            record
        )

        result = {
            "semantic_rank": rank,
            "index_position": int(
                index_position
            ),
            "semantic_similarity":
                round(
                    semantic_similarity,
                    4
                ),
            "ec_score": round(
                ec,
                4
            ),
            "combined_score":
                combined_score(
                    semantic_similarity,
                    ec
                ),
            "evidence_tier":
                record.get(
                    "evidence_tier",
                    "UNKNOWN"
                ),
            "conversation_id":
                record.get(
                    "conversation_id"
                ),
            "customer_query":
                record.get(
                    "customer_query",
                    ""
                ),
            "agent_response":
                record.get(
                    "agent_response",
                    ""
                ),
            "D1":
                record.get(
                    "D1_context_specificity",
                    0.0
                ),
            "D2":
                record.get(
                    "D2_action_specificity",
                    0.0
                ),
            "D3":
                record.get(
                    "D3_dm_redirect",
                    0
                ),
            "D4":
                record.get(
                    "D4_observable_outcome",
                    "unknown"
                ),
        }

        results.append(
            result
        )

    return results


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    query,
    results
):

    print("\n")
    print("=" * 80)
    print("SEMANTIC RETRIEVAL RESULTS")
    print("=" * 80)

    print(
        f"\nCustomer query:\n{query}"
    )

    for result in results:

        print("\n" + "-" * 80)

        print(
            f"Semantic rank       : "
            f"{result['semantic_rank']}"
        )

        print(
            f"Conversation ID     : "
            f"{result['conversation_id']}"
        )

        print(
            f"Semantic similarity : "
            f"{result['semantic_similarity']:.4f}"
        )

        print(
            f"EC score            : "
            f"{result['ec_score']:.4f}"
        )

        print(
            f"Combined score      : "
            f"{result['combined_score']:.4f}"
        )

        print(
            f"Evidence tier       : "
            f"{result['evidence_tier']}"
        )

        print(
            f"D1={result['D1']}  "
            f"D2={result['D2']}  "
            f"D3={result['D3']}  "
            f"D4={result['D4']}"
        )

        print(
            "\nHistorical customer:"
        )

        print(
            result["customer_query"]
        )

        print(
            "\nHistorical agent:"
        )

        print(
            result["agent_response"]
        )


# ============================================================
# INTERACTIVE MODE
# ============================================================

def main():

    print("=" * 80)
    print(
        "EVIDENCE-COMPLETENESS-AWARE RETRIEVAL"
    )
    print("=" * 80)

    index, metadata = load_index()

    records = metadata["records"]

    model = load_model()

    print(
        "\nSystem ready."
    )

    print(
        "\nEnter a customer-support query."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        print("\n" + "-" * 80)

        query = input(
            "Customer query > "
        ).strip()

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit"
        }:
            break

        results = search(
            index,
            records,
            model,
            query,
            TOP_K
        )

        display_results(
            query,
            results
        )

    print(
        "\nRetriever stopped."
    )


if __name__ == "__main__":
    main()