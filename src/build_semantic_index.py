import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "retrieval_corpus.jsonl"
)

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

BATCH_SIZE = 128


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus():

    records = []

    print("\nLoading retrieval corpus...")

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING SEMANTIC RETRIEVAL INDEX")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Retrieval corpus not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    records = load_corpus()

    print(
        f"Records loaded : {len(records):,}"
    )

    if not records:
        raise RuntimeError(
            "Retrieval corpus is empty."
        )

    # --------------------------------------------------------
    # Prepare texts
    # --------------------------------------------------------

    texts = [
        str(record["customer_query"])
        for record in records
    ]

    print(
        f"Texts prepared : {len(texts):,}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        f"\nLoading embedding model:\n"
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print(
        "\nGenerating embeddings..."
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    print(
        f"\nEmbedding matrix shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # Build FAISS index
    # --------------------------------------------------------
    #
    # Because embeddings are normalized, inner product
    # is equivalent to cosine similarity.
    #

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"FAISS vectors stored : "
        f"{index.ntotal:,}"
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # --------------------------------------------------------
    # Save metadata
    #
    # FAISS only stores vectors. We need the original
    # conversation records to interpret search results.
    # --------------------------------------------------------

    metadata = {
        "model_name": MODEL_NAME,
        "dimension": dimension,
        "normalized": True,
        "records": records,
    }

    with METADATA_FILE.open(
        "wb"
    ) as f:

        pickle.dump(
            metadata,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEMANTIC INDEX CREATED")
    print("=" * 70)

    print(
        f"\nModel              : {MODEL_NAME}"
    )

    print(
        f"Embedding dimension: {dimension}"
    )

    print(
        f"Vectors             : "
        f"{index.ntotal:,}"
    )

    print(
        f"\nFAISS index:"
    )

    print(INDEX_FILE)

    print(
        f"\nMetadata:"
    )

    print(METADATA_FILE)

    print("\nDone.")


if __name__ == "__main__":
    main()