"""
src/rag/ingest.py — Embed annotated_results.csv into ChromaDB
=============================================================
Uses sentence-transformers/all-MiniLM-L6-v2 (local, no API needed).

Run once before starting the agent:
    cd "C:\\Users\\kashy\\OneDrive\\Desktop\\Masters Project"
    conda activate masters-project
    python -m src.rag.ingest

Re-running wipes and rebuilds the collection cleanly.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH       = Path("output/reports/annotated_results.csv")
CHROMA_DIR      = Path("chroma_db")
COLLECTION_NAME = "recognition_records"
EMBED_MODEL     = "all-MiniLM-L6-v2"
BATCH_SIZE      = 100


def ingest():
    print("=" * 55)
    print("  RAG Ingest — HR Recognition Records")
    print("  Embeddings: sentence-transformers/all-MiniLM-L6-v2")
    print("=" * 55)

    # 1. Load CSV
    print(f"\n[1/4] Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}\n"
            "Make sure annotated_results.csv is in output/reports/"
        )

    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("").astype(str)

    print(f"      Loaded {len(df)} records")
    print(f"      Columns: {df.columns.tolist()}")

    # 2. Build rich text chunks
    print("\n[2/4] Building text chunks...")
    chunks = (
        "Award: "       + df["award_title"]       + " | " +
        "Recipient: "   + df["recipient_title"]   + " | " +
        "Nominator: "   + df["nominator_title"]   + " | " +
        "Category: "    + df["category_name"]     + " | " +
        "Subcategory: " + df["subcategory_name"]  + " | " +
        "Message: "     + df["message"]
    ).tolist()
    print(f"      Built {len(chunks)} chunks")

    # 3. Embed in batches
    print(f"\n[3/4] Embedding with {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)
    all_embeddings = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        embeddings = embedder.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
        done = min(i + BATCH_SIZE, len(chunks))
        print(f"      Embedded {done}/{len(chunks)}", end="\r")

    print(f"      Embedded {len(all_embeddings)} records ✓          ")

    # 4. Store in ChromaDB
    print(f"\n[4/4] Storing in ChromaDB at {CHROMA_DIR}...")
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Wipe existing collection
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("      Wiped existing collection")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    metadatas = [
        {
            "category_name":    row["category_name"],
            "subcategory_name": row["subcategory_name"],
            "recipient_title":  row["recipient_title"],
            "nominator_title":  row["nominator_title"],
            "award_title":      row["award_title"],
        }
        for _, row in df.iterrows()
    ]

    for i in range(0, len(chunks), BATCH_SIZE):
        collection.add(
            ids        = [str(j) for j in range(i, min(i + BATCH_SIZE, len(chunks)))],
            documents  = chunks[i : i + BATCH_SIZE],
            embeddings = all_embeddings[i : i + BATCH_SIZE],
            metadatas  = metadatas[i : i + BATCH_SIZE],
        )
        done = min(i + BATCH_SIZE, len(chunks))
        print(f"      Stored {done}/{len(chunks)}", end="\r")

    print(f"      Stored {collection.count()} records ✓          ")

    print("\n" + "=" * 55)
    print("  Ingest complete!")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Records    : {collection.count()}")
    print(f"  Embedding  : {EMBED_MODEL}")
    print(f"  Location   : {CHROMA_DIR.resolve()}")
    print("=" * 55)
    print("\nNext: python -m src.rag.agent")


if __name__ == "__main__":
    ingest()
