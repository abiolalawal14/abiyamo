"""
2_build_vector_db.py
──────────────────
Reads the RAG JSON dataset, generates semantic embeddings,
and stores them in a persistent ChromaDB vector database.
"""

import json
import argparse
import chromadb
from sentence_transformers import SentenceTransformer


def build_database(json_path, db_directory="./health_vector_db"):
    # 1. Load the Embedding Model (This downloads a small AI model the first time)
    print("🧠 Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. Initialize Persistent ChromaDB
    print(f"📁 Initializing Vector Database at {db_directory}...")
    client = chromadb.PersistentClient(path=db_directory)

    # Create or connect to our specific collection
    collection = client.get_or_create_collection(name="adolescent_health")

    # 3. Load the JSON Dataset
    print(f"📖 Loading data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents = []
    metadatas = []
    ids = []

    print("⚙️ Preparing chunks and metadata...")
    for chunk in chunks:
        documents.append(chunk["text"])
        ids.append(chunk["chunk_id"])

        metadatas.append({
            "source": chunk["source"],
            "category": chunk.get("category", "general_health"),
            "intent": chunk.get("intent", "awareness"),
        })

    # 4. Generate Embeddings & Insert into DB in batches
    batch_size = 100
    total_chunks = len(documents)

    print(f"🚀 Generating embeddings and storing {total_chunks} chunks...")

    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)

        batch_docs = documents[i:end_idx]
        batch_ids = ids[i:end_idx]
        batch_meta = metadatas[i:end_idx]

        # Turn text into vectors
        batch_embeddings = model.encode(batch_docs).tolist()

        # Save to ChromaDB
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"   ✅ Saved chunks {i} to {end_idx}...")

    print("\n🎉 Vector Database successfully built!")
    print(f"The AI's memory is now stored in the folder: {db_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB from JSON chunks")
    parser.add_argument("json_path", help="Path to your _rag_chunks.json file")

    args = parser.parse_args()
    build_database(args.json_path)