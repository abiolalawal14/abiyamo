"""
3_chatbot.py
────────────
Connects to the ChromaDB memory, enforces safety rules,
and retrieves the best knowledge to answer the user's question.
"""

import chromadb
from sentence_transformers import SentenceTransformer

# 1. Connect to the AI's Memory (ChromaDB)
print("🔌 Connecting to the Vector Database...")
client = chromadb.PersistentClient(path="./health_vector_db")
collection = client.get_collection(name="adolescent_health")

# 2. Load the Embedding Model (so we can turn the user's question into a vector)
print("🧠 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Define Safety Rules
CRISIS_KEYWORDS = ["suicide", "kill", "hurt myself", "rape", "abuse", "danger", "emergency"]
CLINICAL_KEYWORDS = ["dosage", "prescribe", "diagnose", "medication", "treatment"]


def search_knowledge(user_query, n_results=2):
    """Searches the database for the most relevant chunks."""
    # Convert question to vector
    query_vector = model.encode([user_query]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results
    )

    return results


def safe_health_bot(user_query):
    """The main routing engine for the chatbot."""
    query_lower = user_query.lower()

    # Step A: The "Redline" Crisis Filter
    if any(word in query_lower for word in CRISIS_KEYWORDS):
        return {
            "status": "CRISIS",
            "bot_response": "This sounds very serious. Please speak to a trusted adult, a school counselor, or contact a local health professional immediately. You are not alone."
        }

    # Step B: Retrieve Facts from Database
    db_results = search_knowledge(user_query)

    # Extract the best matching text and its metadata
    best_text = db_results['documents'][0][0]
    best_metadata = db_results['metadatas'][0][0]

    # Step C: The "Clinical" Filter (Using your dataset's labels!)
    if best_metadata.get("intent") == "clinical" or any(word in query_lower for word in CLINICAL_KEYWORDS):
        return {
            "status": "CLINICAL",
            "bot_response": "I can share general health information, but I cannot give medical advice or prescribe medicine. Please consult a qualified doctor or visit a youth-friendly health clinic."
        }

    # Step D: Safe to generate a response!
    # (Right now, we just return the raw text. Next, we will add the LLM here)
    return {
        "status": "SAFE",
        "source": best_metadata.get("source"),
        "category": best_metadata.get("category"),
        "raw_knowledge": best_text
    }


# --- TEST THE BOT ---
if __name__ == "__main__":
    print("\n✅ Bot is ready! (Type 'quit' to exit)")
    print("-" * 50)

    while True:
        question = input("\n👤 Teenager asks: ")
        if question.lower() == 'quit':
            break

        answer = safe_health_bot(question)

        if answer["status"] in ["CRISIS", "CLINICAL"]:
            print(f"\n🤖 Bot ({answer['status']}): {answer['bot_response']}")
        else:
            print(f"\n🧠 Source Data: [{answer['source']}]")
            print(f"🤖 Bot (Raw Knowledge): {answer['raw_knowledge']}")