"""
4_generative_chatbot.py
────────────
The final RAG system. Retrieves facts from ChromaDB,
passes them to Google Gemini (using the latest API), and generates a safe, friendly response.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai  # NEW Google GenAI SDK

# --- SETUP YOUR AI ---
# Replace this with your actual API key from Google AI Studio
API_KEY = "AIzaSyCNOpH8IF64NOPGT03Y1_d-EvbcVkpu1iw"
ai_client = genai.Client(api_key=API_KEY)

# Using Google's latest, fastest free-tier model
MODEL_NAME = "gemini-2.5-flash"

# --- CONNECT TO MEMORY ---
print("🔌 Connecting to the Vector Database...")
db_client = chromadb.PersistentClient(path="./health_vector_db")
collection = db_client.get_collection(name="adolescent_health")

print("🧠 Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- SAFETY RULES ---
CRISIS_KEYWORDS = ["suicide", "kill", "hurt myself", "rape", "abuse", "danger", "emergency"]
CLINICAL_KEYWORDS = ["dosage", "prescribe", "diagnose", "medication", "treatment", "cure"]

def search_knowledge(user_query, n_results=1):
    query_vector = embed_model.encode([user_query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=n_results)
    return results

def generate_ai_response(user_query, raw_knowledge):
    """This is where the magic happens. We instruct the LLM on how to behave."""

    prompt = f"""
    You are a friendly, empathetic, and responsible health educator for teenagers in Nigeria.
    Your goal is to answer the teenager's question using ONLY the provided 'Safe Knowledge'.
    
    RULES:
    1. Use simple, easy-to-understand language. Tone should be warm but professional.
    2. DO NOT invent medical advice, dosages, or diagnoses.
    3. If the 'Safe Knowledge' doesn't fully answer the question, admit it gracefully and encourage them to speak to a doctor or trusted adult.
    4. Keep the answer concise (2-4 short paragraphs).
    
    SAFE KNOWLEDGE FROM HEALTH MANUAL:
    {raw_knowledge}
    
    TEENAGER ASKS:
    "{user_query}"
    """

    # Send the prompt to Gemini using the modern SDK
    response = ai_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text

def safe_health_bot(user_query):
    query_lower = user_query.lower()

    # Step A: The "Redline" Crisis Filter
    if any(word in query_lower for word in CRISIS_KEYWORDS):
        return {"status": "CRISIS", "response": "This sounds very serious. Please speak to a trusted adult, a school counselor, or contact a local health professional immediately. You are not alone."}

    # Step B: Retrieve Facts from Database
    db_results = search_knowledge(user_query)

    # If the database is empty or fails
    if not db_results['documents'][0]:
        return {"status": "UNKNOWN", "response": "I'm sorry, I don't have enough verified information to answer that. Please ask a health professional."}

    best_text = db_results['documents'][0][0]
    best_metadata = db_results['metadatas'][0][0]

    # Step C: The "Clinical" Filter
    if best_metadata.get("intent") == "clinical" or any(word in query_lower for word in CLINICAL_KEYWORDS):
        return {"status": "CLINICAL", "response": "I can share general health information, but I cannot give medical advice or prescribe medicine. Please consult a qualified doctor."}

    # Step D: AI Generation!
    print("⏳ AI is thinking and writing the response...")
    final_answer = generate_ai_response(user_query, best_text)

    return {
        "status": "SUCCESS",
        "source": best_metadata.get("source"),
        "response": final_answer
    }

# --- TEST THE FULL SYSTEM ---
if __name__ == "__main__":
    print("\n✅ Generative Bot is ready! (Type 'quit' to exit)")
    print("-" * 50)

    while True:
        question = input("\n👤 Teenager: ")
        if question.lower() == 'quit':
            break

        bot_reply = safe_health_bot(question)

        print("\n🤖 Bot:", bot_reply["response"])
        if bot_reply["status"] == "SUCCESS":
            print(f"   [Source Fact from: {bot_reply['source']}]")