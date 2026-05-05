"""
5_abiyamo_web.py
────────────
The web interface for Abiyamo, the safe adolescent health chatbot.
Built with Streamlit, ChromaDB, and Google Gemini.
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

# --- UI SETUP ---
st.set_page_config(page_title="Abiyamo - Health Educator", page_icon="👩🏾‍⚕️", layout="centered")
st.title("👩🏾‍⚕️ Abiyamo")
st.markdown("Your safe, confidential adolescent health guide.")


# --- CACHED RESOURCES (Loads memory only once for speed) ---
@st.cache_resource
def load_ai_and_db():
    print("Loading resources...")
    # 1. Setup Gemini (PASTE YOUR API KEY HERE)
    API_KEY = st.secrets["GEMINI_API_KEY"]
    ai_client = genai.Client(api_key=API_KEY)

    # 2. Setup ChromaDB
    db_client = chromadb.PersistentClient(path="./health_vector_db")
    collection = db_client.get_collection(name="adolescent_health")

    # 3. Setup Embeddings
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    return ai_client, collection, embed_model


ai_client, collection, embed_model = load_ai_and_db()
MODEL_NAME = "gemini-2.5-flash"

# --- SAFETY RULES ---
CRISIS_KEYWORDS = ["suicide", "kill", "hurt myself", "rape", "abuse", "danger", "emergency"]
CLINICAL_KEYWORDS = ["dosage", "prescribe", "diagnose", "medication", "treatment", "cure"]


# --- CORE LOGIC ---
def search_knowledge(user_query, n_results=1):
    query_vector = embed_model.encode([user_query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=n_results)
    return results


def generate_ai_response(user_query, raw_knowledge):
    prompt = f"""
    You are 'Abiyamo', a friendly, empathetic, and responsible health educator for teenagers in Nigeria.
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
    response = ai_client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


def safe_health_bot(user_query):
    query_lower = user_query.lower()

    if any(word in query_lower for word in CRISIS_KEYWORDS):
        return {"status": "CRISIS",
                "response": "🚨 **This sounds very serious.** Please speak to a trusted adult, a school counselor, or contact a local health professional immediately. You are not alone."}

    db_results = search_knowledge(user_query)

    if not db_results['documents'][0]:
        return {"status": "UNKNOWN",
                "response": "I'm sorry, I don't have enough verified information to answer that. Please ask a health professional."}

    best_text = db_results['documents'][0][0]
    best_metadata = db_results['metadatas'][0][0]

    if best_metadata.get("intent") == "clinical" or any(word in query_lower for word in CLINICAL_KEYWORDS):
        return {"status": "CLINICAL",
                "response": "🩺 I can share general health information, but **I cannot give medical advice or prescribe medicine**. Please consult a qualified doctor or visit a youth-friendly health clinic."}

    final_answer = generate_ai_response(user_query, best_text)

    return {
        "status": "SUCCESS",
        "source": best_metadata.get("source"),
        "response": final_answer
    }


# --- CHAT INTERFACE LOGIC ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "Hello! I am Abiyamo. What questions do you have about your health or growing up today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask Abiyamo a question..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Abiyamo is thinking..."):
            reply_data = safe_health_bot(prompt)
            response_text = reply_data["response"]

            # Add a small footnote with the source if it was a successful data retrieval
            if reply_data["status"] == "SUCCESS":
                response_text += f"\n\n*(📚 Source: {reply_data['source']})*"

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})