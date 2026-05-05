import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

# --- 1. UI & SEMINAR SETUP ---
st.set_page_config(page_title="Abiyamo MVP (Seminar Demo)", page_icon="🛡️", layout="wide")

# Sidebar for the Audience to see "Under the Hood"
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/WHO_logo.svg/1024px-WHO_logo.svg.png",
             width=100)
    st.title("⚙️ System Diagnostics")
    st.markdown("*(Visible for Demo Purposes)*")
    st.markdown("---")
    st.markdown("**Guardrail Status:** `ACTIVE`")
    st.markdown("**Knowledge Base:** `Nigeria FMOH & WHO`")

    # We will use this to show the audience how the system classifies intents live
    st.subheader("Last Query Intent:")
    intent_display = st.empty()
    intent_display.info("Awaiting input...")

# Main Chat Area
st.title("🛡️ Abiyamo: Safe SRH Educator")
st.caption("A deterministic, RAG-powered digital health intervention for Nigerian youth. **Not a diagnostic tool.**")


# --- 2. LOAD RESOURCES ---
@st.cache_resource
def load_system():
    # PASTE YOUR API KEY HERE
    API_KEY = st.secrets["GEMINI_API_KEY"]
    ai_client = genai.Client(api_key=API_KEY)

    # Connect to ChromaDB
    db_client = chromadb.PersistentClient(path="./health_vector_db")
    collection = db_client.get_collection(name="adolescent_health")

    # Load Embedding Model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return ai_client, collection, embed_model


ai_client, collection, embed_model = load_system()

# --- 3. THE "DEMO" INTENT CLASSIFIER ---
# For the MVP, we use Keywords to simulate the DistilBERT layer for speed and 100% reliability live on stage.
CRISIS_WORDS = ["suicide", "kill", "die", "rape", "abuse", "hurt", "emergency"]
CLINICAL_WORDS = ["dosage", "prescribe", "diagnose", "medication", "pill", "cure", "pain"]


def simulate_intent_classification(query):
    query = query.lower()
    if any(word in query for word in CRISIS_WORDS):
        return "CRISIS"
    elif any(word in query for word in CLINICAL_WORDS):
        return "CLINICAL"
    else:
        return "GENERAL_EDUCATION"


# --- 4. CORE PIPELINE ---
def process_query(user_query):
    intent = simulate_intent_classification(user_query)

    # Update the sidebar so the audience sees the classification!
    if intent == "CRISIS":
        intent_display.error("🚨 CRISIS DETECTED - Halting AI")
        return "🚨 **CRISIS ALERT:** Your safety is our priority. Please contact the NAPTIP toll-free line at **0800 7327 3447** or the National Emergency Number at **112** immediately."

    elif intent == "CLINICAL":
        intent_display.warning("⚠️ CLINICAL/DIAGNOSTIC DETECTED - Halting AI")
        return "🩺 **Clinical Boundary Reached:** I am an educational tool and cannot diagnose symptoms or prescribe medication. Please visit a youth-friendly health center or speak to a registered nurse."

    else:
        intent_display.success("✅ GENERAL EDUCATION - Proceeding to RAG")

        # 1. Retrieve Facts
        query_vector = embed_model.encode([user_query]).tolist()
        results = collection.query(query_embeddings=query_vector, n_results=1)

        if not results['documents'][0]:
            return "I don't have verified FMOH/WHO information on that topic."

        raw_fact = results['documents'][0][0]
        source = results['metadatas'][0][0].get("source", "Official Manual")

        # 2. Generate Safe Response
        prompt = f"""
        Act as Abiyamo, an empathetic Nigerian health educator. 
        Answer the teen's question using ONLY this safe knowledge:
        '{raw_fact}'

        Question: {user_query}
        """

        response = ai_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return f"{response.text}\n\n*(📚 Source: {source})*"


# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. I am Abiyamo. Ask me a question about adolescent health."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and retrieving verified data..."):
            reply = process_query(prompt)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})