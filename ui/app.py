import streamlit as st
import requests

# Point to FastAPI backend
API_URL = "http://127.0.0.1:8000"

# Configure the Streamlit page
st.set_page_config(page_title="localRAGvault", page_icon="🗄️", layout="wide")
st.title("🗄️ localRAGvault")
st.markdown("Your privacy-first, fully local document assistant.")


# --- Fetch Dynamic Models from Backend ---
@st.cache_data(
    ttl=60
)  # Cache for 1 minute so it doesn't slam the endpoint on every click
def fetch_available_models():
    try:
        res = requests.get(f"{API_URL}/models/")
        if res.status_code == 200:
            return res.json().get("models", [])
    except requests.exceptions.ConnectionError:
        return []
    return []


available_models = fetch_available_models()

# Fallback defaults if backend is down or Ollama is empty
embedding_options = [m for m in available_models if "embed" in m] or [
    "nomic-embed-text"
]
generation_options = [m for m in available_models if "embed" not in m] or [
    "gemma3",
    "gemma4",
]

# --- Sidebar: Model Configuration & Ingestion ---
with st.sidebar:
    st.header("⚙️ System Configuration")

    # Dropdowns for dynamic model steering
    selected_embed_model = st.selectbox(
        "Select Embedding Model",
        options=embedding_options,
        help="Must match the model space used to search existing collections.",
    )

    selected_gen_model = st.selectbox(
        "Select Generation LLM",
        options=generation_options,
        help="The language model that processes context and generates the reply.",
    )

    st.markdown("---")
    st.header("1. Add to Vault")
    uploaded_file = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])

    if st.button("Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Chunking and embedding document..."):
                # Send the file along with the form parameter for the embedding model
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")
                }
                data = {"embedding_model": selected_embed_model}
                try:
                    res = requests.post(f"{API_URL}/upload/", files=files, data=data)
                    if res.status_code == 200:
                        res_data = res.json()
                        st.success(
                            f"✅ Success! {res_data['chunks_saved']} chunks saved using '{selected_embed_model}'."
                        )
                    else:
                        st.error(f"Failed to ingest: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Backend is unreachable. Is FastAPI running?")
        else:
            st.warning("Please select a file first.")

# --- Main Window: Chat/Search Execution ---
st.header("2. Ask your Vault")
query = st.text_input("What would you like to know about your documents?")

if st.button("Search & Generate"):
    if query:
        with st.spinner("Searching the vault and generating an answer..."):
            payload = {
                "query": query,
                "top_k": 3,
                "embedding_model": selected_embed_model,
                "generation_model": selected_gen_model,
            }
            try:
                # Execution against the full /ask/ orchestration loop
                res = requests.post(f"{API_URL}/ask/", json=payload)

                if res.status_code == 200:
                    res_data = res.json()

                    # Display the Synthesized AI Answer
                    st.markdown("### Answer")
                    st.info(res_data["answer"])

                    # Display the Source Attribution Meta
                    st.markdown("### Sources Cited")
                    for source in res_data["sources"]:
                        st.caption(
                            f"📄 **{source['filename']}** (Similarity Score: {source['similarity']})"
                        )
                else:
                    st.error(f"Error generating answer: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Backend is unreachable. Is FastAPI running?")
    else:
        st.warning("Please enter a question.")
