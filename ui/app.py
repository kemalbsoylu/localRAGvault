import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

from core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="localRAGvault", page_icon="🗄️", layout="wide")
st.title("🗄️ localRAGvault")
st.markdown("Your privacy-first, fully local document assistant.")


@st.cache_data(ttl=60)
def fetch_available_models():
    try:
        res = requests.get(f"{API_URL}/models/")
        if res.status_code == 200:
            return res.json().get("models", [])
    except requests.exceptions.ConnectionError:
        return []
    return []


available_models = fetch_available_models()

# Sort available options dynamically
embedding_options = [m for m in available_models if "embed" in m] or [DEFAULT_EMBEDDING_MODEL]
generation_options = [m for m in available_models if "embed" not in m] or [DEFAULT_GENERATION_MODEL]

default_embed_idx = next(
    (i for i, m in enumerate(embedding_options) if m == DEFAULT_EMBEDDING_MODEL),
    0,
)

default_gen_idx = next(
    (i for i, m in enumerate(generation_options) if m == DEFAULT_GENERATION_MODEL),
    0,
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ System Configuration")

    selected_embed_model = st.selectbox(
        "Select Embedding Model",
        options=embedding_options,
        index=default_embed_idx,
        help="Must match the model space used to search existing collections.",
    )

    selected_gen_model = st.selectbox(
        "Select Generation LLM",
        options=generation_options,
        index=default_gen_idx,
        help="The language model that processes context and generates the reply.",
    )

    st.markdown("---")
    st.header("1. Add to Vault")
    uploaded_file = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])

    if st.button("Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Chunking and embedding document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
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

# --- Main Search Window ---
st.header("2. Search your Vault")

if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

with st.form(key="search_form", clear_on_submit=True):
    query = st.text_input(
        "What would you like to know about your documents?",
        placeholder="Enter your query",
    )
    submit_button = st.form_submit_button(label="Search & Generate")

if submit_button and query:
    if st.session_state.is_processing:
        st.warning("A query is already processing. Please wait a moment.")
    else:
        st.session_state.is_processing = True

        with st.spinner("Searching the vault and generating an answer..."):
            payload = {
                "query": query,
                "top_k": 3,
                "embedding_model": selected_embed_model,
                "generation_model": selected_gen_model,
            }
            try:
                res = requests.post(f"{API_URL}/ask/", json=payload)
                if res.status_code == 200:
                    res_data = res.json()
                    st.session_state.search_history.insert(
                        0,
                        {
                            "query": query,
                            "answer": res_data["answer"],
                            "sources": res_data["sources"],
                            "gen_model": res_data["generation_model"],
                            "embed_model": res_data["embedding_model"],
                        },
                    )
                else:
                    st.error(f"Error generating answer: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Backend is unreachable. Is FastAPI running?")

        st.session_state.is_processing = False
        st.rerun()

st.markdown("---")

for result in st.session_state.search_history:
    with st.container(border=True):
        st.markdown(f"**🔍 Query:** {result['query']}")
        st.info(result["answer"])
        st.caption(
            f"✨ Generated by `{result['gen_model']}` | 🔎 Searched with `{result['embed_model']}`"
        )

        if result["sources"]:
            with st.expander("View Sources Cited"):
                for source in result["sources"]:
                    st.markdown(
                        f"- 📄 **{source['filename']}** (Similarity: {source['similarity']})"
                    )
