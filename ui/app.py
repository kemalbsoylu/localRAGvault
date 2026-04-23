import streamlit as st
import requests

# Point to FastAPI backend
API_URL = "http://127.0.0.1:8000"

# Configure the Streamlit page
st.set_page_config(page_title="localRAGvault", page_icon="🗄️", layout="wide")
st.title("🗄️ localRAGvault")
st.markdown("Your privacy-first, fully local document assistant.")

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("1. Add to Vault")
    uploaded_file = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])

    if st.button("Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Chunking and embedding document..."):
                # Send the file to FastAPI endpoint
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
                try:
                    res = requests.post(f"{API_URL}/upload/", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"✅ Success! {data['chunks_saved']} chunks saved to Postgres.")
                    else:
                        st.error(f"Failed to ingest: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Backend is unreachable. Is FastAPI running?")
        else:
            st.warning("Please select a file first.")

# --- Main Window: Chat/Search ---
st.header("2. Ask your Vault")
query = st.text_input("What would you like to know about your documents?")

if st.button("Search & Generate"):
    if query:
        with st.spinner("Searching the vault and generating an answer..."):
            try:
                # Send the question to FastAPI /ask/ endpoint
                res = requests.post(f"{API_URL}/ask/", json={"query": query, "top_k": 3})

                if res.status_code == 200:
                    data = res.json()

                    # Display the AI Answer
                    st.markdown("### Answer")
                    st.info(data["answer"])

                    # Display the Sources
                    st.markdown("### Sources Cited")
                    for source in data["sources"]:
                        st.caption(f"📄 **{source['filename']}** (Similarity Score: {source['similarity']})")
                else:
                    st.error(f"Error generating answer: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Backend is unreachable. Is FastAPI running?")
    else:
        st.warning("Please enter a question.")
