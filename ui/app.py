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

# --- Session State Initialization ---
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None


def get_error_msg(response: requests.Response) -> str:
    """Safely extracts FastAPI HTTPException detail messages."""
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text


@st.cache_data(ttl=60)
def fetch_available_models():
    try:
        res = requests.get(f"{API_URL}/models/")
        if res.status_code == 200:
            return res.json().get("models", [])
    except requests.exceptions.ConnectionError:
        pass
    return []


def fetch_workspaces():
    try:
        res = requests.get(f"{API_URL}/workspaces/")
        if res.status_code == 200:
            return res.json()
    except requests.exceptions.ConnectionError:
        pass
    return []


# --- Data Loading ---
available_models = fetch_available_models()
workspaces = fetch_workspaces()

embedding_options = [m for m in available_models if "embed" in m] or [DEFAULT_EMBEDDING_MODEL]
generation_options = [m for m in available_models if "embed" not in m] or [DEFAULT_GENERATION_MODEL]

default_embed_idx = next(
    (i for i, m in enumerate(embedding_options) if m == DEFAULT_EMBEDDING_MODEL), 0
)
default_gen_idx = next(
    (i for i, m in enumerate(generation_options) if m == DEFAULT_GENERATION_MODEL), 0
)

# --- Sidebar ---
with st.sidebar:
    st.header("Workspaces")

    # Workspace Selection
    active_workspace = None
    if workspaces:
        ws_options = {ws["id"]: f"{ws['name']} ({ws['embedding_model']})" for ws in workspaces}
        selected_ws_id = st.selectbox(
            "Active Workspace",
            options=list(ws_options.keys()),
            format_func=lambda x: ws_options[x],
            disabled=st.session_state.is_processing,
        )
        active_workspace = next((ws for ws in workspaces if ws["id"] == selected_ws_id), None)
    else:
        st.warning("No workspaces found. Create one below to begin.")

    # Workspace Creation
    with st.expander("➕ Create New Workspace", expanded=not workspaces):
        with st.form("create_workspace_form"):
            new_ws_name = st.text_input("Workspace Name", placeholder="e.g., Financial Reports")
            new_ws_embed = st.selectbox(
                "Embedding Model", options=embedding_options, index=default_embed_idx
            )

            if st.form_submit_button(
                "Create & Lock Dimensions", disabled=st.session_state.is_processing
            ):
                if not new_ws_name.strip():
                    st.error("Workspace name cannot be empty.")
                else:
                    try:
                        res = requests.post(
                            f"{API_URL}/workspaces/",
                            json={"name": new_ws_name, "embedding_model": new_ws_embed},
                        )
                        if res.status_code == 200:
                            st.success(f"Workspace locked to {res.json()['dimension']} dimensions!")
                            st.rerun()
                        else:
                            st.error(f"Error: {get_error_msg(res)}")
                    except requests.exceptions.ConnectionError:
                        st.error("Backend unreachable. Is FastAPI running?")

    st.markdown("---")
    st.header("⚙️ Generation LLM")
    selected_gen_model = st.selectbox(
        "Select LLM",
        options=generation_options,
        index=default_gen_idx,
        help="The language model that reads context and writes replies.",
        disabled=st.session_state.is_processing,
    )

    # Contextual controls (Only visible if a workspace is selected)
    if active_workspace:
        st.markdown("---")
        st.header("Add to Vault")
        with st.form("upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])
            submit_upload = st.form_submit_button(
                "Ingest Document", disabled=st.session_state.is_processing
            )

            if submit_upload and uploaded_file is not None:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
                data = {
                    "workspace_id": active_workspace["id"],
                    "embedding_model": active_workspace["embedding_model"],
                }
                try:
                    res = requests.post(f"{API_URL}/upload/", files=files, data=data)
                    if res.status_code == 200:
                        res_data = res.json()
                        st.success(
                            f"✅ {res_data['chunks_saved']} chunks saved to {active_workspace['name']}."
                        )
                    else:
                        st.error(f"Upload failed: {get_error_msg(res)}")
                except requests.exceptions.ConnectionError:
                    st.error("Backend unreachable.")
            elif submit_upload:
                st.warning("Please select a file first.")

        st.markdown("---")
        st.header(f"📂 Inventory: {active_workspace['name']}")
        if st.button("Refresh Inventory", disabled=st.session_state.is_processing):
            st.rerun()

        try:
            inv_res = requests.get(f"{API_URL}/inventory/{active_workspace['id']}")
            if inv_res.status_code == 200:
                inventory = inv_res.json().get("documents", [])
                if inventory:
                    for doc in inventory:
                        with st.expander(f"📄 {doc['filename']}"):
                            st.caption(f"**Path:** `{doc['file_path']}`")
                            st.caption(f"**Total Chunks:** {doc['total_chunks']}")
                else:
                    st.info("Vault is empty.")
            else:
                st.error(f"Failed to load inventory: {get_error_msg(inv_res)}")
        except requests.exceptions.ConnectionError:
            st.error("Backend unreachable.")


# =====================================================================
# --- MAIN CONTENT AREA: VIEW SWITCHER (Search Mode vs. Chat Mode) ---
# =====================================================================

if not active_workspace:
    st.info("👈 Create and select a workspace from the sidebar to begin.")

# VIEW 1: MULTI-TURN CHAT CONVERSATION VIEW
elif st.session_state.active_thread_id:
    col1, col2 = st.columns([8, 2])
    with col1:
        st.header("💬 Active Conversation")
    with col2:
        # Button to exit chat mode and return to normal vault search
        if st.button("⬅️ Back to Search", use_container_width=True):
            st.session_state.active_thread_id = None
            st.rerun()

    st.caption(
        f"**Thread ID:** `{st.session_state.active_thread_id}` | **Workspace:** `{active_workspace['name']}`"
    )
    st.markdown("---")

    # 1. Fetch and render message history from the backend DB
    try:
        res = requests.get(f"{API_URL}/threads/{st.session_state.active_thread_id}/messages")
        if res.status_code == 200:
            history_data = res.json().get("messages", [])
            for msg in history_data:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander("📚 Sources Cited"):
                            for s in msg["sources"]:
                                st.markdown(
                                    f"- 📄 **{s['filename']}** (Similarity: {s['similarity']})"
                                )
        else:
            st.error(f"Failed to load chat history: {get_error_msg(res)}")
    except requests.exceptions.ConnectionError:
        st.error("Backend unreachable.")

    # 2. Native Chat Input for follow-up questions
    if follow_up_query := st.chat_input("Ask a follow-up question..."):
        with st.chat_message("user"):
            st.markdown(follow_up_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking and checking vault..."):
                payload = {
                    "workspace_id": active_workspace["id"],
                    "query": follow_up_query,
                    "thread_id": st.session_state.active_thread_id,  # Passes thread_id to continue context!
                    "top_k": 3,
                    "embedding_model": active_workspace["embedding_model"],
                    "generation_model": selected_gen_model,
                }
                try:
                    res = requests.post(f"{API_URL}/ask/", json=payload)
                    if res.status_code == 200:
                        res_data = res.json()
                        st.markdown(res_data["answer"])
                        if res_data["sources"]:
                            with st.expander("📚 Sources Cited"):
                                for s in res_data["sources"]:
                                    st.markdown(
                                        f"- 📄 **{s['filename']}** (Similarity: {s['similarity']})"
                                    )
                    else:
                        st.error(f"Error: {get_error_msg(res)}")
                except requests.exceptions.ConnectionError:
                    st.error("Backend unreachable.")
        # Rerun to cleanly re-render from database history
        st.rerun()


# VIEW 2: STANDARD VAULT SEARCH VIEW
else:
    st.header("Search your Vault")
    with st.form(key="search_form"):
        query = st.text_input(
            "What would you like to know?",
            placeholder="Enter your query",
            disabled=st.session_state.is_processing,
        )
        submit_button = st.form_submit_button(
            label="Search & Generate", disabled=st.session_state.is_processing
        )

    if submit_button and query:
        st.session_state.is_processing = True
        st.session_state.current_query = query
        st.rerun()

    if st.session_state.is_processing and st.session_state.current_query:
        with st.spinner("Searching the vault and generating an answer..."):
            payload = {
                "workspace_id": active_workspace["id"],
                "query": st.session_state.current_query,
                "top_k": 3,
                "embedding_model": active_workspace["embedding_model"],
                "generation_model": selected_gen_model,
            }
            try:
                res = requests.post(f"{API_URL}/ask/", json=payload)
                if res.status_code == 200:
                    res_data = res.json()
                    st.session_state.search_history.insert(
                        0,
                        {
                            "thread_id": res_data["thread_id"],
                            "query": st.session_state.current_query,
                            "answer": res_data["answer"],
                            "sources": res_data["sources"],
                            "gen_model": res_data["generation_model"],
                            "embed_model": res_data["embedding_model"],
                        },
                    )
                else:
                    st.error(f"Error generating answer: {get_error_msg(res)}")
            except requests.exceptions.ConnectionError:
                st.error("Backend is unreachable. Is FastAPI running?")

        # Reset processing flag and clear current query cache
        st.session_state.is_processing = False
        st.session_state.current_query = ""
        st.rerun()

    st.markdown("---")

    # Render Search Cards with "Start Conversation" Button
    for result in st.session_state.search_history:
        with st.container(border=True):
            st.markdown(f"**🔍 Query:** {result['query']}")
            st.info(result["answer"])

            col1, col2 = st.columns([7, 3])
            with col1:
                st.caption(
                    f"✨ Generated by `{result['gen_model']}` | 🔎 Searched with `{result['embed_model']}`"
                )
            with col2:
                # THE MAGIC BUTTON: Switches view and binds active thread
                if st.button(
                    "💬 Continue in Chat",
                    key=f"btn_{result['thread_id']}",
                    use_container_width=True,
                ):
                    st.session_state.active_thread_id = result["thread_id"]
                    st.rerun()

            if result["sources"]:
                with st.expander("View Sources Cited"):
                    for source in result["sources"]:
                        st.markdown(
                            f"- 📄 **{source['filename']}** (Similarity: {source['similarity']})"
                        )
