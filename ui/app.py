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
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None
if "selected_gen_model" not in st.session_state:
    st.session_state.selected_gen_model = DEFAULT_GENERATION_MODEL
if "pending_workspace" not in st.session_state:
    st.session_state.pending_workspace = None
if "pending_upload" not in st.session_state:
    st.session_state.pending_upload = None
if "upload_success_msg" not in st.session_state:
    st.session_state.upload_success_msg = None
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0


def get_error_msg(response: requests.Response) -> str:
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


def fetch_workspace_threads(workspace_id: str):
    try:
        res = requests.get(f"{API_URL}/workspaces/{workspace_id}/threads")
        if res.status_code == 200:
            return res.json().get("threads", [])
    except requests.exceptions.ConnectionError:
        pass
    return []


available_models = fetch_available_models()
workspaces = fetch_workspaces()

embedding_options = [m for m in available_models if "embed" in m] or [DEFAULT_EMBEDDING_MODEL]
generation_options = [m for m in available_models if "embed" not in m] or [DEFAULT_GENERATION_MODEL]
default_embed_idx = next(
    (i for i, m in enumerate(embedding_options) if m == DEFAULT_EMBEDDING_MODEL), 0
)
default_gen_idx = next(
    (i for i, m in enumerate(generation_options) if m == st.session_state.selected_gen_model), 0
)

# --- Sidebar (Workspaces & Document Ingestion) ---
with st.sidebar:
    st.header("Workspaces")
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

    # Workspace Creation Form
    with st.expander("➕ Create New Workspace", expanded=not workspaces):
        with st.form("create_workspace_form"):
            new_ws_name = st.text_input(
                "Workspace Name",
                placeholder="e.g., Financial Reports",
                disabled=st.session_state.is_processing,
            )
            new_ws_embed = st.selectbox(
                "Embedding Model",
                options=embedding_options,
                index=default_embed_idx,
                disabled=st.session_state.is_processing,
            )
            submit_workspace = st.form_submit_button(
                "Create & Lock Dimensions", disabled=st.session_state.is_processing
            )

            if submit_workspace:
                if not new_ws_name.strip():
                    st.error("Workspace name cannot be empty.")
                else:
                    st.session_state.is_processing = True
                    st.session_state.pending_workspace = {
                        "name": new_ws_name,
                        "embedding_model": new_ws_embed,
                    }
                    st.rerun()

    if active_workspace:
        st.markdown("---")
        st.header("Add to Vault")

        if st.session_state.upload_success_msg:
            st.success(st.session_state.upload_success_msg)
            st.session_state.upload_success_msg = None

        with st.form("upload_form"):
            uploaded_file = st.file_uploader(
                "Upload a .txt or .md file",
                type=["txt", "md"],
                disabled=st.session_state.is_processing,
                key=f"file_uploader_{st.session_state.file_uploader_key}",
            )
            submit_upload = st.form_submit_button(
                "Ingest Document", disabled=st.session_state.is_processing
            )

            if submit_upload and uploaded_file is not None:
                st.session_state.is_processing = True
                st.session_state.pending_upload = {
                    "filename": uploaded_file.name,
                    "content": uploaded_file.getvalue(),
                    "workspace_id": active_workspace["id"],
                    "embedding_model": active_workspace["embedding_model"],
                }
                st.rerun()
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
# --- BACKGROUND API EXECUTION ---
# =====================================================================

if st.session_state.pending_workspace:
    pw = st.session_state.pending_workspace
    st.session_state.pending_workspace = None
    with st.spinner(f"Probing model and initializing workspace '{pw['name']}'..."):
        try:
            res = requests.post(
                f"{API_URL}/workspaces/",
                json={"name": pw["name"], "embedding_model": pw["embedding_model"]},
            )
            if res.status_code == 200:
                st.success(f"Workspace locked to {res.json()['dimension']} dimensions!")
            else:
                st.error(f"Error: {get_error_msg(res)}")
        except requests.exceptions.ConnectionError:
            st.error("Backend unreachable. Is FastAPI running?")
    st.session_state.is_processing = False
    st.rerun()

if st.session_state.pending_upload:
    pu = st.session_state.pending_upload
    st.session_state.pending_upload = None
    with st.spinner(f"Ingesting '{pu['filename']}' and calculating vector embeddings..."):
        files = {"file": (pu["filename"], pu["content"], "text/plain")}
        data = {
            "workspace_id": pu["workspace_id"],
            "embedding_model": pu["embedding_model"],
        }
        try:
            res = requests.post(f"{API_URL}/upload/", files=files, data=data)
            if res.status_code == 200:
                res_data = res.json()
                st.session_state.upload_success_msg = (
                    f"✅ Success! {res_data['chunks_saved']} chunks saved for '{res_data['filename']}'."
                )
                st.session_state.file_uploader_key += 1
            else:
                st.error(f"Upload failed: {get_error_msg(res)}")
        except requests.exceptions.ConnectionError:
            st.error("Backend unreachable.")
    st.session_state.is_processing = False
    st.rerun()


# =====================================================================
# --- MAIN CONTENT AREA: VIEW SWITCHER (Search Mode vs. Chat Mode) ---
# =====================================================================

threads = fetch_workspace_threads(active_workspace["id"]) if active_workspace else []

# Guard against stale thread IDs when workspace switches
if st.session_state.active_thread_id:
    if not any(t["id"] == st.session_state.active_thread_id for t in threads):
        st.session_state.active_thread_id = None

if not active_workspace:
    st.info("👈 Create and select a workspace from the sidebar to begin.")

# VIEW 1: MULTI-TURN CHAT CONVERSATION VIEW
elif st.session_state.active_thread_id:
    # Resolve current active thread title from fetched threads list
    current_thread = next(
        (t for t in threads if t["id"] == st.session_state.active_thread_id), None
    )
    thread_display_title = (
        current_thread["title"] if current_thread else st.session_state.active_thread_id
    )

    col1, col2 = st.columns([8, 2])
    with col1:
        # Show actual Thread Name as title
        st.header(f"💬 {thread_display_title}")
    with col2:
        if st.button(
            "⬅️ Back to Search", use_container_width=True, disabled=st.session_state.is_processing
        ):
            st.session_state.active_thread_id = None
            st.rerun()

    st.caption(
        f"**Thread ID:** `{st.session_state.active_thread_id}` | **Workspace:** `{active_workspace['name']}`"
    )
    st.markdown("---")

    # Fetch and render message history from DB with timestamps
    try:
        res = requests.get(f"{API_URL}/threads/{st.session_state.active_thread_id}/messages")
        if res.status_code == 200:
            history_data = res.json().get("messages", [])
            for msg in history_data:
                icon = "👤" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=icon):
                    st.markdown(msg["content"])

                    # Format message creation timestamp
                    msg_time = (
                        msg["created_at"][:16].replace("T", " ")
                        if "T" in msg["created_at"]
                        else msg["created_at"][:16]
                    )

                    if msg["role"] == "assistant":
                        st.caption(
                            f"✨ Generated by `{msg['model_used']}` | 🔎 Searched with `{active_workspace['embedding_model']}` | 🕒 `{msg_time}`"
                        )
                    else:
                        st.caption(f"🕒 `{msg_time}`")

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

    st.markdown("---")
    col_model, _ = st.columns([4, 6])
    with col_model:
        st.session_state.selected_gen_model = st.selectbox(
            "⚙️ Generation LLM:",
            options=generation_options,
            index=default_gen_idx,
            help="Switch generation model on the fly during this conversation.",
            disabled=st.session_state.is_processing,
            key="chat_llm_select",
        )

    # Native Chat Input
    if follow_up_query := st.chat_input(
        "Ask a follow-up question...", disabled=st.session_state.is_processing
    ):
        st.session_state.is_processing = True
        st.session_state.current_query = follow_up_query
        st.rerun()

    if (
        st.session_state.is_processing
        and st.session_state.current_query
        and st.session_state.active_thread_id
    ):
        with st.chat_message("user", avatar="👤"):
            st.markdown(st.session_state.current_query)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking and checking vault..."):
                payload = {
                    "workspace_id": active_workspace["id"],
                    "query": st.session_state.current_query,
                    "thread_id": st.session_state.active_thread_id,
                    "top_k": 3,
                    "embedding_model": active_workspace["embedding_model"],
                    "generation_model": st.session_state.selected_gen_model,
                }
                try:
                    res = requests.post(f"{API_URL}/ask/", json=payload)
                    if res.status_code != 200:
                        st.error(f"Error: {get_error_msg(res)}")
                except requests.exceptions.ConnectionError:
                    st.error("Backend unreachable.")

        st.session_state.is_processing = False
        st.session_state.current_query = ""
        st.rerun()


# VIEW 2: STANDARD VAULT SEARCH VIEW
else:
    st.caption(
        f"**Active Workspace:** `{active_workspace['name']}` | "
        f"**Embedding Model:** `{active_workspace['embedding_model']}` | "
        f"**Vector Dimensions:** `{active_workspace['dimension']}`"
    )
    st.markdown("---")

    col_model, _ = st.columns([4, 6])
    with col_model:
        st.session_state.selected_gen_model = st.selectbox(
            "⚙️ Generation LLM:",
            options=generation_options,
            index=default_gen_idx,
            help="The language model that processes context and writes replies.",
            disabled=st.session_state.is_processing,
            key="search_llm_select",
        )

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
                "generation_model": st.session_state.selected_gen_model,
            }
            try:
                res = requests.post(f"{API_URL}/ask/", json=payload)
                if res.status_code != 200:
                    st.error(f"Error generating answer: {get_error_msg(res)}")
            except requests.exceptions.ConnectionError:
                st.error("Backend is unreachable. Is FastAPI running?")

        st.session_state.is_processing = False
        st.session_state.current_query = ""
        st.rerun()

    st.markdown("---")

    if not threads:
        st.info("No conversations yet. Ask a question above to start searching!")
    else:
        for t in threads:
            with st.container(border=True):
                st.subheader(f"💬 {t['title']}")

                st.markdown(f"**👤 Latest Query:** {t['last_query']}")
                st.info(f"**🤖 Latest Reply:** {t['last_answer']}")

                col1, col2 = st.columns([7, 3])
                with col1:
                    last_active_str = (
                        t["updated_at"][:16].replace("T", " ")
                        if "T" in t["updated_at"]
                        else t["updated_at"][:16]
                    )
                    st.caption(
                        f"✨ Generated by `{t['model_used']}` | 🕒 Last Active: `{last_active_str}` | 💬 **{t['message_count']} messages** in thread"
                    )
                with col2:
                    if st.button(
                        "💬 Open Conversation",
                        key=f"btn_{t['id']}",
                        use_container_width=True,
                        disabled=st.session_state.is_processing,
                    ):
                        st.session_state.active_thread_id = t["id"]
                        st.rerun()

                if t.get("sources"):
                    with st.expander("View Sources Cited in Latest Reply"):
                        for source in t["sources"]:
                            st.markdown(
                                f"- 📄 **{source['filename']}** (Similarity: {source['similarity']})"
                            )
