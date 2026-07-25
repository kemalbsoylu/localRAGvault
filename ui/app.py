import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

from core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL, MAX_FILE_SIZE_MB

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
if "pending_batch_upload" not in st.session_state:
    st.session_state.pending_batch_upload = None
if "batch_report" not in st.session_state:
    st.session_state.batch_report = None
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
if "flash_msg" not in st.session_state:
    st.session_state.flash_msg = None
if "ws_name_input_key" not in st.session_state:
    st.session_state.ws_name_input_key = 0
if "search_input_key" not in st.session_state:
    st.session_state.search_input_key = 0


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


# --- Render Sleek Dismissible Flash Feedback Banner ---
if st.session_state.flash_msg:
    msg_type, msg_text = st.session_state.flash_msg

    col_msg, col_close = st.columns([98, 2])
    with col_msg:
        if msg_type == "success":
            st.success(msg_text, icon="✅")
        elif msg_type == "error":
            st.error(msg_text, icon="🚨")
        else:
            st.info(msg_text, icon="ℹ️")
    with col_close:
        if st.button("✕", key="btn_dismiss_flash", type="tertiary", help="Dismiss notification"):
            st.session_state.flash_msg = None
            st.rerun()


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

# --- Sidebar (Workspace Administration & Settings) ---
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
            help="Switch between different indexed document vaults.",
        )
        active_workspace = next((ws for ws in workspaces if ws["id"] == selected_ws_id), None)
    else:
        st.warning("No workspaces found. Create one below to begin.")

    with st.expander("➕ Create New Workspace", expanded=not workspaces):
        with st.form("create_workspace_form"):
            new_ws_name = st.text_input(
                "Workspace Name",
                placeholder="e.g., Financial Reports",
                disabled=st.session_state.is_processing,
                key=f"new_ws_name_input_{st.session_state.ws_name_input_key}",
                help="A unique identifier for your document vault.",
            )
            new_ws_embed = st.selectbox(
                "Embedding Model",
                options=embedding_options,
                index=default_embed_idx,
                disabled=st.session_state.is_processing,
                help="The mathematical vector space engine locked to this workspace. Once created, all documents must be embedded using this exact model to prevent dimension collisions.",
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

        if st.session_state.batch_report:
            rep = st.session_state.batch_report
            with st.container(border=True):
                col_title, col_close = st.columns([9, 1])
                with col_title:
                    st.subheader("📊 Ingestion Report")
                with col_close:
                    if st.button("✕", key="btn_dismiss_rep", type="tertiary", help="Close report"):
                        st.session_state.batch_report = None
                        st.rerun()

                sum_data = rep["summary"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", sum_data["total_files"])
                c2.metric("New", sum_data["successful"])
                c3.metric("Updated", sum_data["upserts"])
                c4.metric("Chunks", sum_data["total_chunks_saved"])

                if sum_data["failed"] > 0:
                    st.error(f"⚠️ {sum_data['failed']} file(s) failed to ingest.")

                with st.expander("📄 View File Breakdown", expanded=(sum_data["failed"] > 0)):
                    for item in rep["results"]:
                        if item["status"] == "failed":
                            st.markdown(
                                f"❌ **{item['filename']}** — *Failed:* `{item['error_message']}`"
                            )
                        elif item["status"] == "upserted":
                            st.markdown(
                                f"🔄 **{item['filename']}** — *Updated* ({item['chunks_saved']} chunks saved)"
                            )
                        else:
                            st.markdown(
                                f"✅ **{item['filename']}** — *New* ({item['chunks_saved']} chunks saved)"
                            )

        # Multi-File Upload Form with Frontend Size Validation
        with st.form("upload_form"):
            uploaded_files = st.file_uploader(
                f"Upload documents (Max {MAX_FILE_SIZE_MB}MB per file)",
                type=["txt", "md", "pdf", "docx", "csv", "json"],
                accept_multiple_files=True,
                disabled=st.session_state.is_processing,
                key=f"file_uploader_{st.session_state.file_uploader_key}",
                help="Select multiple files or folders. Identical file names replace existing versions cleanly without duplicating vector embeddings.",
            )
            submit_upload = st.form_submit_button(
                "Ingest Documents", disabled=st.session_state.is_processing
            )

            if submit_upload and uploaded_files:
                max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
                oversized_files = [f.name for f in uploaded_files if f.size > max_bytes]

                if oversized_files:
                    st.error(
                        f"Upload blocked: The following files exceed the {MAX_FILE_SIZE_MB}MB limit: {', '.join(oversized_files)}"
                    )
                else:
                    st.session_state.is_processing = True
                    st.session_state.pending_batch_upload = {
                        "files": [(f.name, f.getvalue()) for f in uploaded_files],
                        "workspace_id": active_workspace["id"],
                        "embedding_model": active_workspace["embedding_model"],
                    }
                    st.rerun()
            elif submit_upload:
                st.warning("Please select at least one file first.")

        st.markdown("---")
        st.header(f"📂 Inventory: {active_workspace['name']}")
        if st.button(
            "Refresh Inventory", disabled=st.session_state.is_processing, use_container_width=True
        ):
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

                            confirm_del_file = st.checkbox(
                                "Confirm deletion",
                                key=f"chk_{active_workspace['id']}_{doc['filename']}",
                            )
                            if st.button(
                                "🗑️ Delete File",
                                key=f"btn_{active_workspace['id']}_{doc['filename']}",
                                disabled=not confirm_del_file or st.session_state.is_processing,
                                use_container_width=True,
                            ):
                                with st.spinner(f"Deleting '{doc['filename']}'..."):
                                    del_res = requests.delete(
                                        f"{API_URL}/documents/{active_workspace['id']}/{doc['filename']}"
                                    )
                                    if del_res.status_code == 200:
                                        st.session_state.flash_msg = (
                                            "success",
                                            f"Deleted '{doc['filename']}'.",
                                        )
                                        st.rerun()
                                    else:
                                        st.error(f"Deletion failed: {get_error_msg(del_res)}")
                else:
                    st.info("Vault is empty.")
            else:
                st.error(f"Failed to load inventory: {get_error_msg(inv_res)}")
        except requests.exceptions.ConnectionError:
            st.error("Backend unreachable.")

        # --- PER-WORKSPACE RAG SETTINGS ---
        st.markdown("---")
        with st.expander("⚙️ Workspace Settings", expanded=False):
            st.caption("Note: Chunking physics apply strictly to subsequent file ingestions.")
            with st.form(key=f"ws_settings_form_{active_workspace['id']}"):
                cfg_chunk_size = st.slider(
                    "Chunk Size (chars)",
                    min_value=100,
                    max_value=2000,
                    value=active_workspace["chunk_size"],
                    step=50,
                    help="Determines the character length of each individual text block when splitting ingested documents. Larger chunks capture broader context; smaller chunks isolate specific facts.",
                )

                max_allowed_overlap = min(500, int(cfg_chunk_size * 0.5))
                current_overlap = min(active_workspace["chunk_overlap"], max_allowed_overlap)

                cfg_chunk_overlap = st.slider(
                    "Chunk Overlap (chars)",
                    min_value=0,
                    max_value=max_allowed_overlap,
                    value=current_overlap,
                    step=10,
                    help="The number of overlapping characters between adjacent chunks. This acts as a semantic bridge so context is not split awkwardly at chunk boundaries.",
                )

                cfg_top_k = st.slider(
                    "Top-K Retrieval Depth",
                    min_value=1,
                    max_value=20,
                    value=active_workspace["top_k"],
                    step=1,
                    help="The maximum number of relevant document chunks retrieved from PostgreSQL vector search and injected into the LLM's prompt context.",
                )

                cfg_similarity = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=0.8,
                    value=float(active_workspace["similarity_threshold"]),
                    step=0.05,
                    help="The minimum cosine similarity score (0.0 to 1.0) required for a chunk to be considered relevant. Higher values enforce stricter filtering against off-topic chunks.",
                )

                cfg_history_limit = st.slider(
                    "Memory Depth (turns)",
                    min_value=1,
                    max_value=20,
                    value=active_workspace["chat_history_limit"],
                    step=1,
                    help="The number of recent conversational turns (user queries and assistant replies) included in the memory payload to maintain multi-turn context.",
                )

                cfg_system_prompt = st.text_area(
                    "System Instructions / Persona",
                    value=active_workspace["system_prompt"] or "",
                    placeholder="e.g., Respond using concise technical bullet points.",
                    height=100,
                    help="Custom behavioral persona or strict operational instructions injected at the top of the LLM prompt hierarchy.",
                )

                btn_save_settings = st.form_submit_button(
                    "💾 Save Settings", use_container_width=True
                )

                if btn_save_settings:
                    new_prompt_val = cfg_system_prompt.strip() if cfg_system_prompt.strip() else None
                    if (
                        cfg_chunk_size == active_workspace["chunk_size"]
                        and cfg_chunk_overlap == active_workspace["chunk_overlap"]
                        and cfg_top_k == active_workspace["top_k"]
                        and round(cfg_similarity, 4) == round(float(active_workspace["similarity_threshold"]), 4)
                        and cfg_history_limit == active_workspace["chat_history_limit"]
                        and new_prompt_val == active_workspace["system_prompt"]
                    ):
                        st.info("ℹ️ No changes detected. Settings were not modified.")
                    else:
                        patch_payload = {
                            "chunk_size": cfg_chunk_size,
                            "chunk_overlap": cfg_chunk_overlap,
                            "top_k": cfg_top_k,
                            "similarity_threshold": cfg_similarity,
                            "chat_history_limit": cfg_history_limit,
                            "system_prompt": new_prompt_val,
                        }
                        try:
                            patch_res = requests.patch(
                                f"{API_URL}/workspaces/{active_workspace['id']}", json=patch_payload
                            )
                            if patch_res.status_code == 200:
                                st.session_state.flash_msg = (
                                    "success",
                                    "Workspace settings updated successfully!",
                                )
                                st.rerun()
                            else:
                                st.error(f"Failed to update settings: {get_error_msg(patch_res)}")
                        except requests.exceptions.ConnectionError:
                            st.error("Backend is unreachable.")

        # Workspace Deletion UI
        with st.expander("⚠️ Danger Zone: Delete Workspace"):
            st.warning("Permanently deletes workspace, history, and physical files.")
            confirm_del_ws = st.checkbox(
                "Confirm destruction",
                key=f"chk_del_ws_{active_workspace['id']}",
            )
            if st.button(
                "🚨 Delete Workspace Permanently",
                disabled=not confirm_del_ws or st.session_state.is_processing,
                use_container_width=True,
            ):
                with st.spinner(f"Deleting workspace '{active_workspace['name']}'..."):
                    try:
                        del_ws_res = requests.delete(
                            f"{API_URL}/workspaces/{active_workspace['id']}"
                        )
                        if del_ws_res.status_code == 200:
                            st.session_state.active_thread_id = None
                            st.session_state.current_query = ""
                            st.session_state.batch_report = None
                            st.session_state.pop(f"chk_del_ws_{active_workspace['id']}", None)
                            st.session_state.flash_msg = (
                                "success",
                                f"Workspace '{active_workspace['name']}' deleted.",
                            )
                            st.rerun()
                        else:
                            st.error(f"Failed to delete workspace: {get_error_msg(del_ws_res)}")
                    except requests.exceptions.ConnectionError:
                        st.error("Backend unreachable.")


# --- BACKGROUND API EXECUTION ---

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
                st.session_state.ws_name_input_key += 1
                st.session_state.flash_msg = (
                    "success",
                    f"Workspace is created with {res.json()['dimension']} dimensions!",
                )
            else:
                st.session_state.flash_msg = ("error", f"Error: {get_error_msg(res)}")
        except requests.exceptions.ConnectionError:
            st.session_state.flash_msg = ("error", "Backend unreachable. Is FastAPI running?")
    st.session_state.is_processing = False
    st.rerun()

if st.session_state.pending_batch_upload:
    pu = st.session_state.pending_batch_upload
    st.session_state.pending_batch_upload = None
    with st.spinner(f"Batch ingesting {len(pu['files'])} file(s) and calculating embeddings..."):
        files_payload = [
            ("files", (name, content, "application/octet-stream")) for name, content in pu["files"]
        ]
        data_payload = {
            "workspace_id": pu["workspace_id"],
            "embedding_model": pu["embedding_model"],
        }
        try:
            res = requests.post(f"{API_URL}/upload/batch/", files=files_payload, data=data_payload)
            if res.status_code == 200:
                report_data = res.json()
                st.session_state.batch_report = report_data
                st.session_state.file_uploader_key += 1

                sum_data = report_data["summary"]
                if sum_data["failed"] == 0:
                    st.session_state.flash_msg = (
                        "success",
                        f"Batch ingestion complete! Processed {sum_data['successful']} new and {sum_data['upserts']} updated files ({sum_data['total_chunks_saved']} chunks saved).",
                    )
                else:
                    st.session_state.flash_msg = (
                        "error",
                        f"Batch ingestion finished with errors: {sum_data['failed']} file(s) failed to ingest. Check report from the sidebar.",
                    )
            else:
                st.session_state.flash_msg = ("error", f"Batch upload failed: {get_error_msg(res)}")
        except requests.exceptions.ConnectionError:
            st.session_state.flash_msg = ("error", "Backend unreachable.")
    st.session_state.is_processing = False
    st.rerun()


# --- MAIN CONTENT AREA ---

threads = fetch_workspace_threads(active_workspace["id"]) if active_workspace else []

if st.session_state.active_thread_id:
    if not any(t["id"] == st.session_state.active_thread_id for t in threads):
        st.session_state.active_thread_id = None

if not active_workspace:
    st.info("👈 Create and select a workspace from the sidebar to begin.")

# VIEW 1: MULTI-TURN CHAT CONVERSATION VIEW
elif st.session_state.active_thread_id:
    current_thread = next(
        (t for t in threads if t["id"] == st.session_state.active_thread_id), None
    )
    thread_display_title = (
        current_thread["title"] if current_thread else st.session_state.active_thread_id
    )

    col1, col2 = st.columns([8, 2])
    with col1:
        st.header(f"💬 {thread_display_title}")
    with col2:
        if st.button(
            "⬅️ Back to Workspace", use_container_width=True, disabled=st.session_state.is_processing
        ):
            st.session_state.active_thread_id = None
            st.rerun()

    st.caption(
        f"**Thread ID:** `{st.session_state.active_thread_id}` | **Workspace:** `{active_workspace['name']}`"
    )
    st.markdown("---")

    try:
        res = requests.get(f"{API_URL}/threads/{st.session_state.active_thread_id}/messages")
        if res.status_code == 200:
            history_data = res.json().get("messages", [])
            for msg in history_data:
                icon = "👤" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=icon):
                    st.markdown(msg["content"])

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
                                    f"- 📄 **{s['filename']}** (Chunk #{s.get('chunk_index', 1)}, Similarity: {s['similarity']})"
                                )
        else:
            st.error(f"Failed to load chat history: {get_error_msg(res)}")
    except requests.exceptions.ConnectionError:
        st.error("Backend unreachable.")

    st.markdown("---")
    col_model, col_temp, _ = st.columns([2, 2, 6])
    with col_model:
        st.session_state.selected_gen_model = st.selectbox(
            "⚙️ Generation LLM:",
            options=generation_options,
            index=default_gen_idx,
            help="Switch generation model on the fly during this conversation without losing chat context.",
            disabled=st.session_state.is_processing,
            key="chat_llm_select",
        )

    with col_temp:
        override_temp_chat = st.toggle(
            "Override Temperature",
            value=False,
            key="toggle_temp_chat",
            help="Leave off to use the model's native Modelfile calibration. Turn on to force custom sampling randomness.",
        )
        if override_temp_chat:
            chat_temp_val = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="slider_temp_chat",
                help="Higher values increase randomness and creativity; lower values make responses more deterministic and analytical.",
            )
        else:
            chat_temp_val = None

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
                    "embedding_model": active_workspace["embedding_model"],
                    "generation_model": st.session_state.selected_gen_model,
                    "temperature": chat_temp_val,
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
    st.header(f"🗃️ {active_workspace['name']}")

    st.caption(
        f"**Workspace ID:** `{active_workspace['id']}` | "
        f"**Embedding Model:** `{active_workspace['embedding_model']}` | "
        f"**Vector Dimensions:** `{active_workspace['dimension']}`"
    )

    st.markdown("---")

    col_model, col_temp, _ = st.columns([2, 2, 6])
    with col_model:
        st.session_state.selected_gen_model = st.selectbox(
            "⚙️ Generation LLM:",
            options=generation_options,
            index=default_gen_idx,
            help="The language model that synthesizes retrieved vector chunks and writes the response.",
            disabled=st.session_state.is_processing,
            key="search_llm_select",
        )

    with col_temp:
        override_temp_search = st.toggle(
            "Override Temperature",
            value=False,
            key="toggle_temp_search",
            help="Leave off to use the model's native Modelfile calibration. Turn on to force custom sampling randomness.",
        )
        if override_temp_search:
            search_temp_val = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="slider_temp_search",
                help="Higher values increase randomness and creativity; lower values make responses more deterministic and analytical.",
            )
        else:
            search_temp_val = None

    with st.form(key="search_form"):
        query = st.text_input(
            "What would you like to know?",
            placeholder="Enter your query",
            disabled=st.session_state.is_processing,
            key=f"search_query_input_{st.session_state.search_input_key}",
        )
        submit_button = st.form_submit_button(
            label="Search & Generate", disabled=st.session_state.is_processing
        )

    if submit_button and query:
        st.session_state.is_processing = True
        st.session_state.current_query = query
        st.session_state.search_input_key += 1
        st.rerun()

    if st.session_state.is_processing and st.session_state.current_query:
        with st.spinner("Searching the vault and generating an answer..."):
            payload = {
                "workspace_id": active_workspace["id"],
                "query": st.session_state.current_query,
                "embedding_model": active_workspace["embedding_model"],
                "generation_model": st.session_state.selected_gen_model,
                "temperature": search_temp_val,
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
                        key=f"btn_open_{t['id']}",
                        use_container_width=True,
                        disabled=st.session_state.is_processing,
                    ):
                        st.session_state.active_thread_id = t["id"]
                        st.rerun()

                if t.get("sources"):
                    with st.expander("📚 View Sources Cited in Latest Reply"):
                        for source in t["sources"]:
                            st.markdown(
                                f"- 📄 **{source['filename']}** (Chunk #{source.get('chunk_index', 1)}, Similarity: {source['similarity']})"
                            )

                with st.expander("🗑️ Delete Thread"):
                    st.warning("Deletes this conversation thread permanently.")
                    confirm_del = st.checkbox(
                        "Confirm thread deletion",
                        key=f"chk_del_{t['id']}",
                    )
                    if st.button(
                        "🚨 Permanently Delete Thread",
                        key=f"btn_del_{t['id']}",
                        disabled=not confirm_del or st.session_state.is_processing,
                        use_container_width=True,
                    ):
                        with st.spinner("Deleting thread..."):
                            try:
                                del_res = requests.delete(f"{API_URL}/threads/{t['id']}")
                                if del_res.status_code == 200:
                                    st.session_state.pop(f"chk_del_{t['id']}", None)
                                    st.session_state.flash_msg = ("success", "Thread deleted.")
                                    st.rerun()
                                else:
                                    st.error(f"Deletion failed: {get_error_msg(del_res)}")
                            except requests.exceptions.ConnectionError:
                                st.error("Backend unreachable.")
