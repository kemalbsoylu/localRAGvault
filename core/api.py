import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

import psycopg
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from core.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PAGE_LIMIT,
    DEFAULT_PAGE_OFFSET,
    MAX_FILE_SIZE_MB,
)
from core.database import (
    add_message,
    create_thread,
    create_workspace,
    delete_document,
    delete_thread,
    delete_workspace,
    fetch_workspace_inventory,
    get_all_workspaces,
    get_thread,
    get_thread_messages,
    get_workspace,
    get_workspace_threads,
    init_db,
    insert_document_chunks,
    search_vector_db,
    update_workspace_settings,
)
from core.extractors import extract_text_from_file
from core.logging_config import logger
from core.schemas import (
    BatchIngestionResponse,
    BatchIngestionSummary,
    DocumentInventoryItem,
    DocumentSource,
    FileIngestionResult,
    IngestionResponse,
    MessageCard,
    ModelListResponse,
    RAGQueryResponse,
    SearchQuery,
    SearchResultCard,
    ThreadCard,
    ThreadHistoryResponse,
    ThreadListResponse,
    VectorSearchResponse,
    WorkspaceCreate,
    WorkspaceInventoryResponse,
    WorkspaceResponse,
    WorkspaceUpdate,
)
from core.utils import (
    chunk_text,
    delete_physical_file,
    delete_workspace_files,
    ensure_default_models_exist,
    generate_answer,
    get_available_models,
    get_embedding,
    get_embeddings_batch,
    normalize_model_name,
    save_physical_file,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initializing localRAGvault core infrastructure components...")
    init_db()
    ensure_default_models_exist()
    yield
    logger.info("Shutting down localRAGvault runtime application context...")


app = FastAPI(
    title="localRAGvault API",
    description="Local RAG pipeline backend powered by FastAPI, Ollama, and pgvector.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_model=dict)
def health_check() -> dict:
    return {"status": "success", "message": "localRAGvault API is running securely."}


@app.get("/models/", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    """Returns a list of all models currently downloaded in Ollama."""
    try:
        models = get_available_models()
        return ModelListResponse(status="success" if models else "error", models=models)
    except Exception as e:
        logger.error(f"Error fetching models from Ollama daemon: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to fetch available models from Ollama."
        ) from e


# --- WORKSPACE ENDPOINTS ---


@app.get("/workspaces/", response_model=List[WorkspaceResponse])
def list_workspaces() -> List[WorkspaceResponse]:
    """Retrieves all active workspaces."""
    try:
        return [WorkspaceResponse(**ws) for ws in get_all_workspaces()]
    except Exception as e:
        logger.error(f"Error listing workspaces: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch workspaces.") from e


@app.post("/workspaces/", response_model=WorkspaceResponse)
def create_new_workspace(ws: WorkspaceCreate) -> WorkspaceResponse:
    """Creates a new workspace with specified parameters and probes the embedding model."""
    logger.info(f"Probing embedding model '{ws.embedding_model}' for new workspace '{ws.name}'...")

    try:
        probe_vector = get_embedding("test dimension probe", model_name=ws.embedding_model)
    except Exception as e:
        logger.error(f"Ollama failure during vector probe for '{ws.embedding_model}': {e}")
        raise HTTPException(
            status_code=500, detail="Internal error communicating with Ollama for probing."
        ) from e

    if not probe_vector:
        error_msg = f"Failed to generate probe vector. Ensure '{ws.embedding_model}' is valid and pulled locally."
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    dimension = len(probe_vector)
    workspace_id = str(uuid.uuid4())

    try:
        create_workspace(
            workspace_id=workspace_id,
            name=ws.name,
            embedding_model=ws.embedding_model,
            dimension=dimension,
            chunk_size=ws.chunk_size,
            chunk_overlap=ws.chunk_overlap,
            top_k=ws.top_k,
            similarity_threshold=ws.similarity_threshold,
            chat_history_limit=ws.chat_history_limit,
            system_prompt=ws.system_prompt,
        )
        logger.info(f"Workspace '{ws.name}' ({workspace_id}) created with dimension {dimension}.")
    except psycopg.errors.UniqueViolation as unique_err:
        logger.warning(f"Workspace creation rejected: Name '{ws.name}' already exists.")
        raise HTTPException(
            status_code=400,
            detail=f"A workspace named '{ws.name}' already exists. Please choose a unique name.",
        ) from unique_err
    except Exception as e:
        logger.error(f"Failed to persist workspace in database: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to persist workspace in the database."
        ) from e

    created_ws = get_workspace(workspace_id)
    if not created_ws:
        raise HTTPException(status_code=500, detail="Workspace created but failed to load.")
    return WorkspaceResponse(**created_ws)


@app.patch("/workspaces/{workspace_id}", response_model=WorkspaceResponse)
def patch_workspace_settings(workspace_id: str, updates: WorkspaceUpdate) -> WorkspaceResponse:
    """Updates workspace-level RAG hyperparameters and system instructions."""
    try:
        existing_ws = get_workspace(workspace_id)
        if not existing_ws:
            raise HTTPException(status_code=404, detail="Workspace not found.")

        target_chunk_size = updates.chunk_size or existing_ws["chunk_size"]
        target_chunk_overlap = (
            updates.chunk_overlap
            if updates.chunk_overlap is not None
            else existing_ws["chunk_overlap"]
        )

        if target_chunk_overlap >= target_chunk_size:
            raise HTTPException(
                status_code=400,
                detail=f"chunk_overlap ({target_chunk_overlap}) must be strictly less than chunk_size ({target_chunk_size}).",
            )

        updated_ws = update_workspace_settings(workspace_id, updates.model_dump(exclude_unset=True))
        if not updated_ws:
            raise HTTPException(status_code=500, detail="Failed to update workspace settings.")

        logger.info(f"Workspace settings updated for '{workspace_id}'.")
        return WorkspaceResponse(**updated_ws)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating workspace settings for {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update workspace settings.") from e


@app.delete("/workspaces/{workspace_id}", response_model=dict)
def remove_workspace(workspace_id: str) -> dict:
    """Deletes a workspace, all associated database records (documents, threads, messages), and disk files."""
    try:
        deleted = delete_workspace(workspace_id)
        if not deleted:
            logger.warning(f"Workspace deletion failed: Workspace '{workspace_id}' not found.")
            raise HTTPException(status_code=404, detail="Workspace not found.")

        delete_workspace_files(workspace_id)
        logger.info(f"Workspace '{workspace_id}' completely purged from database and disk.")
        return {
            "status": "success",
            "message": f"Workspace '{workspace_id}' and all associated files deleted.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workspace {workspace_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to delete workspace from backend."
        ) from e


# --- CONVERSATION ENDPOINTS ---


@app.get("/workspaces/{workspace_id}/threads", response_model=ThreadListResponse)
def list_workspace_threads(
    workspace_id: str,
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=100),
    offset: int = Query(default=DEFAULT_PAGE_OFFSET, ge=0),
) -> ThreadListResponse:
    """Returns a paginated list of conversation threads inside a workspace."""
    try:
        ws = get_workspace(workspace_id)
        if not ws:
            logger.warning(f"Thread listing failed: Workspace '{workspace_id}' not found.")
            raise HTTPException(status_code=404, detail="Workspace not found.")

        raw_threads, total_count = get_workspace_threads(workspace_id, limit=limit, offset=offset)
        threads = [
            ThreadCard(
                id=t["id"],
                title=t["title"],
                created_at=t["created_at"],
                updated_at=t["updated_at"],
                message_count=t["message_count"],
                last_query=t["last_query"],
                last_answer=t["last_answer"],
                model_used=t["model_used"],
                sources=[DocumentSource(**s) for s in t["sources"]] if t["sources"] else [],
            )
            for t in raw_threads
        ]
        has_more = (offset + len(threads)) < total_count
        return ThreadListResponse(
            workspace_id=workspace_id,
            threads=threads,
            total_count=total_count,
            limit=limit,
            offset=offset,
            has_more=has_more,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching threads for workspace {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation threads.") from e


@app.get("/threads/{thread_id}/messages", response_model=ThreadHistoryResponse)
def get_thread_history(
    thread_id: str,
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=100),
    offset: int = Query(default=DEFAULT_PAGE_OFFSET, ge=0),
) -> ThreadHistoryResponse:
    """Returns the chronological message history for a specific thread, with pagination support."""
    try:
        t = get_thread(thread_id)
        if not t:
            logger.warning(f"Message history fetch failed: Thread '{thread_id}' not found.")
            raise HTTPException(status_code=404, detail="Thread not found.")

        raw_messages, total_count = get_thread_messages(thread_id, limit=limit, offset=offset)
        messages = [
            MessageCard(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                sources=[DocumentSource(**s) for s in m["sources"]] if m["sources"] else [],
                model_used=m["model_used"],
                created_at=m["created_at"],
            )
            for m in raw_messages
        ]
        has_more = (offset + len(messages)) < total_count
        return ThreadHistoryResponse(
            thread_id=thread_id,
            messages=messages,
            total_count=total_count,
            limit=limit,
            offset=offset,
            has_more=has_more,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching messages for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch message history.") from e


@app.delete("/threads/{thread_id}", response_model=dict)
def remove_thread(thread_id: str) -> dict:
    """Deletes a specific conversation thread and all its chronological messages."""
    try:
        deleted = delete_thread(thread_id)
        if not deleted:
            logger.warning(f"Thread deletion failed: Thread '{thread_id}' not found.")
            raise HTTPException(status_code=404, detail="Thread not found.")

        logger.info(f"Thread '{thread_id}' and all associated messages purged from database.")
        return {
            "status": "success",
            "message": f"Thread '{thread_id}' permanently deleted.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation thread.") from e


# --- VAULT & RAG ENDPOINTS ---


@app.get("/inventory/{workspace_id}", response_model=WorkspaceInventoryResponse)
def get_workspace_inventory(
    workspace_id: str,
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=100),
    offset: int = Query(default=DEFAULT_PAGE_OFFSET, ge=0),
) -> WorkspaceInventoryResponse:
    """Returns a paginated list of physical files currently indexed for a given workspace."""
    try:
        ws = get_workspace(workspace_id)
        if not ws:
            logger.warning(f"Inventory fetch failed: Workspace '{workspace_id}' not found.")
            raise HTTPException(status_code=404, detail="Target workspace does not exist.")

        raw_inventory, total_count = fetch_workspace_inventory(
            workspace_id, limit=limit, offset=offset
        )
        documents = [
            DocumentInventoryItem(
                filename=item["filename"],
                file_path=item["file_path"],
                total_chunks=item["total_chunks"],
            )
            for item in raw_inventory
        ]
        has_more = (offset + len(documents)) < total_count
        return WorkspaceInventoryResponse(
            workspace_id=workspace_id,
            documents=documents,
            total_count=total_count,
            limit=limit,
            offset=offset,
            has_more=has_more,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching inventory for {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch workspace inventory.") from e


def _process_single_file_ingestion(
    workspace_id: str,
    filename: str,
    content_bytes: bytes,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int, int]:
    """Internal helper to execute clean upserts, extraction, chunking, and batch DB insertion."""
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content_bytes) > max_bytes:
        raise ValueError(f"File '{filename}' exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB.")

    old_chunks = delete_document(workspace_id, filename)
    if old_chunks > 0:
        logger.info(
            f"Clean upsert triggered: Removed {old_chunks} existing vector chunks for '{filename}'."
        )

    content_text = extract_text_from_file(content_bytes, filename)
    chunks = chunk_text(content_text, chunk_size=chunk_size, overlap=chunk_overlap)
    logger.info(f"Processing '{filename}' -> generated {len(chunks)} text blocks.")

    embeddings = get_embeddings_batch(chunks, model_name=embedding_model)
    chunk_data = [
        (idx, chunk, emb)
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True), start=1)
    ]

    physical_file_path = save_physical_file(workspace_id, filename, content_bytes)
    logger.info(f"Physical file saved to disk: {physical_file_path}")

    inserted_chunks = insert_document_chunks(
        workspace_id=workspace_id,
        filename=filename,
        file_path=str(physical_file_path),
        chunk_data=chunk_data,
    )

    return inserted_chunks, old_chunks


@app.post("/upload/", response_model=IngestionResponse)
def upload_document(
    workspace_id: str = Form(...),
    file: UploadFile = File(...),
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL),
) -> IngestionResponse:
    """Single document ingestion endpoint."""
    supported_extensions = (".txt", ".md", ".pdf", ".docx", ".csv", ".json")
    if not file.filename or not file.filename.lower().endswith(supported_extensions):
        logger.warning(f"Rejected unsupported file upload attempt: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported extensions: {', '.join(supported_extensions)}",
        )

    try:
        ws = get_workspace(workspace_id)
    except Exception as e:
        logger.error(f"Database error while looking up workspace {workspace_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error looking up workspace."
        ) from e

    if not ws:
        logger.warning(f"Upload rejected: Workspace '{workspace_id}' not found.")
        raise HTTPException(status_code=404, detail="Target workspace does not exist.")

    embedding_model = normalize_model_name(embedding_model)
    if ws["embedding_model"] != embedding_model:
        logger.warning(
            f"Vector pollution blocked: Workspace requires {ws['embedding_model']}, got {embedding_model}."
        )
        raise HTTPException(
            status_code=400,
            detail=f"Model mismatch! Workspace '{ws['name']}' is permanently locked to '{ws['embedding_model']}'.",
        )

    try:
        content_bytes = file.file.read()
        inserted_chunks, old_chunks = _process_single_file_ingestion(
            workspace_id,
            file.filename,
            content_bytes,
            embedding_model,
            chunk_size=ws["chunk_size"],
            chunk_overlap=ws["chunk_overlap"],
        )
        logger.info(f"Successfully processed ingestion: {inserted_chunks} chunks saved.")
        return IngestionResponse(
            status="success",
            workspace_id=workspace_id,
            filename=file.filename,
            model_used=embedding_model,
            chunks_saved=inserted_chunks,
            is_upsert=(old_chunks > 0),
            chunks_deleted=old_chunks,
        )
    except ValueError as val_err:
        logger.warning(f"File extraction failed for '{file.filename}': {val_err}")
        raise HTTPException(status_code=400, detail=str(val_err)) from val_err
    except Exception as e:
        logger.error(f"Error during file ingestion processing: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during document processing."
        ) from e


@app.post("/upload/batch/", response_model=BatchIngestionResponse)
def upload_documents_batch(
    workspace_id: str = Form(...),
    files: List[UploadFile] = File(...),
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL),
) -> BatchIngestionResponse:
    """Batch ingestion endpoint for processing multiple files or entire folder selections."""
    supported_extensions = (".txt", ".md", ".pdf", ".docx", ".csv", ".json")

    try:
        ws = get_workspace(workspace_id)
    except Exception as e:
        logger.error(f"Database error while looking up workspace {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail="Database failure verifying workspace.") from e

    if not ws:
        logger.warning(f"Batch upload rejected: Workspace '{workspace_id}' not found.")
        raise HTTPException(status_code=404, detail="Target workspace does not exist.")

    embedding_model = normalize_model_name(embedding_model)
    if ws["embedding_model"] != embedding_model:
        logger.warning(
            f"Batch vector pollution blocked: Workspace requires '{ws['embedding_model']}', got '{embedding_model}'."
        )
        raise HTTPException(
            status_code=400,
            detail=f"Model mismatch! Workspace '{ws['name']}' is locked to '{ws['embedding_model']}'.",
        )

    results: List[FileIngestionResult] = []
    total_chunks = 0
    success_count = 0
    upsert_count = 0
    failed_count = 0

    logger.info(
        f"Starting batch ingestion of {len(files)} file(s) into workspace '{ws['name']}'..."
    )

    for file in files:
        if not file.filename or not file.filename.strip():
            logger.warning("Batch skip: Rejected file stream with missing or empty filename.")
            results.append(
                FileIngestionResult(
                    filename="unnamed_stream",
                    status="failed",
                    error_message="File upload rejected: Missing or empty filename.",
                )
            )
            failed_count += 1
            continue

        filename = file.filename

        if not filename.lower().endswith(supported_extensions):
            logger.warning(f"Batch skip: Unsupported extension for '{filename}'")
            results.append(
                FileIngestionResult(
                    filename=filename,
                    status="failed",
                    error_message=f"Unsupported file type. Supported: {', '.join(supported_extensions)}",
                )
            )
            failed_count += 1
            continue

        try:
            content_bytes = file.file.read()
            inserted_chunks, old_chunks = _process_single_file_ingestion(
                workspace_id,
                filename,
                content_bytes,
                embedding_model,
                chunk_size=ws["chunk_size"],
                chunk_overlap=ws["chunk_overlap"],
            )

            is_upsert = old_chunks > 0
            if is_upsert:
                upsert_count += 1
                status_str = "upserted"
            else:
                success_count += 1
                status_str = "success"

            total_chunks += inserted_chunks
            results.append(
                FileIngestionResult(
                    filename=filename, status=status_str, chunks_saved=inserted_chunks
                )
            )
            logger.info(f"Batch item complete: '{filename}' ({inserted_chunks} chunks)")

        except ValueError as val_err:
            logger.warning(f"Batch extraction failed for '{filename}': {val_err}")
            results.append(
                FileIngestionResult(filename=filename, status="failed", error_message=str(val_err))
            )
            failed_count += 1
        except Exception as err:
            logger.error(f"Batch unexpected error processing '{filename}': {err}")
            results.append(
                FileIngestionResult(
                    filename=filename, status="failed", error_message="Internal processing error."
                )
            )
            failed_count += 1

    summary = BatchIngestionSummary(
        total_files=len(files),
        successful=success_count,
        upserts=upsert_count,
        failed=failed_count,
        total_chunks_saved=total_chunks,
    )

    logger.info(
        f"Batch ingestion complete: {success_count} new, {upsert_count} updated, {failed_count} failed."
    )
    return BatchIngestionResponse(
        status="completed",
        workspace_id=workspace_id,
        model_used=embedding_model,
        summary=summary,
        results=results,
    )


@app.delete("/documents/{workspace_id}/{filename}", response_model=dict)
def remove_document(workspace_id: str, filename: str) -> dict:
    """Deletes a specific document's vector chunks from the database and removes its physical file."""
    try:
        ws = get_workspace(workspace_id)
        if not ws:
            logger.warning(f"Document deletion failed: Workspace '{workspace_id}' not found.")
            raise HTTPException(status_code=404, detail="Workspace not found.")

        deleted_chunks = delete_document(workspace_id, filename)
        file_removed = delete_physical_file(workspace_id, filename)

        if deleted_chunks == 0 and not file_removed:
            logger.warning(
                f"Document deletion failed: '{filename}' not found in workspace '{workspace_id}'."
            )
            raise HTTPException(status_code=404, detail="Document not found in workspace.")

        logger.info(
            f"Document '{filename}' purged from workspace '{workspace_id}' ({deleted_chunks} chunks removed)."
        )
        return {
            "status": "success",
            "workspace_id": workspace_id,
            "filename": filename,
            "chunks_deleted": deleted_chunks,
            "file_removed": file_removed,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {filename} from {workspace_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to delete document from backend."
        ) from e


@app.post("/search/", response_model=VectorSearchResponse)
def search_documents(search: SearchQuery) -> VectorSearchResponse:
    try:
        ws = get_workspace(search.workspace_id)
    except Exception as e:
        logger.error(f"DB Error verifying workspace for search: {e}")
        raise HTTPException(status_code=500, detail="Database failure verifying workspace.") from e

    if not ws:
        logger.warning(f"Search aborted: Workspace '{search.workspace_id}' not found.")
        raise HTTPException(status_code=404, detail="Workspace not found.")

    if ws["embedding_model"] != search.embedding_model:
        logger.warning(
            f"Search model mismatch. Expected '{ws['embedding_model']}', got '{search.embedding_model}'."
        )
        raise HTTPException(
            status_code=400,
            detail=f"Search must use the workspace's locked model: '{ws['embedding_model']}'.",
        )

    effective_top_k = search.top_k if search.top_k is not None else ws["top_k"]
    effective_threshold = (
        search.similarity_threshold
        if search.similarity_threshold is not None
        else ws["similarity_threshold"]
    )

    try:
        query_embedding = get_embedding(search.query, model_name=search.embedding_model)
        if not query_embedding:
            logger.error("Failed to generate search query embedding.")
            raise HTTPException(
                status_code=500, detail="Failed to generate embedding for the query."
            )

        raw_results = search_vector_db(
            workspace_id=search.workspace_id,
            query_embedding=query_embedding,
            top_k=effective_top_k,
            similarity_threshold=effective_threshold,
        )

        results = [
            SearchResultCard(
                id=row["id"],
                filename=row["filename"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                similarity=round(row["similarity"], 4),
            )
            for row in raw_results
        ]

        return VectorSearchResponse(
            workspace_id=search.workspace_id,
            query=search.query,
            embedding_model=search.embedding_model,
            results=results,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during vector search.") from e


@app.post("/ask/", response_model=RAGQueryResponse)
def ask_question(search: SearchQuery) -> RAGQueryResponse:
    logger.info(
        f"Executing RAG pipeline for query: '{search.query}' in workspace {search.workspace_id}"
    )

    try:
        ws = get_workspace(search.workspace_id)
    except Exception as e:
        logger.error(f"DB Error fetching workspace for RAG: {e}")
        raise HTTPException(status_code=500, detail="Database failure resolving workspace.") from e

    if not ws:
        logger.warning(f"Ask aborted: Workspace '{search.workspace_id}' not found.")
        raise HTTPException(status_code=404, detail="Target workspace does not exist.")

    if ws["embedding_model"] != search.embedding_model:
        logger.warning("Ask aborted: Model mismatch preventing vector collision.")
        raise HTTPException(
            status_code=400,
            detail=f"Request embedding model must match workspace: '{ws['embedding_model']}'.",
        )

    effective_top_k = search.top_k if search.top_k is not None else ws["top_k"]
    effective_threshold = (
        search.similarity_threshold
        if search.similarity_threshold is not None
        else ws["similarity_threshold"]
    )
    effective_temp = search.temperature
    effective_history_limit = (
        search.chat_history_limit
        if search.chat_history_limit is not None
        else ws["chat_history_limit"]
    )

    thread_id = search.thread_id
    chat_history = []
    try:
        if thread_id:
            if not get_thread(thread_id):
                logger.warning(f"Ask aborted: Thread '{thread_id}' not found.")
                raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
            chat_history, _ = get_thread_messages(
                thread_id, limit=effective_history_limit, offset=0
            )
            logger.info(f"Loaded {len(chat_history)} historical messages for thread {thread_id}.")
        else:
            thread_id = str(uuid.uuid4())
            title = search.query[:40] + ("..." if len(search.query) > 40 else "")
            create_thread(thread_id=thread_id, workspace_id=search.workspace_id, title=title)
            logger.info(f"Created new conversation thread '{thread_id}' with title '{title}'.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error during thread management: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error managing chat session."
        ) from e

    try:
        query_embedding = get_embedding(search.query, model_name=search.embedding_model)
        if not query_embedding:
            logger.error("Failed to generate query embedding for RAG.")
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

        raw_results = search_vector_db(
            workspace_id=search.workspace_id,
            query_embedding=query_embedding,
            top_k=effective_top_k,
            similarity_threshold=effective_threshold,
        )

        sources = [
            DocumentSource(
                filename=row["filename"],
                chunk_index=row["chunk_index"],
                similarity=round(row["similarity"], 4),
                content=row.get("content"),
            )
            for row in raw_results
        ]

        if not raw_results:
            logger.info("No matching contextual chunks found inside the vault.")
            add_message(thread_id, "user", search.query, search.generation_model)
            add_message(
                thread_id,
                "assistant",
                "No relevant documents found in the vault.",
                search.generation_model,
                sources=[],
            )
            return RAGQueryResponse(
                workspace_id=search.workspace_id,
                thread_id=thread_id,
                query=search.query,
                answer="No relevant documents found in the vault.",
                generation_model=search.generation_model,
                embedding_model=search.embedding_model,
                sources=[],
            )

        llm_response = generate_answer(
            query=search.query,
            context_chunks=raw_results,
            model_name=search.generation_model,
            chat_history=chat_history,
            temperature=effective_temp,
            system_prompt=ws["system_prompt"],
        )

        if not llm_response.is_valid:
            logger.info("LLM returned context failure warning. Hiding document references.")
            sources = []

        logger.info(
            f"RAG generation successful for thread '{thread_id}'. Attached {len(sources)} Relevant Document Chunks."
        )

        add_message(thread_id, "user", search.query, search.generation_model)
        add_message(
            thread_id,
            "assistant",
            llm_response.text,
            search.generation_model,
            sources=[s.model_dump() for s in sources],
        )

        return RAGQueryResponse(
            workspace_id=search.workspace_id,
            thread_id=thread_id,
            query=search.query,
            answer=llm_response.text,
            generation_model=search.generation_model,
            embedding_model=search.embedding_model,
            sources=sources,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG pipeline failure: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during RAG generation."
        ) from e
