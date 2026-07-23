import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

import psycopg
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from core.config import DEFAULT_EMBEDDING_MODEL
from core.database import (
    add_message,
    create_thread,
    create_workspace,
    fetch_workspace_inventory,
    get_all_workspaces,
    get_thread,
    get_thread_messages,
    get_workspace,
    get_workspace_threads,
    init_db,
    insert_document_chunks,
    search_vector_db,
)
from core.logging_config import logger
from core.schemas import (
    DocumentInventoryItem,
    DocumentSource,
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
)
from core.utils import (
    chunk_text,
    ensure_default_models_exist,
    generate_answer,
    get_available_models,
    get_embedding,
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
    """Creates a new workspace and probes the embedding model to lock in vector dimensions."""
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
        create_workspace(workspace_id, ws.name, ws.embedding_model, dimension)
        logger.info(f"Workspace '{ws.name}' ({workspace_id}) locked with dimension {dimension}.")
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

    return WorkspaceResponse(
        id=workspace_id, name=ws.name, embedding_model=ws.embedding_model, dimension=dimension
    )


# --- CONVERSATION ENDPOINTS ---


@app.get("/workspaces/{workspace_id}/threads", response_model=ThreadListResponse)
def list_workspace_threads(workspace_id: str) -> ThreadListResponse:
    """Returns a list of all conversation threads inside a workspace."""
    try:
        ws = get_workspace(workspace_id)
        if not ws:
            logger.warning(f"Thread listing failed: Workspace '{workspace_id}' not found.")
            raise HTTPException(status_code=404, detail="Workspace not found.")
        raw_threads = get_workspace_threads(workspace_id)
        threads = [ThreadCard(**t) for t in raw_threads]
        return ThreadListResponse(workspace_id=workspace_id, threads=threads)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching threads for workspace {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation threads.") from e


@app.get("/threads/{thread_id}/messages", response_model=ThreadHistoryResponse)
def get_thread_history(thread_id: str) -> ThreadHistoryResponse:
    """Returns the full chronological message history for a specific thread."""
    try:
        t = get_thread(thread_id)
        if not t:
            logger.warning(f"Message history fetch failed: Thread '{thread_id}' not found.")
            raise HTTPException(status_code=404, detail="Thread not found.")
        raw_messages = get_thread_messages(thread_id, limit=50)
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
        return ThreadHistoryResponse(thread_id=thread_id, messages=messages)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching messages for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch message history.") from e


# --- VAULT & RAG ENDPOINTS ---


@app.get("/inventory/{workspace_id}", response_model=WorkspaceInventoryResponse)
def get_workspace_inventory(workspace_id: str) -> WorkspaceInventoryResponse:
    """Returns a list of physical files currently indexed for a given workspace."""
    try:
        ws = get_workspace(workspace_id)
        if not ws:
            logger.warning(f"Inventory fetch failed: Workspace '{workspace_id}' not found.")
            raise HTTPException(status_code=404, detail="Target workspace does not exist.")

        raw_inventory = fetch_workspace_inventory(workspace_id)
        documents = [
            DocumentInventoryItem(
                filename=item["filename"],
                file_path=item["file_path"],
                total_chunks=item["total_chunks"],
            )
            for item in raw_inventory
        ]
        return WorkspaceInventoryResponse(workspace_id=workspace_id, documents=documents)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching inventory for {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch workspace inventory.") from e


@app.post("/upload/", response_model=IngestionResponse)
async def upload_document(
    workspace_id: str = Form(...),
    file: UploadFile = File(...),
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL),
) -> IngestionResponse:
    if not file.filename or not file.filename.endswith((".txt", ".md")):
        logger.warning(f"Rejected malicious or invalid file upload attempt: {file.filename}")
        raise HTTPException(
            status_code=400, detail="Only .txt and .md files are supported for now."
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

    content_bytes = await file.read()
    try:
        content_text = content_bytes.decode("utf-8")
    except UnicodeDecodeError as err:
        logger.error(f"Encoding conversion breakdown during reading: {file.filename}")
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text.") from err

    try:
        physical_file_path = save_physical_file(workspace_id, file.filename, content_bytes)
        logger.info(f"Physical file saved to disk: {physical_file_path}")

        chunks = chunk_text(content_text)
        logger.info(f"Processing '{file.filename}' -> generated {len(chunks)} text blocks.")

        chunk_data = []
        for chunk in chunks:
            embedding = get_embedding(chunk, model_name=embedding_model)
            chunk_data.append((chunk, embedding))

        inserted_chunks = insert_document_chunks(
            workspace_id=workspace_id,
            filename=file.filename,
            file_path=str(physical_file_path),
            chunk_data=chunk_data,
        )

        logger.info(
            f"Successfully processed ingestion: {inserted_chunks}/{len(chunks)} chunks saved."
        )
        return IngestionResponse(
            status="success",
            workspace_id=workspace_id,
            filename=file.filename,
            model_used=embedding_model,
            chunks_saved=inserted_chunks,
        )
    except Exception as e:
        logger.error(f"Error during file ingestion processing: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during document processing."
        ) from e


@app.post("/search/", response_model=VectorSearchResponse)
async def search_documents(search: SearchQuery) -> VectorSearchResponse:
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
            top_k=search.top_k,
        )

        results = [
            SearchResultCard(
                id=row["id"],
                filename=row["filename"],
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
async def ask_question(search: SearchQuery) -> RAGQueryResponse:
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
        raise HTTPException(status_code=404, detail="Workspace not found.")

    if ws["embedding_model"] != search.embedding_model:
        logger.warning("Ask aborted: Model mismatch preventing vector collision.")
        raise HTTPException(
            status_code=400,
            detail=f"Request embedding model must match workspace: '{ws['embedding_model']}'.",
        )

    # 1. Thread Management: Resolve existing thread or create a new one
    thread_id = search.thread_id
    chat_history = []
    try:
        if thread_id:
            if not get_thread(thread_id):
                logger.warning(f"Ask aborted: Thread '{thread_id}' not found.")
                raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
            # Load previous turns to pass to the LLM for conversational awareness
            chat_history = get_thread_messages(thread_id, limit=6)
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
            top_k=search.top_k,
        )

        retrieved_chunks = [row["content"] for row in raw_results]
        sources = [
            DocumentSource(filename=row["filename"], similarity=round(row["similarity"], 4))
            for row in raw_results
        ]

        if not retrieved_chunks:
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

        # 2. Execute LLM with retrieved context AND chat history
        llm_response = generate_answer(
            query=search.query,
            context_chunks=retrieved_chunks,
            model_name=search.generation_model,
            chat_history=chat_history,
        )

        if not llm_response.is_valid:
            logger.info("LLM returned context failure warning. Hiding document references.")
            sources = []

        # 3. Persist turns to Database
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
