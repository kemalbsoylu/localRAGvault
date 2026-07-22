from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from core.config import DEFAULT_EMBEDDING_MODEL
from core.database import (
    fetch_workspace_inventory,
    get_db_connection,
    init_db,
    insert_document_chunks,
)
from core.logging_config import logger
from core.schemas import (
    DocumentInventoryItem,
    DocumentSource,
    IngestionResponse,
    ModelListResponse,
    RAGQueryResponse,
    SearchQuery,
    SearchResultCard,
    VectorSearchResponse,
    WorkspaceInventoryResponse,
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
    models = get_available_models()
    status = "success" if models else "error"
    return ModelListResponse(status=status, models=models)


@app.get("/inventory/{workspace_id}", response_model=WorkspaceInventoryResponse)
def get_workspace_inventory(workspace_id: str) -> WorkspaceInventoryResponse:
    """Returns a list of physical files currently indexed for a given workspace."""
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


@app.post("/upload/", response_model=IngestionResponse)
async def upload_document(
    file: UploadFile = File(...), embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL)
) -> IngestionResponse:
    if not file.filename or not file.filename.endswith((".txt", ".md")):
        logger.warning(f"Rejected malicious or invalid file upload attempt: {file.filename}")
        raise HTTPException(
            status_code=400, detail="Only .txt and .md files are supported for now."
        )

    embedding_model = normalize_model_name(embedding_model)
    content_bytes = await file.read()

    try:
        content_text = content_bytes.decode("utf-8")
    except UnicodeDecodeError as err:
        logger.error(f"Encoding conversion breakdown during reading: {file.filename}")
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text.") from err

    # Hardcode workspace_id until workspaces are implemented
    workspace_id = "default"
    physical_file_path = save_physical_file(workspace_id, file.filename, content_bytes)
    logger.info(f"Physical file saved to disk: {physical_file_path}")

    chunks = chunk_text(content_text)
    logger.info(f"Processing '{file.filename}' -> generated {len(chunks)} text blocks.")

    chunk_data = []
    for chunk in chunks:
        embedding = get_embedding(chunk, model_name=embedding_model)
        chunk_data.append((chunk, embedding))

    inserted_chunks = insert_document_chunks(
        filename=file.filename,
        file_path=str(physical_file_path),
        embedding_model=embedding_model,
        chunk_data=chunk_data,
    )

    logger.info(f"Successfully processed ingestion: {inserted_chunks}/{len(chunks)} chunks saved.")

    return IngestionResponse(
        status="success",
        filename=file.filename,
        model_used=embedding_model,
        chunks_saved=inserted_chunks,
    )


@app.post("/search/", response_model=VectorSearchResponse)
async def search_documents(search: SearchQuery) -> VectorSearchResponse:
    query_embedding = get_embedding(search.query, model_name=search.embedding_model)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate embedding for the query.")

    results: List[SearchResultCard] = []

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE embedding_model = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """,
                (
                    query_embedding,
                    search.embedding_model,
                    query_embedding,
                    search.top_k,
                ),
            )

            rows = cur.fetchall()
            for row in rows:
                results.append(
                    SearchResultCard(
                        id=row[0],
                        filename=row[1],
                        content=row[2],
                        similarity=round(row[3], 4),
                    )
                )

    return VectorSearchResponse(
        query=search.query, embedding_model=search.embedding_model, results=results
    )


@app.post("/ask/", response_model=RAGQueryResponse)
async def ask_question(search: SearchQuery) -> RAGQueryResponse:
    logger.info(f"Executing RAG pipeline invocation for user query: '{search.query}'")
    query_embedding = get_embedding(search.query, model_name=search.embedding_model)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

    retrieved_chunks: List[str] = []
    sources: List[DocumentSource] = []

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT filename, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE embedding_model = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """,
                (
                    query_embedding,
                    search.embedding_model,
                    query_embedding,
                    search.top_k,
                ),
            )

            rows = cur.fetchall()
            for row in rows:
                filename, content, similarity = row[0], row[1], row[2]
                retrieved_chunks.append(content)
                sources.append(DocumentSource(filename=filename, similarity=round(similarity, 4)))

    if not retrieved_chunks:
        logger.info("No matching contextual chunks found inside the database vaults.")
        return RAGQueryResponse(
            query=search.query,
            answer="No relevant documents found in the vault.",
            generation_model=search.generation_model,
            embedding_model=search.embedding_model,
            sources=[],
        )

    llm_response = generate_answer(
        query=search.query,
        context_chunks=retrieved_chunks,
        model_name=search.generation_model,
    )

    if not llm_response.is_valid:
        logger.info("LLM returned context failure warning. Hiding document references.")
        sources = []

    return RAGQueryResponse(
        query=search.query,
        answer=llm_response.text,
        generation_model=search.generation_model,
        embedding_model=search.embedding_model,
        sources=sources,
    )
