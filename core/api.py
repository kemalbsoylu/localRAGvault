from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

import ollama
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL
from core.database import get_db_connection, init_db
from core.logging_config import logger
from core.schemas import (
    DocumentSource,
    IngestionResponse,
    ModelListResponse,
    RAGQueryResponse,
    SearchQuery,
    SearchResultCard,
    VectorSearchResponse,
)
from core.utils import chunk_text, generate_answer, get_available_models, get_embedding


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initializing localRAGvault core infrastructure components...")
    init_db()

    logger.info("Verifying required Ollama models...")
    try:
        available_models = get_available_models()
        required_models = [DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL]

        for model in required_models:
            target_model = model if ":" in model else f"{model}:latest"

            if target_model not in available_models:
                logger.info(
                    f"Model '{target_model}' missing locally. Initiating auto-pull (this may take a few minutes)..."
                )
                try:
                    ollama.pull(model)
                    logger.info(f"Successfully downloaded and registered '{target_model}'.")
                except Exception as pull_err:
                    logger.error(
                        f"Failed to pull '{target_model}': {pull_err}. Proceeding with startup."
                    )
            else:
                logger.info(f"Model '{target_model}' is already available.")

    except Exception as e:
        logger.error(f"Failed to communicate with local Ollama daemon during startup: {e}")

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


@app.post("/upload/", response_model=IngestionResponse)
async def upload_document(
    file: UploadFile = File(...), embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL)
) -> IngestionResponse:
    if not file.filename or not file.filename.endswith((".txt", ".md")):
        logger.warning(f"Rejected malicious or invalid file upload attempt: {file.filename}")
        raise HTTPException(
            status_code=400, detail="Only .txt and .md files are supported for now."
        )

    content_bytes = await file.read()
    try:
        content_text = content_bytes.decode("utf-8")
    except UnicodeDecodeError as err:
        logger.error(f"Encoding conversion breakdown during reading: {file.filename}")
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text.") from err

    chunks = chunk_text(content_text)
    logger.info(f"Processing '{file.filename}' -> generated {len(chunks)} text blocks.")

    inserted_chunks = 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                embedding = get_embedding(chunk, model_name=embedding_model)
                if embedding:
                    cur.execute(
                        """
                        INSERT INTO documents (filename, content, embedding_model, embedding)
                        VALUES (%s, %s, %s, %s)
                    """,
                        (file.filename, chunk, embedding_model, embedding),
                    )
                    inserted_chunks += 1

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
