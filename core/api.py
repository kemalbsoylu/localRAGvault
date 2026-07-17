from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form

from core.database import init_db, get_db_connection
from core.schemas import SearchQuery
from core.utils import chunk_text, get_embedding, generate_answer, get_available_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    print("Starting up localRAGvault...")
    init_db()

    yield  # This yields control back to FastAPI to run the application

    # --- Shutdown Logic ---
    print("Shutting down localRAGvault...")


# Initialize FastAPI application with the lifespan manager
app = FastAPI(
    title="localRAGvault API",
    description="Local RAG pipeline backend powered by FastAPI, Ollama, and pgvector.",
    version="0.1.0",
    lifespan=lifespan
)


# --- Endpoints ---
@app.get("/")
def health_check():
    return {"status": "success", "message": "localRAGvault API is running securely."}


@app.get("/models/")
def list_models():
    """Returns a list of all models currently downloaded in Ollama."""
    return get_available_models()


@app.post("/upload/")
async def upload_document(
    file: UploadFile = File(...),
    embedding_model: str = Form("nomic-embed-text")
):
    # Validate file type
    if not file.filename or not file.filename.endswith((".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported for now.")

    # Read the content
    content_bytes = await file.read()
    try:
        content_text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text.")

    # Chunk the text
    chunks = chunk_text(content_text, chunk_size=1000, overlap=200)

    # Embed and Save to Database
    inserted_chunks = 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                # Generate embedding using Ollama, use the selected model
                embedding = get_embedding(chunk, model_name=embedding_model)

                if embedding:
                    # Insert into pgvector database
                    cur.execute("""
                        INSERT INTO documents (filename, content, embedding_model, embedding)
                        VALUES (%s, %s, %s, %s)
                    """, (file.filename, chunk, embedding_model, embedding))
                    inserted_chunks += 1

    return {
        "status": "success",
        "filename": file.filename,
        "model_used": embedding_model,
        "chunks_saved": inserted_chunks
    }


@app.post("/search/")
async def search_documents(search: SearchQuery):
    # Embed the user's query using the specified embedding model
    query_embedding = get_embedding(search.query, model_name=search.embedding_model)

    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate embedding for the query.")

    results = []

    # Search Postgres using Cosine Distance (<=>) filtering by model name
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Distance gives cosine similarity (1.0 is a perfect match)
            cur.execute("""
                SELECT id, filename, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE embedding_model = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, search.embedding_model, query_embedding, search.top_k))

            rows = cur.fetchall()

            for row in rows:
                results.append({
                    "id": row[0],
                    "filename": row[1],
                    "content": row[2],
                    "similarity": round(row[3], 4)
                })

    return {
        "query": search.query,
        "embedding_model": search.embedding_model,
        "results": results
    }


@app.post("/ask/")
async def ask_question(search: SearchQuery):
    # Embed the question using the selected embedding model
    query_embedding = get_embedding(search.query, model_name=search.embedding_model)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

    retrieved_chunks = []
    sources = []

    # Retrieve the most relevant chunks from Postgres
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Added WHERE embedding_model = %s to prevent vector pollution
            cur.execute("""
                SELECT filename, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE embedding_model = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, search.embedding_model, query_embedding, search.top_k))

            rows = cur.fetchall()
            for row in rows:
                filename, content, similarity = row[0], row[1], row[2]
                retrieved_chunks.append(content)
                sources.append({"filename": filename, "similarity": round(similarity, 4)})

    if not retrieved_chunks:
        return {"query": search.query, "answer": "No relevant documents found in the vault.", "sources": []}

    # Generate the answer using the selected generation model
    final_answer = generate_answer(
        query=search.query,
        context_chunks=retrieved_chunks,
        model_name=search.generation_model
    )

    return {
        "query": search.query,
        "answer": final_answer,
        "generation_model": search.generation_model,
        "sources": sources
    }
