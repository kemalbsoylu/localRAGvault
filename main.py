from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from database import init_db, get_db_connection
from utils import chunk_text, get_embedding
from pydantic import BaseModel


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


# --- Request Schemas ---
class SearchQuery(BaseModel):
    query: str
    top_k: int = 3  # Return the top 3 closest chunks by default


# --- Endpoints ---
@app.get("/")
def health_check():
    return {"status": "success", "message": "localRAGvault API is running securely."}


@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported for now.")

    # Read the content
    content_bytes = await file.read()
    try:
        content_text = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text.")

    # Chunk the text
    chunks = chunk_text(content_text, chunk_size=1000, overlap=200)

    # Embed and Save to Database
    inserted_chunks = 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                # Generate embedding using Ollama
                embedding = get_embedding(chunk, model_name="nomic-embed-text")

                if embedding:
                    # Insert into pgvector database
                    cur.execute("""
                                INSERT INTO documents (filename, content, embedding)
                                VALUES (%s, %s, %s)
                                """, (file.filename, chunk, embedding))
                    inserted_chunks += 1

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_processed": len(chunks),
        "chunks_saved": inserted_chunks
    }


@app.post("/search/")
async def search_documents(search: SearchQuery):
    # Embed the user's question
    query_embedding = get_embedding(search.query, model_name="nomic-embed-text")

    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate embedding for the query.")

    results = []

    # Search Postgres using Cosine Distance (<=>)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 1 - distance gives us cosine similarity (1.0 is a perfect match)
            cur.execute("""
                        SELECT id, filename, content, 1 - (embedding <=> %s::vector) AS similarity
                        FROM documents
                        ORDER BY embedding <=> %s::vector
                            LIMIT %s;
                        """, (query_embedding, query_embedding, search.top_k))

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
        "results": results
    }
