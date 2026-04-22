from contextlib import asynccontextmanager
from fastapi import FastAPI
from database import init_db


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


@app.get("/")
def health_check():
    return {"status": "success", "message": "localRAGvault API is running securely."}
