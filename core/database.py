from typing import List, Tuple

import psycopg
from pgvector.psycopg import register_vector

from core.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from core.logging_config import logger


def get_db_connection() -> psycopg.Connection:
    """Establishes a connection to the database and registers the vector type."""
    try:
        conn = psycopg.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            autocommit=True,
        )
        register_vector(conn)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to the database engine: {e}")
        raise


def init_db() -> None:
    """Initializes the database schema if not present."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        content TEXT NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding vector,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
        logger.info("Database initialized successfully. 'documents' table is ready.")
    except Exception as e:
        logger.error(f"Critical error during database schema creation: {e}")
        raise


def insert_document_chunks(
    filename: str, file_path: str, embedding_model: str, chunk_data: List[Tuple[str, List[float]]]
) -> int:
    """Bulk inserts text chunks and their corresponding vector embeddings."""
    inserted_chunks = 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for chunk, embedding in chunk_data:
                if embedding:
                    cur.execute(
                        """
                        INSERT INTO documents (filename, file_path, content, embedding_model, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                    """,
                        (filename, file_path, chunk, embedding_model, embedding),
                    )
                    inserted_chunks += 1
    return inserted_chunks


def fetch_workspace_inventory(workspace_id: str) -> List[dict]:
    """Retrieves file aggregates for a given workspace."""
    # Note: We will use workspace_id in the WHERE clause when workspaces are implemented
    inventory = []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT filename, file_path, COUNT(*) as total_chunks
                FROM documents
                GROUP BY filename, file_path
                ORDER BY filename ASC;
                """
            )
            rows = cur.fetchall()
            for row in rows:
                inventory.append({"filename": row[0], "file_path": row[1], "total_chunks": row[2]})
    return inventory


def search_vector_db(query_embedding: List[float], embedding_model: str, top_k: int) -> List[dict]:
    """Performs a vector similarity search against the document chunks."""
    results = []
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
                (query_embedding, embedding_model, query_embedding, top_k),
            )
            rows = cur.fetchall()
            for row in rows:
                results.append(
                    {"id": row[0], "filename": row[1], "content": row[2], "similarity": row[3]}
                )
    return results


if __name__ == "__main__":
    init_db()
