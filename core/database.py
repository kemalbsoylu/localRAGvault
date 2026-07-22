from typing import List, Optional, Tuple

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
                    CREATE TABLE IF NOT EXISTS workspaces (
                        id VARCHAR(50) PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        embedding_model TEXT NOT NULL,
                        dimension INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        workspace_id VARCHAR(50) REFERENCES workspaces(id) ON DELETE CASCADE,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        content TEXT NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding vector,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
        logger.info(
            "Database initialized successfully. 'workspaces' and 'documents' tables are ready."
        )
    except Exception as e:
        logger.error(f"Critical error during database schema creation: {e}")
        raise


def create_workspace(workspace_id: str, name: str, embedding_model: str, dimension: int) -> None:
    """Creates a new workspace, locking in its dimension size."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO workspaces (id, name, embedding_model, dimension)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (workspace_id, name, embedding_model, dimension),
                )
    except Exception as e:
        logger.error(f"Database error while creating workspace '{name}': {e}")
        raise


def get_workspace(workspace_id: str) -> Optional[dict]:
    """Retrieves workspace metadata by ID."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, embedding_model, dimension FROM workspaces WHERE id = %s",
                    (workspace_id,),
                )
                row = cur.fetchone()
                if isinstance(row, tuple):
                    return {
                        "id": row[0],
                        "name": row[1],
                        "embedding_model": row[2],
                        "dimension": row[3],
                    }
        return None
    except Exception as e:
        logger.error(f"Database error while fetching workspace {workspace_id}: {e}")
        raise


def get_all_workspaces() -> List[dict]:
    """Retrieves all available workspaces."""
    workspaces = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, embedding_model, dimension FROM workspaces ORDER BY created_at DESC;"
                )
                for row in cur.fetchall():
                    workspaces.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "embedding_model": row[2],
                            "dimension": row[3],
                        }
                    )
        return workspaces
    except Exception as e:
        logger.error(f"Database error while fetching all workspaces: {e}")
        raise


def insert_document_chunks(
    workspace_id: str,
    filename: str,
    file_path: str,
    embedding_model: str,
    chunk_data: List[Tuple[str, List[float]]],
) -> int:
    """Bulk inserts text chunks and their corresponding vector embeddings."""
    inserted_chunks = 0
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in chunk_data:
                    if embedding:
                        cur.execute(
                            """
                            INSERT INTO documents (workspace_id, filename, file_path, content, embedding_model, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (workspace_id, filename, file_path, chunk, embedding_model, embedding),
                        )
                        inserted_chunks += 1
        return inserted_chunks
    except Exception as e:
        logger.error(f"Database error while inserting document chunks for {filename}: {e}")
        raise


def fetch_workspace_inventory(workspace_id: str) -> List[dict]:
    """Retrieves file aggregates for a given workspace."""
    inventory = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT filename, file_path, COUNT(*) as total_chunks
                    FROM documents
                    WHERE workspace_id = %s
                    GROUP BY filename, file_path
                    ORDER BY filename ASC;
                    """,
                    (workspace_id,),
                )
                rows = cur.fetchall()
                for row in rows:
                    inventory.append(
                        {"filename": row[0], "file_path": row[1], "total_chunks": row[2]}
                    )
        return inventory
    except Exception as e:
        logger.error(f"Database error while fetching inventory for workspace {workspace_id}: {e}")
        raise


def search_vector_db(
    workspace_id: str, query_embedding: List[float], embedding_model: str, top_k: int
) -> List[dict]:
    """Performs a vector similarity search against the document chunks."""
    results = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, filename, content, 1 - (embedding <=> %s::vector) AS similarity
                    FROM documents
                    WHERE workspace_id = %s AND embedding_model = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_embedding, workspace_id, embedding_model, query_embedding, top_k),
                )
                rows = cur.fetchall()
                for row in rows:
                    results.append(
                        {"id": row[0], "filename": row[1], "content": row[2], "similarity": row[3]}
                    )
        return results
    except Exception as e:
        logger.error(f"Database error during vector search in workspace {workspace_id}: {e}")
        raise


if __name__ == "__main__":
    init_db()
