from typing import List, Optional, Tuple

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

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
                        embedding vector,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS threads (
                        id VARCHAR(50) PRIMARY KEY,
                        workspace_id VARCHAR(50) REFERENCES workspaces(id) ON DELETE CASCADE,
                        title TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        thread_id VARCHAR(50) REFERENCES threads(id) ON DELETE CASCADE,
                        role VARCHAR(20) NOT NULL,
                        content TEXT NOT NULL,
                        sources JSONB NULL,
                        model_used TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
        logger.info(
            "Database initialized successfully. 'workspaces', 'documents', 'threads', and 'messages' tables are ready."
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
                            INSERT INTO documents (workspace_id, filename, file_path, content, embedding)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (workspace_id, filename, file_path, chunk, embedding),
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


def search_vector_db(workspace_id: str, query_embedding: List[float], top_k: int) -> List[dict]:
    """Performs a vector similarity search against the document chunks."""
    results = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, filename, content, 1 - (embedding <=> %s::vector) AS similarity
                    FROM documents
                    WHERE workspace_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_embedding, workspace_id, query_embedding, top_k),
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


def create_thread(thread_id: str, workspace_id: str, title: str) -> None:
    """Creates a new conversation thread inside a workspace."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO threads (id, workspace_id, title) VALUES (%s, %s, %s)",
                    (thread_id, workspace_id, title),
                )
    except Exception as e:
        logger.error(f"Database error creating thread {thread_id}: {e}")
        raise


def get_thread(thread_id: str) -> Optional[dict]:
    """Retrieves thread metadata by ID."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, workspace_id, title FROM threads WHERE id = %s", (thread_id,)
                )
                row = cur.fetchone()
                if isinstance(row, tuple):
                    return {"id": row[0], "workspace_id": row[1], "title": row[2]}
        return None
    except Exception as e:
        logger.error(f"Database error fetching thread {thread_id}: {e}")
        raise


def get_workspace_threads(workspace_id: str) -> List[dict]:
    """Retrieves all conversation threads for a workspace, ordered by most recently active."""
    threads = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT t.id, t.title, t.created_at, t.updated_at,
                           (SELECT COUNT(*) FROM messages m WHERE m.thread_id = t.id) as msg_count
                    FROM threads t
                    WHERE t.workspace_id = %s
                    ORDER BY t.updated_at DESC LIMIT 50;
                    """,
                    (workspace_id,),
                )
                thread_rows = cur.fetchall()
                for row in thread_rows:
                    t_id = row[0]

                    # Get latest user query
                    cur.execute(
                        "SELECT content FROM messages WHERE thread_id = %s AND role = 'user' ORDER BY created_at DESC LIMIT 1;",
                        (t_id,),
                    )
                    u_row = cur.fetchone()
                    last_query = u_row[0] if isinstance(u_row, tuple) else row[1]

                    # Get latest assistant answer & metadata
                    cur.execute(
                        "SELECT content, model_used, sources FROM messages WHERE thread_id = %s AND role = 'assistant' ORDER BY created_at DESC LIMIT 1;",
                        (t_id,),
                    )
                    a_row = cur.fetchone()
                    if isinstance(a_row, tuple):
                        last_answer = a_row[0]
                        model_used = a_row[1]
                        sources = a_row[2] if a_row[2] else []
                    else:
                        last_answer = "No response recorded."
                        model_used = "unknown"
                        sources = []

                    threads.append(
                        {
                            "id": t_id,
                            "title": row[1],
                            "created_at": str(row[2]),
                            "updated_at": str(row[3]),
                            "message_count": row[4],
                            "last_query": last_query,
                            "last_answer": last_answer,
                            "model_used": model_used,
                            "sources": sources,
                        }
                    )
        return threads
    except Exception as e:
        logger.error(f"Database error fetching threads for workspace {workspace_id}: {e}")
        raise


def add_message(
    thread_id: str, role: str, content: str, model_used: str, sources: Optional[list] = None
) -> None:
    """Appends a message (user or assistant) to a thread."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (thread_id, role, content, sources, model_used)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        thread_id,
                        role,
                        content,
                        Jsonb(sources) if sources is not None else None,
                        model_used,
                    ),
                )
                cur.execute(
                    "UPDATE threads SET updated_at = CURRENT_TIMESTAMP WHERE id = %s;",
                    (thread_id,),
                )
    except Exception as e:
        logger.error(f"Database error saving message to thread {thread_id}: {e}")
        raise


def get_thread_messages(thread_id: str, limit: int = 10) -> List[dict]:
    """Retrieves conversation history for a thread in chronological order."""
    messages = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Subquery gets latest N messages, outer query sorts them chronologically (oldest -> newest) for LLM context
                cur.execute(
                    """
                    SELECT id, role, content, sources, model_used, created_at
                    FROM (
                        SELECT id, role, content, sources, model_used, created_at
                        FROM messages
                        WHERE thread_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    ) sub
                    ORDER BY created_at ASC;
                    """,
                    (thread_id, limit),
                )
                for row in cur.fetchall():
                    messages.append(
                        {
                            "id": row[0],
                            "role": row[1],
                            "content": row[2],
                            "sources": row[3] if row[3] else [],
                            "model_used": row[4],
                            "created_at": str(row[5]),
                        }
                    )
        return messages
    except Exception as e:
        logger.error(f"Database error fetching messages for thread {thread_id}: {e}")
        raise


def delete_workspace(workspace_id: str) -> bool:
    """Deletes a workspace from the database. Cascade constraints automatically purge child documents and threads."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM workspaces WHERE id = %s RETURNING id;", (workspace_id,))
                row = cur.fetchone()
                return row is not None
    except Exception as e:
        logger.error(f"Database error while deleting workspace '{workspace_id}': {e}")
        raise


def delete_document(workspace_id: str, filename: str) -> int:
    """Deletes all vector chunks associated with a specific document in a workspace."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE workspace_id = %s AND filename = %s;",
                    (workspace_id, filename),
                )
                return cur.rowcount
    except Exception as e:
        logger.error(
            f"Database error deleting document '{filename}' in workspace '{workspace_id}': {e}"
        )
        raise


if __name__ == "__main__":
    init_db()
