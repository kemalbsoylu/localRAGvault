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


if __name__ == "__main__":
    init_db()
