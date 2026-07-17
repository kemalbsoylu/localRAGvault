import os
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def get_db_connection():
    """Establishes a connection to the database and registers the vector type."""
    # Pulling credentials securely from the environment
    conn = psycopg.connect(
        dbname=os.getenv("DB_NAME", "localragvault"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        autocommit=True
    )
    # Register pgvector so psycopg knows how to handle vector types
    register_vector(conn)
    return conn

def init_db():
    """Initializes the database schema."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding vector(768),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("Database initialized successfully. 'documents' table is ready.")

if __name__ == "__main__":
    # Test the connection and create the table
    init_db()
