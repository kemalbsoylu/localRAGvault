import os
import pytest
import psycopg

# OVERRIDE THE DATABASE NAME GLOBALLY
os.environ["DB_NAME"] = "localragvault_test"

from core.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
from core.database import get_db_connection, init_db


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    Runs ONCE per test session.
    Automatically provisions the test database and vector extension if missing.
    """
    try:
        # 1. Connect to default db to create the test database
        conn = psycopg.connect(
            dbname="postgres", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT, autocommit=True
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'localragvault_test'")
            if not cur.fetchone():
                print("\nProvisioning isolated test database: localragvault_test...")
                cur.execute("CREATE DATABASE localragvault_test")
        conn.close()

        # 2. Connect to the test database to check/add the vector extension
        conn_test = psycopg.connect(
            dbname="localragvault_test", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT, autocommit=True
        )
        with conn_test.cursor() as cur:
            # Query the system catalog to see if 'vector' is already installed
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if not cur.fetchone():
                try:
                    cur.execute("CREATE EXTENSION vector")
                except psycopg.errors.InsufficientPrivilege:
                    pytest.exit(
                        "\n❌ Missing superuser permissions to install pgvector.\n"
                        "Run this command in your terminal once to fix it:\n"
                        "sudo -u postgres psql -d localragvault_test -c \"CREATE EXTENSION vector;\""
                    )
        conn_test.close()

        # 3. Initialize our application tables
        init_db()

    except Exception as e:
        pytest.exit(f"Critical failure provisioning the test database: {e}")


@pytest.fixture(autouse=True)
def clean_database():
    """
    Runs BEFORE AND AFTER EVERY test.
    Ensures that tests do not pollute each other's vector space.
    """
    yield

    # After the test finishes, wipe the table completely clean
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
