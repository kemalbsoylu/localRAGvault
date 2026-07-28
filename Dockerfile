# ==========================================
# Stage 1: Base Environment & Tooling
# ==========================================
FROM python:3.12-slim AS base

# Copy uv package manager binary directly from Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configure working directory and Python runtime behavior
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install essential system utilities required by psycopg[binary] and network checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# Stage 2: Dependency Layer Caching
# ==========================================
FROM base AS dependencies

# Copy dependency files first to exploit Docker cache layers
# (Changes to source code won't trigger re-downloading packages)
COPY pyproject.toml uv.lock* ./

# Install production dependencies without dev/test packages
RUN uv sync --no-dev --no-install-project

# Copy application source code
COPY . .

# Install the localRAGvault project package itself
RUN uv sync --no-dev

# ==========================================
# Stage 3: FastAPI Backend Target
# ==========================================
FROM base AS api

# Copy the compiled virtual environment and source code from the dependency stage
COPY --from=dependencies /app /app
WORKDIR /app

# Expose FastAPI REST port
EXPOSE 8000

# Launch Uvicorn server
ENTRYPOINT ["uvicorn", "core.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ==========================================
# Stage 4: Streamlit UI Target
# ==========================================
FROM base AS ui

# Copy the compiled virtual environment and source code from the dependency stage
COPY --from=dependencies /app /app
WORKDIR /app

# Expose Streamlit dashboard port
EXPOSE 8501

# Launch Streamlit server with container-safe bindings
ENTRYPOINT ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
