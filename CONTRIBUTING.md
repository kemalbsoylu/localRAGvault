# Contributing to localRAGvault

Thank you for your interest in contributing to **localRAGvault**! Our main goal is to build solid, readable, and defensible software. Whether you are fixing a bug, adding a new document extractor, or improving retrieval physics, your contributions are welcome.

---

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/localRAGvault.git
   cd localRAGvault
   ```

2. **Sync the environment using `uv`:**
   We strictly use `uv` for package management to guarantee deterministic builds.
   ```bash
   uv sync
   ```

3. **Set up your test environment:**
   Ensure you have a local PostgreSQL instance running with the `pgvector` extension enabled. Copy the `.env.example` file:
   ```bash
   cp .env.example .env
   ```

---

## Engineering Standards

To keep the codebase maintainable and accessible for future engineers, we enforce strict quality gates:

1. **Defensive Type Safety:** Always use Python 3.12+ type annotations for function arguments and return types (`List[str]`, `Optional[dict]`, etc.). Avoid `Any` unless dealing with arbitrary JSON payloads.
2. **Pydantic Contracts:** All API inputs and outputs must pass through Pydantic v2 schemas defined in `core/schemas.py`. Never accept raw dictionaries in endpoint routes.
3. **Domain Isolation:** Keep route handlers in `api.py` thin. Place database queries in `database.py`, parsing logic in `extractors.py`, and LLM mechanics in `utils.py`.
4. **No Silent Failures:** Use our centralized logger (`from core.logging_config import logger`) to capture warnings and errors with clear domain context.

---

## Git Workflow & Pull Requests

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests for your changes:**
   * Add mocked unit tests for any new utility functions or endpoints.
   * If adding functionality that interacts directly with Ollama, tag the test with `@pytest.mark.integration`.

3. **Run the pre-commit verification suite:**
   Before pushing your branch, you **must** ensure the full verification pipeline passes locally:
   ```bash
   uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest
   ```

4. **Submit a Pull Request:**
   * Push your branch to your fork and open a Pull Request against our `main` branch.
   * Provide a clear summary of what your PR changes and why the design trade-offs were chosen.
   * Ensure all automated CI checks pass.
