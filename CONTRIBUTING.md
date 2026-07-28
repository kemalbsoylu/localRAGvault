# Contributing to localRAGvault

Thank you for your interest in contributing to **localRAGvault**! Our main goal is to build solid, readable, and defensible software. Whether you are fixing a bug, adding a new document extractor, improving retrieval physics, or refining container architectures, your contributions are welcome.

---

## Development Setup

We support two primary development environments: containerized development using Docker Compose (ideal for testing systems integration) and standard native local development using `uv` (ideal for rapid core logic hacking).

### 1. Fork and Clone the Repository
```bash
git clone https://github.com/your-username/localRAGvault.git
cd localRAGvault
```

### Option A: Standard Native Local Setup (Recommended for Core Logic)
We strictly use **`uv`** for Python package and environment management to guarantee deterministic builds. For full step-by-step database provisioning and split terminal launch commands, refer directly to our **[Native Local Development Guide](DEVELOPMENT.md#native-local-development-setup-without-docker)**.
1. Ensure your local PostgreSQL instance is running with `pgvector` enabled.
2. Copy the configuration template: `cp .env.example .env`
3. Sync environment dependencies: `uv sync`

### Option B: Containerized Setup (Recommended for Systems/DevOps)
If you are testing network isolation, Dockerfile multi-stage layers, or UI deployment behavior:
1. Ensure Ollama is configured on your OS to allow container connections (`OLLAMA_HOST=0.0.0.0`).
2. Build and launch the container stack from source:
   ```bash
   docker compose up -d --build
   ```

---

## Engineering Standards

To keep the codebase maintainable, secure, and accessible for future engineers, we enforce strict quality gates:

1. **Defensive Type Safety:** Always use Python 3.12+ type annotations for function arguments and return types (`List[str]`, `Optional[dict]`, etc.). Avoid `Any` unless dealing with arbitrary JSON payloads.
2. **Pydantic Contracts:** All API inputs and outputs must pass through Pydantic v2 schemas defined in `core/schemas.py`. Never accept raw dictionaries in endpoint routes.
3. **Domain Isolation:** Keep route handlers in `api.py` thin. Place database queries in `database.py`, parsing logic in `extractors.py`, and LLM mechanics in `utils.py`.
4. **Container Hygiene:** When modifying `Dockerfile` or `docker-compose.yml`, preserve our multi-stage build structure. Do not introduce unnecessary layer bloating or run containers as root where preventable.
5. **No Silent Failures:** Use our centralized logger (`from core.logging_config import logger`) to capture warnings and errors with clear domain context.

---

## Automated CI/CD & Quality Verification

Whenever you push code or open a Pull Request against our `main` branch, our automated GitHub Actions workflow (`ci.yml`) will execute:
* A live `pgvector/pgvector:pg16` database service container for isolated testing.
* Automated code formatting and linting verification via **Ruff**.
* Static type checking via **Ty**.
* Full unit test execution via **Pytest** using our CI-optimized SDK mocking architecture.

### Running Pre-Commit Checks Locally
Before submitting a pull request, you **must** ensure the verification pipeline passes on your local machine:

```bash
uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest
```

---

## Git Workflow & Submitting Pull Requests

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Write tests for your changes:**
   * Add mocked unit tests for any new utility functions, extractors, or API endpoints.
   * If adding functionality that interacts directly with live Ollama daemons, tag the test explicitly with `@pytest.mark.integration`.
3. **Verify Locally:** Run the pre-commit verification suite shown above.
4. **Submit your PR:**
   * Push your branch to your fork and open a Pull Request against our `main` branch.
   * Provide a clear summary of what your PR changes and the architectural trade-offs behind your decisions.
   * Ensure all GitHub Actions CI checks turn green!

