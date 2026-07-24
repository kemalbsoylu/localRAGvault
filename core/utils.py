import os
import shutil
from pathlib import Path
from typing import List, Optional

import ollama

from core.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    UPLOAD_DIR,
)
from core.logging_config import logger
from core.schemas import LLMInternalResponse


def normalize_model_name(model_name: str) -> str:
    """Ensures model names always carry an explicit tag (defaulting to ':latest' if omitted)."""
    if not model_name:
        return model_name
    return model_name if ":" in model_name else f"{model_name}:latest"


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Splits text into chunks of a specific size, with a defined overlap."""
    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def get_embedding(text: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    """Generates a vector embedding for the given text using local Ollama."""
    target_model = normalize_model_name(model_name)
    try:
        response = ollama.embeddings(model=target_model, prompt=text)
        return response["embedding"]
    except Exception as e:
        logger.error(f"Ollama vector embedding engine failure [{target_model}]: {e}")
        raise


def get_embeddings_batch(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 50,
) -> List[List[float]]:
    """Generates vector embeddings for a list of texts in batches using local Ollama."""
    if not texts:
        return []

    target_model = normalize_model_name(model_name)
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = ollama.embed(model=target_model, input=batch)
            all_embeddings.extend(response["embeddings"])
        except Exception as e:
            logger.error(f"Ollama batch vector embedding failure [{target_model}]: {e}")
            raise

    return all_embeddings


def generate_answer(
    query: str,
    context_chunks: List[dict],
    model_name: str = DEFAULT_GENERATION_MODEL,
    chat_history: Optional[List[dict]] = None,
) -> LLMInternalResponse:
    """Sends retrieved context, enriched chat history, and user query to the local LLM using optimized attention placement."""
    target_model = normalize_model_name(model_name)
    fallback_msg = "I cannot answer this based on the provided documents."

    # 1. Format active context chunks with clear visual boundaries
    formatted_context_blocks = []
    for item in context_chunks:
        block = (
            f"[Document: {item['filename']} | Chunk #{item.get('chunk_index', '?')}]\n"
            f"{item['content']}"
        )
        formatted_context_blocks.append(block)

    context_text = "\n\n---\n\n".join(formatted_context_blocks)

    # 2. Format conversation history and inject historical source attribution
    history_block = ""
    if chat_history:
        formatted_turns = []
        for msg in chat_history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            turn_text = f"{role_label}: {msg['content']}"

            # Inject historical sources for assistant turns so the LLM remembers its citations
            if msg["role"] == "assistant" and msg.get("sources"):
                source_labels = [
                    f"{s['filename']} (Chunk #{s.get('chunk_index', '?')})"
                    for s in msg["sources"][:3]
                ]
                if source_labels:
                    turn_text += f"\n  [Cited Documents: {', '.join(source_labels)}]"

            formatted_turns.append(turn_text)

        if formatted_turns:
            history_str = "\n".join(formatted_turns)
            history_block = f"\n### Previous Conversation History:\n{history_str}\n"

    # 3. Assemble prompt: Rules -> Memory -> Active Context -> Question (Optimal Causal Order)
    prompt = f"""You are a knowledgeable, analytical assistant for a private document vault.
Your task is to answer the user's question by synthesizing and explaining information from the provided document chunks and conversation history.

### Strict Operating Rules:
1. Ground your reasoning strictly in the provided context chunks and conversation history. Do NOT use outside knowledge or assume facts not directly supported by the text.
2. When referencing information from specific documents, cite the document name inline where appropriate.
3. Synthesize and explain concepts in clear, natural language—do not copy-paste raw sentences verbatim unless quoting specific data or technical terms.
4. If the user asks a follow-up question or a meta-question about previous answers (e.g., "which document did you find that in?", "can you summarize what we just discussed?"), answer directly using the Previous Conversation History and its [Cited Documents] tags.
5. If the context contains relevant information that only partially answers the question, explain what the documents reveal and note what details are missing.
6. If the provided context and conversation history are completely irrelevant or contain no information to answer the question, respond EXACTLY with this string: "{fallback_msg}"
{history_block}
### Active Retrieved Document Context:
{context_text}

### Current User Question:
{query}

### Answer:"""

    try:
        response = ollama.generate(model=target_model, prompt=prompt)
        answer_text = response["response"].strip()
        is_valid = fallback_msg not in answer_text

        return LLMInternalResponse(text=answer_text, is_valid=is_valid)

    except ollama.ResponseError as e:
        logger.error(
            f"Ollama API ResponseError under [{target_model}] (status code: {e.status_code}): {e.error}"
        )

        if e.status_code == 400:
            friendly_msg = f"Bad request (status code: 400): Invalid parameters or model payload for '{target_model}'."
        elif e.status_code == 401:
            friendly_msg = (
                "Authentication required (status code: 401). Run 'ollama signin' in your terminal."
            )
        elif e.status_code == 403:
            friendly_msg = f"Subscription required for model '{target_model}' (status code: 403). Upgrade access at https://ollama.com/upgrade"
        elif e.status_code == 404:
            friendly_msg = f"Model '{target_model}' not found (status code: 404). Run 'ollama pull {target_model}' first."
        elif e.status_code == 410:
            friendly_msg = (
                f"Model '{target_model}' has been retired by its provider (status code: 410)."
            )
        elif e.status_code == 429:
            friendly_msg = (
                "Too many requests (status code: 429). Rate limit exceeded on Ollama Cloud."
            )
        elif e.status_code == 500:
            friendly_msg = f"Internal server error (status code: 500): The local engine process crashed while running model '{target_model}'."
        elif e.status_code == 502:
            friendly_msg = f"Bad gateway (status code: 502): Could not reach cloud model endpoints for '{target_model}'."
        else:
            friendly_msg = f"Ollama service error (status code: {e.status_code}): {e.error}"

        return LLMInternalResponse(text=friendly_msg, is_valid=False)

    except Exception as e:
        logger.error(f"Unexpected execution error under [{target_model}]: {e}")
        return LLMInternalResponse(
            text="Connection to local Ollama daemon failed. Ensure the Ollama service is running locally.",
            is_valid=False,
        )


def get_available_models() -> List[str]:
    """Fetches a list of installed models directly from local Ollama."""
    try:
        response = ollama.list()
        return [normalize_model_name(m.model) for m in response.models if m.model is not None]
    except Exception as e:
        logger.error(f"Failed to fetch model catalog from local Ollama service: {e}")
        raise


def ensure_default_models_exist() -> None:
    """Verifies required Ollama models and auto-pulls them if missing."""
    logger.info("Verifying required Ollama models...")
    try:
        available_models = get_available_models()
        required_models = [DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL]

        for model in required_models:
            target_model = normalize_model_name(model)

            if target_model not in available_models:
                logger.info(
                    f"Model '{target_model}' missing locally. Initiating auto-pull (this may take a few minutes)..."
                )
                try:
                    ollama.pull(target_model)
                    logger.info(f"Successfully downloaded and registered '{target_model}'.")
                except Exception as pull_err:
                    logger.error(
                        f"Failed to pull '{target_model}': {pull_err}. Proceeding with startup."
                    )
            else:
                logger.info(f"Model '{target_model}' is already available.")

    except Exception as e:
        logger.error(f"Failed to communicate with local Ollama daemon during startup: {e}")


def save_physical_file(workspace_id: str, filename: str, content_bytes: bytes) -> Path:
    """Saves raw bytes to the physical disk inside the workspace directory."""
    workspace_dir = UPLOAD_DIR / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)

    physical_file_path = workspace_dir / filename

    with open(physical_file_path, "wb") as f:
        f.write(content_bytes)

    return physical_file_path


def delete_workspace_files(workspace_id: str) -> None:
    """Removes the entire physical workspace directory and all its contents from disk."""
    workspace_dir = UPLOAD_DIR / workspace_id
    try:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir, ignore_errors=True)
            logger.info(f"Physical directory removed from disk: {workspace_dir}")
    except Exception as e:
        logger.error(f"Failed to delete physical directory '{workspace_dir}': {e}")


def delete_physical_file(workspace_id: str, filename: str) -> bool:
    """Removes a specific physical file from the workspace directory on disk."""
    file_path = UPLOAD_DIR / workspace_id / filename
    try:
        if file_path.exists():
            os.remove(file_path)
            logger.info(f"Physical file removed from disk: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete physical file '{file_path}': {e}")
        raise
