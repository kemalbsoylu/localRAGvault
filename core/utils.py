from typing import List

import ollama

from core.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
)
from core.logging_config import logger
from core.schemas import LLMInternalResponse


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
    try:
        response = ollama.embeddings(model=model_name, prompt=text)
        return response["embedding"]
    except Exception as e:
        logger.error(f"Ollama vector embedding engine failure [{model_name}]: {e}")
        return []


def generate_answer(
    query: str, context_chunks: List[str], model_name: str = DEFAULT_GENERATION_MODEL
) -> LLMInternalResponse:
    """Sends the retrieved context and user query to the local LLM using a strict validation gate."""
    context_text = "\n---\n".join(context_chunks)
    fallback_msg = "I cannot answer this based on the provided documents."

    prompt = f"""You are a helpful, precise assistant. Answer the user's question using ONLY the provided context. 
If the answer is not contained in the context, say exactly: "{fallback_msg}" Do not use outside knowledge.

Context:
{context_text}

Question:
{query}

Answer:"""

    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        answer_text = response["response"].strip()
        is_valid = fallback_msg not in answer_text

        return LLMInternalResponse(text=answer_text, is_valid=is_valid)

    except ollama.ResponseError as e:
        logger.error(
            f"Ollama API ResponseError under [{model_name}] (status code: {e.status_code}): {e.error}"
        )

        if e.status_code == 400:
            friendly_msg = f"Bad request (status code: 400): Invalid parameters or model payload for '{model_name}'."
        elif e.status_code == 401:
            friendly_msg = (
                "Authentication required (status code: 401). Run 'ollama signin' in your terminal."
            )
        elif e.status_code == 403:
            friendly_msg = f"Subscription required for model '{model_name}' (status code: 403). Upgrade access at https://ollama.com/upgrade"
        elif e.status_code == 404:
            friendly_msg = f"Model '{model_name}' not found (status code: 404). Run 'ollama pull {model_name}' first."
        elif e.status_code == 410:
            friendly_msg = (
                f"Model '{model_name}' has been retired by its provider (status code: 410)."
            )
        elif e.status_code == 429:
            friendly_msg = (
                "Too many requests (status code: 429). Rate limit exceeded on Ollama Cloud."
            )
        elif e.status_code == 500:
            friendly_msg = f"Internal server error (status code: 500): The local engine process crashed while running model '{model_name}'."
        elif e.status_code == 502:
            friendly_msg = f"Bad gateway (status code: 502): Could not reach cloud model endpoints for '{model_name}'."
        else:
            friendly_msg = f"Ollama service error (status code: {e.status_code}): {e.error}"

        return LLMInternalResponse(text=friendly_msg, is_valid=False)

    except Exception as e:
        logger.error(f"Unexpected execution error under [{model_name}]: {e}")
        return LLMInternalResponse(
            text="Connection to local Ollama daemon failed. Ensure the Ollama service is running locally.",
            is_valid=False,
        )


def get_available_models() -> List[str]:
    """Fetches a list of installed models directly from local Ollama."""
    try:
        models_response = ollama.list()
        return [model["model"] for model in models_response["models"]]
    except Exception as e:
        logger.error(f"Failed to fetch model catalog from local Ollama service: {e}")
        return []
