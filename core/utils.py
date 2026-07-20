import ollama
from typing import List
from core.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from core.schemas import LLMInternalResponse
from core.logging_config import logger


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
    except Exception as e:
        logger.error(f"Ollama LLM generations breakdown under [{model_name}]: {e}")
        return LLMInternalResponse(
            text="Sorry, I encountered an internal error while generating the response.",
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
