import ollama


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Splits text into chunks of a specific size, with a defined overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move the start forward, but step back by the overlap amount
        start += (chunk_size - overlap)

    return chunks


def get_embedding(text: str, model_name: str = "embeddinggemma") -> list[float]:
    """
    Generates a vector embedding for the given text using local Ollama.
    """
    try:
        response = ollama.embeddings(model=model_name, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def generate_answer(
    query: str, context_chunks: list[str], model_name: str = "gemma3"
) -> dict:
    """
    Sends the retrieved context and user query to the local LLM.
    Returns a dict with the text answer and a boolean indicating if a valid answer was found.
    """
    # Combine the retrieved chunks into a single string
    context_text = "\n---\n".join(context_chunks)

    # Define the exact fallback string here as a single source of truth
    fallback_msg = "I cannot answer this based on the provided documents."

    prompt = f"""You are a helpful, precise assistant. Answer the user's question using ONLY the provided context. 
If the answer is not contained in the context, say exactly: "{fallback_msg}" Do not use outside knowledge.

Context:
{context_text}

Question:
{query}

Answer:"""

    try:
        # Call Ollama to generate the text
        response = ollama.generate(model=model_name, prompt=prompt)
        answer_text = response["response"].strip()

        is_valid = fallback_msg not in answer_text

        return {"text": answer_text, "is_valid": is_valid}

    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            "text": "Sorry, I encountered an internal error while generating the response.",
            "is_valid": False,
        }


def get_available_models() -> dict:
    """Fetches a list of installed models directly from local Ollama."""
    try:
        models_response = ollama.list()
        # Extract just the model names from the response
        model_names = [model["model"] for model in models_response["models"]]
        return {"status": "success", "models": model_names}
    except Exception as e:
        print(f"Error fetching models: {e}")
        return {"status": "error", "models": []}
