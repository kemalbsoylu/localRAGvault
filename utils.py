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


def get_embedding(text: str, model_name: str = "nomic-embed-text") -> list[float]:
    """
    Generates a vector embedding for the given text using local Ollama.
    """
    try:
        response = ollama.embeddings(model=model_name, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []
