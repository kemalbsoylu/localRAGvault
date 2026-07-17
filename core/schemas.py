from pydantic import BaseModel


class SearchQuery(BaseModel):
    query: str
    top_k: int = 3
    embedding_model: str = "embeddinggemma"
    generation_model: str = "gemma3"
