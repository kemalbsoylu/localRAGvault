from pydantic import BaseModel


class SearchQuery(BaseModel):
    query: str
    top_k: int = 3
    embedding_model: str = "nomic-embed-text"
    generation_model: str = "gemma3"
