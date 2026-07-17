from pydantic import BaseModel


class SearchQuery(BaseModel):
    query: str
    top_k: int = 3  # Return the top 3 closest chunks by default
