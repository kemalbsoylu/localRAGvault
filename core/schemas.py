from typing import Any, List

from pydantic import BaseModel, Field, field_validator

from core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL


class SearchQuery(BaseModel):
    query: str = Field(..., description="The query string used for matching.")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of context chunks to pull.")
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL, description="Target vector space model."
    )
    generation_model: str = Field(
        default=DEFAULT_GENERATION_MODEL, description="Target text generation model."
    )

    @field_validator("embedding_model", "generation_model", mode="before")
    @classmethod
    def enforce_model_tag(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value if ":" in value else f"{value}:latest"
        return value


class DocumentSource(BaseModel):
    filename: str
    similarity: float


class SearchResultCard(BaseModel):
    id: int
    filename: str
    content: str
    similarity: float


class VectorSearchResponse(BaseModel):
    query: str
    embedding_model: str
    results: List[SearchResultCard]


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    generation_model: str
    embedding_model: str
    sources: List[DocumentSource]


class IngestionResponse(BaseModel):
    status: str
    filename: str
    model_used: str
    chunks_saved: int


class ModelListResponse(BaseModel):
    status: str
    models: List[str]


class LLMInternalResponse(BaseModel):
    text: str
    is_valid: bool


class DocumentInventoryItem(BaseModel):
    filename: str
    file_path: str
    total_chunks: int


class WorkspaceInventoryResponse(BaseModel):
    workspace_id: str
    documents: List[DocumentInventoryItem]
