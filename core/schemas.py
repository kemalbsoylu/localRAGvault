from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator

from core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL


def normalize_tag(value: str) -> str:
    """Helper to ensure models have a tag"""
    return value if ":" in value else f"{value}:latest"


class WorkspaceCreate(BaseModel):
    name: str = Field(..., description="Human-readable workspace name.")
    embedding_model: str = Field(..., description="The embedding model locked to this workspace.")

    @field_validator("embedding_model", mode="before")
    @classmethod
    def enforce_model_tag(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalize_tag(value)
        return value


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    embedding_model: str
    dimension: int


class SearchQuery(BaseModel):
    workspace_id: str = Field(..., description="Target workspace to search within.")
    query: str = Field(..., description="The query string used for matching.")
    thread_id: Optional[str] = Field(
        default=None, description="Optional thread ID to continue an existing conversation."
    )
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
            return normalize_tag(value)
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
    workspace_id: str
    query: str
    embedding_model: str
    results: List[SearchResultCard]


class RAGQueryResponse(BaseModel):
    workspace_id: str
    thread_id: str
    query: str
    answer: str
    generation_model: str
    embedding_model: str
    sources: List[DocumentSource]


class ThreadCard(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_query: str
    last_answer: str
    model_used: str
    sources: List[DocumentSource]


class ThreadListResponse(BaseModel):
    workspace_id: str
    threads: List[ThreadCard]


class MessageCard(BaseModel):
    id: int
    role: str
    content: str
    sources: List[DocumentSource]
    model_used: str
    created_at: str


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    messages: List[MessageCard]


class IngestionResponse(BaseModel):
    status: str
    workspace_id: str
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
