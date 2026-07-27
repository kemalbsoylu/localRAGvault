from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from core.config import (
    DEFAULT_CHAT_HISTORY_LIMIT,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)


def normalize_tag(value: str) -> str:
    """Helper to ensure models have an explicit tag."""
    return value if ":" in value else f"{value}:latest"


class WorkspaceCreate(BaseModel):
    name: str = Field(..., description="Human-readable workspace name.")
    embedding_model: str = Field(..., description="The embedding model locked to this workspace.")
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=100, le=2000)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0, le=1000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0)
    chat_history_limit: int = Field(default=DEFAULT_CHAT_HISTORY_LIMIT, ge=1, le=20)
    system_prompt: Optional[str] = Field(default=None)

    @field_validator("embedding_model", mode="before")
    @classmethod
    def enforce_model_tag(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalize_tag(value)
        return value

    @model_validator(mode="after")
    def validate_overlap_bounds(self) -> "WorkspaceCreate":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        return self


class WorkspaceUpdate(BaseModel):
    chunk_size: Optional[int] = Field(default=None, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    chat_history_limit: Optional[int] = Field(default=None, ge=1, le=20)
    system_prompt: Optional[str] = None

    @model_validator(mode="after")
    def validate_overlap_bounds(self) -> "WorkspaceUpdate":
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        return self


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    embedding_model: str
    dimension: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    similarity_threshold: float
    chat_history_limit: int
    system_prompt: Optional[str] = None


class SearchQuery(BaseModel):
    workspace_id: str = Field(..., description="Target workspace to search within.")
    query: str = Field(..., description="The query string used for matching.")
    thread_id: Optional[str] = Field(
        default=None, description="Optional thread ID to continue an existing conversation."
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20, description="Override context chunks to pull."
    )
    similarity_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Override minimum similarity score."
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Optional override for LLM temperature."
    )
    chat_history_limit: Optional[int] = Field(
        default=None, ge=1, le=20, description="Override historical message context limit."
    )
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
    chunk_index: int
    similarity: float
    content: Optional[str] = Field(
        default=None, description="The text excerpt preview of the retrieved document chunk."
    )


class SearchResultCard(BaseModel):
    id: int
    filename: str
    chunk_index: int
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


class ThreadUpdate(BaseModel):
    title: str = Field(
        ..., min_length=1, max_length=100, description="New title for the conversation thread."
    )

    @field_validator("title", mode="after")
    @classmethod
    def strip_and_validate_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Thread title cannot be empty or consist only of whitespace.")
        return cleaned


class ThreadResponse(BaseModel):
    id: str
    workspace_id: str
    title: str


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
    total_count: int = 0
    limit: int = 5
    offset: int = 0
    has_more: bool = False


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
    total_count: int = 0
    limit: int = 10
    offset: int = 0
    has_more: bool = False


class IngestionResponse(BaseModel):
    status: str
    workspace_id: str
    filename: str
    model_used: str
    chunks_saved: int
    is_upsert: bool = False
    chunks_deleted: int = 0


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
    total_count: int = 0
    limit: int = 5
    offset: int = 0
    has_more: bool = False


class FileIngestionResult(BaseModel):
    filename: str
    status: str
    chunks_saved: int = 0
    error_message: Optional[str] = None


class BatchIngestionSummary(BaseModel):
    total_files: int
    successful: int
    upserts: int
    failed: int
    total_chunks_saved: int


class BatchIngestionResponse(BaseModel):
    status: str
    workspace_id: str
    model_used: str
    summary: BatchIngestionSummary
    results: List[FileIngestionResult]
