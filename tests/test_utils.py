import pytest

from core.utils import chunk_text, normalize_model_name


def test_normalize_model_name() -> None:
    """Verify models without explicit tags default to ':latest' while existing tags remain untouched."""
    assert normalize_model_name("gemma4") == "gemma4:latest"
    assert normalize_model_name("embeddinggemma:latest") == "embeddinggemma:latest"
    assert normalize_model_name("custom-model:v2") == "custom-model:v2"
    assert normalize_model_name("") == ""


def test_chunk_text_standard_split() -> None:
    """Verify character chunking splits text accurately according to size and overlap parameters."""
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=50)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_invalid_overlap_raises() -> None:
    """Verify setting chunk overlap greater than or equal to chunk size raises a ValueError."""
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        chunk_text("Sample text content for boundary test.", chunk_size=100, overlap=100)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        chunk_text("Sample text content for boundary test.", chunk_size=100, overlap=150)
