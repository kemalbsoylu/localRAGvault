import csv
import io
import json
from typing import Any, List

import docx
import pypdf

from core.logging_config import logger


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Dispatches file bytes to the appropriate format-specific parser.
    Returns cleaned UTF-8 text ready for chunking.
    """
    lower_name = filename.lower()
    try:
        if lower_name.endswith((".txt", ".md")):
            return _extract_txt(file_bytes, filename)
        elif lower_name.endswith(".pdf"):
            return _extract_pdf(file_bytes, filename)
        elif lower_name.endswith(".docx"):
            return _extract_docx(file_bytes, filename)
        elif lower_name.endswith(".csv"):
            return _extract_csv(file_bytes, filename)
        elif lower_name.endswith(".json"):
            return _extract_json(file_bytes, filename)
        else:
            raise ValueError(
                f"Unsupported file format: '{filename}'. Supported extensions: .txt, .md, .pdf, .docx, .csv, .json"
            )
    except Exception as e:
        logger.error(f"Extraction failure for file '{filename}': {e}")
        raise


def _extract_txt(file_bytes: bytes, filename: str) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError as err:
        logger.warning(
            f"UTF-8 decode failed for '{filename}', attempting fallback latin-1 decoding..."
        )
        try:
            return file_bytes.decode("latin-1")
        except Exception:
            raise ValueError(f"File '{filename}' could not be decoded as text.") from err


def _extract_pdf(file_bytes: bytes, filename: str) -> str:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    page_texts: List[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        cleaned_text = text.strip()
        if cleaned_text:
            # Explicit page boundaries prevent chunk fragmentation across unmapped breaks
            page_texts.append(f"--- [Page {i + 1}] ---\n{cleaned_text}")

    if not page_texts:
        raise ValueError(
            f"No extractable text found in PDF '{filename}'. It may be a scanned image."
        )
    return "\n\n".join(page_texts)


def _extract_docx(file_bytes: bytes, filename: str) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    content_parts: List[str] = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            content_parts.append(text)

    # Extract structured tables into pipe-delimited text rows
    for table in doc.tables:
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_data:
                content_parts.append(" | ".join(row_data))

    if not content_parts:
        raise ValueError(f"No extractable text found in DOCX '{filename}'.")
    return "\n\n".join(content_parts)


def _extract_csv(file_bytes: bytes, filename: str) -> str:
    text_stream = io.StringIO(file_bytes.decode("utf-8", errors="replace"))
    reader = csv.DictReader(text_stream)

    if not reader.fieldnames:
        raise ValueError(f"CSV file '{filename}' appears to be empty or missing headers.")

    sentences: List[str] = []
    for row_idx, row in enumerate(reader, start=1):
        row_statements = [
            f"{key.strip()}: {val.strip()}" for key, val in row.items() if key and val
        ]
        if row_statements:
            sentences.append(f"[Row {row_idx}] " + ", ".join(row_statements) + ".")

    if not sentences:
        raise ValueError(f"No valid data rows found in CSV '{filename}'.")
    return "\n".join(sentences)


def _extract_json(file_bytes: bytes, filename: str) -> str:
    text = file_bytes.decode("utf-8", errors="replace")
    data = json.loads(text)

    def _flatten_to_sentences(obj: Any, prefix: str = "") -> List[str]:
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                key_label = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, (dict, list)):
                    lines.extend(_flatten_to_sentences(v, key_label))
                else:
                    lines.append(f"{key_label}: {v}")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                item_label = f"{prefix}[{idx}]"
                if isinstance(item, (dict, list)):
                    lines.extend(_flatten_to_sentences(item, item_label))
                else:
                    lines.append(f"{item_label}: {item}")
        else:
            lines.append(f"{prefix}: {obj}")
        return lines

    if isinstance(data, list):
        formatted_items = []
        for idx, item in enumerate(data, start=1):
            if isinstance(item, dict):
                statements = [
                    f"{k}: {v}" for k, v in item.items() if not isinstance(v, (dict, list))
                ]
                formatted_items.append(f"[Record {idx}] " + ", ".join(statements))
            else:
                formatted_items.append(f"[Record {idx}] {item}")
        return "\n".join(formatted_items)
    elif isinstance(data, dict):
        return "\n".join(_flatten_to_sentences(data))
    else:
        return str(data)
