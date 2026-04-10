"""
PDF parsing and chunking for the official MTG comprehensive rules.

Strategy:
  1. Extract full text from the PDF.
  2. Split by MTG rule section numbers (e.g. "100.", "101.1.", "702.6a.").
  3. If no section markers are found, fall back to fixed 500-word chunks
     with 50-word overlap.
"""

import re
from pathlib import Path

from pypdf import PdfReader

from config import settings

# Matches rule numbers at the start of a line: e.g. "100.", "100.1.", "100.1a."
_SECTION_RE = re.compile(r"(?m)^(\d{3}\.[\da-z]*\.?)\s")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Read every page of a PDF and return the concatenated text.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Full text extracted from all pages.
    """
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def chunk_by_section(text: str) -> list[dict]:
    """
    Split text into chunks delimited by MTG rule section numbers.

    Args:
        text: Full rules text.

    Returns:
        List of dicts, each with 'text' and 'metadata' keys.
        Returns an empty list when no section markers are found.
    """
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return []

    chunks: list[dict] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        section_id = match.group(1)

        if section_text:
            chunks.append(
                {
                    "text": section_text,
                    "metadata": {"source": "rules", "section": section_id},
                }
            )

    return chunks


def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    Fallback chunker — splits text into fixed-size word windows with overlap.

    Args:
        text: Full text to chunk.
        chunk_size: Target number of words per chunk.
        overlap: Number of words to repeat between consecutive chunks.

    Returns:
        List of dicts, each with 'text' and 'metadata' keys.
    """
    words = text.split()
    chunks: list[dict] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "source": "rules",
                        "chunk_start": start,
                        "chunk_end": min(end, len(words)),
                    },
                }
            )

        next_start = end - overlap
        if next_start <= start:
            # Guard against infinite loop when overlap >= chunk_size
            next_start = start + 1
        start = next_start

    return chunks


def load_rules_chunks() -> list[dict]:
    """
    Parse the MTG rules PDF and return a list of text chunks.

    Attempts section-based splitting first; falls back to fixed-size
    chunking when the PDF layout does not contain standard rule numbers.

    Returns:
        List of chunk dicts with 'text' and 'metadata'.

    Raises:
        FileNotFoundError: When the PDF is missing from the configured path.
    """
    pdf_path = settings.RULES_PDF_PATH
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Rules PDF not found at '{pdf_path}'. "
            "Place mtg-rules.pdf in the project root before ingesting."
        )

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_by_section(text)

    if not chunks:
        chunks = chunk_fixed(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    return chunks
