"""
PDF parsing and chunking for the official MTG comprehensive rules.

Strategy:
  1. Extract full text from the PDF.
  2. Cut away the table of contents and credits; split the remainder into the
     numbered rules and the glossary.
  3. Group each rule (e.g. "702.19.") with its subrules ("702.19a", ...), and
     prefix every chunk with its section title ("702. Keyword Abilities") so a
     subrule never loses the context of what it is about.
  4. Rules too long for the embedding model's window are split between
     subrules, repeating the parent rule's opening words as a header.
  5. Each glossary term becomes its own chunk.
  6. If the document has no recognisable rule numbers, fall back to fixed
     word windows with overlap.
"""

import re
from pathlib import Path

from pypdf import PdfReader

from config import settings

# A rules entry starts a paragraph with a section ("702."), rule ("702.19.") or
# subrule ("702.19a") number followed by text. Wrapped cross-references such as
# "...see rule\n202.2." or "119.1d. In ..." do not match this shape.
_ENTRY_RE = re.compile(r"^(\d{3}\.(?:\d+\.|\d+[a-z])?)\s+\S")
# Chapter headings like "7. Additional Rules" carry no rule content.
_CHAPTER_RE = re.compile(r"^\d\.\s+[A-Z]")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n")
_GLOSSARY_RE = re.compile(r"(?m)^Glossary[ \t]*$")
_CREDITS_RE = re.compile(r"(?m)^Credits[ \t]*$")
_FIRST_RULE_RE = re.compile(r"(?m)^100\.1\.\s")
_FIRST_SECTION_RE = re.compile(r"(?m)^100\.\s")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Read every page of a PDF and return the concatenated text."""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines and unwrap hard line breaks inside each paragraph."""
    out = []
    for raw in _PARAGRAPH_SPLIT_RE.split(text):
        para = " ".join(line.strip() for line in raw.splitlines() if line.strip())
        if para:
            out.append(para)
    return out


def _split_document(text: str) -> tuple[str, str]:
    """
    Return (rules_body, glossary) with the table of contents and credits removed.

    The table of contents repeats every section heading, so the body is taken
    to start at the "100." heading immediately preceding the first "100.1." rule.
    When the expected landmarks are missing, the whole text is returned as the
    body and the glossary is empty.
    """
    first_rule = _FIRST_RULE_RE.search(text)
    if not first_rule:
        return text, ""
    sections_before = [m.start() for m in _FIRST_SECTION_RE.finditer(text, 0, first_rule.start())]
    start = sections_before[-1] if sections_before else first_rule.start()

    glossaries = [m for m in _GLOSSARY_RE.finditer(text) if m.start() > start]
    credits = [m for m in _CREDITS_RE.finditer(text) if m.start() > start]
    if not glossaries:
        return text[start:], ""
    glossary_start = glossaries[-1]
    glossary_end = credits[-1].start() if credits and credits[-1].start() > glossary_start.end() else len(text)
    return text[start : glossary_start.start()], text[glossary_start.end() : glossary_end]


def _parse_entries(body: str) -> list[tuple[str, str]]:
    """
    Parse the rules body into (rule_id, text) entries.

    Paragraphs that do not start with a rule number (examples, text continued
    across a page break) are appended to the preceding entry.
    """
    entries: list[tuple[str, str]] = []
    for para in _paragraphs(body):
        match = _ENTRY_RE.match(para)
        if match:
            entries.append((match.group(1).rstrip("."), para))
        elif _CHAPTER_RE.match(para):
            continue
        elif entries:
            rule_id, prev = entries[-1]
            entries[-1] = (rule_id, f"{prev}\n{para}")
    return entries


def _truncate_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " ..."


def _range_label(ids: list[str]) -> str:
    return ids[0] if len(ids) == 1 else f"{ids[0]}–{ids[-1]}"


def _chunk_rule_group(
    header: str,
    group: list[tuple[str, str]],
    max_words: int,
) -> list[dict]:
    """Chunk one rule plus its subrules, splitting between subrules if too long."""

    def make(ids: list[str], texts: list[str], prefix: str) -> dict:
        return {
            "text": "\n".join([prefix, *texts]),
            "metadata": {"source": "rules", "section": _range_label(ids)},
        }

    all_texts = [t for _, t in group]
    if len(" ".join([header, *all_texts]).split()) <= max_words:
        return [make([group[0][0]], all_texts, header)]

    # Too long: the parent rule leads the first window; later windows repeat
    # its opening words so each chunk still says which rule it belongs to.
    parent_id, parent_text = group[0]
    continuation = f"{header}\n{_truncate_words(parent_text, 30)}"
    chunks: list[dict] = []
    ids, texts, prefix = [parent_id], [parent_text], header
    budget = len(prefix.split()) + len(parent_text.split())

    for rule_id, text in group[1:]:
        n = len(text.split())
        if budget + n > max_words and texts:
            chunks.append(make(ids, texts, prefix))
            ids, texts, prefix = [], [], continuation
            budget = len(prefix.split())
        ids.append(rule_id)
        texts.append(text)
        budget += n

    if texts:
        chunks.append(make(ids, texts, prefix))
    return chunks


def _chunk_glossary(glossary: str) -> list[dict]:
    """One chunk per glossary term: the first line of each paragraph is the term."""
    chunks = []
    for raw in _PARAGRAPH_SPLIT_RE.split(glossary):
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        term, definition = lines[0], " ".join(lines[1:])
        chunks.append(
            {
                "text": f"Glossary — {term}: {definition}",
                "metadata": {"source": "rules", "glossary_term": term},
            }
        )
    return chunks


def chunk_by_section(text: str, max_words: int | None = None) -> list[dict]:
    """
    Split the comprehensive rules into rule-group and glossary chunks.

    Args:
        text: Full rules text.
        max_words: Soft word cap per rules chunk (defaults to
            settings.RULE_CHUNK_MAX_WORDS). A single subrule longer than this
            is kept whole.

    Returns:
        List of dicts, each with 'text' and 'metadata' keys.
        Returns an empty list when no rule numbers are found.
    """
    max_words = max_words or settings.RULE_CHUNK_MAX_WORDS
    body, glossary = _split_document(text)
    entries = _parse_entries(body)
    if not entries:
        return []

    chunks: list[dict] = []
    header = ""
    group: list[tuple[str, str]] = []

    def flush() -> None:
        if group:
            chunks.extend(_chunk_rule_group(header, group, max_words))
            group.clear()

    for rule_id, entry_text in entries:
        if "." not in rule_id:  # section heading, e.g. "702"
            flush()
            header = entry_text
        elif rule_id[-1].isdigit():  # rule, e.g. "702.19"
            flush()
            group.append((rule_id, entry_text))
        else:  # subrule, e.g. "702.19a"
            group.append((rule_id, entry_text))
    flush()

    return chunks + _chunk_glossary(glossary)


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

    Attempts structured rule/glossary chunking first; falls back to fixed-size
    chunking when the PDF layout does not contain standard rule numbers.

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
