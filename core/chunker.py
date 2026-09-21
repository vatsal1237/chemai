"""
Markdown-aware text chunker for scientific papers.
Splits on headings and page breaks while preserving figure description blocks
as atomic chunks with metadata.
"""

import re
from dataclasses import dataclass, field
from config.settings import MAX_CHUNK_CHARS, OVERLAP_CHARS


@dataclass
class Chunk:
    """A single text chunk with metadata for retrieval."""
    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        page = self.metadata.get("page", "?")
        heading = self.metadata.get("heading", "")
        return f"Chunk(page={page}, heading='{heading}', len={len(self.text)})"


def _extract_page_number(text: str) -> int | None:
    """Extract page number from <!-- Page N --> comment."""
    match = re.search(r"<!--\s*Page\s+(\d+)\s*-->", text)
    return int(match.group(1)) if match else None


def _split_into_sections(markdown: str) -> list[dict]:
    """
    Split markdown into sections based on headings (##, #) and page breaks (---).
    Each section carries its heading and page number.
    """
    sections = []
    current_page = None
    current_heading = ""
    current_lines = []

    for line in markdown.split("\n"):
        # Track page markers
        page_match = re.match(r"<!--\s*Page\s+(\d+)\s*-->", line.strip())
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # Skip page separator lines
        if line.strip() == "---":
            if current_lines:
                sections.append({
                    "text": "\n".join(current_lines).strip(),
                    "heading": current_heading,
                    "page": current_page,
                })
                current_lines = []
            continue

        # Skip markdown code fences (```markdown)
        if line.strip().startswith("```"):
            continue

        # Detect headings
        heading_match = re.match(r"^(#{1,3})\s+(.+)", line.strip())
        if heading_match:
            # Save previous section
            if current_lines:
                sections.append({
                    "text": "\n".join(current_lines).strip(),
                    "heading": current_heading,
                    "page": current_page,
                })
                current_lines = []
            current_heading = heading_match.group(2).strip()

        current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        sections.append({
            "text": "\n".join(current_lines).strip(),
            "heading": current_heading,
            "page": current_page,
        })

    return [s for s in sections if s["text"].strip()]


def _split_long_section(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """
    Split a long section into sub-chunks at sentence boundaries
    with overlap for context continuity.
    """
    if len(text) <= max_chars:
        return [text]

    # Split at sentence boundaries (period followed by space/newline)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_len + sentence_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Keep overlap from the end of current chunk
            overlap_text = " ".join(current_chunk)
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap_chars:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)

            current_chunk = overlap_sentences + [sentence]
            current_len = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_markdown(markdown: str) -> list[Chunk]:
    """
    Split parsed Markdown into retrieval-optimized chunks.

    Strategy:
    1. Split on headings and page breaks → sections
    2. Keep figure description blocks as atomic chunks
    3. Split oversized sections at sentence boundaries with overlap
    4. Attach metadata (page, heading, chunk_index) to each chunk

    Args:
        markdown: Full parsed Markdown text.

    Returns:
        List of Chunk objects with text and metadata.
    """
    sections = _split_into_sections(markdown)
    chunks = []
    chunk_idx = 0

    for section in sections:
        text = section["text"].strip()
        if not text:
            continue

        # Check if this is a figure description block — keep atomic
        is_figure = bool(re.search(
            r"\*\*\[Figure.*?Description\].*?\*\*", text, re.IGNORECASE
        )) or text.startswith("**[Figure")

        if is_figure or len(text) <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(
                text=text,
                metadata={
                    "page": section["page"],
                    "heading": section["heading"],
                    "chunk_index": chunk_idx,
                    "is_figure": is_figure,
                },
            ))
            chunk_idx += 1
        else:
            # Split long sections
            sub_texts = _split_long_section(text, MAX_CHUNK_CHARS, OVERLAP_CHARS)
            for sub_text in sub_texts:
                chunks.append(Chunk(
                    text=sub_text,
                    metadata={
                        "page": section["page"],
                        "heading": section["heading"],
                        "chunk_index": chunk_idx,
                        "is_figure": False,
                    },
                ))
                chunk_idx += 1

    print(f"[+] Chunked into {len(chunks)} chunks", flush=True)
    return chunks
