"""
Parent-Document Retrieval Chunker for Scientific Papers.
Splits Markdown into large parent sections (preserving complete reaction mechanisms,
tables, and figure descriptions) and breaks them into small, dense, highly searchable
child blocks (~100 tokens) linked by parent_id.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from config.settings import CHILD_CHUNK_CHARS, CHILD_OVERLAP_CHARS, PARENT_MAX_CHARS


@dataclass
class ParentDoc:
    """A full contextual parent document section."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParentDoc":
        return cls(id=data["id"], text=data["text"], metadata=data.get("metadata", {}))


@dataclass
class ChildChunk:
    """A dense, searchable child chunk pointing to its parent."""
    id: str
    parent_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_into_parent_sections(markdown: str) -> list[dict]:
    """
    Split markdown into parent sections based on headings (##, #),
    page boundaries, and figure descriptions.
    """
    sections = []
    current_page = None
    current_heading = "Introduction"
    current_lines = []

    for line in markdown.split("\n"):
        # Track page markers
        page_match = re.match(r"<!--\s*Page\s+(\d+)\s*-->", line.strip())
        if page_match:
            current_page = int(page_match.group(1))
            continue

        if line.strip() == "---":
            if current_lines:
                sections.append({
                    "text": "\n".join(current_lines).strip(),
                    "heading": current_heading,
                    "page": current_page,
                })
                current_lines = []
            continue

        # Detect headings
        heading_match = re.match(r"^(#{1,4})\s+(.+)", line.strip())
        if heading_match:
            if current_lines:
                sections.append({
                    "text": "\n".join(current_lines).strip(),
                    "heading": current_heading,
                    "page": current_page,
                })
                current_lines = []
            current_heading = heading_match.group(2).strip()

        current_lines.append(line)

    if current_lines:
        sections.append({
            "text": "\n".join(current_lines).strip(),
            "heading": current_heading,
            "page": current_page,
        })

    # Filter empty sections and split oversized parents if needed
    parent_blocks = []
    for s in sections:
        text = s["text"].strip()
        if not text:
            continue

        # If a single section exceeds PARENT_MAX_CHARS, split at paragraph boundaries
        if len(text) > PARENT_MAX_CHARS:
            paragraphs = text.split("\n\n")
            curr_para_block = []
            curr_len = 0
            for p in paragraphs:
                p_len = len(p)
                if curr_len + p_len > PARENT_MAX_CHARS and curr_para_block:
                    parent_blocks.append({
                        "text": "\n\n".join(curr_para_block).strip(),
                        "heading": s["heading"],
                        "page": s["page"],
                    })
                    curr_para_block = [p]
                    curr_len = p_len
                else:
                    curr_para_block.append(p)
                    curr_len += p_len + 2
            if curr_para_block:
                parent_blocks.append({
                    "text": "\n\n".join(curr_para_block).strip(),
                    "heading": s["heading"],
                    "page": s["page"],
                })
        else:
            parent_blocks.append(s)

    return parent_blocks


def _split_into_child_blocks(text: str, max_chars: int = CHILD_CHUNK_CHARS, overlap_chars: int = CHILD_OVERLAP_CHARS) -> list[str]:
    """
    Split text into small, dense ~100-token child chunks (~350-400 chars)
    split at sentence or clause boundaries with slight overlap.
    """
    if len(text) <= max_chars:
        return [text]

    # Sentence boundary regex
    sentences = re.split(r'(?<=[.!?;\n])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)
        if current_len + s_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk).strip())
            # Overlap from end of current chunk
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
            current_len += s_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return [c for c in chunks if c.strip()]


def chunk_parent_child(
    markdown: str,
    source_id: str = "doc",
) -> tuple[dict[str, ParentDoc], list[ChildChunk]]:
    """
    Splits markdown into Parent Documents and associated Child Chunks.

    Args:
        markdown: Extracted markdown text of the paper.
        source_id: Unique identifier for the paper/document.

    Returns:
        Tuple of (parent_docstore: dict[parent_id, ParentDoc], child_chunks: list[ChildChunk]).
    """
    parent_sections = _split_into_parent_sections(markdown)
    parent_store: dict[str, ParentDoc] = {}
    child_chunks: list[ChildChunk] = []

    child_idx = 0
    for p_idx, sec in enumerate(parent_sections):
        parent_id = f"{source_id}_parent_{p_idx}"
        parent_text = sec["text"]
        is_figure = "**[Figure Description]:**" in parent_text or parent_text.startswith("**[Figure")

        p_meta = {
            "source_id": source_id,
            "page": sec.get("page") or 1,
            "heading": sec.get("heading") or "",
            "parent_index": p_idx,
            "is_figure": is_figure,
        }

        parent_doc = ParentDoc(id=parent_id, text=parent_text, metadata=p_meta)
        parent_store[parent_id] = parent_doc

        # Generate dense ~100 token child chunks from this parent section
        child_texts = _split_into_child_blocks(parent_text, CHILD_CHUNK_CHARS, CHILD_OVERLAP_CHARS)
        for c_text in child_texts:
            child_id = f"{source_id}_child_{child_idx}"
            child_meta = {
                "parent_id": parent_id,
                "source_id": source_id,
                "page": sec.get("page") or 1,
                "heading": sec.get("heading") or "",
                "child_index": child_idx,
            }
            child_chunks.append(ChildChunk(
                id=child_id,
                parent_id=parent_id,
                text=c_text,
                metadata=child_meta,
            ))
            child_idx += 1

    print(f"[+] Parent-Document chunking complete: {len(parent_store)} parent sections, {len(child_chunks)} searchable child chunks.", flush=True)
    return parent_store, child_chunks
