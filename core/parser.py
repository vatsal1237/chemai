"""
PDF → Markdown parser.
Wraps parse_paper_qwen_vl.py and caches results to avoid re-parsing.
"""

import hashlib
from pathlib import Path

from config.settings import PARSED_DIR, OLLAMA_BASE_URL, VISION_MODEL, PARSE_DPI

# Import functions from the existing parser
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parse_paper_qwen_vl import (
    extract_pdf_with_qwen_vl,
    download_pdf,
    parse_page_selection,
)


def _cache_key(pdf_path: str, pages: str | None) -> str:
    """Generate a deterministic cache filename from PDF path and page selection."""
    pdf_hash = hashlib.md5(Path(pdf_path).resolve().read_bytes()[:4096]).hexdigest()[:12]
    page_tag = pages.replace(",", "_").replace("-", "to") if pages else "all"
    stem = Path(pdf_path).stem
    return f"{stem}_{pdf_hash}_p{page_tag}.md"


def parse_pdf(pdf_path: str, pages: str | None = None, force: bool = False) -> str:
    """
    Parse a PDF into Markdown using Qwen2.5-VL.

    Args:
        pdf_path: Path to the PDF file.
        pages: Page range string (e.g. '4-5', '1,3,5-7') or None for all pages.
        force: If True, re-parse even if cached result exists.

    Returns:
        Extracted Markdown string.
    """
    cache_name = _cache_key(pdf_path, pages)
    cache_path = PARSED_DIR / cache_name

    # Return cached if available
    if cache_path.exists() and not force:
        print(f"[+] Using cached parsed output: {cache_path}", flush=True)
        return cache_path.read_text(encoding="utf-8")

    # Parse fresh
    print(f"[+] Parsing PDF with Qwen2.5-VL (DPI={PARSE_DPI})...", flush=True)
    output_path = str(cache_path)

    markdown = extract_pdf_with_qwen_vl(
        pdf_path=pdf_path,
        output_markdown_path=output_path,
        model=VISION_MODEL,
        api_base=OLLAMA_BASE_URL,
        page_number=pages,
    )

    print(f"[+] Parsed output cached at: {cache_path}", flush=True)
    return markdown
