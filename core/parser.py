"""
PDF -> Markdown parser using 100% Local Vision Models (Ollama: llama3.2-vision or qwen2.5-vl).
Rasterizes PDF pages using pdf2image (with fallback to pymupdf) and queries Ollama locally.
Zero Gemini API dependencies.
"""

import os
import io
import base64
import hashlib
from pathlib import Path
from typing import Callable, Optional
import requests
from PIL import Image

from config.settings import PARSED_DIR, OLLAMA_BASE_URL, VISION_MODEL, PARSE_DPI


def _cache_key(pdf_path: str, pages: str | None) -> str:
    """Generate a deterministic cache filename from PDF path and page selection."""
    pdf_bytes = Path(pdf_path).resolve().read_bytes()[:4096]
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()[:12]
    page_tag = pages.replace(",", "_").replace("-", "to") if pages else "all"
    stem = Path(pdf_path).stem
    return f"{stem}_{pdf_hash}_p{page_tag}.md"


def _rasterize_pdf_page(pdf_path: str, page_index: int, dpi: int = PARSE_DPI) -> str:
    """
    Rasterize a single PDF page to a base64 JPEG string.
    Tries pdf2image first (requires poppler), falls back cleanly to pymupdf (fitz).
    """
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_index + 1,
            last_page=page_index + 1,
        )
        if images:
            img = images[0]
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        # Fallback to PyMuPDF (always self-contained)
        pass

    import pymupdf
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("jpeg")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def _get_pdf_total_pages(pdf_path: str) -> int:
    """Get total page count of a PDF."""
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path)
        return int(info.get("Pages", 1))


def _query_local_vision_model(
    base64_image: str,
    model: str = VISION_MODEL,
    ollama_base: str = OLLAMA_BASE_URL,
) -> str:
    """
    Send rasterized page to local Ollama vision model (llama3.2-vision or qwen2.5vl).
    """
    prompt = (
        "You are an expert scientific document transcription assistant specializing in chemistry papers.\n"
        "Please transcribe this page into clean, structured Markdown.\n"
        "Strict Requirements:\n"
        "1. Follow natural multi-column reading order (left column first, then right column).\n"
        "2. Format all tables as clean Markdown tables with header separators.\n"
        "3. Enclose all chemical formulas, structures, and mathematical equations in LaTeX notation:\n"
        "   - Inline math: $...$ (e.g., $\\mathrm{C}_{60}$, $\\mathrm{H}_2\\mathrm{SO}_4$, $R^2 = 0.99$)\n"
        "   - Block equations: $$...$$\n"
        "4. Preserve all section headings (#, ##, ###).\n"
        "5. For EVERY figure, scheme, chart, diagram, or molecular structure, provide a comprehensive block:\n"
        "   **[Figure Description]:** <Detailed explanation of reaction mechanism, axes, labels, structures, catalysts>\n"
        "6. Do not include introductory conversational filler. Output only the transcribed markdown."
    )

    url = f"{ollama_base.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64_image],
            }
        ],
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "temperature": 0.1,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
        if response.status_code == 200:
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
        else:
            # Try fallback OpenAI-compatible endpoint
            alt_url = f"{ollama_base.rstrip('/')}/v1/chat/completions"
            alt_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                "temperature": 0.1,
            }
            alt_resp = requests.post(alt_url, json=alt_payload, timeout=300)
            if alt_resp.status_code == 200:
                return alt_resp.json()["choices"][0]["message"]["content"].strip()
            raise RuntimeError(f"Ollama Vision error ({response.status_code}): {response.text}")
    except Exception as exc:
        print(f"[!] Vision parsing request failed: {exc}", flush=True)
        raise


def parse_pdf_fast(pdf_path: str) -> str:
    """
    Rapidly extract structured Markdown from digital PDFs using PyMuPDF.
    Preserves page boundaries, section headings, and tables in ~1-2 seconds.
    """
    import pymupdf
    doc = pymupdf.open(pdf_path)
    full_markdown = []

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text").strip()
        if text:
            # Format headings and page markers
            lines = text.split("\n")
            formatted_lines = []
            for line in lines:
                l_strip = line.strip()
                if l_strip.isupper() and len(l_strip) > 3 and len(l_strip) < 60:
                    formatted_lines.append(f"\n## {l_strip}\n")
                else:
                    formatted_lines.append(line)
            page_text = "\n".join(formatted_lines)
            full_markdown.append(f"<!-- Page {i+1} -->\n{page_text}")
        else:
            full_markdown.append(f"<!-- Page {i+1} -->\n> [Scanned / Image Page]")

    doc.close()
    return "\n\n---\n\n".join(full_markdown)


def parse_pdf(
    pdf_path: str,
    pages: str | None = None,
    force: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    vision_model: str | None = None,
    force_vision: bool = False,
) -> str:
    """
    Parse a PDF into Markdown.
    If the document has digital text and force_vision is False, extracts text rapidly in 1-2s.
    If the document is scanned or force_vision is True, executes 100% local vision transcription via Ollama.
    """
    active_model = vision_model or VISION_MODEL
    cache_name = _cache_key(pdf_path, pages)
    cache_path = PARSED_DIR / cache_name

    if cache_path.exists() and not force:
        print(f"[+] Using cached parsed output: {cache_path}", flush=True)
        if progress_callback:
            progress_callback(1, 1)
        return cache_path.read_text(encoding="utf-8")

    # Check if PDF contains extractable text
    import pymupdf
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    has_text = any(len(doc[i].get_text("text").strip()) > 80 for i in range(min(3, total_pages)))
    doc.close()

    if has_text and not force_vision:
        print(f"[+] PDF contains digital text layer. Fast parsing {total_pages} page(s)...", flush=True)
        final_text = parse_pdf_fast(pdf_path)
        cache_path.write_text(final_text, encoding="utf-8")
        if progress_callback:
            progress_callback(total_pages, total_pages)
        return final_text

    total_pages = _get_pdf_total_pages(pdf_path)
    page_indices = list(range(total_pages))

    # Parse page range if specified
    if pages:
        selected = set()
        for part in pages.split(","):
            part = part.strip()
            if "-" in part:
                s, e = map(int, part.split("-"))
                selected.update(range(s - 1, min(e, total_pages)))
            elif part.isdigit():
                p = int(part) - 1
                if 0 <= p < total_pages:
                    selected.add(p)
        if selected:
            page_indices = sorted(list(selected))

    print(f"[+] Parsing {len(page_indices)} page(s) with local vision model '{active_model}'...", flush=True)
    full_markdown = []

    for idx, page_idx in enumerate(page_indices):
        page_num = page_idx + 1
        if progress_callback:
            progress_callback(idx + 1, len(page_indices))

        print(f"    Rasterizing & transcribing Page {page_num}/{total_pages}...", flush=True)
        try:
            b64_img = _rasterize_pdf_page(pdf_path, page_idx, dpi=PARSE_DPI)
            page_md = _query_local_vision_model(b64_img, model=active_model)
            full_markdown.append(f"<!-- Page {page_num} -->\n{page_md}")
        except Exception as exc:
            print(f"[!] Error on page {page_num}: {exc}. Using text fallback...", flush=True)
            try:
                import pymupdf
                doc = pymupdf.open(pdf_path)
                fallback_text = doc[page_idx].get_text()
                doc.close()
                full_markdown.append(f"<!-- Page {page_num} -->\n{fallback_text}")
            except Exception:
                full_markdown.append(f"<!-- Page {page_num} -->\n> ⚠️ Error transcribing page {page_num}.")

    final_text = "\n\n---\n\n".join(full_markdown)
    cache_path.write_text(final_text, encoding="utf-8")
    print(f"[+] Successfully parsed and cached output at: {cache_path}", flush=True)
    return final_text
