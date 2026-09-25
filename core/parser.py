"""
PDF → Markdown parser using Google Gemini API.
Caches results to avoid re-parsing.
"""

import hashlib
import json
import base64
import requests
from pathlib import Path
import fitz
from typing import Callable, Optional

from config.settings import PARSED_DIR, GEMINI_API_KEY, VISION_MODEL, PARSE_DPI

def _cache_key(pdf_path: str, pages: str | None) -> str:
    """Generate a deterministic cache filename from PDF path and page selection."""
    pdf_hash = hashlib.md5(Path(pdf_path).resolve().read_bytes()[:4096]).hexdigest()[:12]
    page_tag = pages.replace(",", "_").replace("-", "to") if pages else "all"
    stem = Path(pdf_path).stem
    return f"{stem}_{pdf_hash}_p{page_tag}.md"

def parse_pdf(pdf_path: str, pages: str | None = None, force: bool = False, progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
    """
    Parse a PDF into Markdown using Google Gemini API.

    Args:
        pdf_path: Path to the PDF file.
        pages: Page range string (ignored for now, does all pages) or None.
        force: If True, re-parse even if cached result exists.

    Returns:
        Extracted Markdown string.
    """
    cache_name = _cache_key(pdf_path, pages)
    cache_path = PARSED_DIR / cache_name

    # Return cached if available
    if cache_path.exists() and not force:
        print(f"[+] Using cached parsed output: {cache_path}", flush=True)
        if progress_callback:
            progress_callback(1, 1) # simulate 100% completion for cache
        return cache_path.read_text(encoding="utf-8")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

    print(f"[+] Parsing PDF with Gemini API ({VISION_MODEL})...", flush=True)
    doc = fitz.open(pdf_path)
    
    # Process all pages
    page_nums = range(len(doc))
    full_markdown = []
    
    # Gemini API URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{VISION_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    for i in page_nums:
        if progress_callback:
            progress_callback(i, len(doc))
        print(f"    Transcribing Page {i+1}/{len(doc)}...", flush=True)
        page = doc[i]
        pix = page.get_pixmap(dpi=PARSE_DPI)
        img_data = pix.tobytes("jpeg")
        b64_image = base64.b64encode(img_data).decode("utf-8")
        
        prompt = (
            "You are an expert scientific document transcription assistant. "
            "Please transcribe the following page into Markdown format. "
            "For any figures, images, or charts, provide a detailed description "
            "in the following format: \\n\\n**[Figure Description]:** <your detailed description>\\n\\n"
            "Do not output anything other than the transcribed markdown."
        )
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": b64_image
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1
            }
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                full_markdown.append(f"<!-- Page {i+1} -->\n{text}")
            except (KeyError, IndexError):
                print(f"[!] Error extracting text from Gemini response on page {i+1}")
        else:
            print(f"[!] API Error on page {i+1}: {response.status_code} {response.text}")
            if response.status_code == 429:
                full_markdown.append(f"<!-- FAILED_PAGE_{i+1} -->\n> ⚠️ **Page {i+1} was not parsed due to Gemini API rate limits/quota.**")
            else:
                full_markdown.append(f"<!-- FAILED_PAGE_{i+1} -->\n> ⚠️ **Page {i+1} was not parsed due to API Error {response.status_code}.**")
            
    if progress_callback:
        progress_callback(len(doc), len(doc))
        
    final_text = "\n\n---\n\n".join(full_markdown)
    cache_path.write_text(final_text, encoding="utf-8")
    print(f"[+] Parsed output cached at: {cache_path}", flush=True)
    return final_text
