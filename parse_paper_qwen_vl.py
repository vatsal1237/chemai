#!/usr/bin/env python3
"""
Qwen2.5-VL Scientific Paper PDF Ingestion Pipeline
Renders PDF pages to high-resolution images and uses Qwen2.5-VL (Vision-Language Model)
to transcribe complex multi-column layouts, chemical formulas, and LaTeX equations into Markdown.
Supports local execution (Ollama/Transformers/vLLM) and free API providers (OpenRouter/HuggingFace/DashScope).
"""

import os
import sys
import argparse
import base64
import io
import requests
from dotenv import load_dotenv
import pymupdf

# Load environment variables
load_dotenv()


def pdf_page_to_base64(page: pymupdf.Page, dpi: int = 150) -> str:
    """Renders a PyMuPDF page to a JPEG image encoded in base64."""
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("jpeg")
    return base64.b64encode(img_bytes).decode("utf-8")


def download_pdf(url_or_path: str, temp_filename: str = "temp_qwen_paper.pdf") -> str:
    """Downloads PDF from URL or validates local path."""
    if os.path.isfile(url_or_path):
        print(f"[+] Using local PDF file: '{url_or_path}'", flush=True)
        return url_or_path

    url = url_or_path
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url
        print(f"[!] Protocol scheme missing. Auto-prepended 'https://' -> {url}", flush=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
    }

    print(f"[+] Fetching PDF from: {url}", flush=True)
    try:
        res = requests.get(url, headers=headers, stream=True, timeout=30)
        res.raise_for_status()

        with open(temp_filename, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return temp_filename
    except Exception as exc:
        raise RuntimeError(f"Failed to download PDF: {exc}") from exc


def query_qwen_vl_api(
    base64_image: str,
    api_key: str = None,
    api_base: str = "http://localhost:11434",
    model: str = "qwen2.5vl",
) -> str:
    """
    Sends a page image to Qwen2.5-VL model via Ollama native API or OpenAI/OpenRouter API.
    """
    prompt = (
        "Transcribe this research paper page into clean, structured Markdown.\n"
        "- Follow the correct 2-column reading order (left column completely first, then right column).\n"
        "- Format all data tables as strict Markdown tables.\n"
        "- Wrap all chemical formulas, reaction schemes, and math equations in LaTeX ($ or $$).\n"
        "- Preserve headings (#, ##) and section structures.\n"
        "- For EVERY figure, diagram, schematic, flowchart, chemical structure, or image on the page:\n"
        "  - Write the figure caption exactly as shown.\n"
        "  - Then provide a detailed description of what the figure shows, including all labels, arrows,\n"
        "    components, flow directions, chemical structures, apparatus parts, and relationships.\n"
        "  - Format as: **[Figure Description]:** followed by the detailed description.\n"
        "  - Do NOT skip or omit any visual element. Describe every part of the image thoroughly."
    )

    # Detect local Ollama endpoint (localhost on default port 11434)
    is_ollama = "localhost" in api_base and "11434" in api_base
    if is_ollama:
        url = api_base.rstrip("/")
        if not url.endswith("/api/chat"):
            url = url.split("/v1")[0] + "/api/chat"

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
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    # Otherwise use standard OpenAI / OpenRouter format
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
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

    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def parse_page_selection(pages_arg, total_pages: int) -> list:
    """Parses page selection string or int into 0-indexed page indices."""
    if pages_arg is None:
        return list(range(total_pages))
    if isinstance(pages_arg, int):
        idx = pages_arg - 1
        return [idx] if 0 <= idx < total_pages else []
    pages_str = str(pages_arg).strip()
    if not pages_str:
        return list(range(total_pages))
    indices = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            sub = part.split("-")
            if len(sub) == 2:
                try:
                    s, e = int(sub[0].strip()), int(sub[1].strip())
                    for p in range(s, e + 1):
                        if 0 <= p - 1 < total_pages:
                            indices.add(p - 1)
                except ValueError:
                    pass
        else:
            try:
                p = int(part)
                if 0 <= p - 1 < total_pages:
                    indices.add(p - 1)
            except ValueError:
                pass
    return sorted(list(indices))


def extract_pdf_with_qwen_vl(
    pdf_path: str,
    output_markdown_path: str = "extracted_paper_qwen_vl.md",
    model: str = "qwen2.5vl",
    api_base: str = "http://localhost:11434",
    page_number=None,
) -> str:
    """Extracts Markdown from PDF using Qwen2.5-VL."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    print(f"[+] Opening PDF document for Qwen2.5-VL parsing: '{pdf_path}'", flush=True)
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    print(f"[+] Total pages in document: {total_pages}", flush=True)

    markdown_pages = []
    pages_to_process = parse_page_selection(page_number, total_pages)
    if not pages_to_process:
        print(f"[!] No valid pages selected out of {total_pages} total pages.", flush=True)

    for page_idx in pages_to_process:
        page_num = page_idx + 1
        print(f"[+] Transcribing Page {page_num}/{total_pages} with Qwen2.5-VL...", flush=True)

        page = doc.load_page(page_idx)
        img_b64 = pdf_page_to_base64(page)

        try:
            page_text = query_qwen_vl_api(img_b64, api_key=api_key, api_base=api_base, model=model)
        except Exception as err:
            print(f"[!] Qwen-VL API call failed on page {page_num}: {err}. Falling back to PyMuPDF text.", flush=True)
            page_text = page.get_text()

        markdown_pages.append(f"<!-- Page {page_num} -->\n" + page_text)

    doc.close()
    full_markdown = "\n\n---\n\n".join(markdown_pages)

    print(f"[+] Writing Qwen2.5-VL extracted Markdown to '{output_markdown_path}'...", flush=True)
    with open(output_markdown_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    print(f"[+] Successfully saved Qwen2.5-VL result to '{output_markdown_path}'.", flush=True)
    return full_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract scientific PDF papers into Markdown using Qwen2.5-VL vision model."
    )
    parser.add_argument("target", nargs="?", help="URL or local file path of the PDF paper.")
    parser.add_argument(
        "--output",
        "-o",
        default="extracted_paper_qwen_vl.md",
        help="Path to save the extracted Markdown file.",
    )
    parser.add_argument(
        "--page",
        "--pages",
        "-p",
        type=str,
        default=None,
        help="Specific 1-indexed page number or range to transcribe (e.g. --page 4-5 or --page 2 or --page 1,3,4-5).",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5vl",
        help="Qwen vision model identifier (default: qwen2.5vl).",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:11434",
        help="OpenAI-compatible API base URL or Ollama URL (default: http://localhost:11434).",
    )

    args = parser.parse_args()

    if not args.target:
        print("Usage: python parse_paper_qwen_vl.py <PDF_URL_OR_LOCAL_PATH> [--page 4-5] [--output extracted_paper_qwen_vl.md]")
        sys.exit(1)

    is_temp = not os.path.isfile(args.target)
    pdf_path = download_pdf(args.target)

    try:
        extract_pdf_with_qwen_vl(
            pdf_path,
            args.output,
            model=args.model,
            api_base=args.api_base,
            page_number=args.page,
        )
    finally:
        if is_temp and os.path.exists(pdf_path) and pdf_path.startswith("temp_qwen"):
            os.remove(pdf_path)


if __name__ == "__main__":
    main()
