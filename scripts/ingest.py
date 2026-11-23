#!/usr/bin/env python3
"""
CLI script to ingest any scientific PDF paper into Chem-RAG.
Usage:
    python scripts/ingest.py /path/to/paper.pdf [--vision]
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.parser import parse_pdf
from core.chunker import chunk_parent_child
from core.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF research paper into Chem-RAG.")
    parser.add_argument("pdf_path", help="Path to local PDF file.")
    parser.add_argument("--vision", action="store_true", help="Force local vision model (qwen2.5vl/llama3.2-vision) OCR on all pages.")
    parser.add_argument("--force", action="store_true", help="Force re-parsing even if a cached parsed file already exists.")
    args = parser.parse_args()

    pdf_file = Path(args.pdf_path).resolve()
    if not pdf_file.exists():
        print(f"[!] Error: File '{pdf_file}' not found.")
        sys.exit(1)

    print(f"[+] Ingesting: {pdf_file.name}")
    print("[1/3] Parsing PDF into Markdown...")
    markdown = parse_pdf(str(pdf_file), force=args.force or args.vision, force_vision=args.vision)

    print("[2/3] Chunking into Parent Sections & Dense Child Blocks...")
    source_id = pdf_file.stem
    parent_store, child_chunks = chunk_parent_child(markdown, source_id=source_id)

    print("[3/3] Generating embeddings & storing in ChromaDB...")
    store = VectorStore()
    ingested = store.ingest_parents_and_children(
        parent_store=parent_store,
        child_chunks=child_chunks,
        pdf_path=str(pdf_file),
    )

    print(f"\n[✓] Successfully ingested '{pdf_file.name}'!")
    print(f"    - Child vectors in ChromaDB: {ingested}")
    print(f"    - Parent sections stored: {len(parent_store)}")
    print(f"    - Total database vectors: {store.count}")
    print("    This paper is now permanently available for everyone to query!")


if __name__ == "__main__":
    main()

