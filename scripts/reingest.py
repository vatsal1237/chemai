#!/usr/bin/env bash
"""
Re-ingest script — rebuild the ChromaDB vector store with Parent-Document Retrieval.
Usage:
    python scripts/reingest.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PARSED_DIR
from core.vectorstore import VectorStore
from core.chunker import chunk_parent_child


def main():
    store = VectorStore()
    print("[*] Clearing old vector store and parent docstore...")
    store.clear()

    parsed_files = list(PARSED_DIR.glob("*.md"))
    if not parsed_files:
        print("[!] No parsed markdown files found in data/parsed/.")
        return

    for md_path in parsed_files:
        print(f"\n[+] Re-ingesting with Parent-Document retrieval: {md_path.name}")
        markdown = md_path.read_text(encoding="utf-8")
        source_id = md_path.stem
        parent_store, child_chunks = chunk_parent_child(markdown, source_id=source_id)

        if child_chunks:
            ingested = store.ingest_parents_and_children(
                parent_store=parent_store,
                child_chunks=child_chunks,
                pdf_path=str(md_path),
            )
            print(f"    → Ingested {ingested} child vectors linked to {len(parent_store)} parent sections.")
        else:
            print("    → No chunks produced.")

    print(f"\n[✓] Done! Total child vectors: {store.count}, parent sections: {len(store.parent_docstore)}")


if __name__ == "__main__":
    main()
