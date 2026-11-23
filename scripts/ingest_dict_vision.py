#!/usr/bin/env python3
"""
Batch Ingest Script for dict/ folder using 100% Deep Vision Model (qwen2.5vl).
Renders every page of each PDF, extracts text, tables, chemical formulas,
LaTeX equations, and figure descriptions via Ollama Vision,
then chunks into Parent Sections & Dense Child Vectors and ingests into ChromaDB.
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.parser import parse_pdf
from core.chunker import chunk_parent_child
from core.vectorstore import VectorStore


def main():
    dict_dir = PROJECT_ROOT / "dict"
    if not dict_dir.exists():
        print(f"[!] Error: Directory '{dict_dir}' not found.")
        sys.exit(1)

    pdf_files = sorted(list(dict_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"[!] No PDF files found in '{dict_dir}'.")
        sys.exit(1)

    print("═════════════════════════════════════════════════════════════════════")
    print(f"  🧪  Deep Vision Ingestion Pipeline ({len(pdf_files)} papers found in dict/)")
    print("      Vision Model: qwen2.5vl (Ollama)")
    print("      Embedding Model: nomic-embed-text")
    print("      Storage: ChromaDB (Child Vectors) + Parent Docstore")
    print("═════════════════════════════════════════════════════════════════════\n")

    store = VectorStore()
    total_start = time.time()
    results_summary = []

    for i, pdf_path in enumerate(pdf_files, 1):
        file_name = pdf_path.name
        source_id = pdf_path.stem
        print(f"\n[{i}/{len(pdf_files)}] 📄 Processing: '{file_name}'")
        paper_start = time.time()

        # Step 1: Deep Vision Parsing
        print(f"  [1/3] Deep Vision Transcription with qwen2.5vl...")
        try:
            markdown = parse_pdf(
                str(pdf_path),
                force=True,         # Re-parse to ensure 100% vision extraction
                force_vision=True,  # Bypass fast text extraction
                vision_model="qwen2.5vl",
            )
        except Exception as e:
            print(f"  [!] Failed to parse '{file_name}': {e}")
            continue

        # Step 2: Parent-Document Chunking
        print(f"  [2/3] Chunking into Parent Sections & ~100-token Child Chunks...")
        parent_store, child_chunks = chunk_parent_child(markdown, source_id=source_id)
        if not child_chunks:
            print(f"  [!] Warning: No chunks produced for '{file_name}'.")
            continue

        # Step 3: Vector Ingestion
        print(f"  [3/3] Generating nomic-embed-text embeddings & saving...")
        ingested_count = store.ingest_parents_and_children(
            parent_store=parent_store,
            child_chunks=child_chunks,
            pdf_path=str(pdf_path),
        )

        elapsed = time.time() - paper_start
        print(f"  ✓ Completed '{file_name}' in {elapsed:.1f}s: "
              f"{len(parent_store)} parent sections, {ingested_count} child vectors.")

        results_summary.append({
            "name": file_name,
            "parents": len(parent_store),
            "children": ingested_count,
            "time": elapsed,
        })

    total_time = time.time() - total_start
    print("\n═════════════════════════════════════════════════════════════════════")
    print("  ✅  ALL DOCUMENTS IN dict/ SUCCESSFULLY INGESTED!")
    print(f"  ⏱️  Total Duration: {total_time / 60:.1f} minutes")
    print(f"  📊  Total Child Vectors in ChromaDB: {store.count}")
    print(f"  📚  Total Parent Sections in Docstore: {len(store.parent_docstore)}")
    print("─────────────────────────────────────────────────────────────────────")
    for r in results_summary:
        print(f"  • {r['name']}: {r['parents']} parents | {r['children']} vectors ({r['time']:.1f}s)")
    print("═════════════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()

