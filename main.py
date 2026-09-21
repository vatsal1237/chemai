#!/usr/bin/env python3
"""
RAG Pipeline CLI — ingest PDFs, query, or chat from the terminal.

Usage:
    python main.py ingest <pdf_path> [--pages 4-5] [--force]
    python main.py query "What is laser ablation?"
    python main.py chat
    python main.py clear
"""

import sys
import argparse

from config.settings import PARSED_DIR
from core.parser import parse_pdf
from core.chunker import chunk_markdown
from core.vectorstore import VectorStore
from core.retriever import retrieve
from core.rag_chain import RAGChain


def cmd_ingest(args) -> None:
    """Ingest a PDF into the vector store."""
    pdf_path = args.pdf
    pages = args.pages
    force = args.force

    print("=" * 70)
    print("  RAG PIPELINE — PDF INGESTION")
    print("=" * 70)

    # Step 1: Parse PDF → Markdown
    print(f"\n[Step 1/3] Parsing PDF: {pdf_path}", flush=True)
    if pages:
        print(f"           Pages: {pages}", flush=True)
    markdown = parse_pdf(pdf_path, pages=pages, force=force)
    print(f"           Parsed {len(markdown)} characters of Markdown", flush=True)

    # Step 2: Chunk the Markdown
    print(f"\n[Step 2/3] Chunking Markdown...", flush=True)
    chunks = chunk_markdown(markdown)
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i}: page={chunk.metadata.get('page', '?')}, "
              f"heading='{chunk.metadata.get('heading', '')}', "
              f"len={len(chunk.text)}", flush=True)

    # Step 3: Embed and store
    print(f"\n[Step 3/3] Embedding and storing in ChromaDB...", flush=True)
    store = VectorStore()
    n = store.ingest(chunks, pdf_path, pages)

    print(f"\n{'=' * 70}")
    print(f"  INGESTION COMPLETE — {n} chunks stored")
    print(f"  Total documents in store: {store.count}")
    print(f"{'=' * 70}\n")


def cmd_query(args) -> None:
    """Run a single query against the RAG pipeline."""
    question = args.question

    print(f"\n[?] Query: {question}\n", flush=True)

    store = VectorStore()
    if store.count == 0:
        print("[!] No documents ingested yet. Run 'python main.py ingest <pdf>' first.")
        sys.exit(1)

    chain = RAGChain(store)
    answer = chain.ask(question)

    print(f"\n{'─' * 70}")
    print(f"Answer:\n")
    print(answer)
    print(f"{'─' * 70}\n")


def cmd_chat(args) -> None:
    """Interactive chat loop."""
    store = VectorStore()
    if store.count == 0:
        print("[!] No documents ingested yet. Run 'python main.py ingest <pdf>' first.")
        sys.exit(1)

    chain = RAGChain(store)

    print("=" * 70)
    print("  RAG PIPELINE — INTERACTIVE CHAT")
    print("  Type 'quit' or 'exit' to stop. Type 'clear' to reset history.")
    print("=" * 70)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            chain.clear_history()
            print("[+] Conversation history cleared.\n")
            continue

        print("\nAssistant: ", end="", flush=True)
        answer = chain.ask(user_input)
        print(answer)
        print()


def cmd_clear(args) -> None:
    """Clear the vector store."""
    store = VectorStore()
    store.clear()
    print("[+] All documents removed from the vector store.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline for Scientific Paper Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py ingest paper.pdf --pages 4-5\n"
            "  python main.py query 'What is laser ablation?'\n"
            "  python main.py chat\n"
            "  python main.py clear\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a PDF into the vector store")
    p_ingest.add_argument("pdf", help="Path to the PDF file")
    p_ingest.add_argument("--pages", "-p", type=str, default=None,
                          help="Page range (e.g. '4-5', '1,3,5-7')")
    p_ingest.add_argument("--force", "-f", action="store_true",
                          help="Force re-parse even if cached")
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = subparsers.add_parser("query", help="Run a single query")
    p_query.add_argument("question", help="Question to ask")
    p_query.set_defaults(func=cmd_query)

    # chat
    p_chat = subparsers.add_parser("chat", help="Interactive chat session")
    p_chat.set_defaults(func=cmd_chat)

    # clear
    p_clear = subparsers.add_parser("clear", help="Clear the vector store")
    p_clear.set_defaults(func=cmd_clear)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
