"""
ChromaDB vector store manager.
Handles ingestion, search, and persistence of document chunks.
"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import CHROMA_DIR, CHROMA_COLLECTION_NAME
from core.chunker import Chunk
from core.embeddings import get_embeddings_batch, get_embedding


class VectorStore:
    """Persistent ChromaDB-backed vector store for RAG retrieval."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[+] ChromaDB collection '{CHROMA_COLLECTION_NAME}' loaded "
              f"({self._collection.count()} existing documents)", flush=True)

    @property
    def count(self) -> int:
        """Number of documents currently stored."""
        return self._collection.count()

    def is_ingested(self, pdf_path: str, pages: str | None = None) -> bool:
        """Check if chunks from this PDF+page combo are already ingested."""
        source_id = self._source_id(pdf_path, pages)
        results = self._collection.get(
            where={"source_id": source_id},
            limit=1,
        )
        return len(results["ids"]) > 0

    def ingest(self, chunks: list[Chunk], pdf_path: str, pages: str | None = None) -> int:
        """
        Embed and store chunks into ChromaDB.

        Args:
            chunks: List of Chunk objects to ingest.
            pdf_path: Source PDF path (for dedup tracking).
            pages: Page range string used during parsing.

        Returns:
            Number of chunks ingested.
        """
        if not chunks:
            print("[!] No chunks to ingest.", flush=True)
            return 0

        source_id = self._source_id(pdf_path, pages)

        # Remove old chunks from same source before re-ingesting
        existing = self._collection.get(where={"source_id": source_id})
        if existing["ids"]:
            print(f"[+] Removing {len(existing['ids'])} old chunks from same source...", flush=True)
            self._collection.delete(ids=existing["ids"])

        # Prepare data
        texts = [c.text for c in chunks]
        ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = []
        for c in chunks:
            meta = {**c.metadata, "source_id": source_id}
            # ChromaDB requires metadata values to be str, int, float, or bool
            for k, v in meta.items():
                if v is None:
                    meta[k] = ""
                elif isinstance(v, bool):
                    pass  # booleans are fine
                elif not isinstance(v, (str, int, float)):
                    meta[k] = str(v)
            metadatas.append(meta)

        # Generate embeddings
        print(f"[+] Generating embeddings for {len(texts)} chunks...", flush=True)
        embeddings = get_embeddings_batch(texts)

        # Store in ChromaDB
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"[+] Ingested {len(chunks)} chunks into ChromaDB "
              f"(total: {self._collection.count()})", flush=True)
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys: text, metadata, distance.
        """
        if self._collection.count() == 0:
            return []

        query_embedding = get_embedding(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return hits

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[+] Vector store cleared.", flush=True)

    @staticmethod
    def _source_id(pdf_path: str, pages: str | None) -> str:
        """Generate a deterministic source ID for dedup."""
        stem = Path(pdf_path).stem
        page_tag = pages if pages else "all"
        raw = f"{stem}_{page_tag}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
