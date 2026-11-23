"""
ChromaDB vector store with Parent-Document Retrieval.
Stores small child chunk embeddings in ChromaDB and resolves parent document
sections from a persistent docstore for full context injection.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    PARENT_DOCSTORE_PATH,
    TOP_K,
    SIMILARITY_THRESHOLD,
)
from core.chunker import ParentDoc, ChildChunk
from core.embeddings import get_embeddings_batch, get_embedding


class VectorStore:
    """Persistent ChromaDB-backed vector store with Parent-Document resolution."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.parent_docstore: dict[str, dict] = self._load_parent_docstore()
        print(
            f"[+] ChromaDB collection '{CHROMA_COLLECTION_NAME}' loaded "
            f"({self._collection.count()} child chunks, {len(self.parent_docstore)} parent sections)",
            flush=True,
        )

    def _load_parent_docstore(self) -> dict[str, dict]:
        """Load parent document sections from disk."""
        if PARENT_DOCSTORE_PATH.exists():
            try:
                with open(PARENT_DOCSTORE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Warning: failed to load parent docstore: {e}", flush=True)
        return {}

    def _save_parent_docstore(self) -> None:
        """Persist parent document sections to disk."""
        try:
            with open(PARENT_DOCSTORE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.parent_docstore, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Warning: failed to save parent docstore: {e}", flush=True)

    @property
    def count(self) -> int:
        """Number of child chunks stored."""
        return self._collection.count()

    def is_ingested(self, pdf_path: str, pages: str | None = None) -> bool:
        """Check if chunks from this PDF are already ingested."""
        source_id = self._source_id(pdf_path, pages)
        results = self._collection.get(
            where={"source_id": source_id},
            limit=1,
        )
        return len(results.get("ids", [])) > 0

    def ingest_parents_and_children(
        self,
        parent_store: dict[str, ParentDoc],
        child_chunks: list[ChildChunk],
        pdf_path: str,
        pages: str | None = None,
    ) -> int:
        """
        Embed and store child chunks into ChromaDB, save parent sections to docstore.

        Args:
            parent_store: Dictionary mapping parent_id to ParentDoc.
            child_chunks: List of ChildChunk objects.
            pdf_path: Source PDF path.
            pages: Optional page string.

        Returns:
            Number of child chunks ingested.
        """
        if not child_chunks:
            print("[!] No child chunks to ingest.", flush=True)
            return 0

        source_id = self._source_id(pdf_path, pages)

        # Remove old entries for same source if re-ingesting
        existing = self._collection.get(where={"source_id": source_id})
        if existing and existing.get("ids"):
            print(f"[+] Removing {len(existing['ids'])} existing child chunks for source {source_id}...", flush=True)
            self._collection.delete(ids=existing["ids"])

        # Update parent docstore
        for pid, pdoc in parent_store.items():
            self.parent_docstore[pid] = pdoc.to_dict()
        self._save_parent_docstore()

        # Prepare child chunk vectors
        texts = [c.text for c in child_chunks]
        ids = [c.id for c in child_chunks]
        metadatas = []
        for c in child_chunks:
            meta = {**c.metadata, "source_id": source_id, "parent_id": c.parent_id}
            # Normalize metadata types for ChromaDB
            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    clean_meta[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        print(f"[+] Generating nomic-embed-text embeddings for {len(texts)} child chunks...", flush=True)
        embeddings = get_embeddings_batch(texts)

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"[+] Successfully ingested {len(child_chunks)} child chunks and {len(parent_store)} parent documents.", flush=True)
        return len(child_chunks)

    @staticmethod
    def _clean_paper_name(src: str) -> str:
        """Format raw source ID into a clean PDF paper name."""
        import re
        cleaned = re.sub(r'_[a-f0-9]{8,16}_p.*$', '', src)
        if not cleaned.lower().endswith(".pdf"):
            cleaned += ".pdf"
        return cleaned

    def search_parent_docs(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """
        Parent-Document Retrieval:
        1. Embeds query and searches small, dense child chunks in ChromaDB.
        2. Filters child chunks by cosine similarity threshold.
        3. Maps matching child chunks back to their full parent sections.
        4. Deduplicates parent sections, ranking by highest child similarity.

        Returns:
            List of dicts:
            [{"parent_id": str, "paper_name": str, "source_id": str, "text": str, "heading": str, "page": int, "similarity": float}]
        """
        if self._collection.count() == 0:
            return []

        query_embedding = get_embedding(query)
        # Query more child chunks to find diverse parent sections
        n_results = min(top_k * 4, self._collection.count())

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        parent_hits: dict[str, dict] = {}

        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            if similarity < threshold:
                continue

            metadata = results["metadatas"][0][i]
            parent_id = metadata.get("parent_id", "")
            raw_source = metadata.get("source_id", "")

            # If parent is found in docstore
            if parent_id in self.parent_docstore:
                parent_data = self.parent_docstore[parent_id]
                p_meta = parent_data.get("metadata", {})
                src = p_meta.get("source_id", raw_source)
                if parent_id not in parent_hits or similarity > parent_hits[parent_id]["similarity"]:
                    parent_hits[parent_id] = {
                        "parent_id": parent_id,
                        "paper_name": self._clean_paper_name(src),
                        "source_id": src,
                        "text": parent_data["text"],
                        "heading": p_meta.get("heading", metadata.get("heading", "")),
                        "page": p_meta.get("page", metadata.get("page", 1)),
                        "similarity": round(similarity, 4),
                        "best_child_match": results["documents"][0][i],
                    }
            else:
                # Fallback to child text if parent wasn't found
                child_id = results["ids"][0][i]
                if child_id not in parent_hits:
                    parent_hits[child_id] = {
                        "parent_id": child_id,
                        "paper_name": self._clean_paper_name(raw_source),
                        "source_id": raw_source,
                        "text": results["documents"][0][i],
                        "heading": metadata.get("heading", ""),
                        "page": metadata.get("page", 1),
                        "similarity": round(similarity, 4),
                        "best_child_match": results["documents"][0][i],
                    }

        # Rank parents by best similarity score
        sorted_parents = sorted(parent_hits.values(), key=lambda x: x["similarity"], reverse=True)
        return sorted_parents[:top_k]

    def clear(self) -> None:
        """Clear all child vectors and parent docstore."""
        self._client.delete_collection(CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.parent_docstore = {}
        self._save_parent_docstore()
        print("[+] Vector store and parent docstore cleared.", flush=True)

    @staticmethod
    def _source_id(pdf_path: str, pages: str | None) -> str:
        """Deterministic hash for PDF tracking."""
        stem = Path(pdf_path).stem
        page_tag = pages if pages else "all"
        raw = f"{stem}_{page_tag}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
