"""
Retriever module.
Wraps VectorStore search with similarity threshold filtering and context formatting.
Supports multi-query retrieval for better recall.
"""

from core.vectorstore import VectorStore
from config.settings import TOP_K, SIMILARITY_THRESHOLD


def retrieve(
    store: VectorStore,
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks for a single query, filtered by similarity threshold.

    Args:
        store: VectorStore instance.
        query: User's search query.
        top_k: Max results to return.
        threshold: Minimum similarity (1 - distance) to include a result.

    Returns:
        Tuple of (formatted context string, raw hits list).
        If no relevant chunks found, context string is "No relevant chunks found."
    """
    raw_hits = store.search(query, top_k=top_k)

    # Filter by similarity threshold (ChromaDB returns cosine distance, lower = more similar)
    filtered = [h for h in raw_hits if (1.0 - h["distance"]) >= threshold]

    if not filtered:
        return "No relevant chunks found.", []

    context_str = _format_hits(filtered)
    return context_str, filtered


def retrieve_multi(
    store: VectorStore,
    queries: list[str],
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    final_k: int | None = None,
) -> tuple[str, list[dict], dict[str, list[dict]]]:
    """
    Retrieve relevant chunks using MULTIPLE queries, merge and deduplicate.
    Each query retrieves top_k results; results are merged by best similarity
    and capped at final_k total results.

    Args:
        store: VectorStore instance.
        queries: List of search queries (from query rewriter).
        top_k: Max results per query.
        threshold: Minimum similarity to include.
        final_k: Max total results after merging (defaults to top_k).

    Returns:
        Tuple of (formatted context string, merged hits list, per-query hits dict).
    """
    if final_k is None:
        final_k = top_k

    # Collect all hits across queries, keyed by chunk text to deduplicate
    seen: dict[str, dict] = {}
    per_query_hits: dict[str, list[dict]] = {}

    for q in queries:
        raw_hits = store.search(q, top_k=top_k)
        query_filtered = []
        for hit in raw_hits:
            sim = 1.0 - hit["distance"]
            if sim < threshold:
                continue

            query_filtered.append(hit)
            chunk_key = hit["text"][:200]  # dedup key
            if chunk_key not in seen or sim > (1.0 - seen[chunk_key]["distance"]):
                seen[chunk_key] = hit  # keep the best similarity score

        per_query_hits[q] = query_filtered

    if not seen:
        return "No relevant chunks found.", [], per_query_hits

    # Sort by similarity (best first) and take final_k
    merged = sorted(seen.values(), key=lambda h: h["distance"])[:final_k]

    context_str = _format_hits(merged)
    return context_str, merged, per_query_hits


def _format_hits(hits: list[dict]) -> str:
    """Format a list of hits into a numbered context string."""
    context_parts = []
    for i, hit in enumerate(hits, 1):
        page = hit["metadata"].get("page", "?")
        heading = hit["metadata"].get("heading", "")
        similarity = 1.0 - hit["distance"]

        header = f"[Chunk {i} | Page {page}"
        if heading:
            header += f" | {heading}"
        header += f" | Similarity: {similarity:.2f}]"

        context_parts.append(f"{header}\n{hit['text']}")

    return "\n\n".join(context_parts)
