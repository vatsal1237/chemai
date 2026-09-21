"""
Retriever module.
Wraps VectorStore search with similarity threshold filtering and context formatting.
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
    Retrieve relevant chunks for a query, filtered by similarity threshold.

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

    # Format as numbered context blocks
    context_parts = []
    for i, hit in enumerate(filtered, 1):
        page = hit["metadata"].get("page", "?")
        heading = hit["metadata"].get("heading", "")
        similarity = 1.0 - hit["distance"]

        header = f"[Chunk {i} | Page {page}"
        if heading:
            header += f" | {heading}"
        header += f" | Similarity: {similarity:.2f}]"

        context_parts.append(f"{header}\n{hit['text']}")

    context_str = "\n\n".join(context_parts)
    return context_str, filtered
