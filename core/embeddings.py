"""
Embedding generation via Ollama REST API.
Uses nomic-embed-text for high-quality local embeddings.
"""

import requests
from config.settings import OLLAMA_BASE_URL, EMBEDDING_MODEL


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for a single text string.

    Args:
        text: Input text to embed.

    Returns:
        Embedding vector as list of floats.
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    return data["embeddings"][0]


def get_embeddings_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.
    Processes in sub-batches to avoid memory issues.

    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts per API call.

    Returns:
        List of embedding vectors.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"    Embedding batch {batch_num}/{total_batches}...", flush=True)

        url = f"{OLLAMA_BASE_URL}/api/embed"
        payload = {
            "model": EMBEDDING_MODEL,
            "input": batch,
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        all_embeddings.extend(data["embeddings"])

    return all_embeddings
