"""
Embedding generation via Ollama's local embedding API.
Uses nomic-embed-text by default.
"""

import time
import requests
from config.settings import OLLAMA_BASE_URL, EMBEDDING_MODEL


def get_embedding(text: str, retries: int = 3) -> list[float]:
    """
    Generate an embedding vector for a single text string using Ollama.
    Includes retry logic for transient errors.
    """
    base_url = OLLAMA_BASE_URL.rstrip("/")
    url = f"{base_url}/api/embed"
    payload = {
        "model": EMBEDDING_MODEL,
        "input": [text],
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                return data["embeddings"][0]
            elif response.status_code in (429, 503):
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited (HTTP {response.status_code}), retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                response.raise_for_status()
        except requests.exceptions.ConnectionError:
            wait = 2 ** (attempt + 1)
            print(f"    Connection error, retrying in {wait}s...", flush=True)
            time.sleep(wait)

    # Final attempt without catching
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["embeddings"][0]


def get_embeddings_batch(texts: list[str], batch_size: int = 50, retries: int = 5) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts using Ollama's /api/embed endpoint.
    Sends up to batch_size texts in a single HTTP request.
    """
    base_url = OLLAMA_BASE_URL.rstrip("/")
    url = f"{base_url}/api/embed"
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"    Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks in 1 request)...", flush=True)

        payload = {
            "model": EMBEDDING_MODEL,
            "input": batch,
        }

        success = False
        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    all_embeddings.extend(data["embeddings"])
                    success = True
                    break
                elif response.status_code in (429, 503):
                    wait = 2 ** (attempt + 1) + 1
                    print(f"    Rate limit/transient error ({response.status_code}), sleeping {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    response.raise_for_status()
            except requests.exceptions.ConnectionError:
                wait = 2 ** (attempt + 1) + 1
                print(f"    Connection error, sleeping {wait}s...", flush=True)
                time.sleep(wait)

        if not success:
            # Fallback to individual calls
            print("    Batch failed, falling back to individual embedding calls with backoff...", flush=True)
            for t in batch:
                all_embeddings.append(get_embedding(t))
                time.sleep(0.5)

    return all_embeddings
