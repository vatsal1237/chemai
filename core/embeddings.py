"""
Embedding generation via Google Gemini REST API.
Uses gemini-embedding-001.
"""

import time
import requests
from config.settings import GEMINI_API_KEY, EMBEDDING_MODEL


def get_embedding(text: str, retries: int = 3) -> list[float]:
    """
    Generate an embedding vector for a single text string.
    Includes retry logic for transient API errors (429, 503).
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {
            "parts": [{"text": text}]
        }
    }

    for attempt in range(retries):
        response = requests.post(url, json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()
            return data["embedding"]["values"]
        elif response.status_code in (429, 503):
            wait = 2 ** (attempt + 1)
            print(f"    Rate limited (HTTP {response.status_code}), retrying in {wait}s...", flush=True)
            time.sleep(wait)
        else:
            response.raise_for_status()

    # Final attempt without catching
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["embedding"]["values"]


def get_embeddings_batch(texts: list[str], batch_size: int = 50, retries: int = 5) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts using Google Gemini batchEmbedContents.
    Sends up to batch_size texts in a single HTTP request to avoid rate limits.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    all_embeddings = []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:batchEmbedContents?key={GEMINI_API_KEY}"

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"    Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks in 1 request)...", flush=True)

        requests_payload = [
            {
                "model": f"models/{EMBEDDING_MODEL}",
                "content": {"parts": [{"text": t}]}
            }
            for t in batch
        ]
        payload = {"requests": requests_payload}

        success = False
        for attempt in range(retries):
            response = requests.post(url, json=payload, timeout=90)
            if response.status_code == 200:
                data = response.json()
                batch_values = [e["values"] for e in data["embeddings"]]
                all_embeddings.extend(batch_values)
                success = True
                break
            elif response.status_code in (429, 503):
                wait = 2 ** (attempt + 1) + 1
                print(f"    Rate limit/transient error ({response.status_code}), sleeping {wait}s...", flush=True)
                time.sleep(wait)
            else:
                response.raise_for_status()

        if not success:
            # Fallback to individual calls if batchEmbedContents had an issue
            print("    Batch failed, falling back to individual embedding calls with backoff...", flush=True)
            for t in batch:
                all_embeddings.append(get_embedding(t))
                time.sleep(1.0)

        # Gentle delay between batches
        time.sleep(0.5)

    return all_embeddings
