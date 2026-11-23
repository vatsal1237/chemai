"""
LLM interface via Ollama's OpenAI-compatible API.
Supports both streaming and non-streaming responses, with model fallback detection.
"""

import json
from typing import Generator
import requests
from config.settings import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

# Cache for verified available model
_AVAILABLE_MODEL_CACHE: str | None = None


def get_available_llm_model(preferred: str = LLM_MODEL) -> str:
    """Check Ollama for the preferred model, falling back to other installed models if needed."""
    global _AVAILABLE_MODEL_CACHE
    if _AVAILABLE_MODEL_CACHE:
        return _AVAILABLE_MODEL_CACHE

    base_url = OLLAMA_BASE_URL.rstrip("/")
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            # Exact match or prefix match
            for m in models:
                if preferred in m or m in preferred:
                    _AVAILABLE_MODEL_CACHE = m
                    return m
            # If preferred not found, look for llama3 variants or qwen
            for fallback in ["llama3.1:8b", "llama3.2:latest", "llama3:latest", "qwen2.5:latest"]:
                for m in models:
                    if fallback in m:
                        print(f"[!] Preferred model '{preferred}' not found. Using installed model '{m}'.", flush=True)
                        _AVAILABLE_MODEL_CACHE = m
                        return m
            if models:
                _AVAILABLE_MODEL_CACHE = models[0]
                return models[0]
    except Exception:
        pass

    _AVAILABLE_MODEL_CACHE = preferred
    return preferred


def query_llm(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    stream: bool = False,
    history: list[dict] | None = None,
) -> str | Generator[str, None, None]:
    """
    Send a prompt (with optional conversation history) to the Ollama LLM.
    Uses Ollama's OpenAI-compatible /v1/chat/completions endpoint.
    """
    active_model = model or get_available_llm_model()
    base_url = OLLAMA_BASE_URL.rstrip("/")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        for msg in history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": active_model,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "stream": stream,
    }

    url = f"{base_url}/v1/chat/completions"

    if stream:
        return _stream_response(url, payload)
    else:
        return _blocking_response(url, payload)


def _blocking_response(url: str, payload: dict) -> str:
    """Make a request and accumulate tokens into the full response string."""
    payload["stream"] = True
    full_tokens = []
    for token in _stream_response(url, payload):
        full_tokens.append(token)
    return "".join(full_tokens)


def _stream_response(url: str, payload: dict) -> Generator[str, None, None]:
    """Make a streaming request and yield response tokens."""
    response = requests.post(url, json=payload, timeout=300, stream=True)
    if response.status_code != 200:
        try:
            err = response.json().get("error", {}).get("message", response.text)
        except Exception:
            err = response.text
        raise RuntimeError(f"Ollama API error ({response.status_code}): {err}")

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith("data: "):
                json_str = line_str[6:]
                if json_str.strip() == "" or json_str.strip() == "[DONE]":
                    continue
                try:
                    data = json.loads(json_str)
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue
