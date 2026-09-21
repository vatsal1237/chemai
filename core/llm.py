"""
LLM interface via Ollama REST API.
Supports both streaming and non-streaming responses.
"""

import json
from typing import Generator

import requests
from config.settings import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE


def query_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = LLM_MODEL,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Send a prompt to the Ollama LLM and return the response.

    Args:
        prompt: User prompt text.
        system_prompt: System-level instruction.
        model: Ollama model name.
        stream: If True, returns a generator yielding response tokens.

    Returns:
        Complete response string, or generator of token strings if streaming.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": LLM_TEMPERATURE,
        },
    }

    if stream:
        return _stream_response(url, payload)
    else:
        return _blocking_response(url, payload)


def _blocking_response(url: str, payload: dict) -> str:
    """Make a non-streaming request and return the full response."""
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def _stream_response(url: str, payload: dict) -> Generator[str, None, None]:
    """Make a streaming request and yield response tokens."""
    response = requests.post(url, json=payload, timeout=300, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                continue
