"""
LLM interface via Google Gemini REST API.
Supports both streaming and non-streaming responses.
"""

import json
from typing import Generator
import requests
from config.settings import GEMINI_API_KEY, LLM_MODEL, LLM_TEMPERATURE


def query_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = LLM_MODEL,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Send a prompt to the Gemini LLM and return the response.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": LLM_TEMPERATURE
        }
    }

    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    if stream:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
        return _stream_response(url, payload)
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        return _blocking_response(url, payload)


def _blocking_response(url: str, payload: dict) -> str:
    """Make a non-streaming request and return the full response."""
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return ""


def _stream_response(url: str, payload: dict) -> Generator[str, None, None]:
    """Make a streaming request and yield response tokens."""
    response = requests.post(url, json=payload, timeout=300, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith("data: "):
                json_str = line_str[6:]
                if json_str.strip() == "":
                    continue
                try:
                    data = json.loads(json_str)
                    if "candidates" in data and len(data["candidates"]) > 0:
                        parts = data["candidates"][0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            yield parts[0]["text"]
                except json.JSONDecodeError:
                    continue
