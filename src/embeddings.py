import os

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
MAX_EMBED_CHARS = 2000


def embed_text(text: str) -> list[float]:
    cleaned_text = text.strip() or "(empty)"
    prompt = cleaned_text[:MAX_EMBED_CHARS]
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": prompt}

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama embeddings unavailable at {OLLAMA_URL}"
        ) from exc

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama embeddings request failed with status {response.status_code}"
        ) from exc

    data = response.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("Ollama embeddings response missing embedding vector.")

    return [float(value) for value in embedding]
