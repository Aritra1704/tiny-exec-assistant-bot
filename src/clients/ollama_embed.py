import requests

from src.config import get_config


def embed(text: str) -> list[float]:
    cfg = get_config(validate=True, require_telegram=False)
    cleaned_text = text.strip() or "(empty)"
    prompt = cleaned_text[: cfg.EMBED_MAX_CHARS]
    payload = {"model": cfg.OLLAMA_EMBED_MODEL, "prompt": prompt}

    try:
        response = requests.post(
            f"{cfg.OLLAMA_URL}/api/embeddings",
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama not reachable at {cfg.OLLAMA_URL}. Is ollama running?"
        ) from exc

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama embeddings request failed with status {response.status_code}."
        ) from exc

    data = response.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("Ollama embeddings response missing embedding vector.")

    return [float(value) for value in embedding]
