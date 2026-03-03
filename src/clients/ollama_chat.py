import requests

from src.config import get_config


def _ollama_not_reachable_message() -> str:
    cfg = get_config(validate=True, require_telegram=False)
    return f"Ollama not reachable at {cfg.OLLAMA_URL}. Is ollama running?"


def chat(messages: list[dict], model: str | None = None, timeout: int = 120) -> str:
    cfg = get_config(validate=True, require_telegram=False)
    payload = {
        "model": model or cfg.OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
    }
    try:
        response = requests.post(
            f"{cfg.OLLAMA_URL}/api/chat",
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_ollama_not_reachable_message()) from exc

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama chat request failed with status {response.status_code}."
        ) from exc

    data = response.json()
    message = data.get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama chat response missing message content.")
    return content


def check_ollama_health(timeout: int = 10) -> dict:
    cfg = get_config(validate=True, require_telegram=False)
    try:
        response = requests.get(f"{cfg.OLLAMA_URL}/api/tags", timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(_ollama_not_reachable_message()) from exc

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama health request failed with status {response.status_code}."
        ) from exc

    data = response.json()
    models = data.get("models")
    model_count = len(models) if isinstance(models, list) else 0
    return {"ok": True, "model_count": model_count}
