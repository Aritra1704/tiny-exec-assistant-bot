import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

def chat(messages, timeout=120):
    """
    messages: list[dict] with keys: role in {"system","user","assistant"} and content.
    """
    payload = {"model": MODEL, "messages": messages, "stream": False}
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]