from src.clients.ollama_embed import embed


def embed_text(text: str) -> list[float]:
    return embed(text)


__all__ = ["embed_text"]
