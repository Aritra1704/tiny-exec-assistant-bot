from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class ConfigError(RuntimeError):
    pass


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, "true" if default else "false").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false.")


@dataclass(slots=True)
class Config:
    OLLAMA_URL: str
    OLLAMA_CHAT_MODEL: str
    OLLAMA_EMBED_MODEL: str
    OLLAMA_CREATIVE_MODEL: str
    RAG_TOP_K: int
    EMBED_MAX_CHARS: int
    CREATIVE_INTENT_ROUTING: bool
    PG_HOST: str
    PG_PORT: int
    PG_DATABASE: str
    PG_USER: str
    PG_PASSWORD: str
    PG_SCHEMA: str
    TELEGRAM_BOT_TOKEN: str
    LOG_LEVEL: str
    CHAT_CONTEXT_CHAR_LIMIT: int
    _summary_printed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            OLLAMA_URL=os.getenv("OLLAMA_URL", "http://localhost:11434").strip(),
            OLLAMA_CHAT_MODEL=os.getenv("OLLAMA_MODEL", "").strip(),
            OLLAMA_EMBED_MODEL=os.getenv("OLLAMA_EMBED_MODEL", "").strip(),
            OLLAMA_CREATIVE_MODEL=os.getenv("OLLAMA_CREATIVE_MODEL", "").strip(),
            RAG_TOP_K=_get_int("RAG_TOP_K", 5),
            EMBED_MAX_CHARS=_get_int("EMBED_MAX_CHARS", 2000),
            CREATIVE_INTENT_ROUTING=_get_bool("CREATIVE_INTENT_ROUTING", True),
            PG_HOST=os.getenv("PG_HOST", "localhost").strip(),
            PG_PORT=_get_int("PG_PORT", 5432),
            PG_DATABASE=os.getenv("PG_DATABASE", "postgres").strip(),
            PG_USER=os.getenv("PG_USER", "postgres").strip(),
            PG_PASSWORD=os.getenv("PG_PASSWORD", ""),
            PG_SCHEMA=os.getenv("PG_SCHEMA", "tinyse").strip(),
            TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
            CHAT_CONTEXT_CHAR_LIMIT=_get_int("CHAT_CONTEXT_CHAR_LIMIT", 1200),
        )

    def validate(self, require_telegram: bool = True) -> "Config":
        missing = []
        if not self.OLLAMA_CHAT_MODEL:
            missing.append("OLLAMA_MODEL")
        if not self.OLLAMA_EMBED_MODEL:
            missing.append("OLLAMA_EMBED_MODEL")
        if require_telegram and not self.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if missing:
            raise ConfigError("Missing required environment variables: " + ", ".join(missing))

        if self.RAG_TOP_K < 1 or self.RAG_TOP_K > 20:
            raise ConfigError("RAG_TOP_K must be between 1 and 20.")
        if self.EMBED_MAX_CHARS < 100 or self.EMBED_MAX_CHARS > 20000:
            raise ConfigError("EMBED_MAX_CHARS must be between 100 and 20000.")
        if self.PG_PORT < 1 or self.PG_PORT > 65535:
            raise ConfigError("PG_PORT must be between 1 and 65535.")
        if self.CHAT_CONTEXT_CHAR_LIMIT < 200 or self.CHAT_CONTEXT_CHAR_LIMIT > 10000:
            raise ConfigError("CHAT_CONTEXT_CHAR_LIMIT must be between 200 and 10000.")
        if not self.OLLAMA_URL:
            raise ConfigError("OLLAMA_URL must not be empty.")
        if not self.PG_HOST or not self.PG_DATABASE or not self.PG_USER or not self.PG_SCHEMA:
            raise ConfigError("PG_HOST, PG_DATABASE, PG_USER, and PG_SCHEMA must not be empty.")

        if not self._summary_printed:
            print(
                "config loaded "
                f"chat_model={self.OLLAMA_CHAT_MODEL} "
                f"embed_model={self.OLLAMA_EMBED_MODEL} "
                f"creative_model={self.OLLAMA_CREATIVE_MODEL or self.OLLAMA_CHAT_MODEL} "
                f"rag_top_k={self.RAG_TOP_K} "
                f"ollama_url={self.OLLAMA_URL} "
                f"pg={self.PG_HOST}/{self.PG_DATABASE}/{self.PG_SCHEMA}"
            )
            self._summary_printed = True
        return self


_CONFIG: Config | None = None


def get_config(validate: bool = False, require_telegram: bool = True) -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config.from_env()
    if validate:
        _CONFIG.validate(require_telegram=require_telegram)
    return _CONFIG
