import unittest
from unittest.mock import MagicMock, patch

import requests

from src.clients import ollama_embed
from src.config import Config


class EmbeddingsSmokeTests(unittest.TestCase):
    @patch(
        "src.clients.ollama_embed.get_config",
        return_value=Config(
            OLLAMA_URL="http://localhost:11434",
            OLLAMA_CHAT_MODEL="llama3.1:8b",
            OLLAMA_EMBED_MODEL="nomic-embed-text",
            OLLAMA_CREATIVE_MODEL="llama3.1:70b",
            RAG_TOP_K=5,
            EMBED_MAX_CHARS=2000,
            CREATIVE_INTENT_ROUTING=True,
            PG_HOST="localhost",
            PG_PORT=5432,
            PG_DATABASE="postgres",
            PG_USER="postgres",
            PG_PASSWORD="",
            PG_SCHEMA="tinyse",
            TELEGRAM_BOT_TOKEN="telegram-token",
            LOG_LEVEL="INFO",
            CHAT_CONTEXT_CHAR_LIMIT=1200,
        ),
    )
    @patch("src.clients.ollama_embed.requests.post")
    def test_embed_text_truncates_prompt_and_returns_vector(self, post_mock, _config_mock):
        response = MagicMock()
        response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        response.raise_for_status.return_value = None
        post_mock.return_value = response

        vector = ollama_embed.embed("x" * 3000)

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(len(payload["prompt"]), 2000)
        self.assertEqual(payload["model"], "nomic-embed-text")

    @patch(
        "src.clients.ollama_embed.get_config",
        return_value=Config(
            OLLAMA_URL="http://localhost:11434",
            OLLAMA_CHAT_MODEL="llama3.1:8b",
            OLLAMA_EMBED_MODEL="nomic-embed-text",
            OLLAMA_CREATIVE_MODEL="llama3.1:70b",
            RAG_TOP_K=5,
            EMBED_MAX_CHARS=2000,
            CREATIVE_INTENT_ROUTING=True,
            PG_HOST="localhost",
            PG_PORT=5432,
            PG_DATABASE="postgres",
            PG_USER="postgres",
            PG_PASSWORD="",
            PG_SCHEMA="tinyse",
            TELEGRAM_BOT_TOKEN="telegram-token",
            LOG_LEVEL="INFO",
            CHAT_CONTEXT_CHAR_LIMIT=1200,
        ),
    )
    @patch("src.clients.ollama_embed.requests.post", side_effect=requests.RequestException("boom"))
    def test_embed_text_raises_clear_error_when_ollama_unreachable(self, _post_mock, _config_mock):
        with self.assertRaises(RuntimeError):
            ollama_embed.embed("hello")
