import unittest
from unittest.mock import MagicMock, patch

import requests

from src import embeddings


class EmbeddingsSmokeTests(unittest.TestCase):
    @patch("src.embeddings.requests.post")
    def test_embed_text_truncates_prompt_and_returns_vector(self, post_mock):
        response = MagicMock()
        response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        response.raise_for_status.return_value = None
        post_mock.return_value = response

        vector = embeddings.embed_text("x" * 3000)

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(len(payload["prompt"]), embeddings.MAX_EMBED_CHARS)
        self.assertEqual(payload["model"], embeddings.OLLAMA_EMBED_MODEL)

    @patch("src.embeddings.requests.post", side_effect=requests.RequestException("boom"))
    def test_embed_text_raises_clear_error_when_ollama_unreachable(self, _post_mock):
        with self.assertRaises(RuntimeError):
            embeddings.embed_text("hello")
