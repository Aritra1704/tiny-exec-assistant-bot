import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.config import Config
from src.memory import store


def _mock_connection(fetchone=None, fetchall=None):
    conn = MagicMock()
    cursor = MagicMock()

    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    cursor.fetchone.return_value = fetchone
    cursor.fetchall.return_value = fetchall

    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    conn.cursor.return_value = cursor
    return conn, cursor


class StoreSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config_patcher = patch(
            "src.memory.store.get_config",
            return_value=Config(
                OLLAMA_URL="http://localhost:11434",
                OLLAMA_CHAT_MODEL="llama3.1:8b",
                OLLAMA_EMBED_MODEL="nomic-embed-text",
                RAG_TOP_K=5,
                EMBED_MAX_CHARS=2000,
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
        self.config_patcher.start()

    def tearDown(self):
        self.config_patcher.stop()

    def test_save_message_inserts_chat_row(self):
        conn, cursor = _mock_connection(fetchone={"id": 42})

        with patch("src.memory.store._conn", return_value=conn):
            message_id = store.save_message(321, "user", "hello")

        self.assertEqual(message_id, 42)
        cursor.execute.assert_called_once()
        sql, params = cursor.execute.call_args.args
        self.assertIn("INSERT INTO", sql)
        self.assertIn("chat_messages", sql)
        self.assertEqual(params, (321, "user", "hello"))
        conn.commit.assert_called_once_with()

    def test_get_recent_messages_returns_rows(self):
        rows = [
            {
                "id": 1,
                "created_at": datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc),
                "chat_id": 321,
                "role": "user",
                "content": "hello",
            }
        ]
        conn, cursor = _mock_connection(fetchall=rows)

        with patch("src.memory.store._conn", return_value=conn):
            result = store.get_recent_messages(321, limit=99)

        self.assertEqual(result, rows)
        sql, params = cursor.execute.call_args.args
        self.assertIn("chat_messages", sql)
        self.assertEqual(params, (321, 50))

    def test_save_message_rejects_invalid_role(self):
        with self.assertRaises(ValueError):
            store.save_message(321, "tool", "hello")

    def test_get_last_summary_returns_latest_summary_row(self):
        conn, cursor = _mock_connection(
            fetchone={
                "id": 3,
                "chat_id": 321,
                "summary_text": "Summary text",
                "last_message_id_covered": 22,
            }
        )

        with patch("src.memory.store._conn", return_value=conn):
            result = store.get_last_summary(321)

        self.assertEqual(result["summary_text"], "Summary text")
        sql, params = cursor.execute.call_args.args
        self.assertIn("conversation_summaries", sql)
        self.assertEqual(params, (321,))

    def test_save_summary_inserts_summary_row(self):
        conn, cursor = _mock_connection(fetchone={"id": 9})

        with patch("src.memory.store._conn", return_value=conn):
            summary_id = store.save_summary(321, "Concise summary", 44)

        self.assertEqual(summary_id, 9)
        sql, params = cursor.execute.call_args.args
        self.assertIn("conversation_summaries", sql)
        self.assertEqual(params, (321, "Concise summary", 44))
        conn.commit.assert_called_once_with()

    def test_get_messages_after_returns_ordered_rows(self):
        rows = [
            {
                "id": 11,
                "created_at": datetime(2026, 3, 3, 10, 0, tzinfo=timezone.utc),
                "chat_id": 321,
                "role": "assistant",
                "content": "hello again",
            }
        ]
        conn, cursor = _mock_connection(fetchall=rows)

        with patch("src.memory.store._conn", return_value=conn):
            result = store.get_messages_after(321, 10)

        self.assertEqual(result, rows)
        sql, params = cursor.execute.call_args.args
        self.assertIn("chat_messages", sql)
        self.assertEqual(params, (321, 10))

    def test_get_user_preferences_returns_defaults_if_missing(self):
        conn, cursor = _mock_connection(fetchone=None)

        with patch("src.memory.store._conn", return_value=conn):
            result = store.get_user_preferences(321)

        self.assertEqual(result["tone"], "calm")
        self.assertEqual(result["verbosity"], "medium")
        self.assertEqual(result["timezone"], "Asia/Kolkata")
        self.assertEqual(result["executive_mode"], True)
        cursor.execute.assert_called_once()
        conn.commit.assert_not_called()

    def test_upsert_user_preferences_updates_requested_fields(self):
        conn, cursor = _mock_connection(
            fetchone={
                "chat_id": 321,
                "tone": "strict",
                "verbosity": "medium",
                "timezone": "Asia/Kolkata",
                "executive_mode": True,
            }
        )

        with patch("src.memory.store._conn", return_value=conn):
            result = store.upsert_user_preferences(321, {"tone": "strict"})

        self.assertEqual(result["tone"], "strict")
        sql, params = cursor.execute.call_args.args
        self.assertIn("user_preferences", sql)
        self.assertEqual(params, [321, "strict"])

    def test_save_message_embedding_upserts_vector_row(self):
        conn, cursor = _mock_connection(fetchone={"id": 77})

        with patch("src.memory.store._conn", return_value=conn):
            embedding_id = store.save_message_embedding(321, 9, "hello", [0.1, 0.2])

        self.assertEqual(embedding_id, 77)
        sql, params = cursor.execute.call_args.args
        self.assertIn("message_embeddings", sql)
        self.assertEqual(params, (321, 9, "hello", "[0.1,0.2]"))

    def test_search_similar_messages_returns_hits(self):
        rows = [{"message_id": 9, "content": "hello", "distance": 0.12}]
        conn, cursor = _mock_connection(fetchall=rows)

        with patch("src.memory.store._conn", return_value=conn):
            result = store.search_similar_messages(321, [0.1, 0.2], top_k=5)

        self.assertEqual(result, rows)
        sql, params = cursor.execute.call_args.args
        self.assertIn("message_embeddings", sql)
        self.assertEqual(params, ("[0.1,0.2]", 321, "[0.1,0.2]", 5))

    def test_embedding_exists_checks_for_existing_row(self):
        conn, cursor = _mock_connection(fetchone={"?column?": 1})

        with patch("src.memory.store._conn", return_value=conn):
            result = store.embedding_exists(321, 9)

        self.assertTrue(result)
        sql, params = cursor.execute.call_args.args
        self.assertIn("message_embeddings", sql)
        self.assertEqual(params, (321, 9))

    def test_iter_chat_messages_returns_ordered_batch(self):
        rows = [
            {"id": 11, "chat_id": 321, "role": "user", "content": "hello"},
            {"id": 12, "chat_id": 321, "role": "assistant", "content": "hi"},
        ]
        conn, cursor = _mock_connection(fetchall=rows)

        with patch("src.memory.store._conn", return_value=conn):
            result = store.iter_chat_messages(after_id=10, batch_size=2500)

        self.assertEqual(result, rows)
        sql, params = cursor.execute.call_args.args
        self.assertIn("chat_messages", sql)
        self.assertEqual(params, (10, 1000))
