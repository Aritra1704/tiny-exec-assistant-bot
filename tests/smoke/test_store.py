import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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
