import unittest
from unittest.mock import patch

from scripts import backfill_embeddings


class BackfillEmbeddingsSmokeTests(unittest.TestCase):
    def test_backfill_embeddings_skips_system_and_existing_rows(self):
        rows = [
            {"id": 1, "chat_id": 321, "role": "user", "content": "Discuss payroll timing."},
            {"id": 2, "chat_id": 321, "role": "system", "content": "internal system note"},
            {"id": 3, "chat_id": 321, "role": "assistant", "content": "Vendor review on Thursday."},
            {"id": 4, "chat_id": 321, "role": "user", "content": "   "},
        ]

        with (
            patch("scripts.backfill_embeddings.init_db") as init_db_mock,
            patch(
                "scripts.backfill_embeddings.iter_chat_messages",
                side_effect=[rows[:2], rows[2:], []],
            ) as iter_mock,
            patch(
                "scripts.backfill_embeddings.embedding_exists",
                side_effect=[False, True],
            ) as exists_mock,
            patch(
                "scripts.backfill_embeddings.embed_text",
                return_value=[0.1, 0.2, 0.3],
            ) as embed_mock,
            patch("scripts.backfill_embeddings.save_message_embedding") as save_mock,
            patch("scripts.backfill_embeddings.time.sleep") as sleep_mock,
            patch("builtins.print"),
        ):
            result = backfill_embeddings.backfill_embeddings(batch_size=2, sleep_seconds=0.05)

        init_db_mock.assert_called_once_with()
        self.assertEqual(iter_mock.call_count, 3)
        self.assertEqual(exists_mock.call_count, 2)
        embed_mock.assert_called_once_with("Discuss payroll timing.")
        save_mock.assert_called_once_with(321, 1, "Discuss payroll timing.", [0.1, 0.2, 0.3])
        sleep_mock.assert_called_once_with(0.05)
        self.assertEqual(
            result,
            {"scanned": 4, "embedded": 1, "skipped": 3, "last_seen_id": 4},
        )
