import unittest
from unittest.mock import patch

from src.memory.summarizer import maybe_summarize


def _message_row(message_id: int, role: str, content: str) -> dict:
    return {"id": message_id, "role": role, "content": content}


class SummarizerSmokeTests(unittest.TestCase):
    @patch("src.memory.summarizer.save_summary")
    @patch("src.memory.summarizer.chat")
    @patch("src.memory.summarizer.get_last_summary", return_value=None)
    @patch(
        "src.memory.summarizer.get_messages_after",
        side_effect=[
            [_message_row(i, "user" if i % 2 else "assistant", f"message {i}") for i in range(1, 21)],
        ],
    )
    def test_maybe_summarize_skips_below_threshold(
        self,
        _get_messages_after_mock,
        _get_last_summary_mock,
        chat_mock,
        save_summary_mock,
    ):
        result = maybe_summarize(321)

        self.assertEqual(
            result,
            {"ok": False, "reason": "below_threshold", "compressed_count": 0},
        )
        chat_mock.assert_not_called()
        save_summary_mock.assert_not_called()

    @patch("builtins.print")
    @patch("src.memory.summarizer.save_summary")
    @patch("src.memory.summarizer.chat", return_value="Rolled-up summary")
    @patch(
        "src.memory.summarizer.get_last_summary",
        return_value={
            "chat_id": 321,
            "summary_text": "Earlier summary",
            "last_message_id_covered": 10,
        },
    )
    @patch(
        "src.memory.summarizer.get_messages_after",
        side_effect=[
            [_message_row(i, "user" if i % 2 else "assistant", f"message {i}") for i in range(1, 33)],
            [_message_row(i, "user" if i % 2 else "assistant", f"message {i}") for i in range(11, 33)],
        ],
    )
    def test_maybe_summarize_rolls_summary_forward(
        self,
        get_messages_after_mock,
        _get_last_summary_mock,
        chat_mock,
        save_summary_mock,
        print_mock,
    ):
        result = maybe_summarize(321)

        self.assertEqual(result["ok"], True)
        self.assertEqual(result["compressed_count"], 7)
        self.assertEqual(result["last_message_id_covered"], 17)
        save_summary_mock.assert_called_once_with(321, "Rolled-up summary", 17)
        self.assertEqual(get_messages_after_mock.call_count, 2)
        prompt_messages = chat_mock.call_args.args[0]
        self.assertIn("Earlier summary", prompt_messages[1]["content"])
        print_mock.assert_called_once_with(
            "summarization ran for chat_id 321; compressed 7 messages"
        )
