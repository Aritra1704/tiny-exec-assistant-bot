import unittest
from unittest.mock import patch

from src.router import decide, parse_tool_call


class RouterSmokeTests(unittest.TestCase):
    def test_parse_tool_call_accepts_direct_json(self):
        parsed = parse_tool_call('{"tool":"save_note","args":{"text":"buy oat milk"}}')

        self.assertEqual(
            parsed,
            {"tool": "save_note", "args": {"text": "buy oat milk"}},
        )

    def test_parse_tool_call_extracts_first_json_block(self):
        parsed = parse_tool_call('Use this payload: {"tool":"list_notes","args":{"limit":3}} thanks')

        self.assertEqual(
            parsed,
            {"tool": "list_notes", "args": {"limit": 3}},
        )

    def test_parse_tool_call_rejects_invalid_schema(self):
        parsed = parse_tool_call('{"tool":"save_note"}')

        self.assertIsNone(parsed)

    @patch("src.router.chat", return_value='{"tool":"save_note","args":{"text":"  buy oat milk  "}}')
    def test_decide_returns_validated_save_note_tool(self, _chat_mock):
        decision = decide("Save a note to buy oat milk")

        self.assertEqual(
            decision,
            {
                "type": "tool",
                "tool": "save_note",
                "args": {"text": "buy oat milk"},
            },
        )

    @patch("src.router.chat", return_value='{"tool":"list_notes","args":{"limit":"5"}}')
    def test_decide_coerces_numeric_tool_args(self, _chat_mock):
        decision = decide("Show my last 5 notes")

        self.assertEqual(
            decision,
            {
                "type": "tool",
                "tool": "list_notes",
                "args": {"limit": 5},
            },
        )

    @patch("src.router.chat", return_value="In how many minutes?")
    def test_decide_returns_plain_text_when_no_json(self, _chat_mock):
        decision = decide("Remind me later")

        self.assertEqual(decision, {"type": "text", "text": "In how many minutes?"})

    @patch("src.router.chat", return_value='{"tool":"unknown_tool","args":{}}')
    def test_decide_rejects_unknown_tool_payload(self, _chat_mock):
        decision = decide("Do something unsupported")

        self.assertEqual(
            decision,
            {"type": "text", "text": '{"tool":"unknown_tool","args":{}}'},
        )

    @patch("src.router.chat", return_value='Here you go: {"tool":"save_note","args":{"text":"buy milk"}}')
    def test_decide_parses_tool_call_embedded_in_text(self, _chat_mock):
        decision = decide("Save buy milk as a note")

        self.assertEqual(
            decision,
            {
                "type": "tool",
                "tool": "save_note",
                "args": {"text": "buy milk"},
            },
        )
