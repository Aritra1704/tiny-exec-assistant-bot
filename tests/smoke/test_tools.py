import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.tools.alarms import schedule_reminder
from src.tools.notes import tool_list_notes, tool_save_note


class NotesToolSmokeTests(unittest.TestCase):
    @patch("src.tools.notes.add_note", return_value=12)
    def test_tool_save_note_trims_text_and_returns_id(self, add_note_mock):
        result = tool_save_note("  buy oat milk  ")

        add_note_mock.assert_called_once_with("buy oat milk")
        self.assertEqual(
            result,
            {"ok": True, "note_id": 12, "saved": "buy oat milk"},
        )

    def test_tool_save_note_rejects_empty_text(self):
        result = tool_save_note("   ")

        self.assertEqual(
            result,
            {"ok": False, "error": "Note text cannot be empty."},
        )

    @patch(
        "src.tools.notes.list_notes",
        return_value=[
            {
                "id": 9,
                "created_at": datetime(2026, 3, 2, 12, 30, tzinfo=timezone.utc),
                "text": "buy oat milk",
            }
        ],
    )
    def test_tool_list_notes_normalizes_rows(self, list_notes_mock):
        result = tool_list_notes(200)

        list_notes_mock.assert_called_once_with(50)
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["limit"], 50)
        self.assertEqual(result["notes"][0]["id"], 9)
        self.assertEqual(result["notes"][0]["text"], "buy oat milk")
        self.assertEqual(
            result["notes"][0]["created_at"],
            "2026-03-02T12:30:00+00:00",
        )


class _FakeJob:
    def __init__(self, job_id: str):
        self.id = job_id


class _FakeScheduler:
    def __init__(self):
        self.trigger = None
        self.run_date = None
        self.callback = None

    def add_job(self, callback, trigger, run_date):
        self.callback = callback
        self.trigger = trigger
        self.run_date = run_date
        return _FakeJob("job-123")


class ReminderToolSmokeTests(unittest.TestCase):
    @patch("src.tools.alarms._get_scheduler")
    def test_schedule_reminder_returns_job_info_and_callback(self, get_scheduler_mock):
        sent_messages = []
        fake_scheduler = _FakeScheduler()
        get_scheduler_mock.return_value = fake_scheduler

        result = schedule_reminder(
            send_fn=lambda chat_id, text: sent_messages.append((chat_id, text)),
            chat_id=77,
            in_minutes=2,
            message="  stretch  ",
        )

        self.assertEqual(result["ok"], True)
        self.assertEqual(result["job_id"], "job-123")
        self.assertEqual(result["in_minutes"], 2)
        self.assertEqual(result["message"], "stretch")
        self.assertEqual(fake_scheduler.trigger, "date")
        self.assertIsNotNone(fake_scheduler.run_date)

        fake_scheduler.callback()
        self.assertEqual(sent_messages, [(77, "Reminder: stretch")])

    def test_schedule_reminder_rejects_invalid_minutes(self):
        result = schedule_reminder(
            send_fn=lambda _chat_id, _text: None,
            chat_id=77,
            in_minutes=0,
            message="stretch",
        )

        self.assertEqual(
            result,
            {
                "ok": False,
                "error": "Reminder time must be greater than 0 minutes.",
            },
        )
