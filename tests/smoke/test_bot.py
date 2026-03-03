import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src import bot
from src.prompt import SYSTEM_PROMPT


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


class _DummyApplication:
    def __init__(self):
        self.tasks = []

    def create_task(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task


class _DummyContext:
    def __init__(self, args=None):
        self.bot = SimpleNamespace(send_message=AsyncMock())
        self.application = _DummyApplication()
        self.args = args or []


class _DummyMessage:
    def __init__(self, text: str):
        self.text = text
        self.reply_text = AsyncMock()


class _DummyUpdate:
    def __init__(self, text: str, chat_id: int = 321):
        self.message = _DummyMessage(text)
        self.effective_chat = SimpleNamespace(id=chat_id)


class BotSmokeTests(unittest.IsolatedAsyncioTestCase):
    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_on_message_executes_tool_and_logs_result(self):
        update = _DummyUpdate("Save a note: buy oat milk")
        context = _DummyContext()

        with (
            patch(
                "src.bot.decide",
                return_value={
                    "type": "tool",
                    "tool": "save_note",
                    "args": {"text": "buy oat milk"},
                },
            ),
            patch("src.bot._run_tool_and_reply", new=AsyncMock()) as run_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
        ):
            await bot.on_message(update, context)

        run_tool_mock.assert_awaited_once_with(
            update,
            context,
            "save_note",
            {"text": "buy oat milk"},
            debug_source="tool routing",
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "Save a note: buy oat milk"),
            ],
        )
        update.message.reply_text.assert_not_awaited()

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_on_message_uses_normal_chat_for_non_tool_text(self):
        update = _DummyUpdate("Summarize my day")
        context = _DummyContext()
        history_rows = [
            {"role": "assistant", "content": f"assistant {index}"}
            if index % 2 == 0
            else {"role": "user", "content": f"user {index}"}
            for index in range(1, 19)
        ]

        with (
            patch(
                "src.bot.decide",
                return_value={"type": "text", "text": "Let me think about that."},
            ),
            patch("src.bot.save_message") as save_message_mock,
            patch(
                "src.bot.get_last_summary",
                return_value={
                    "summary_text": "You planned payroll and follow-ups.",
                    "last_message_id_covered": 40,
                },
            ),
            patch("src.bot.get_messages_after", return_value=history_rows) as get_messages_after_mock,
            patch("src.bot.chat", return_value="Here is the concise summary.") as chat_mock,
            patch("builtins.print") as print_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.on_message(update, context)

        get_messages_after_mock.assert_called_once_with(321, 40)
        chat_mock.assert_called_once()
        messages = chat_mock.call_args.args[0]
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(
            messages[1],
            {
                "role": "system",
                "content": "Conversation summary so far: You planned payroll and follow-ups.",
            },
        )
        self.assertEqual(
            messages[2:],
            [
                {
                    "role": row["role"],
                    "content": row["content"],
                }
                for row in history_rows[-15:]
            ],
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "Summarize my day"),
                unittest.mock.call(321, "assistant", "Here is the concise summary."),
            ],
        )
        print_mock.assert_any_call("Loaded 15 previous messages for chat_id 321")
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with("Here is the concise summary.")

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_on_message_uses_router_question_directly(self):
        update = _DummyUpdate("Remind me later")
        context = _DummyContext()

        with (
            patch(
                "src.bot.decide",
                return_value={"type": "text", "text": "In how many minutes?"},
            ),
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot.chat") as chat_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.on_message(update, context)

        chat_mock.assert_not_called()
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "Remind me later"),
                unittest.mock.call(321, "assistant", "In how many minutes?"),
            ],
        )
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with("In how many minutes?")

    def test_build_chat_context_includes_summary_and_trims_large_messages(self):
        long_content = "x" * (bot.CHAT_CONTEXT_CHAR_LIMIT + 25)

        messages = bot._build_chat_context(
            [
                {"role": "assistant", "content": "Short reply"},
                {"role": "user", "content": long_content},
            ],
            summary_text="Decisions and commitments so far.",
        )

        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(
            messages[1],
            {
                "role": "system",
                "content": "Conversation summary so far: Decisions and commitments so far.",
            },
        )
        self.assertEqual(messages[2], {"role": "assistant", "content": "Short reply"})
        self.assertEqual(len(messages[3]["content"]), bot.CHAT_CONTEXT_CHAR_LIMIT)
        self.assertTrue(messages[3]["content"].endswith("..."))

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_note_command_runs_save_note_tool(self):
        update = _DummyUpdate("/note buy oat milk")
        context = _DummyContext(args=["buy", "oat", "milk"])

        with (
            patch("src.bot._run_tool", new=AsyncMock(return_value={"ok": True, "note_id": 7})) as run_tool_mock,
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.note_command(update, context)

        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "/note buy oat milk"),
                unittest.mock.call(321, "assistant", "Noted (ID 7)."),
            ],
        )
        run_tool_mock.assert_awaited_once_with("save_note", {"text": "buy oat milk"}, context, 321)
        log_tool_mock.assert_awaited_once_with("save_note", {"text": "buy oat milk"}, {"ok": True, "note_id": 7})
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with("Noted (ID 7).")

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_notes_command_runs_list_notes_tool(self):
        update = _DummyUpdate("/notes 5")
        context = _DummyContext(args=["5"])

        with (
            patch(
                "src.bot._run_tool",
                new=AsyncMock(
                    return_value={
                        "ok": True,
                        "notes": [
                            {"id": 9, "text": "buy oat milk"},
                            {"id": 8, "text": "call the bank"},
                        ],
                    }
                ),
            ) as run_tool_mock,
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.notes_command(update, context)

        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "/notes 5"),
                unittest.mock.call(321, "assistant", "9: buy oat milk\n8: call the bank"),
            ],
        )
        run_tool_mock.assert_awaited_once_with("list_notes", {"limit": 5}, context, 321)
        log_tool_mock.assert_awaited_once()
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with(
            "9: buy oat milk\n8: call the bank"
        )

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_remind_command_runs_set_reminder_tool(self):
        update = _DummyUpdate("/remind 2 stretch now")
        context = _DummyContext(args=["2", "stretch", "now"])

        with (
            patch(
                "src.bot._run_tool",
                new=AsyncMock(
                    return_value={"ok": True, "run_at": "2026-03-02T10:15:00+00:00"}
                ),
            ) as run_tool_mock,
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.remind_command(update, context)

        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "/remind 2 stretch now"),
                unittest.mock.call(
                    321, "assistant", "Done. I'll remind you at 2026-03-02T10:15:00+00:00."
                ),
            ],
        )
        run_tool_mock.assert_awaited_once_with(
            "set_reminder",
            {"in_minutes": 2, "message": "stretch now"},
            context,
            321,
        )
        log_tool_mock.assert_awaited_once()
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with(
            "Done. I'll remind you at 2026-03-02T10:15:00+00:00."
        )


class _FakeApp:
    def __init__(self):
        self.handlers = []
        self.run_polling = MagicMock()

    def add_handler(self, handler):
        self.handlers.append(handler)


class _FakeBuilder:
    def __init__(self, app):
        self.app = app
        self.token_value = None

    def token(self, token_value):
        self.token_value = token_value
        return self

    def build(self):
        return self.app


class BotStartupSmokeTests(unittest.TestCase):
    def test_main_initializes_db_and_starts_polling(self):
        fake_app = _FakeApp()
        fake_builder = _FakeBuilder(fake_app)

        with (
            patch("src.bot.init_db") as init_db_mock,
            patch("src.bot.ApplicationBuilder", return_value=fake_builder),
            patch("src.bot.os.getenv", return_value="telegram-token"),
        ):
            bot.main()

        init_db_mock.assert_called_once_with()
        self.assertEqual(fake_builder.token_value, "telegram-token")
        self.assertEqual(len(fake_app.handlers), 4)
        fake_app.run_polling.assert_called_once()
