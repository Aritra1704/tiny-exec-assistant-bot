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
    def __init__(self):
        self.bot = SimpleNamespace(send_message=AsyncMock())
        self.application = _DummyApplication()


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
            patch(
                "src.bot._execute_tool",
                return_value={"ok": True, "note_id": 7, "saved": "buy oat milk"},
            ) as execute_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot.log_tool") as log_tool_mock,
        ):
            await bot.on_message(update, context)

        execute_tool_mock.assert_called_once()
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "Save a note: buy oat milk"),
                unittest.mock.call(321, "assistant", "Saved note 7."),
            ],
        )
        log_tool_mock.assert_called_once_with(
            tool="save_note",
            args_json={"text": "buy oat milk"},
            result_json={"ok": True, "note_id": 7, "saved": "buy oat milk"},
        )
        update.message.reply_text.assert_awaited_once_with("Saved note 7.")

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_on_message_uses_normal_chat_for_non_tool_text(self):
        update = _DummyUpdate("Summarize my day")
        context = _DummyContext()
        history_rows = [
            {"role": "assistant", "content": "Yesterday you planned three tasks."},
            {"role": "user", "content": "Summarize my day"},
        ]

        with (
            patch(
                "src.bot.decide",
                return_value={"type": "text", "text": "Let me think about that."},
            ),
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot.get_recent_messages", return_value=history_rows),
            patch("src.bot.chat", return_value="Here is the concise summary.") as chat_mock,
        ):
            await bot.on_message(update, context)

        chat_mock.assert_called_once()
        messages = chat_mock.call_args.args[0]
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(
            messages[1:],
            [
                {"role": "assistant", "content": "Yesterday you planned three tasks."},
                {"role": "user", "content": "Summarize my day"},
            ],
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "Summarize my day"),
                unittest.mock.call(321, "assistant", "Here is the concise summary."),
            ],
        )
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
        update.message.reply_text.assert_awaited_once_with("In how many minutes?")

    def test_build_chat_context_trims_large_messages(self):
        long_content = "x" * (bot.CHAT_CONTEXT_CHAR_LIMIT + 25)

        messages = bot._build_chat_context(
            [
                {"role": "assistant", "content": "Short reply"},
                {"role": "user", "content": long_content},
            ]
        )

        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[1], {"role": "assistant", "content": "Short reply"})
        self.assertEqual(len(messages[2]["content"]), bot.CHAT_CONTEXT_CHAR_LIMIT)
        self.assertTrue(messages[2]["content"].endswith("..."))


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
        self.assertEqual(len(fake_app.handlers), 1)
        fake_app.run_polling.assert_called_once()
