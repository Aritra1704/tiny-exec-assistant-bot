import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src import bot
from src.config import Config
from src.prompt import SYSTEM_PROMPT


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


def _test_config() -> Config:
    return Config(
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
    )


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
            patch("src.bot.get_config", return_value=_test_config()),
            patch("src.bot._get_persona", new=AsyncMock(return_value=bot._normalize_persona(None))),
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
            patch("src.bot.get_config", return_value=_test_config()),
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
            patch(
                "src.bot.get_user_preferences",
                return_value={
                    "tone": "strict",
                    "verbosity": "short",
                    "timezone": "UTC",
                    "executive_mode": True,
                    "mode": "exec",
                },
            ),
            patch("src.bot._get_persona", new=AsyncMock(return_value={"name": "Akira", "voice": "calm executive assistant", "humor_level": 1, "creativity_level": 4, "signature": ""})),
            patch("src.bot._get_relevant_memory", new=AsyncMock(return_value=["Discussed payroll cutoff."])) as relevant_memory_mock,
            patch("src.bot.get_messages_after", return_value=history_rows) as get_messages_after_mock,
            patch("src.bot.chat", return_value="Here is the concise summary.") as chat_mock,
            patch("src.bot._log_llm_call", new=AsyncMock()) as log_llm_call_mock,
            patch("builtins.print") as print_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.on_message(update, context)

        get_messages_after_mock.assert_called_once_with(321, 40)
        relevant_memory_mock.assert_awaited_once()
        chat_mock.assert_called_once_with(unittest.mock.ANY, "llama3.1:8b")
        messages = chat_mock.call_args.args[0]
        self.assertIn(SYSTEM_PROMPT, messages[0]["content"])
        self.assertIn("Your name is Akira.", messages[0]["content"])
        self.assertIn("Tone: strict, direct, and disciplined.", messages[0]["content"])
        self.assertIn("Be concise.", messages[0]["content"])
        self.assertIn("Assume the user's timezone is UTC", messages[0]["content"])
        self.assertEqual(
            messages[1],
            {
                "role": "system",
                "content": "Conversation summary so far: You planned payroll and follow-ups.",
            },
        )
        self.assertEqual(
            messages[2],
            {
                "role": "system",
                "content": "Relevant memory (semantic):\n- Discussed payroll cutoff.",
            },
        )
        self.assertEqual(
            messages[3:],
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
        print_mock.assert_any_call(
            'loaded preferences for chat_id 321: {"executive_mode": true, "mode": "exec", "timezone": "UTC", "tone": "strict", "verbosity": "short"}'
        )
        print_mock.assert_any_call("Loaded 15 previous messages for chat_id 321")
        log_llm_call_mock.assert_awaited_once_with("llama3.1:8b", "exec", False, True, "Here is the concise summary.")
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with("Here is the concise summary.")

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_on_message_uses_router_question_directly(self):
        update = _DummyUpdate("Remind me later")
        context = _DummyContext()

        with (
            patch("src.bot.get_config", return_value=_test_config()),
            patch(
                "src.bot.decide",
                return_value={"type": "text", "text": "In how many minutes?"},
            ),
            patch("src.bot._get_persona", new=AsyncMock(return_value=bot._normalize_persona(None))),
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot.chat") as chat_mock,
            patch("src.bot._log_llm_call", new=AsyncMock()) as log_llm_call_mock,
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
        log_llm_call_mock.assert_awaited_once_with("llama3.1:8b", "exec", False, True, "In how many minutes?")
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with("In how many minutes?")

    def test_build_chat_context_includes_summary_memory_and_trims_large_messages(self):
        with patch("src.bot.get_config", return_value=_test_config()):
            long_content = "x" * (_test_config().CHAT_CONTEXT_CHAR_LIMIT + 25)

            messages = bot._build_chat_context(
                [
                    {"role": "assistant", "content": "Short reply"},
                    {"role": "user", "content": long_content},
                ],
                summary_text="Decisions and commitments so far.",
                preferences={
                    "tone": "casual",
                    "verbosity": "detailed",
                    "timezone": "America/New_York",
                    "executive_mode": False,
                    "mode": "creative",
                },
                persona={"name": "Akira", "voice": "calm executive assistant", "humor_level": 2, "creativity_level": 7, "signature": "Akira"},
                creative_active=True,
                relevant_memory=["Discussed quarterly staffing."],
            )

        self.assertIn(SYSTEM_PROMPT, messages[0]["content"])
        self.assertIn("Your name is Akira.", messages[0]["content"])
        self.assertIn("Tone: casual, conversational, and approachable.", messages[0]["content"])
        self.assertIn("Give more detail.", messages[0]["content"])
        self.assertIn("Use vivid but concise language.", messages[0]["content"])
        self.assertEqual(
            messages[1],
            {
                "role": "system",
                "content": "Conversation summary so far: Decisions and commitments so far.",
            },
        )
        self.assertEqual(
            messages[2],
            {
                "role": "system",
                "content": "Relevant memory (semantic):\n- Discussed quarterly staffing.",
            },
        )
        self.assertEqual(messages[3], {"role": "assistant", "content": "Short reply"})
        self.assertEqual(len(messages[4]["content"]), _test_config().CHAT_CONTEXT_CHAR_LIMIT)
        self.assertTrue(messages[4]["content"].endswith("..."))

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_on_message_uses_creative_model_when_mode_is_creative(self):
        update = _DummyUpdate("Write a short birthday greeting for my sister.")
        context = _DummyContext()

        with (
            patch("src.bot.get_config", return_value=_test_config()),
            patch("src.bot.decide", return_value={"type": "text", "text": "Let me think about that."}),
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot.get_last_summary", return_value=None),
            patch(
                "src.bot.get_user_preferences",
                return_value={
                    "tone": "calm",
                    "verbosity": "medium",
                    "timezone": "Asia/Kolkata",
                    "executive_mode": True,
                    "mode": "creative",
                },
            ),
            patch("src.bot._get_persona", new=AsyncMock(return_value=bot._normalize_persona(None))),
            patch("src.bot._get_relevant_memory", new=AsyncMock(return_value=[])),
            patch("src.bot.get_messages_after", return_value=[]),
            patch("src.bot.chat", return_value="Happy birthday to the brightest light in every room.") as chat_mock,
            patch("src.bot._log_llm_call", new=AsyncMock()) as log_llm_call_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()),
        ):
            await bot.on_message(update, context)

        chat_mock.assert_called_once_with(unittest.mock.ANY, "llama3.1:70b")
        log_llm_call_mock.assert_awaited_once_with(
            "llama3.1:70b",
            "creative",
            True,
            True,
            "Happy birthday to the brightest light in every room.",
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "Write a short birthday greeting for my sister."),
                unittest.mock.call(321, "assistant", "Happy birthday to the brightest light in every room."),
            ],
        )

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_save_chat_message_also_saves_embedding_for_natural_language(self):
        with (
            patch("src.bot.save_message", return_value=55) as save_message_mock,
            patch("src.bot._save_message_embedding", new=AsyncMock()) as save_embedding_mock,
        ):
            message_id = await bot._save_chat_message(321, "user", "hello")

        self.assertEqual(message_id, 55)
        save_message_mock.assert_called_once_with(321, "user", "hello")
        save_embedding_mock.assert_awaited_once_with(321, 55, "hello")

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_get_relevant_memory_filters_current_message_and_logs_hits(self):
        with (
            patch("src.bot.get_config", return_value=_test_config()),
            patch("src.bot.embed_text", return_value=[0.1, 0.2]),
            patch(
                "src.bot.search_similar_messages",
                return_value=[
                    {"message_id": 7, "content": "current question", "distance": 0.0},
                    {"message_id": 5, "content": "Remember payroll deadline", "distance": 0.1},
                    {"message_id": 6, "content": "Remember payroll deadline", "distance": 0.2},
                    {"message_id": 4, "content": "Budget review is tomorrow", "distance": 0.3},
                ],
            ),
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
        ):
            snippets = await bot._get_relevant_memory(321, "What is due tomorrow?", 7)

        self.assertEqual(
            snippets,
            ["Remember payroll deadline", "Budget review is tomorrow"],
        )
        log_tool_mock.assert_awaited_once_with(
            "semantic_retrieval",
            {"chat_id": 321, "top_k": 5},
            {"hits": 2},
        )

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

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_set_tone_command_updates_preferences(self):
        update = _DummyUpdate("/set_tone strict")
        context = _DummyContext(args=["strict"])

        with (
            patch(
                "src.bot.upsert_user_preferences",
                return_value={"tone": "strict", "verbosity": "medium", "timezone": "Asia/Kolkata", "executive_mode": True, "mode": "exec"},
            ),
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.set_tone_command(update, context)

        log_tool_mock.assert_awaited_once_with(
            "set_prefs",
            {"tone": "strict"},
            {"tone": "strict", "verbosity": "medium", "timezone": "Asia/Kolkata", "executive_mode": True, "mode": "exec"},
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "/set_tone strict"),
                unittest.mock.call(321, "assistant", "Tone set to strict."),
            ],
        )
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with("Tone set to strict.")

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_set_verbosity_command_updates_preferences(self):
        update = _DummyUpdate("/set_verbosity short")
        context = _DummyContext(args=["short"])

        with (
            patch(
                "src.bot.upsert_user_preferences",
                return_value={"tone": "calm", "verbosity": "short", "timezone": "Asia/Kolkata", "executive_mode": True, "mode": "exec"},
            ),
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.set_verbosity_command(update, context)

        log_tool_mock.assert_awaited_once_with(
            "set_prefs",
            {"verbosity": "short"},
            {"tone": "calm", "verbosity": "short", "timezone": "Asia/Kolkata", "executive_mode": True, "mode": "exec"},
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "/set_verbosity short"),
                unittest.mock.call(321, "assistant", "Verbosity set to short."),
            ],
        )
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with("Verbosity set to short.")

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_set_timezone_command_updates_preferences(self):
        update = _DummyUpdate("/set_timezone UTC")
        context = _DummyContext(args=["UTC"])

        with (
            patch(
                "src.bot.upsert_user_preferences",
                return_value={"tone": "calm", "verbosity": "medium", "timezone": "UTC", "executive_mode": True, "mode": "exec"},
            ),
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.set_timezone_command(update, context)

        log_tool_mock.assert_awaited_once_with(
            "set_prefs",
            {"timezone": "UTC"},
            {"tone": "calm", "verbosity": "medium", "timezone": "UTC", "executive_mode": True, "mode": "exec"},
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "/set_timezone UTC"),
                unittest.mock.call(321, "assistant", "Timezone set to UTC."),
            ],
        )
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with("Timezone set to UTC.")

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_exec_mode_command_updates_preferences(self):
        update = _DummyUpdate("/exec_mode off")
        context = _DummyContext(args=["off"])

        with (
            patch(
                "src.bot.upsert_user_preferences",
                return_value={"tone": "calm", "verbosity": "medium", "timezone": "Asia/Kolkata", "executive_mode": False, "mode": "exec"},
            ),
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.exec_mode_command(update, context)

        log_tool_mock.assert_awaited_once_with(
            "set_prefs",
            {"executive_mode": False},
            {"tone": "calm", "verbosity": "medium", "timezone": "Asia/Kolkata", "executive_mode": False, "mode": "exec"},
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "/exec_mode off"),
                unittest.mock.call(321, "assistant", "Executive mode disabled."),
            ],
        )
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with("Executive mode disabled.")

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_prefs_command_shows_current_preferences(self):
        update = _DummyUpdate("/prefs")
        context = _DummyContext()

        with (
            patch(
                "src.bot.get_user_preferences",
                return_value={
                    "tone": "calm",
                    "verbosity": "medium",
                    "timezone": "Asia/Kolkata",
                    "executive_mode": True,
                    "mode": "exec",
                },
            ),
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
            patch("builtins.print"),
        ):
            await bot.prefs_command(update, context)

        log_tool_mock.assert_awaited_once_with(
            "get_prefs",
            {"chat_id": 321},
            {
                "ok": True,
                "tone": "calm",
                "verbosity": "medium",
                "timezone": "Asia/Kolkata",
                "executive_mode": True,
                "mode": "exec",
            },
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "/prefs"),
                unittest.mock.call(
                    321,
                    "assistant",
                    "Current preferences:\ntone=calm\nverbosity=medium\ntimezone=Asia/Kolkata\nexecutive_mode=True\nmode=exec",
                ),
            ],
        )
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with(
            "Current preferences:\ntone=calm\nverbosity=medium\ntimezone=Asia/Kolkata\nexecutive_mode=True\nmode=exec"
        )

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_memory_command_returns_ranked_hits_and_logs(self):
        update = _DummyUpdate("/memory supplier invoices")
        context = _DummyContext(args=["supplier", "invoices"])

        with (
            patch("src.bot._save_chat_message", new=AsyncMock(side_effect=[88, None])) as save_chat_message_mock,
            patch("src.bot.embed_text", return_value=[0.1, 0.2]),
            patch(
                "src.bot.search_similar_messages",
                return_value=[
                    {"message_id": 88, "content": "/memory supplier invoices", "distance": 0.0},
                    {"message_id": 10, "content": "Neha handles vendor invoices.", "distance": 0.12},
                    {"message_id": 11, "content": "Accounts payable closes Friday.", "distance": 0.31},
                ],
            ) as search_mock,
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.memory_command(update, context)

        search_mock.assert_called_once_with(321, [0.1, 0.2], 6)
        log_tool_mock.assert_awaited_once_with(
            "memory_inspect",
            {"chat_id": 321, "query": "supplier invoices", "top_k": 5},
            {"hits": 2},
        )
        self.assertEqual(save_chat_message_mock.await_count, 2)
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with(
            "1. score=0.880 dist=0.120 Neha handles vendor invoices.\n"
            "2. score=0.690 dist=0.310 Accounts payable closes Friday."
        )

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_mode_command_updates_mode(self):
        update = _DummyUpdate("/mode creative")
        context = _DummyContext(args=["creative"])

        with (
            patch(
                "src.bot.upsert_user_preferences",
                return_value={
                    "tone": "calm",
                    "verbosity": "medium",
                    "timezone": "Asia/Kolkata",
                    "executive_mode": True,
                    "mode": "creative",
                },
            ),
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.mode_command(update, context)

        log_tool_mock.assert_awaited_once_with(
            "set_prefs",
            {"mode": "creative"},
            {
                "tone": "calm",
                "verbosity": "medium",
                "timezone": "Asia/Kolkata",
                "executive_mode": True,
                "mode": "creative",
            },
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "/mode creative"),
                unittest.mock.call(321, "assistant", "Mode set to creative."),
            ],
        )
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with("Mode set to creative.")

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_persona_command_shows_current_persona(self):
        update = _DummyUpdate("/persona")
        context = _DummyContext()

        with (
            patch(
                "src.bot.get_persona",
                return_value={
                    "name": "Akira",
                    "voice": "calm executive assistant",
                    "humor_level": 2,
                    "creativity_level": 6,
                    "signature": "",
                },
            ),
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot.save_message") as save_message_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.persona_command(update, context)

        log_tool_mock.assert_awaited_once_with(
            "get_persona",
            {"chat_id": 321},
            {
                "ok": True,
                "name": "Akira",
                "voice": "calm executive assistant",
                "humor_level": 2,
                "creativity_level": 6,
                "signature": "",
            },
        )
        self.assertEqual(
            save_message_mock.call_args_list,
            [
                unittest.mock.call(321, "user", "/persona"),
                unittest.mock.call(
                    321,
                    "assistant",
                    "Persona:\nname=Akira\nvoice=calm executive assistant\nhumor_level=2\ncreativity_level=6\nsignature=(none)",
                ),
            ],
        )
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with(
            "Persona:\nname=Akira\nvoice=calm executive assistant\nhumor_level=2\ncreativity_level=6\nsignature=(none)"
        )

    @patch("src.bot.asyncio.to_thread", new=_run_inline)
    async def test_models_command_reports_runtime_models(self):
        update = _DummyUpdate("/models")
        context = _DummyContext()

        with (
            patch("src.bot.get_config", return_value=_test_config()),
            patch("src.bot.get_embedding_dim", return_value=768),
            patch("src.bot._save_chat_message", new=AsyncMock()) as save_chat_message_mock,
            patch("src.bot._log_tool_result", new=AsyncMock()) as log_tool_mock,
            patch("src.bot._maybe_summarize", new=AsyncMock()) as maybe_summarize_mock,
        ):
            await bot.models_command(update, context)

        log_tool_mock.assert_awaited_once_with(
            "models_info",
            {"chat_id": 321},
            {
                "ok": True,
                "chat_model": "llama3.1:8b",
                "embed_model": "nomic-embed-text",
                "creative_model": "llama3.1:70b",
                "rag_top_k": 5,
                "embedding_dim": 768,
                "ollama_url": "http://localhost:11434",
                "creative_intent_routing": True,
            },
        )
        self.assertEqual(save_chat_message_mock.await_count, 2)
        maybe_summarize_mock.assert_awaited_once_with(321)
        update.message.reply_text.assert_awaited_once_with(
            "Models:\n"
            "chat=llama3.1:8b\n"
            "embed=nomic-embed-text\n"
            "creative=llama3.1:70b\n"
            "rag_top_k=5\n"
            "embedding_dim=768\n"
            "creative_intent_routing=True\n"
            "ollama_url=http://localhost:11434"
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
            patch("src.bot.get_config", return_value=_test_config()) as get_config_mock,
            patch("src.bot.init_db") as init_db_mock,
            patch("src.bot._startup_health_check") as health_check_mock,
            patch("src.bot.ApplicationBuilder", return_value=fake_builder),
        ):
            bot.main()

        get_config_mock.assert_called_once_with(validate=True)
        init_db_mock.assert_called_once_with()
        health_check_mock.assert_called_once_with()
        self.assertEqual(fake_builder.token_value, "telegram-token")
        self.assertEqual(len(fake_app.handlers), 16)
        fake_app.run_polling.assert_called_once()
