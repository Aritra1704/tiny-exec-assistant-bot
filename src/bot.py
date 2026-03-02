import asyncio
import json
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.llm import chat
from src.memory.store import get_recent_messages, init_db, log_tool, save_message
from src.prompt import SYSTEM_PROMPT
from src.router import decide
from src.tools.alarms import schedule_reminder
from src.tools.notes import tool_save_note, tool_list_notes

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
CHAT_HISTORY_LIMIT = max(1, min(int(os.getenv("CHAT_HISTORY_LIMIT", "12")), 20))
CHAT_CONTEXT_CHAR_LIMIT = max(200, min(int(os.getenv("CHAT_CONTEXT_CHAR_LIMIT", "1200")), 4000))
DEBUG_TEXT_LIMIT = 80


def _execute_tool(tool: str, args: dict, send_fn, chat_id: int) -> dict:
    if tool == "set_reminder":
        return schedule_reminder(
            send_fn=send_fn,
            chat_id=chat_id,
            in_minutes=int(args["in_minutes"]),
            message=str(args["message"]),
        )
    if tool == "save_note":
        return tool_save_note(text=str(args["text"]))
    if tool == "list_notes":
        return tool_list_notes(limit=int(args.get("limit", 10)))
    return {"ok": False, "error": f"Unknown tool: {tool}"}


def _format_tool_reply(tool: str, result: dict) -> str:
    if not result.get("ok"):
        return f"Couldn’t complete that: {result.get('error', 'unknown error')}"

    if tool == "set_reminder":
        return f"Reminder scheduled for {result['run_at']}."
    if tool == "save_note":
        return f"Saved note {result['note_id']}."
    if tool == "list_notes":
        notes = result.get("notes", [])
        if not notes:
            return "No notes saved."
        lines = [f"{note['id']}: {note['text']}" for note in notes]
        return "Latest notes:\n" + "\n".join(lines)
    return "Done."


def _should_use_router_text(router_text: str) -> bool:
    stripped = router_text.strip()
    return bool(stripped) and stripped.endswith("?")


def _truncate_for_debug(text: str, limit: int = DEBUG_TEXT_LIMIT) -> str:
    cleaned_text = text.strip()
    if len(cleaned_text) <= limit:
        return cleaned_text
    return cleaned_text[: limit - 3].rstrip() + "..."


def _trim_content_for_context(content: str) -> str:
    cleaned_content = content.strip()
    if len(cleaned_content) <= CHAT_CONTEXT_CHAR_LIMIT:
        return cleaned_content
    return cleaned_content[: CHAT_CONTEXT_CHAR_LIMIT - 3].rstrip() + "..."


def _build_chat_context(history_rows: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for row in history_rows:
        role = row.get("role")
        content = row.get("content")
        if role not in {"user", "assistant", "system"} or not isinstance(content, str):
            continue

        trimmed_content = _trim_content_for_context(content)
        if trimmed_content:
            messages.append({"role": role, "content": trimmed_content})
    return messages


async def _save_chat_message(chat_id: int, role: str, content: str) -> None:
    try:
        await asyncio.to_thread(save_message, chat_id, role, content)
    except Exception as exc:
        print(f"chat memory save failed for role={role}: {type(exc).__name__}")


async def _get_chat_history(chat_id: int) -> list[dict]:
    try:
        return await asyncio.to_thread(get_recent_messages, chat_id, CHAT_HISTORY_LIMIT)
    except Exception as exc:
        print(f"chat memory load failed: {type(exc).__name__}")
        return []


def _make_send_telegram(context: ContextTypes.DEFAULT_TYPE) -> Callable[[int, str], None]:
    loop = asyncio.get_running_loop()

    def send_telegram(target_chat_id: int, text: str) -> None:
        def _enqueue() -> None:
            context.application.create_task(
                context.bot.send_message(chat_id=target_chat_id, text=text)
            )

        loop.call_soon_threadsafe(_enqueue)

    return send_telegram


async def _run_tool_and_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    tool: str,
    args: dict,
    debug_source: str,
) -> None:
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    print(f'{debug_source} tool name="{tool}" args={json.dumps(args, sort_keys=True)}')
    send_telegram = _make_send_telegram(context)

    try:
        result = await asyncio.to_thread(_execute_tool, tool, args, send_telegram, chat_id)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}

    try:
        await asyncio.to_thread(log_tool, tool=tool, args_json=args, result_json=result)
        print("tool logged to postgres")
    except Exception as exc:
        print(f"log_tool failed for {tool}: {type(exc).__name__}")

    reply = _format_tool_reply(tool, result)
    await update.message.reply_text(reply)
    await _save_chat_message(chat_id, "assistant", reply)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    chat_id = update.effective_chat.id
    print(
        f'incoming message chat_id={chat_id} user_text="{_truncate_for_debug(user_text)}"'
    )
    await _save_chat_message(chat_id, "user", user_text)

    decision = await asyncio.to_thread(decide, user_text)
    print(f'router decision type="{decision["type"]}"')

    if decision["type"] == "tool":
        await _run_tool_and_reply(
            update,
            context,
            decision["tool"],
            decision["args"],
            debug_source="tool routing",
        )
        return

    if _should_use_router_text(decision.get("text", "")):
        reply = decision["text"]
    else:
        history_rows = await _get_chat_history(chat_id)
        messages = _build_chat_context(history_rows)
        reply = await asyncio.to_thread(chat, messages)

    await update.message.reply_text(reply)
    await _save_chat_message(chat_id, "assistant", reply)


async def note_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print(f'command /note chat_id={chat_id} user_text="{_truncate_for_debug(user_text)}"')
    await _save_chat_message(chat_id, "user", user_text)

    text = " ".join(context.args).strip()
    args = {"text": text}
    await _run_tool_and_reply(update, context, "save_note", args, debug_source="command /note")


async def notes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print(f'command /notes chat_id={chat_id} user_text="{_truncate_for_debug(user_text)}"')
    await _save_chat_message(chat_id, "user", user_text)

    raw_limit = context.args[0] if context.args else "10"
    try:
        limit = int(raw_limit)
    except ValueError:
        reply = "Use /notes or /notes <limit>."
        await update.message.reply_text(reply)
        await _save_chat_message(chat_id, "assistant", reply)
        return

    args = {"limit": limit}
    await _run_tool_and_reply(update, context, "list_notes", args, debug_source="command /notes")


async def remind_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print(f'command /remind chat_id={chat_id} user_text="{_truncate_for_debug(user_text)}"')
    await _save_chat_message(chat_id, "user", user_text)

    if len(context.args) < 2:
        reply = "Use /remind <minutes> <message>."
        await update.message.reply_text(reply)
        await _save_chat_message(chat_id, "assistant", reply)
        return

    raw_minutes = context.args[0]
    message = " ".join(context.args[1:]).strip()
    try:
        minutes = int(raw_minutes)
    except ValueError:
        reply = "Use /remind <minutes> <message>."
        await update.message.reply_text(reply)
        await _save_chat_message(chat_id, "assistant", reply)
        return

    args = {"in_minutes": minutes, "message": message}
    await _run_tool_and_reply(
        update,
        context,
        "set_reminder",
        args,
        debug_source="command /remind",
    )


def main():
    init_db()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("note", note_command))
    app.add_handler(CommandHandler("notes", notes_command))
    app.add_handler(CommandHandler("remind", remind_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
