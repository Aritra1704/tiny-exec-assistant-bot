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

from src.embeddings import embed_text
from src.llm import chat
from src.memory.store import (
    get_last_summary,
    get_messages_after,
    get_user_preferences,
    init_db,
    log_tool,
    save_message_embedding,
    save_message,
    search_similar_messages,
    upsert_user_preferences,
)
from src.memory.summarizer import maybe_summarize
from src.prompt import SYSTEM_PROMPT
from src.router import decide
from src.tools.alarms import schedule_reminder
from src.tools.notes import tool_save_note, tool_list_notes

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
CHAT_HISTORY_LIMIT = 20
DETAILED_HISTORY_LIMIT = 15
CHAT_CONTEXT_CHAR_LIMIT = max(200, min(int(os.getenv("CHAT_CONTEXT_CHAR_LIMIT", "1200")), 4000))
DEBUG_TEXT_LIMIT = 80
RAG_TOP_K = max(1, min(int(os.getenv("RAG_TOP_K", "5")), 10))
VALID_TONES = {"calm", "strict", "casual"}
VALID_VERBOSITY = {"short", "medium", "detailed"}


def _normalize_preferences(preferences: dict | None) -> dict:
    base = {
        "tone": "calm",
        "verbosity": "medium",
        "timezone": "Asia/Kolkata",
        "executive_mode": True,
    }
    if preferences:
        base.update(
            {
                "tone": preferences.get("tone", base["tone"]),
                "verbosity": preferences.get("verbosity", base["verbosity"]),
                "timezone": preferences.get("timezone", base["timezone"]),
                "executive_mode": bool(preferences.get("executive_mode", base["executive_mode"])),
            }
        )
    return base


def _build_dynamic_system_prompt(preferences: dict | None) -> str:
    normalized = _normalize_preferences(preferences)
    additions = [SYSTEM_PROMPT]

    tone = normalized["tone"]
    if tone == "strict":
        additions.append("Tone: strict, direct, and disciplined.")
    elif tone == "casual":
        additions.append("Tone: casual, conversational, and approachable.")
    else:
        additions.append("Tone: calm, steady, and practical.")

    verbosity = normalized["verbosity"]
    if verbosity == "short":
        additions.append("Be concise.")
    elif verbosity == "detailed":
        additions.append("Give more detail.")
    else:
        additions.append("Keep responses medium length unless more detail is necessary.")

    if normalized["executive_mode"]:
        additions.append("Maintain an executive assistant mindset and prioritize decisions, tasks, and outcomes.")
    else:
        additions.append("Do not force executive assistant phrasing unless the user asks for it.")

    additions.append(f"Assume the user's timezone is {normalized['timezone']} when time context matters.")
    return "\n\n".join(additions)


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


def _trim_relevant_memory(content: str) -> str:
    trimmed_content = _trim_content_for_context(content)
    if len(trimmed_content) <= 180:
        return trimmed_content
    return trimmed_content[:177].rstrip() + "..."


def _build_chat_context(
    history_rows: list[dict],
    summary_text: str | None = None,
    preferences: dict | None = None,
    relevant_memory: list[str] | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": _build_dynamic_system_prompt(preferences)}]
    if summary_text:
        messages.append(
            {
                "role": "system",
                "content": f"Conversation summary so far: {summary_text.strip()}",
            }
        )
    if relevant_memory:
        messages.append(
            {
                "role": "system",
                "content": "Relevant memory (semantic):\n"
                + "\n".join(f"- {snippet}" for snippet in relevant_memory),
            }
        )
    for row in history_rows:
        role = row.get("role")
        content = row.get("content")
        if role not in {"user", "assistant", "system"} or not isinstance(content, str):
            continue

        trimmed_content = _trim_content_for_context(content)
        if trimmed_content:
            messages.append({"role": role, "content": trimmed_content})
    return messages


async def _save_message_embedding(chat_id: int, message_id: int, content: str) -> None:
    try:
        embedding = await asyncio.to_thread(embed_text, content)
        await asyncio.to_thread(
            save_message_embedding,
            chat_id,
            message_id,
            content,
            embedding,
        )
    except Exception as exc:
        print(f"embedding save failed for chat_id {chat_id}: {type(exc).__name__}")


async def _save_chat_message(chat_id: int, role: str, content: str) -> int | None:
    try:
        message_id = await asyncio.to_thread(save_message, chat_id, role, content)
    except Exception as exc:
        print(f"chat memory save failed for role={role}: {type(exc).__name__}")
        return None

    if role in {"user", "assistant"} and isinstance(message_id, int):
        await _save_message_embedding(chat_id, message_id, content)
    return message_id


async def _maybe_summarize(chat_id: int) -> None:
    try:
        await asyncio.to_thread(maybe_summarize, chat_id)
    except Exception as exc:
        print(f"summarization failed for chat_id {chat_id}: {type(exc).__name__}")


async def _get_preferences(chat_id: int) -> dict:
    try:
        preferences = await asyncio.to_thread(get_user_preferences, chat_id)
        normalized = _normalize_preferences(preferences)
        print(f"loaded preferences for chat_id {chat_id}: {json.dumps(normalized, sort_keys=True)}")
        return normalized
    except Exception as exc:
        print(f"preferences load failed for chat_id {chat_id}: {type(exc).__name__}")
        return _normalize_preferences(None)


async def _get_chat_history(chat_id: int) -> tuple[str | None, list[dict]]:
    try:
        last_summary = await asyncio.to_thread(get_last_summary, chat_id)
        last_message_id_covered = (
            int(last_summary["last_message_id_covered"]) if last_summary is not None else 0
        )
        history_rows = await asyncio.to_thread(get_messages_after, chat_id, last_message_id_covered)
        detailed_rows = history_rows[-DETAILED_HISTORY_LIMIT:]
        print(f"Loaded {len(detailed_rows)} previous messages for chat_id {chat_id}")
        summary_text = str(last_summary["summary_text"]) if last_summary is not None else None
        return summary_text, detailed_rows
    except Exception as exc:
        print(f"chat memory load failed: {type(exc).__name__}")
        return None, []


async def _get_relevant_memory(
    chat_id: int,
    user_text: str,
    current_message_id: int | None,
) -> list[str]:
    try:
        query_embedding = await asyncio.to_thread(embed_text, user_text)
        raw_hits = await asyncio.to_thread(
            search_similar_messages,
            chat_id,
            query_embedding,
            RAG_TOP_K + 3,
        )
        snippets = []
        seen = set()
        for hit in raw_hits:
            if current_message_id is not None and int(hit.get("message_id", 0)) == current_message_id:
                continue

            snippet = _trim_relevant_memory(str(hit.get("content", "")))
            if not snippet:
                continue

            dedupe_key = snippet.lower()
            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            snippets.append(snippet)
            if len(snippets) == RAG_TOP_K:
                break

        await _log_tool_result(
            "semantic_retrieval",
            {"chat_id": chat_id, "top_k": RAG_TOP_K},
            {"hits": len(snippets)},
        )
        return snippets
    except Exception as exc:
        print(f"semantic retrieval failed for chat_id {chat_id}: {type(exc).__name__}")
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


async def _run_tool(tool: str, args: dict, context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> dict:
    send_telegram = _make_send_telegram(context)
    try:
        return await asyncio.to_thread(_execute_tool, tool, args, send_telegram, chat_id)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _log_tool_result(tool: str, args: dict, result: dict) -> None:
    try:
        await asyncio.to_thread(log_tool, tool=tool, args_json=args, result_json=result)
        print("tool logged to postgres")
    except Exception as exc:
        print(f"log_tool failed for {tool}: {type(exc).__name__}")


async def _run_preference_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    fields: dict,
    reply_text: str | None = None,
) -> None:
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    try:
        result = await asyncio.to_thread(upsert_user_preferences, chat_id, fields)
        await _log_tool_result("set_prefs", fields, result)
        preferences = _normalize_preferences(result)
        reply = reply_text or (
            "Preferences updated:\n"
            f"tone={preferences['tone']}\n"
            f"verbosity={preferences['verbosity']}\n"
            f"timezone={preferences['timezone']}\n"
            f"executive_mode={preferences['executive_mode']}"
        )
    except Exception as exc:
        reply = f"Couldn’t update preferences: {type(exc).__name__}"

    await update.message.reply_text(reply)
    await _save_chat_message(chat_id, "assistant", reply)
    await _maybe_summarize(chat_id)


def _format_command_reply(tool: str, result: dict) -> str:
    if not result.get("ok"):
        return f"Couldn’t complete that: {result.get('error', 'unknown error')}"

    if tool == "save_note":
        return f"Noted (ID {result['note_id']})."
    if tool == "list_notes":
        notes = result.get("notes", [])
        if not notes:
            return "No notes yet."
        return "\n".join(f"{note['id']}: {note['text']}" for note in notes)
    if tool == "set_reminder":
        return f"Done. I'll remind you at {result['run_at']}."
    return "Done."


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
    result = await _run_tool(tool, args, context, chat_id)
    await _log_tool_result(tool, args, result)

    reply = _format_tool_reply(tool, result)
    await update.message.reply_text(reply)
    await _save_chat_message(chat_id, "assistant", reply)
    await _maybe_summarize(chat_id)


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
    user_message_id = await _save_chat_message(chat_id, "user", user_text)
    preferences = await _get_preferences(chat_id)

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
        summary_text, history_rows = await _get_chat_history(chat_id)
        relevant_memory = await _get_relevant_memory(chat_id, user_text, user_message_id)
        messages = _build_chat_context(
            history_rows,
            summary_text=summary_text,
            preferences=preferences,
            relevant_memory=relevant_memory,
        )
        reply = await asyncio.to_thread(chat, messages)

    await update.message.reply_text(reply)
    await _save_chat_message(chat_id, "assistant", reply)
    await _maybe_summarize(chat_id)


async def note_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print(f"CMD /note args={json.dumps(context.args)}")
    await _save_chat_message(chat_id, "user", user_text)

    text = " ".join(context.args).strip()
    args = {"text": text}
    result = await _run_tool("save_note", args, context, chat_id)
    await _log_tool_result("save_note", args, result)
    reply = _format_command_reply("save_note", result)
    await update.message.reply_text(reply)
    await _save_chat_message(chat_id, "assistant", reply)
    await _maybe_summarize(chat_id)


async def notes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print(f"CMD /notes args={json.dumps(context.args)}")
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
    result = await _run_tool("list_notes", args, context, chat_id)
    await _log_tool_result("list_notes", args, result)
    reply = _format_command_reply("list_notes", result)
    await update.message.reply_text(reply)
    await _save_chat_message(chat_id, "assistant", reply)
    await _maybe_summarize(chat_id)


async def remind_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print(f"CMD /remind args={json.dumps(context.args)}")
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
    result = await _run_tool("set_reminder", args, context, chat_id)
    await _log_tool_result("set_reminder", args, result)
    reply = _format_command_reply("set_reminder", result)
    await update.message.reply_text(reply)
    await _save_chat_message(chat_id, "assistant", reply)
    await _maybe_summarize(chat_id)


async def set_tone_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print(f"CMD /set_tone args={json.dumps(context.args)}")
    await _save_chat_message(chat_id, "user", user_text)

    tone = (context.args[0].strip().lower() if context.args else "")
    if tone not in VALID_TONES:
        reply = "Use /set_tone calm, /set_tone strict, or /set_tone casual."
        await update.message.reply_text(reply)
        await _save_chat_message(chat_id, "assistant", reply)
        await _maybe_summarize(chat_id)
        return

    await _run_preference_command(
        update,
        context,
        {"tone": tone},
        reply_text=f"Tone set to {tone}.",
    )


async def set_verbosity_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print(f"CMD /set_verbosity args={json.dumps(context.args)}")
    await _save_chat_message(chat_id, "user", user_text)

    verbosity = (context.args[0].strip().lower() if context.args else "")
    if verbosity not in VALID_VERBOSITY:
        reply = "Use /set_verbosity short, /set_verbosity medium, or /set_verbosity detailed."
        await update.message.reply_text(reply)
        await _save_chat_message(chat_id, "assistant", reply)
        await _maybe_summarize(chat_id)
        return

    await _run_preference_command(
        update,
        context,
        {"verbosity": verbosity},
        reply_text=f"Verbosity set to {verbosity}.",
    )


async def set_timezone_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print(f"CMD /set_timezone args={json.dumps(context.args)}")
    await _save_chat_message(chat_id, "user", user_text)

    timezone_value = " ".join(context.args).strip()
    if not timezone_value:
        reply = "Use /set_timezone <tz>."
        await update.message.reply_text(reply)
        await _save_chat_message(chat_id, "assistant", reply)
        await _maybe_summarize(chat_id)
        return

    await _run_preference_command(
        update,
        context,
        {"timezone": timezone_value},
        reply_text=f"Timezone set to {timezone_value}.",
    )


async def exec_mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print(f"CMD /exec_mode args={json.dumps(context.args)}")
    await _save_chat_message(chat_id, "user", user_text)

    raw_value = (context.args[0].strip().lower() if context.args else "")
    if raw_value not in {"on", "off"}:
        reply = "Use /exec_mode on or /exec_mode off."
        await update.message.reply_text(reply)
        await _save_chat_message(chat_id, "assistant", reply)
        await _maybe_summarize(chat_id)
        return

    executive_mode = raw_value == "on"
    await _run_preference_command(
        update,
        context,
        {"executive_mode": executive_mode},
        reply_text=f"Executive mode {'enabled' if executive_mode else 'disabled'}.",
    )


async def prefs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    print("CMD /prefs args=[]")
    await _save_chat_message(chat_id, "user", user_text)

    preferences = await _get_preferences(chat_id)
    result = {"ok": True, **preferences}
    await _log_tool_result("get_prefs", {"chat_id": chat_id}, result)
    reply = (
        "Current preferences:\n"
        f"tone={preferences['tone']}\n"
        f"verbosity={preferences['verbosity']}\n"
        f"timezone={preferences['timezone']}\n"
        f"executive_mode={preferences['executive_mode']}"
    )
    await update.message.reply_text(reply)
    await _save_chat_message(chat_id, "assistant", reply)
    await _maybe_summarize(chat_id)


def main():
    init_db()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("note", note_command))
    app.add_handler(CommandHandler("notes", notes_command))
    app.add_handler(CommandHandler("remind", remind_command))
    app.add_handler(CommandHandler("set_tone", set_tone_command))
    app.add_handler(CommandHandler("set_verbosity", set_verbosity_command))
    app.add_handler(CommandHandler("set_timezone", set_timezone_command))
    app.add_handler(CommandHandler("exec_mode", exec_mode_command))
    app.add_handler(CommandHandler("prefs", prefs_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    print("tinySe is running ✅ (polling started)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
