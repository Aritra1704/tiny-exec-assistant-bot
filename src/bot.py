import os
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from src.prompt import SYSTEM_PROMPT
from src.router import decide
from src.tools.alarms import schedule_reminder
from src.tools.notes import tool_save_note, tool_list_notes
from src.memory.store import init_db, log_tool
from src.llm import chat

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = (update.message.text or "").strip()
    chat_id = update.message.chat_id

    # Helper for scheduler jobs (background threads) to send telegram messages safely
    def send_telegram(chat_id: int, text: str):
        context.application.create_task(
            context.bot.send_message(chat_id=chat_id, text=text)
        )

    decision = decide(user_text)

    if decision["type"] == "tool":
        tool = decision["tool"]
        args = decision["args"]

        try:
            if tool == "set_reminder":
                res = schedule_reminder(
                    send_fn=send_telegram,
                    chat_id=chat_id,
                    in_minutes=int(args["in_minutes"]),
                    message=str(args["message"]),
                )
            elif tool == "save_note":
                res = tool_save_note(text=str(args["text"]))
            elif tool == "list_notes":
                res = tool_list_notes(limit=int(args.get("limit", 10)))
            else:
                res = {"ok": False, "error": f"Unknown tool: {tool}"}

            log_tool(tool=tool, args_json=json.dumps(args), result_json=json.dumps(res))

            if res.get("ok"):
                if tool == "set_reminder":
                    reply = f"Done. I’ll remind you at {res['run_at']}."
                elif tool == "save_note":
                    reply = f"Noted (ID {res['note_id']})."
                elif tool == "list_notes":
                    notes = res["notes"]
                    if not notes:
                        reply = "No notes yet."
                    else:
                        lines = [f"{n['id']}: {n['text']}" for n in notes]
                        reply = "Your latest notes:\n" + "\n".join(lines)
                else:
                    reply = "Done."
            else:
                reply = f"Couldn’t complete that: {res.get('error','unknown error')}"

        except Exception as e:
            reply = f"Tool error: {e}"

        await update.message.reply_text(reply)
        return

    # Normal chat (with personality)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    reply = chat(messages)
    await update.message.reply_text(reply)


def main():
    init_db()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
