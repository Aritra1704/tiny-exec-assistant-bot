from src.llm import chat
from src.memory.store import get_last_summary, get_messages_after, save_summary

SUMMARY_TRIGGER_THRESHOLD = 30
DETAILED_MESSAGES_TO_KEEP = 15

SUMMARY_SYSTEM_PROMPT = """
Summarize this conversation in concise executive style.
Preserve decisions, tasks, commitments.
Return plain text only.
""".strip()


def _format_messages_for_summary(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        role = row.get("role", "assistant")
        content = str(row.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def maybe_summarize(chat_id: int) -> dict:
    all_messages = get_messages_after(chat_id, 0)
    if len(all_messages) <= SUMMARY_TRIGGER_THRESHOLD:
        return {"ok": False, "reason": "below_threshold", "compressed_count": 0}

    last_summary = get_last_summary(chat_id)
    last_message_id_covered = (
        int(last_summary["last_message_id_covered"]) if last_summary is not None else 0
    )
    unsummarized_messages = get_messages_after(chat_id, last_message_id_covered)
    messages_to_compress = unsummarized_messages[:-DETAILED_MESSAGES_TO_KEEP]
    if not messages_to_compress:
        return {"ok": False, "reason": "nothing_to_compress", "compressed_count": 0}

    prompt_sections = []
    if last_summary is not None:
        prompt_sections.append(
            "Existing summary:\n" + str(last_summary["summary_text"]).strip()
        )
    prompt_sections.append(
        "Messages to fold into the summary:\n"
        + _format_messages_for_summary(messages_to_compress)
    )

    summary_text = chat(
        [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(prompt_sections)},
        ]
    ).strip()

    last_message_id = int(messages_to_compress[-1]["id"])
    save_summary(chat_id, summary_text, last_message_id)
    print(
        f"summarization ran for chat_id {chat_id}; compressed {len(messages_to_compress)} messages"
    )
    return {
        "ok": True,
        "chat_id": int(chat_id),
        "compressed_count": len(messages_to_compress),
        "last_message_id_covered": last_message_id,
        "summary_text": summary_text,
    }
