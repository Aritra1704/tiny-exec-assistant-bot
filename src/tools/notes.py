from src.memory.store import add_note, list_notes


def tool_save_note(text: str) -> dict:
    cleaned_text = text.strip()
    if not cleaned_text:
        return {"ok": False, "error": "Note text cannot be empty."}

    note_id = add_note(cleaned_text)
    return {"ok": True, "note_id": note_id, "saved": cleaned_text}


def tool_list_notes(limit: int = 10) -> dict:
    safe_limit = max(1, min(int(limit), 50))
    rows = list_notes(safe_limit)
    items = [
        {
            "id": int(row["id"]),
            "created_at": row["created_at"].isoformat(),
            "text": row["text"],
        }
        for row in rows
    ]
    return {"ok": True, "limit": safe_limit, "notes": items}
