from src.memory.store import add_note, list_notes

def tool_save_note(text: str):
    note_id = add_note(text)
    return {"ok": True, "note_id": note_id, "saved": text}

def tool_list_notes(limit: int = 10):
    rows = list_notes(limit)
    items = [{"id": r["id"], "created_at": str(r["created_at"]), "text": r["text"]} for r in rows]
    return {"ok": True, "notes": items}