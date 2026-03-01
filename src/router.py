import json
from src.llm import chat

TOOL_PROMPT = """
You are Aritra's calm executive assistant with tool access.

If the user is asking to perform an action, output ONLY valid JSON in this schema:
{
  "tool": "<tool_name>",
  "args": { ... }
}

Available tools:
1) set_reminder: args { "in_minutes": int, "message": string }
2) save_note: args { "text": string }
3) list_notes: args { "limit": int }

Rules:
- Output ONLY JSON when using a tool. No extra text.
- If not using a tool, respond normally (plain text).
- If user request is ambiguous (missing minutes, unclear note content), ask one short question.
""".strip()

def decide(user_text: str):
    out = chat([
        {"role": "system", "content": TOOL_PROMPT},
        {"role": "user", "content": user_text},
    ])

    try:
        data = json.loads(out)
        if isinstance(data, dict) and "tool" in data and "args" in data:
            return {"type": "tool", "tool": data["tool"], "args": data["args"], "raw": out}
    except Exception:
        pass

    return {"type": "text", "text": out}