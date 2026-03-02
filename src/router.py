import json
from typing import Any

from src.llm import chat

ALLOWED_TOOLS = {"set_reminder", "save_note", "list_notes"}

TOOL_PROMPT = """
You are Aritra's calm executive assistant with tool access.

If the user is asking to perform an action with an available tool, output ONLY valid JSON and nothing else:
{
  "tool": "<tool_name>",
  "args": { ... }
}

Available tools:
1) set_reminder: args { "in_minutes": integer, "message": string }
2) save_note: args { "text": string }
3) list_notes: args { "limit": integer }

Rules:
- Output ONLY JSON when using a tool.
- Do not wrap the JSON in markdown, prose, code fences, or explanations.
- If the user is not asking for a tool, respond normally in plain text.
- If a tool request is ambiguous or missing required details, ask one short clarifying question in plain text.
- Do not invent tool arguments.
""".strip()


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _looks_like_tool_call_attempt(text: str) -> bool:
    stripped = text.strip()
    return (
        stripped.startswith("{")
        or '"tool"' in stripped
        or "'tool'" in stripped
        or '"args"' in stripped
        or "'args'" in stripped
    )


def parse_tool_call(text: str) -> dict[str, Any] | None:
    # Quick sanity checks:
    # parse_tool_call('{"tool":"save_note","args":{"text":"buy milk"}}') -> {"tool": ..., "args": ...}
    # parse_tool_call('Sure: {"tool":"list_notes","args":{"limit":3}}') -> {"tool": ..., "args": ...}
    # parse_tool_call("No JSON here") -> None
    raw_text = text.strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = None

    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("tool"), str)
        and isinstance(parsed.get("args"), dict)
    ):
        return parsed

    extracted = _extract_first_json_object(raw_text)
    if not extracted:
        return None

    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError:
        return None

    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("tool"), str)
        and isinstance(parsed.get("args"), dict)
    ):
        return parsed

    return None


def _validate_tool_payload(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None

    tool = data.get("tool")
    args = data.get("args")
    if tool not in ALLOWED_TOOLS or not isinstance(args, dict):
        return None

    if tool == "set_reminder":
        in_minutes = args.get("in_minutes")
        message = args.get("message")
        if not isinstance(message, str) or not message.strip():
            return None
        try:
            parsed_minutes = int(in_minutes)
        except (TypeError, ValueError):
            return None
        return {
            "type": "tool",
            "tool": tool,
            "args": {
                "in_minutes": parsed_minutes,
                "message": message.strip(),
            },
        }

    if tool == "save_note":
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        return {
            "type": "tool",
            "tool": tool,
            "args": {"text": text.strip()},
        }

    if tool == "list_notes":
        raw_limit = args.get("limit", 10)
        try:
            parsed_limit = int(raw_limit)
        except (TypeError, ValueError):
            return None
        return {
            "type": "tool",
            "tool": tool,
            "args": {"limit": parsed_limit},
        }

    return None


def decide(user_text: str) -> dict[str, Any]:
    output = chat(
        [
            {"role": "system", "content": TOOL_PROMPT},
            {"role": "user", "content": user_text},
        ]
    ).strip()

    if _looks_like_tool_call_attempt(output):
        print(f"router raw model output: {output}")

    parsed = parse_tool_call(output)
    if parsed is None:
        if "{" in output and not _looks_like_tool_call_attempt(output):
            print(f"router raw model output: {output}")
        return {"type": "text", "text": output}

    decision = _validate_tool_payload(parsed)
    if decision is not None:
        return decision

    return {"type": "text", "text": output}
