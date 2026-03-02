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

Examples:
- User: Remind me in 15 minutes to join the standup
  Output: {"tool":"set_reminder","args":{"in_minutes":15,"message":"join the standup"}}
- User: Note: buy milk
  Output: {"tool":"save_note","args":{"text":"buy milk"}}
- User: Show my notes
  Output: {"tool":"list_notes","args":{"limit":10}}

Rules:
- Output ONLY JSON when using a tool.
- If you use a tool, output ONLY JSON. No markdown, no explanation.
- Do not wrap the JSON in markdown, prose, code fences, or explanations.
- If the user is not asking for a tool, respond normally in plain text.
- If a tool request is ambiguous or missing required details, ask one short clarifying question in plain text.
- Do not invent tool arguments.
""".strip()

def _looks_like_tool_call_attempt(text: str) -> bool:
    stripped = text.strip()
    return (
        stripped.startswith("{")
        or '"tool"' in stripped
        or "'tool'" in stripped
        or '"args"' in stripped
        or "'args'" in stripped
    )


def _parse_tool_object(candidate_text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate_text)
    except json.JSONDecodeError:
        return None

    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("tool"), str)
        and isinstance(parsed.get("args"), dict)
    ):
        return parsed

    return None


def parse_tool_call(output_text: str) -> dict[str, Any]:
    # Quick sanity checks:
    # parse_tool_call('{"tool":"save_note","args":{"text":"buy milk"}}') -> {"type":"tool", ...}
    # parse_tool_call('Sure {"tool":"list_notes","args":{"limit":3}} thanks') -> {"type":"tool", ...}
    # parse_tool_call("No JSON here") -> {"type":"text","text":"No JSON here"}
    raw_text = output_text.strip()

    parsed = _parse_tool_object(raw_text)
    if parsed is None:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and start < end:
            parsed = _parse_tool_object(raw_text[start : end + 1])

    if parsed is not None:
        return {
            "type": "tool",
            "tool": parsed["tool"],
            "args": parsed["args"],
        }

    return {"type": "text", "text": output_text}


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

    decision = parse_tool_call(output)
    if decision["type"] != "tool":
        if "{" in output and not _looks_like_tool_call_attempt(output):
            print(f"router raw model output: {output}")
        return decision

    validated = _validate_tool_payload(
        {"tool": decision["tool"], "args": decision["args"]}
    )
    if validated is not None:
        return validated

    return {"type": "text", "text": output}
