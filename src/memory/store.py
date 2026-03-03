import os
import json
from pathlib import Path

from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE", "postgres")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_SCHEMA = os.getenv("PG_SCHEMA", "tinyse")
ALLOWED_CHAT_ROLES = {"system", "user", "assistant"}

def _conn():
    if PG_PASSWORD:
        conninfo = (
            f"host={PG_HOST} port={PG_PORT} dbname={PG_DATABASE} "
            f"user={PG_USER} password={PG_PASSWORD}"
        )
    else:
        conninfo = f"host={PG_HOST} port={PG_PORT} dbname={PG_DATABASE} user={PG_USER}"

    return psycopg.connect(conninfo, row_factory=dict_row)

def init_db():
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {PG_SCHEMA};

    CREATE TABLE IF NOT EXISTS {PG_SCHEMA}.notes (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      text TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS {PG_SCHEMA}.tool_logs (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      tool TEXT NOT NULL,
      args JSONB NOT NULL,
      result JSONB NOT NULL
    );

    CREATE TABLE IF NOT EXISTS {PG_SCHEMA}.chat_messages (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      chat_id BIGINT NOT NULL,
      role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
      content TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS {PG_SCHEMA}.conversation_summaries (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      chat_id BIGINT NOT NULL,
      summary_text TEXT NOT NULL,
      last_message_id_covered BIGINT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS {PG_SCHEMA}.user_preferences (
      chat_id BIGINT PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      tone TEXT DEFAULT 'calm',
      verbosity TEXT DEFAULT 'medium',
      timezone TEXT DEFAULT 'Asia/Kolkata',
      executive_mode BOOLEAN DEFAULT true
    );
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

def add_note(text: str) -> int:
    q = f"INSERT INTO {PG_SCHEMA}.notes(text) VALUES (%s) RETURNING id;"
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (text,))
            note_id = cur.fetchone()["id"]
        conn.commit()
        return int(note_id)

def list_notes(limit: int = 20):
    q = f"""
    SELECT id, created_at, text
    FROM {PG_SCHEMA}.notes
    ORDER BY id DESC
    LIMIT %s;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (limit,))
            return cur.fetchall()

def log_tool(tool: str, args_json: str, result_json: str):
    args = json.loads(args_json) if isinstance(args_json, str) else args_json
    result = json.loads(result_json) if isinstance(result_json, str) else result_json

    q = f"""
    INSERT INTO {PG_SCHEMA}.tool_logs(tool, args, result)
    VALUES (%s, %s::jsonb, %s::jsonb);
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (tool, json.dumps(args), json.dumps(result)))
        conn.commit()


def save_message(chat_id: int, role: str, content: str) -> int:
    cleaned_role = role.strip()
    cleaned_content = content.strip()
    if cleaned_role not in ALLOWED_CHAT_ROLES:
        raise ValueError(f"Unsupported chat role: {cleaned_role}")
    if not cleaned_content:
        raise ValueError("Chat message content cannot be empty.")

    q = f"""
    INSERT INTO {PG_SCHEMA}.chat_messages(chat_id, role, content)
    VALUES (%s, %s, %s)
    RETURNING id;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id), cleaned_role, cleaned_content))
            message_id = cur.fetchone()["id"]
        conn.commit()
    return int(message_id)


def get_recent_messages(chat_id: int, limit: int = 20) -> list[dict]:
    safe_limit = max(1, min(int(limit), 50))
    q = f"""
    SELECT id, created_at, chat_id, role, content
    FROM (
      SELECT id, created_at, chat_id, role, content
      FROM {PG_SCHEMA}.chat_messages
      WHERE chat_id = %s
      ORDER BY id DESC
      LIMIT %s
    ) recent
    ORDER BY id ASC;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id), safe_limit))
            return cur.fetchall()


def get_last_summary(chat_id: int) -> dict | None:
    q = f"""
    SELECT id, created_at, chat_id, summary_text, last_message_id_covered
    FROM {PG_SCHEMA}.conversation_summaries
    WHERE chat_id = %s
    ORDER BY last_message_id_covered DESC, id DESC
    LIMIT 1;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id),))
            return cur.fetchone()


def save_summary(chat_id: int, summary_text: str, last_message_id: int) -> int:
    cleaned_summary = summary_text.strip()
    if not cleaned_summary:
        raise ValueError("Summary text cannot be empty.")

    q = f"""
    INSERT INTO {PG_SCHEMA}.conversation_summaries(chat_id, summary_text, last_message_id_covered)
    VALUES (%s, %s, %s)
    RETURNING id;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id), cleaned_summary, int(last_message_id)))
            summary_id = cur.fetchone()["id"]
        conn.commit()
    return int(summary_id)


def get_messages_after(chat_id: int, last_message_id: int) -> list[dict]:
    q = f"""
    SELECT id, created_at, chat_id, role, content
    FROM {PG_SCHEMA}.chat_messages
    WHERE chat_id = %s AND id > %s
    ORDER BY id ASC;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id), int(last_message_id)))
            return cur.fetchall()


def get_user_preferences(chat_id: int) -> dict:
    select_q = f"""
    SELECT chat_id, created_at, tone, verbosity, timezone, executive_mode
    FROM {PG_SCHEMA}.user_preferences
    WHERE chat_id = %s;
    """
    insert_q = f"""
    INSERT INTO {PG_SCHEMA}.user_preferences(chat_id)
    VALUES (%s)
    ON CONFLICT (chat_id) DO NOTHING
    RETURNING chat_id, created_at, tone, verbosity, timezone, executive_mode;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(select_q, (int(chat_id),))
            row = cur.fetchone()
            if row is None:
                cur.execute(insert_q, (int(chat_id),))
                row = cur.fetchone()
                if row is None:
                    cur.execute(select_q, (int(chat_id),))
                    row = cur.fetchone()
        conn.commit()
    return row


def upsert_user_preferences(chat_id: int, fields_dict: dict) -> dict:
    allowed_fields = {"tone", "verbosity", "timezone", "executive_mode"}
    updates = {key: value for key, value in fields_dict.items() if key in allowed_fields}
    if not updates:
        return get_user_preferences(chat_id)

    columns = ", ".join(updates.keys())
    placeholders = ", ".join(["%s"] * len(updates))
    assignments = ", ".join(f"{column} = EXCLUDED.{column}" for column in updates.keys())
    q = f"""
    INSERT INTO {PG_SCHEMA}.user_preferences(chat_id, {columns})
    VALUES (%s, {placeholders})
    ON CONFLICT (chat_id) DO UPDATE
    SET {assignments}
    RETURNING chat_id, created_at, tone, verbosity, timezone, executive_mode;
    """
    params = [int(chat_id), *updates.values()]
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            row = cur.fetchone()
        conn.commit()
    return row
