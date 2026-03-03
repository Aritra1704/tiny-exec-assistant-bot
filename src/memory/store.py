import json

import psycopg
from psycopg.rows import dict_row

from src.config import get_config
from src.embeddings import embed_text

ALLOWED_CHAT_ROLES = {"system", "user", "assistant"}
DEFAULT_USER_PREFERENCES = {
    "tone": "calm",
    "verbosity": "medium",
    "timezone": "Asia/Kolkata",
    "executive_mode": True,
    "mode": "exec",
}
DEFAULT_PERSONA = {
    "name": "Akira",
    "voice": "calm executive assistant",
    "humor_level": 1,
    "creativity_level": 4,
    "signature": "",
}


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.12g}" for value in values) + "]"


def _conn():
    cfg = get_config(validate=True, require_telegram=False)
    if cfg.PG_PASSWORD:
        conninfo = (
            f"host={cfg.PG_HOST} port={cfg.PG_PORT} dbname={cfg.PG_DATABASE} "
            f"user={cfg.PG_USER} password={cfg.PG_PASSWORD}"
        )
    else:
        conninfo = (
            f"host={cfg.PG_HOST} port={cfg.PG_PORT} dbname={cfg.PG_DATABASE} "
            f"user={cfg.PG_USER}"
        )
    return psycopg.connect(conninfo, row_factory=dict_row)


def get_embedding_dim() -> int:
    cfg = get_config(validate=True, require_telegram=False)
    q = f"""
    SELECT dim
    FROM {cfg.PG_SCHEMA}.embedding_meta
    WHERE model = %s;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (cfg.OLLAMA_EMBED_MODEL,))
            row = cur.fetchone()
            if row is not None:
                return int(row["dim"])

            dim = len(embed_text("dimension check"))
            cur.execute(
                f"""
                INSERT INTO {cfg.PG_SCHEMA}.embedding_meta(model, dim)
                VALUES (%s, %s)
                ON CONFLICT (model) DO UPDATE SET dim = EXCLUDED.dim
                RETURNING dim;
                """,
                (cfg.OLLAMA_EMBED_MODEL, dim),
            )
            stored = cur.fetchone()
        conn.commit()
    return int(stored["dim"])


def init_db():
    cfg = get_config(validate=True, require_telegram=False)
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE SCHEMA IF NOT EXISTS {cfg.PG_SCHEMA};

    CREATE TABLE IF NOT EXISTS {cfg.PG_SCHEMA}.notes (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      text TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS {cfg.PG_SCHEMA}.tool_logs (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      tool TEXT NOT NULL,
      args JSONB NOT NULL,
      result JSONB NOT NULL
    );

    CREATE TABLE IF NOT EXISTS {cfg.PG_SCHEMA}.chat_messages (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      chat_id BIGINT NOT NULL,
      role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
      content TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS {cfg.PG_SCHEMA}.conversation_summaries (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      chat_id BIGINT NOT NULL,
      summary_text TEXT NOT NULL,
      last_message_id_covered BIGINT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS {cfg.PG_SCHEMA}.user_preferences (
      chat_id BIGINT PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      tone TEXT DEFAULT 'calm',
      verbosity TEXT DEFAULT 'medium',
      timezone TEXT DEFAULT 'Asia/Kolkata',
      executive_mode BOOLEAN DEFAULT true,
      mode TEXT DEFAULT 'exec'
    );

    CREATE TABLE IF NOT EXISTS {cfg.PG_SCHEMA}.persona (
      chat_id BIGINT PRIMARY KEY,
      name TEXT DEFAULT 'Akira',
      voice TEXT DEFAULT 'calm executive assistant',
      humor_level INT DEFAULT 1,
      creativity_level INT DEFAULT 4,
      signature TEXT DEFAULT '',
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS {cfg.PG_SCHEMA}.embedding_meta (
      model TEXT PRIMARY KEY,
      dim INT NOT NULL
    );
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(
                f"""
                ALTER TABLE {cfg.PG_SCHEMA}.user_preferences
                ADD COLUMN IF NOT EXISTS mode TEXT DEFAULT 'exec';
                """
            )
        conn.commit()

    embedding_dim = get_embedding_dim()
    embedding_ddl = f"""
    CREATE TABLE IF NOT EXISTS {cfg.PG_SCHEMA}.message_embeddings (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      chat_id BIGINT NOT NULL,
      message_id BIGINT NOT NULL,
      content TEXT NOT NULL,
      embedding vector({embedding_dim}) NOT NULL
    );

    CREATE INDEX IF NOT EXISTS message_embeddings_chat_id_idx
    ON {cfg.PG_SCHEMA}.message_embeddings(chat_id);

    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'message_embeddings_chat_id_message_id_key'
      ) THEN
        ALTER TABLE {cfg.PG_SCHEMA}.message_embeddings
        ADD CONSTRAINT message_embeddings_chat_id_message_id_key
        UNIQUE (chat_id, message_id);
      END IF;
    END $$;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(embedding_ddl)
        conn.commit()


def add_note(text: str) -> int:
    cfg = get_config(validate=True, require_telegram=False)
    q = f"INSERT INTO {cfg.PG_SCHEMA}.notes(text) VALUES (%s) RETURNING id;"
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (text,))
            note_id = cur.fetchone()["id"]
        conn.commit()
        return int(note_id)


def list_notes(limit: int = 20):
    cfg = get_config(validate=True, require_telegram=False)
    q = f"""
    SELECT id, created_at, text
    FROM {cfg.PG_SCHEMA}.notes
    ORDER BY id DESC
    LIMIT %s;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (limit,))
            return cur.fetchall()


def log_tool(tool: str, args_json: str, result_json: str):
    cfg = get_config(validate=True, require_telegram=False)
    args = json.loads(args_json) if isinstance(args_json, str) else args_json
    result = json.loads(result_json) if isinstance(result_json, str) else result_json

    q = f"""
    INSERT INTO {cfg.PG_SCHEMA}.tool_logs(tool, args, result)
    VALUES (%s, %s::jsonb, %s::jsonb);
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (tool, json.dumps(args), json.dumps(result)))
        conn.commit()


def save_message(chat_id: int, role: str, content: str) -> int:
    cfg = get_config(validate=True, require_telegram=False)
    cleaned_role = role.strip()
    cleaned_content = content.strip()
    if cleaned_role not in ALLOWED_CHAT_ROLES:
        raise ValueError(f"Unsupported chat role: {cleaned_role}")
    if not cleaned_content:
        raise ValueError("Chat message content cannot be empty.")

    q = f"""
    INSERT INTO {cfg.PG_SCHEMA}.chat_messages(chat_id, role, content)
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
    cfg = get_config(validate=True, require_telegram=False)
    safe_limit = max(1, min(int(limit), 50))
    q = f"""
    SELECT id, created_at, chat_id, role, content
    FROM (
      SELECT id, created_at, chat_id, role, content
      FROM {cfg.PG_SCHEMA}.chat_messages
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
    cfg = get_config(validate=True, require_telegram=False)
    q = f"""
    SELECT id, created_at, chat_id, summary_text, last_message_id_covered
    FROM {cfg.PG_SCHEMA}.conversation_summaries
    WHERE chat_id = %s
    ORDER BY last_message_id_covered DESC, id DESC
    LIMIT 1;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id),))
            return cur.fetchone()


def save_summary(chat_id: int, summary_text: str, last_message_id: int) -> int:
    cfg = get_config(validate=True, require_telegram=False)
    cleaned_summary = summary_text.strip()
    if not cleaned_summary:
        raise ValueError("Summary text cannot be empty.")

    q = f"""
    INSERT INTO {cfg.PG_SCHEMA}.conversation_summaries(chat_id, summary_text, last_message_id_covered)
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
    cfg = get_config(validate=True, require_telegram=False)
    q = f"""
    SELECT id, created_at, chat_id, role, content
    FROM {cfg.PG_SCHEMA}.chat_messages
    WHERE chat_id = %s AND id > %s
    ORDER BY id ASC;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id), int(last_message_id)))
            return cur.fetchall()


def get_user_preferences(chat_id: int) -> dict:
    cfg = get_config(validate=True, require_telegram=False)
    q = f"""
    SELECT chat_id, created_at, tone, verbosity, timezone, executive_mode, mode
    FROM {cfg.PG_SCHEMA}.user_preferences
    WHERE chat_id = %s;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id),))
            row = cur.fetchone()
    if row is None:
        return {"chat_id": int(chat_id), **DEFAULT_USER_PREFERENCES}
    return row


def upsert_user_preferences(chat_id: int, fields_dict: dict) -> dict:
    cfg = get_config(validate=True, require_telegram=False)
    allowed_fields = {"tone", "verbosity", "timezone", "executive_mode", "mode"}
    updates = {key: value for key, value in fields_dict.items() if key in allowed_fields}
    if not updates:
        return get_user_preferences(chat_id)

    columns = ", ".join(updates.keys())
    placeholders = ", ".join(["%s"] * len(updates))
    assignments = ", ".join(f"{column} = EXCLUDED.{column}" for column in updates.keys())
    q = f"""
    INSERT INTO {cfg.PG_SCHEMA}.user_preferences(chat_id, {columns})
    VALUES (%s, {placeholders})
    ON CONFLICT (chat_id) DO UPDATE
    SET {assignments}
    RETURNING chat_id, created_at, tone, verbosity, timezone, executive_mode, mode;
    """
    params = [int(chat_id), *updates.values()]
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            row = cur.fetchone()
        conn.commit()
    return row


def save_message_embedding(
    chat_id: int,
    message_id: int,
    content: str,
    embedding: list[float],
) -> int:
    cfg = get_config(validate=True, require_telegram=False)
    q = f"""
    INSERT INTO {cfg.PG_SCHEMA}.message_embeddings(chat_id, message_id, content, embedding)
    VALUES (%s, %s, %s, %s::vector)
    ON CONFLICT (chat_id, message_id) DO UPDATE
    SET content = EXCLUDED.content,
        embedding = EXCLUDED.embedding
    RETURNING id;
    """
    vector_value = _vector_literal(embedding)
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id), int(message_id), content.strip(), vector_value))
            row = cur.fetchone()
        conn.commit()
    return int(row["id"])


def search_similar_messages(
    chat_id: int,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    cfg = get_config(validate=True, require_telegram=False)
    safe_top_k = max(1, min(int(top_k), 20))
    vector_value = _vector_literal(query_embedding)
    q = f"""
    SELECT message_id, content, embedding <=> %s::vector AS distance
    FROM {cfg.PG_SCHEMA}.message_embeddings
    WHERE chat_id = %s
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (vector_value, int(chat_id), vector_value, safe_top_k))
            return cur.fetchall()


def embedding_exists(chat_id: int, message_id: int) -> bool:
    cfg = get_config(validate=True, require_telegram=False)
    q = f"""
    SELECT 1
    FROM {cfg.PG_SCHEMA}.message_embeddings
    WHERE chat_id = %s AND message_id = %s
    LIMIT 1;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id), int(message_id)))
            return cur.fetchone() is not None


def iter_chat_messages(after_id: int = 0, batch_size: int = 100) -> list[dict]:
    cfg = get_config(validate=True, require_telegram=False)
    safe_batch_size = max(1, min(int(batch_size), 1000))
    q = f"""
    SELECT id, chat_id, role, content
    FROM {cfg.PG_SCHEMA}.chat_messages
    WHERE id > %s
    ORDER BY id ASC
    LIMIT %s;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(after_id), safe_batch_size))
            return cur.fetchall()


def get_persona(chat_id: int) -> dict:
    cfg = get_config(validate=True, require_telegram=False)
    q = f"""
    SELECT chat_id, created_at, name, voice, humor_level, creativity_level, signature
    FROM {cfg.PG_SCHEMA}.persona
    WHERE chat_id = %s;
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (int(chat_id),))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    f"""
                    INSERT INTO {cfg.PG_SCHEMA}.persona(chat_id)
                    VALUES (%s)
                    ON CONFLICT (chat_id) DO NOTHING
                    RETURNING chat_id, created_at, name, voice, humor_level, creativity_level, signature;
                    """,
                    (int(chat_id),),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(q, (int(chat_id),))
                    row = cur.fetchone()
        conn.commit()
    if row is None:
        return {"chat_id": int(chat_id), **DEFAULT_PERSONA}
    normalized = {"chat_id": int(chat_id), **DEFAULT_PERSONA}
    normalized.update(row)
    return normalized


def upsert_persona(chat_id: int, fields_dict: dict) -> dict:
    cfg = get_config(validate=True, require_telegram=False)
    allowed_fields = {"name", "voice", "humor_level", "creativity_level", "signature"}
    updates = {key: value for key, value in fields_dict.items() if key in allowed_fields}
    if not updates:
        return get_persona(chat_id)

    columns = ", ".join(updates.keys())
    placeholders = ", ".join(["%s"] * len(updates))
    assignments = ", ".join(f"{column} = EXCLUDED.{column}" for column in updates.keys())
    q = f"""
    INSERT INTO {cfg.PG_SCHEMA}.persona(chat_id, {columns})
    VALUES (%s, {placeholders})
    ON CONFLICT (chat_id) DO UPDATE
    SET {assignments}
    RETURNING chat_id, created_at, name, voice, humor_level, creativity_level, signature;
    """
    params = [int(chat_id), *updates.values()]
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            row = cur.fetchone()
        conn.commit()
    normalized = {"chat_id": int(chat_id), **DEFAULT_PERSONA}
    normalized.update(row)
    return normalized
