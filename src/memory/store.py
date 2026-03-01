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