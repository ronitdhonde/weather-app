"""
Lakebase (Databricks-managed Postgres) connection helper.

Connects using a single LAKEBASE_URL (a standard Postgres connection URL,
e.g. postgresql://role:password@host:5432/databricks_postgres?sslmode=require)
pointing at a native Postgres role with a static, non-expiring password.
This keeps setup to a single secret instead of five separate env vars.
"""
import base64
import os
from contextlib import contextmanager

import psycopg2
from databricks.sdk import WorkspaceClient
from psycopg2.extras import RealDictCursor, execute_values
from sqlalchemy import create_engine

_w = WorkspaceClient()
_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

WEATHER_DOCUMENTS_TABLE = "weather_documents"
WEATHER_EMBEDDINGS_TABLE = "weather_embeddings"


def _lakebase_url() -> str:
    """Fetch and decode the Lakebase connection URL from the Databricks secret scope."""
    secret = _w.secrets.get_secret(scope=_SCOPE, key=_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """Return a SQLAlchemy engine for Lakebase."""
    return create_engine(_lakebase_url())


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


# ---------------------------------------------------------------------
# Weather-specific helpers (weather_documents / weather_embeddings)
# ---------------------------------------------------------------------

def upsert_weather_documents(documents: list[dict]) -> int:
    """Bulk upsert normalized weather documents (from weather_client.py)
    into weather_documents, keyed on the stable `id` dedup key. Re-syncing
    the same alert/forecast period updates it in place rather than
    duplicating it.

    Each dict must have: id, location, source_type, headline,
    narrative_text, issued_at, effective_at, payload, synced_at.
    """
    if not documents:
        return 0

    rows = [
        (
            doc["id"],
            doc["location"],
            doc["source_type"],
            doc.get("headline"),
            doc["narrative_text"],
            doc.get("issued_at"),
            doc.get("effective_at"),
            psycopg2.extras.Json(doc["payload"]),
            doc.get("synced_at"),
        )
        for doc in documents
    ]

    sql = f"""
        INSERT INTO {WEATHER_DOCUMENTS_TABLE}
            (id, location, source_type, headline, narrative_text,
             issued_at, effective_at, payload, synced_at)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            location       = EXCLUDED.location,
            source_type    = EXCLUDED.source_type,
            headline       = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            issued_at      = EXCLUDED.issued_at,
            effective_at   = EXCLUDED.effective_at,
            payload        = EXCLUDED.payload,
            synced_at      = EXCLUDED.synced_at
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)
            conn.commit()
            return cur.rowcount


def upsert_weather_embeddings(chunks: list[dict]) -> int:
    """Bulk upsert chunk-level embeddings into weather_embeddings, keyed on
    (document_id, chunk_index) so re-embedding a document replaces its
    prior chunks rather than duplicating them.

    Each dict must have: document_id, chunk_index, chunk_text, embedding
    (a list/tuple of floats matching the table's VECTOR(384) column).
    """
    if not chunks:
        return 0

    rows = [
        (
            chunk["document_id"],
            chunk["chunk_index"],
            chunk["chunk_text"],
            list(chunk["embedding"]),
        )
        for chunk in chunks
    ]

    sql = f"""
        INSERT INTO {WEATHER_EMBEDDINGS_TABLE}
            (document_id, chunk_index, chunk_text, embedding)
        VALUES %s
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding  = EXCLUDED.embedding
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, template="(%s, %s, %s, %s::vector)")
            conn.commit()
            return cur.rowcount


def search_weather_embeddings(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Cosine-similarity search over weather_embeddings using pgvector's
    <=> operator, joined back to the parent document for display fields.
    """
    sql = f"""
        SELECT
            d.id, d.location, d.source_type, d.headline, d.narrative_text,
            d.issued_at, d.effective_at,
            e.chunk_text, e.chunk_index,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM {WEATHER_EMBEDDINGS_TABLE} e
        JOIN {WEATHER_DOCUMENTS_TABLE} d ON d.id = e.document_id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    return run_query(sql, (query_embedding, query_embedding, top_k))
