-- weather_schema.sql
-- Mirrors the ticker_news_documents / ticker_news_embeddings pattern:
-- one table for the raw normalized document, one table for its chunk
-- embeddings (a document can produce >1 chunk/row if narrative_text is long).

CREATE EXTENSION IF NOT EXISTS vector;

-- Raw, normalized weather documents (one row per alert or forecast period)
CREATE TABLE IF NOT EXISTS weather_documents (
    id             TEXT PRIMARY KEY,               -- NWS alert id, or hash(location|source_type|period|issued_at)
    location       TEXT NOT NULL,                  -- "Chicago, IL" or "41.8781,-87.6298"
    source_type    TEXT NOT NULL,                  -- 'alert' | 'forecast' | 'forecast_hourly'
    headline       TEXT,                           -- alert event name, or forecast period name
    narrative_text TEXT NOT NULL,                  -- the free text that gets chunked/embedded
    issued_at      TIMESTAMPTZ,
    effective_at   TIMESTAMPTZ,
    payload        JSONB NOT NULL,                 -- raw NWS feature/period JSON, for provenance
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location    ON weather_documents (location);
CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type ON weather_documents (source_type);
CREATE INDEX IF NOT EXISTS idx_weather_documents_issued_at   ON weather_documents (issued_at);

-- Chunk-level embeddings. 384 dims matches sentence-transformers
-- all-MiniLM-L6-v2 (the project's existing embedding convention) --
-- adjust the vector() dimension if a different model is used.
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id           BIGSERIAL PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES weather_documents (id) ON DELETE CASCADE,
    chunk_index  INT  NOT NULL,
    chunk_text   TEXT NOT NULL,
    embedding    VECTOR(384) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Cosine-similarity ANN index, matching pgvector's <=> operator used at query time.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_cosine
    ON weather_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
