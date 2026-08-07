"""
Chunking + embedding helpers for weather documents.

Mirrors whatever chunk/embed step the news pipeline uses ahead of writing
into ticker_news_embeddings: split narrative_text into word-bounded
chunks, embed each with sentence-transformers, and shape the output for
lakebase.upsert_weather_embeddings().
"""
from sentence_transformers import SentenceTransformer

# Model name is stored on every embedding row (weather_embeddings.model_name)
# so re-embedding with a different model is traceable and queryable.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, matches VECTOR(384)
_SENTENCE_TRANSFORMERS_NAME = "all-MiniLM-L6-v2"  # short form the library expects

# Character-based sliding window, matching the existing news pipeline's
# ingest_ticker_news_embeddings.py convention. Most NWS narrative_text
# (a single forecast period, or a short alert) is well under CHUNK_SIZE
# and comes back as a single chunk; chunking mainly kicks in for combined
# alert description + instruction text on severe/complex warnings.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_SENTENCE_TRANSFORMERS_NAME)
    return _model


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping character-bounded chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts. Returns one 384-dim vector per input."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single search query the same way chunks are embedded, so
    cosine similarity between query and chunk vectors is meaningful.
    """
    return embed_texts([query])[0]


def documents_to_embedding_rows(documents: list[dict]) -> list[dict]:
    """Chunk + embed a batch of normalized weather documents (from
    weather_client.get_documents()) into rows ready for
    lakebase.upsert_weather_embeddings().
    """
    all_chunks: list[str] = []
    chunk_meta: list[tuple[str, int]] = []  # (document_id, chunk_index)

    for doc in documents:
        chunks = chunk_text(doc["narrative_text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_meta.append((doc["id"], i))

    if not all_chunks:
        return []

    vectors = embed_texts(all_chunks)

    return [
        {
            "document_id": doc_id,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text_value,
            "embedding": vector,
            "model_name": MODEL_NAME,
        }
        for (doc_id, chunk_index), chunk_text_value, vector in zip(
            chunk_meta, all_chunks, vectors
        )
    ]
