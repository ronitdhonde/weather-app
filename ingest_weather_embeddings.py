"""
notebooks/ingest_weather_embeddings.py

Standalone batch job: reads weather_documents rows that don't yet have
embeddings, chunks + embeds their narrative_text, and writes vectors into
weather_embeddings via psycopg2. Mirrors ingest_ticker_news_embeddings.py's
role in the news pipeline, but written as a plain Python script using the
project's get_connection() helper — NOT spark.write.jdbc, which isn't
reliable against this Lakebase instance.

Run as a Databricks notebook cell, a scheduled job, or directly:

    python notebooks/ingest_weather_embeddings.py [--limit N] [--batch-size N]

Safe to re-run: get_unembedded_documents() only returns documents with
zero rows in weather_embeddings, and upsert_weather_embeddings() upserts
on (document_id, chunk_index), so a partial prior run won't duplicate rows.
"""
import argparse
import sys

from lakebase import get_unembedded_documents, upsert_weather_embeddings
from weather_embeddings import MODEL_NAME, chunk_text, embed_texts


def build_embedding_rows(documents: list[dict]) -> list[dict]:
    """Chunk + embed a batch of unembedded documents into rows ready for
    upsert_weather_embeddings(). Embedding is batched across all chunks
    from all documents in one model.encode() call for throughput.
    """
    all_chunks: list[str] = []
    chunk_meta: list[tuple[str, int]] = []

    for doc in documents:
        text = doc.get("narrative_text") or ""
        chunks = chunk_text(text)
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
            "chunk_text": chunk_value,
            "embedding": vector,
            "model_name": MODEL_NAME,
        }
        for (doc_id, chunk_index), chunk_value, vector in zip(
            chunk_meta, all_chunks, vectors
        )
    ]


def run(limit: int | None = None, batch_size: int = 200) -> None:
    documents = get_unembedded_documents(limit=limit)
    if not documents:
        print("No unembedded weather_documents found. Nothing to do.")
        return

    print(f"Found {len(documents)} unembedded document(s). Embedding with {MODEL_NAME}...")

    total_chunks_written = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        rows = build_embedding_rows(batch)
        written = upsert_weather_embeddings(rows)
        total_chunks_written += written
        print(
            f"  batch {start // batch_size + 1}: "
            f"{len(batch)} document(s) -> {written} chunk(s) written"
        )

    print(
        f"Done. {len(documents)} document(s) embedded, "
        f"{total_chunks_written} chunk row(s) written to weather_embeddings."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max unembedded documents to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Documents per embed/write batch (keeps memory + write size bounded)",
    )
    args = parser.parse_args()

    try:
        run(limit=args.limit, batch_size=args.batch_size)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)
