"""
Flask routes for the weather pipeline: harvest + embed (/weather/sync) and
semantic retrieval (/weather/search). Mirrors the news blueprint's
/news/sync shape — register this blueprint on the app the same way.

    # in app.py:
    from weather_routes import weather_bp
    app.register_blueprint(weather_bp)
"""
from flask import Blueprint, jsonify, request

from lakebase import search_weather_embeddings, upsert_weather_documents
from weather_client import WeatherClient
from weather_embeddings import embed_query

weather_bp = Blueprint("weather", __name__, url_prefix="/weather")

_MIN_TOP_K, _MAX_TOP_K, _DEFAULT_TOP_K = 1, 20, 5


@weather_bp.post("/sync")
def sync_weather():
    """
    POST /weather/sync
    Body: {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50, "include_hourly": false}

    Fetches active alerts + forecast periods for each location, normalizes
    them, and upserts into weather_documents. Returns a count of documents
    synced. Embedding is a separate step — run
    notebooks/ingest_weather_embeddings.py afterward to populate
    weather_embeddings for any newly synced documents.
    """
    body = request.get_json(force=True) or {}
    locations = body.get("locations")
    if not locations:
        return jsonify({"error": "locations is required, e.g. ['Chicago, IL']"}), 400
    limit = body.get("limit", 50)
    include_hourly = body.get("include_hourly", False)

    client = WeatherClient()
    try:
        documents = client.get_documents(
            locations, limit=limit, include_hourly=include_hourly
        )
    except Exception as exc:  # geocoding/gridpoint/network failures for a bad location, etc.
        return jsonify({"error": f"weather fetch failed: {exc}"}), 502

    documents_synced = upsert_weather_documents(documents)

    return jsonify({"locations": locations, "documents_synced": documents_synced})


@weather_bp.post("/search")
def search_weather():
    """
    POST /weather/search
    Body: {"query": "flash flood risk this weekend", "top_k": 5}

    Embeds the query and returns the most semantically similar weather
    document chunks, ranked by cosine similarity via pgvector's <=> operator.
    """
    body = request.get_json(force=True) or {}
    query = body.get("query")
    if not query or not str(query).strip():
        return jsonify({"error": "query is required"}), 400

    try:
        top_k = int(body.get("top_k", _DEFAULT_TOP_K))
    except (TypeError, ValueError):
        top_k = _DEFAULT_TOP_K
    top_k = max(_MIN_TOP_K, min(_MAX_TOP_K, top_k))

    query_vector = embed_query(query)
    try:
        results = search_weather_embeddings(query_vector, top_k=top_k)
    except Exception as exc:
        # e.g. weather_embeddings is empty / relation not yet populated —
        # surface an empty result set rather than a 500.
        return jsonify({"query": query, "results": [], "warning": str(exc)})

    return jsonify({"query": query, "results": results})
