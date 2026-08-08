"""
Flask app entry point.

Databricks Apps runs this directly (see app.yaml: command: ["python", "app.py"]).
Registers the weather blueprint (POST /weather/sync, POST /weather/search).
If you already have other blueprints (e.g. a /news blueprint from the
reference app), register them here the same way.
"""
import os

from flask import Flask, jsonify

from weather_routes import weather_bp

app = Flask(__name__)
app.register_blueprint(weather_bp)


@app.get("/")
def health():
    """Simple health check so you can confirm the app is up by hitting
    its root URL in a browser before testing the real endpoints."""
    return jsonify({"status": "ok", "service": "weather-intel-app"})


if __name__ == "__main__":
    # Databricks Apps injects the port to listen on as DATABRICKS_APP_PORT —
    # not a generic PORT var. Falls back to 8000 for local testing.
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    app.run(host="0.0.0.0", port=port)
