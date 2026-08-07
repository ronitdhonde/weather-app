"""
Client for the National Weather Service (NWS) API — api.weather.gov.

No API key is required. NWS does require a descriptive User-Agent header
identifying the application and a contact (their policy asks for an email
or website so they can reach you if a client misbehaves). Set these via
env vars so nothing sensitive is hardcoded, mirroring how massive_client.py
keeps auth out of source:

    WEATHER_USER_AGENT_APP   e.g. "lakebase-weather-app"
    WEATHER_USER_AGENT_EMAIL e.g. "ron@example.com"

Unlike Massive, NWS has no ticker-style symbol — locations are resolved
through a two-step lookup:
  1. geocode "City, ST" -> (lat, lon) via the free, keyless US Census
     geocoder (geocoding.geo.census.gov). Skip this step if the caller
     already passes lat/lon.
  2. lat/lon -> NWS gridpoint (office, gridX, gridY) + forecast zone /
     county via GET /points/{lat},{lon}, which is what every other NWS
     endpoint (alerts, forecast, forecast/hourly) is keyed on.
"""
import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Any

import requests

_USER_AGENT_APP = os.environ.get("WEATHER_USER_AGENT_APP", "lakebase-weather-app")
_USER_AGENT_EMAIL = os.environ.get("WEATHER_USER_AGENT_EMAIL", "contact@example.com")
_NWS_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
_CENSUS_GEOCODER_URL = os.environ.get(
    "CENSUS_GEOCODER_URL",
    "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
)
_DEFAULT_TIMEOUT = 30

# crude "City, ST" -> state abbreviation extractor for the /alerts/active?area= param
_STATE_RE = re.compile(r",\s*([A-Za-z]{2})\s*$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WeatherClient:
    """Thin wrapper around the NWS API with a retry-friendly session.

    Mirrors MassiveClient's shape (self._session, .get(), paginated-style
    helpers) but swaps bearer-token auth for the NWS User-Agent contract
    and adds a geocode + gridpoint resolution step in front of every fetch.
    """

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _NWS_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": f"({_USER_AGENT_APP}, {_USER_AGENT_EMAIL})",
                "Accept": "application/geo+json",
            }
        )

    def get(self, path_or_url: str, params: dict[str, Any] | None = None) -> Any:
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}{path_or_url}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------------
    # Location resolution
    # ---------------------------------------------------------------

    def geocode(self, location: str) -> tuple[float, float]:
        """Resolve a free-text "City, ST" address to (lat, lon) using the
        keyless US Census geocoder. Raises if no match is found."""
        params = {
            "address": location,
            "benchmark": "Public_AR_Current",
            "format": "json",
        }
        resp = requests.get(_CENSUS_GEOCODER_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        matches = resp.json().get("result", {}).get("addressMatches", [])
        if not matches:
            raise ValueError(f"Could not geocode location: {location!r}")
        coords = matches[0]["coordinates"]
        return float(coords["y"]), float(coords["x"])  # lat, lon

    def resolve_gridpoint(self, lat: float, lon: float) -> dict[str, Any]:
        """GET /points/{lat},{lon} -> office/grid + forecast URLs for that point."""
        data = self.get(f"/points/{lat:.4f},{lon:.4f}")
        props = data.get("properties", {})
        return {
            "grid_id": props.get("gridId"),
            "grid_x": props.get("gridX"),
            "grid_y": props.get("gridY"),
            "forecast_url": props.get("forecast"),
            "forecast_hourly_url": props.get("forecastHourly"),
            "state": props.get("relativeLocation", {})
            .get("properties", {})
            .get("state"),
        }

    def resolve_location(self, location: str | tuple[float, float]) -> dict[str, Any]:
        """Given "City, ST" or an (lat, lon) tuple, return lat/lon + gridpoint info."""
        if isinstance(location, tuple):
            lat, lon = location
            label = f"{lat:.4f},{lon:.4f}"
        else:
            lat, lon = self.geocode(location)
            label = location
        grid = self.resolve_gridpoint(lat, lon)
        state_match = _STATE_RE.search(label)
        return {
            "label": label,
            "lat": lat,
            "lon": lon,
            "state": grid.get("state") or (state_match.group(1).upper() if state_match else None),
            **grid,
        }

    # ---------------------------------------------------------------
    # Fetches (raw)
    # ---------------------------------------------------------------

    def get_active_alerts(self, state: str) -> list[dict]:
        """GET /alerts/active?area={state} -> list of alert GeoJSON features."""
        data = self.get("/alerts/active", params={"area": state})
        return data.get("features", [])

    def get_forecast(self, forecast_url: str) -> list[dict]:
        """GET the location's forecast URL -> list of narrative forecast periods."""
        data = self.get(forecast_url)
        return data.get("properties", {}).get("periods", [])

    def get_forecast_hourly(self, forecast_hourly_url: str) -> list[dict]:
        """GET the location's hourly forecast URL -> list of narrative periods."""
        data = self.get(forecast_hourly_url)
        return data.get("properties", {}).get("periods", [])

    # ---------------------------------------------------------------
    # Normalization -> document schema (mirrors get_news()'s "results" shape)
    # ---------------------------------------------------------------

    def normalize_alert(self, feature: dict, location_label: str) -> dict:
        props = feature.get("properties", {})
        narrative = " ".join(
            filter(None, [props.get("description"), props.get("instruction")])
        ).strip()
        return {
            "id": props.get("id") or feature.get("id"),
            "location": location_label,
            "source_type": "alert",
            "headline": props.get("event"),
            "narrative_text": narrative,
            "issued_at": props.get("sent"),
            "effective_at": props.get("effective"),
            "payload": feature,
            "synced_at": _now_iso(),
        }

    def normalize_forecast_period(
        self, period: dict, location_label: str, source_type: str = "forecast"
    ) -> dict:
        issued_at = period.get("startTime")
        dedup_key = f"{location_label}|{source_type}|{period.get('number')}|{issued_at}"
        stable_id = hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()
        return {
            "id": stable_id,
            "location": location_label,
            "source_type": source_type,
            "headline": period.get("name"),
            "narrative_text": period.get("detailedForecast") or period.get("shortForecast"),
            "issued_at": issued_at,
            "effective_at": period.get("endTime"),
            "payload": period,
            "synced_at": _now_iso(),
        }

    # ---------------------------------------------------------------
    # High-level: one location -> normalized documents
    # ---------------------------------------------------------------

    def get_documents_for_location(
        self,
        location: str | tuple[float, float],
        limit: int = 50,
        include_hourly: bool = False,
    ) -> list[dict]:
        """Fetch + normalize alerts and forecast periods for a single
        location in a small, bounded set of API calls (geocode + points +
        alerts + forecast [+ hourly forecast]).
        """
        resolved = self.resolve_location(location)
        label = resolved["label"]
        docs: list[dict] = []

        if resolved.get("state"):
            for feature in self.get_active_alerts(resolved["state"]):
                docs.append(self.normalize_alert(feature, label))

        if resolved.get("forecast_url"):
            for period in self.get_forecast(resolved["forecast_url"]):
                docs.append(self.normalize_forecast_period(period, label, "forecast"))

        if include_hourly and resolved.get("forecast_hourly_url"):
            for period in self.get_forecast_hourly(resolved["forecast_hourly_url"]):
                docs.append(
                    self.normalize_forecast_period(period, label, "forecast_hourly")
                )

        return docs[:limit] if limit else docs

    def get_documents(
        self, locations: list[str], limit: int = 50, include_hourly: bool = False
    ) -> list[dict]:
        """Fetch + normalize documents across multiple locations. Mirrors
        MassiveClient.get_news() but takes a list since NWS has no single
        multi-location endpoint the way ticker news does.
        """
        all_docs: list[dict] = []
        for location in locations:
            all_docs.extend(
                self.get_documents_for_location(
                    location, limit=limit, include_hourly=include_hourly
                )
            )
        return all_docs
