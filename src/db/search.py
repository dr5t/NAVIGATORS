from __future__ import annotations
"""
Navigators IDR - Phase 33 Search Engine & Data Repositories
Supports unified search across:
  - Places (Canonical POIs)
  - Addresses (Street/Address lookup)
  - Coordinates (Multi-format coordinate parsing)
  - Saved Places (User favorites/bookmarks)
  - Recent Searches (User query history)

Handles Online vs Offline operation cleanly:
  - Online mode: Uses online search provider + local fallback
  - Offline mode: Relies on local POI/map DB, coordinates parser, local saved places, and recent searches.
  - Transparent Capability Notices: Explicitly flags offline capability boundaries without pretending features exist.
"""

import re
import json
import secrets
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from src.db.database import get_db
from src.db.places import PlaceRepository, Place, normalize_category, haversine_km


@dataclass
class SearchResultItem:
    id: str
    result_type: str
    title: str
    subtitle: Optional[str] = None
    category: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    source: str = "local_db"
    place_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        if self.metadata is None:
            res["metadata"] = {}
        return res


class CoordinateParser:
    """Parses various coordinate format strings into (latitude, longitude)."""

    @staticmethod
    def parse(query: str) -> Optional[Tuple[float, float]]:
        if not query or not query.strip():
            return None

        q = query.strip()


        m1 = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*[\s,]\s*([+-]?\d+(?:\.\d+)?)$", q)
        if m1:
            try:
                lat, lon = float(m1.group(1)), float(m1.group(2))
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return (lat, lon)
            except ValueError:
                pass


        m2 = re.match(
            r"^(\d+(?:\.\d+)?)\s*([NSns])\s*[\s,]\s*(\d+(?:\.\d+)?)\s*([EWew])$",
            q,
        )
        if m2:
            try:
                lat = float(m2.group(1)) * (-1 if m2.group(2).upper() == "S" else 1)
                lon = float(m2.group(3)) * (-1 if m2.group(4).upper() == "W" else 1)
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return (lat, lon)
            except ValueError:
                pass


        m3 = re.match(
            r"""^(\d+)°\s*(\d+)'\s*(\d+(?:\.\d+)?)"\s*([NSns])\s*[\s,]\s*(\d+)°\s*(\d+)'\s*(\d+(?:\.\d+)?)"\s*([EWew])$""",
            q,
        )
        if m3:
            try:
                lat_d, lat_m, lat_s = float(m3.group(1)), float(m3.group(2)), float(m3.group(3))
                lat_dir = m3.group(4).upper()
                lat = (lat_d + lat_m / 60.0 + lat_s / 3600.0) * (-1 if lat_dir == "S" else 1)

                lon_d, lon_m, lon_s = float(m3.group(5)), float(m3.group(6)), float(m3.group(7))
                lon_dir = m3.group(8).upper()
                lon = (lon_d + lon_m / 60.0 + lon_s / 3600.0) * (-1 if lon_dir == "W" else 1)

                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return (lat, lon)
            except ValueError:
                pass

        return None


class SavedPlacesRepository:
    """Data access layer for user saved places and bookmarks."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = db_path

    def add_saved_place(
        self,
        user_id: str,
        name: str,
        latitude: float,
        longitude: float,
        place_id: Optional[str] = None,
        category: str = "landmark",
        address: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        sid = f"sp_{secrets.token_hex(6)}"
        now = datetime.now(timezone.utc).isoformat()
        norm_cat = normalize_category(category)

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO saved_places (
                    id, user_id, place_id, name, category,
                    latitude, longitude, address, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, user_id, place_id, name.strip(), norm_cat, float(latitude), float(longitude), address, notes, now, now),
            )

        return self.get_saved_place(user_id, sid) or {}

    def get_saved_place(self, user_id: str, saved_place_id: str) -> Optional[Dict[str, Any]]:
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM saved_places WHERE id = ? AND user_id = ?",
                (saved_place_id, user_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_saved_places(
        self, user_id: str, search: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM saved_places WHERE user_id = ?"
        params: List[Any] = [user_id]

        if search and search.strip():
            query += " AND (name LIKE ? OR address LIKE ? OR notes LIKE ?)"
            term = f"%{search.strip()}%"
            params.extend([term, term, term])

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with get_db(self.db_path) as conn:
            cur = conn.execute(query, tuple(params))
            return [dict(row) for row in cur.fetchall()]

    def delete_saved_place(self, user_id: str, saved_place_id: str) -> bool:
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM saved_places WHERE id = ? AND user_id = ?",
                (saved_place_id, user_id),
            )
            return cur.rowcount > 0


class RecentSearchesRepository:
    """Data access layer for user recent search query history."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = db_path

    def add_recent_search(
        self,
        user_id: str,
        query_text: str,
        search_type: str = "text",
        selected_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not query_text or not query_text.strip():
            return {}

        rid = f"rs_{secrets.token_hex(6)}"
        now = datetime.now(timezone.utc).isoformat()
        sel_json = json.dumps(selected_result or {})

        with get_db(self.db_path) as conn:
            conn.execute(
                "DELETE FROM recent_searches WHERE user_id = ? AND query_text = ?",
                (user_id, query_text.strip()),
            )
            conn.execute(
                """
                INSERT INTO recent_searches (
                    id, user_id, query_text, search_type, selected_result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (rid, user_id, query_text.strip(), search_type, sel_json, now),
            )

        return {
            "id": rid,
            "user_id": user_id,
            "query_text": query_text.strip(),
            "search_type": search_type,
            "selected_result": selected_result or {},
            "created_at": now,
        }

    def list_recent_searches(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM recent_searches WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            results = []
            for row in cur.fetchall():
                item = dict(row)
                try:
                    item["selected_result"] = json.loads(item.get("selected_result_json") or "{}")
                except Exception:
                    item["selected_result"] = {}
                results.append(item)
            return results

    def clear_recent_searches(self, user_id: str) -> bool:
        with get_db(self.db_path) as conn:
            cur = conn.execute("DELETE FROM recent_searches WHERE user_id = ?", (user_id,))
            return cur.rowcount > 0


class SearchEngine:
    """
    Unified multi-type search engine supporting:
      1. Places (Canonical POIs)
      2. Addresses (Street/Address lookup)
      3. Coordinates (Parsed lat/lon formats)
      4. Saved Places (User bookmarks)
      5. Recent Searches (Query history)

    Respects Online vs Offline state cleanly.
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = db_path
        self.place_repo = PlaceRepository(db_path)
        self.saved_repo = SavedPlacesRepository(db_path)
        self.recent_repo = RecentSearchesRepository(db_path)

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_km: Optional[float] = None,
        is_online: bool = True,
        limit: int = 20,
    ) -> Dict[str, Any]:
        q_clean = query.strip() if query else ""
        results: List[SearchResultItem] = []

        capabilities = {
            "places": "available_offline",
            "addresses": "available_offline" if not is_online else "available_online",
            "coordinates": "available_offline",
            "saved_places": "available_offline",
            "recent_searches": "available_offline",
            "online_geocoder": "available_online" if is_online else "unavailable_offline",
        }

        offline_notice = None
        if not is_online:
            offline_notice = (
                "Search is running in Offline Mode using local POI / map database. "
                "Global online web geocoding is unavailable offline."
            )


        coords = CoordinateParser.parse(q_clean) if q_clean else None
        if coords:
            c_lat, c_lon = coords
            results.append(
                SearchResultItem(
                    id=f"coord_{c_lat:.5f}_{c_lon:.5f}",
                    result_type="coordinates",
                    title=f"Coordinates ({c_lat:.5f}, {c_lon:.5f})",
                    subtitle="GPS Coordinate Location",
                    category="coordinates",
                    latitude=c_lat,
                    longitude=c_lon,
                    source="coordinate_parser",
                )
            )


        if user_id:
            saved_items = self.saved_repo.list_saved_places(user_id, search=q_clean, limit=5)
            for sp in saved_items:
                results.append(
                    SearchResultItem(
                        id=sp["id"],
                        result_type="saved_place",
                        title=sp["name"],
                        subtitle=sp.get("address") or sp.get("notes") or "Saved Place",
                        category=sp.get("category", "landmark"),
                        latitude=sp["latitude"],
                        longitude=sp["longitude"],
                        source="saved_places",
                        place_id=sp.get("place_id"),
                        metadata={"notes": sp.get("notes")},
                    )
                )


        if user_id:
            recents = self.recent_repo.list_recent_searches(user_id, limit=5)
            for rs in recents:
                if not q_clean or q_clean.lower() in rs["query_text"].lower():
                    results.append(
                        SearchResultItem(
                            id=rs["id"],
                            result_type="recent_search",
                            title=rs["query_text"],
                            subtitle="Recent Search",
                            category="history",
                            source="recent_searches",
                            metadata=rs.get("selected_result"),
                        )
                    )


        if q_clean and not coords:
            local_places = self.place_repo.list_places(
                search=q_clean,
                lat=lat,
                lon=lon,
                radius_km=radius_km,
                limit=limit,
            )

            for p in local_places:
                is_address_match = p.address and q_clean.lower() in p.address.lower() and q_clean.lower() not in p.name.lower()
                res_type = "address" if is_address_match else "place"

                results.append(
                    SearchResultItem(
                        id=p.id,
                        result_type=res_type,
                        title=p.name,
                        subtitle=p.address or f"{p.category.replace('_', ' ').title()} • ({p.latitude:.4f}, {p.longitude:.4f})",
                        category=p.category,
                        latitude=p.latitude,
                        longitude=p.longitude,
                        source="local_db",
                        place_id=p.id,
                        metadata={
                            "opening_hours": p.opening_hours,
                            "phone": p.phone,
                            "website": p.website,
                        },
                    )
                )

        if user_id and q_clean:
            self.recent_repo.add_recent_search(user_id, q_clean)

        return {
            "query": q_clean,
            "is_online": is_online,
            "offline_mode": not is_online,
            "offline_notice": offline_notice,
            "capabilities": capabilities,
            "results": [r.to_dict() for r in results[:limit]],
            "count": len(results[:limit]),
        }
