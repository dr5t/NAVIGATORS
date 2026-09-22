"""
Navigators IDR - Canonical Places Repository & Versioned Audit Layer
Manages canonical map places, immutable version history snapshots,
and soft delete / archiving to prevent casual data loss.
"""

import json
import secrets
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from src.db.database import get_db
from src.db.audit import AuditRepository


@dataclass
class Place:
    id: str
    name: str
    category: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    opening_hours: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    metadata_json: str = "{}"
    status: str = "published"
    version: int = 1
    is_deleted: int = 0
    created_by: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        try:
            res["metadata"] = json.loads(self.metadata_json)
        except (ValueError, TypeError):
            res["metadata"] = {}
        return res


@dataclass
class PlaceHistory:
    id: str
    place_id: str
    version: int
    action: str
    changed_by: Optional[str]
    snapshot_json: str
    change_summary: Optional[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        try:
            res["snapshot"] = json.loads(self.snapshot_json)
        except (ValueError, TypeError):
            res["snapshot"] = {}
        return res


class PlaceRepository:
    """Data access and audit layer for canonical map places."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self._db_path = db_path
        self.audit = AuditRepository(db_path)

    @property
    def db_path(self) -> Optional[str | Path]:
        return self._db_path

    @db_path.setter
    def db_path(self, val: Optional[str | Path]):
        self._db_path = val
        self.audit.db_path = val

    def create_place(
        self,
        name: str,
        category: str,
        latitude: float,
        longitude: float,
        address: Optional[str] = None,
        opening_hours: Optional[str] = None,
        phone: Optional[str] = None,
        website: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
        place_id: Optional[str] = None,
    ) -> Place:
        """
        Create a new published canonical place record and write version 1 history.
        """
        pid = place_id or f"plc_{secrets.token_hex(6)}"
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata or {})

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO places (
                    id, name, category, latitude, longitude,
                    address, opening_hours, phone, website,
                    metadata_json, status, version, is_deleted,
                    created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'published', 1, 0, ?, ?, ?)
                """,
                (
                    pid, name.strip(), category.strip().lower(), float(latitude), float(longitude),
                    address.strip() if address else None,
                    opening_hours.strip() if opening_hours else None,
                    phone.strip() if phone else None,
                    website.strip() if website else None,
                    meta_str, created_by, now, now
                ),
            )

        place = self.get_place(pid, include_deleted=True)
        if not place:
            raise RuntimeError(f"Failed to create place {pid}")

        # Record version 1 snapshot in place_history
        self._record_history(
            place=place,
            action="created",
            changed_by=created_by,
            summary="Initial canonical place publication",
        )

        # Log platform audit record
        try:
            self.audit.log(
                action="CREATE",
                resource_type="place",
                resource_id=place.id,
                actor_id=created_by,
                new_state="published",
                metadata={"name": place.name, "category": place.category},
            )
        except Exception:
            pass

        return place

    def get_place(self, place_id: str, include_deleted: bool = False) -> Optional[Place]:
        """Fetch canonical place by ID. Filters out soft-deleted places unless requested."""
        query = "SELECT * FROM places WHERE id = ?"
        if not include_deleted:
            query += " AND is_deleted = 0 AND status = 'published'"

        with get_db(self.db_path) as conn:
            cur = conn.execute(query, (place_id,))
            row = cur.fetchone()
            if not row:
                return None
            return Place(**dict(row))

    def update_place(
        self,
        place_id: str,
        changed_by: Optional[str] = None,
        name: Optional[str] = None,
        category: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        address: Optional[str] = None,
        opening_hours: Optional[str] = None,
        phone: Optional[str] = None,
        website: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        change_summary: Optional[str] = None,
    ) -> Place:
        """
        Update canonical place attributes, increment version, and record history snapshot.
        """
        current = self.get_place(place_id, include_deleted=True)
        if not current:
            raise ValueError(f"Place '{place_id}' not found.")

        fields: List[str] = []
        values: List[Any] = []

        if name is not None:
            fields.append("name = ?")
            values.append(name.strip())
        if category is not None:
            fields.append("category = ?")
            values.append(category.strip().lower())
        if latitude is not None:
            fields.append("latitude = ?")
            values.append(float(latitude))
        if longitude is not None:
            fields.append("longitude = ?")
            values.append(float(longitude))
        if address is not None:
            fields.append("address = ?")
            values.append(address.strip())
        if opening_hours is not None:
            fields.append("opening_hours = ?")
            values.append(opening_hours.strip())
        if phone is not None:
            fields.append("phone = ?")
            values.append(phone.strip())
        if website is not None:
            fields.append("website = ?")
            values.append(website.strip())
        if metadata is not None:
            fields.append("metadata_json = ?")
            values.append(json.dumps(metadata))

        new_version = current.version + 1
        fields.append("version = ?")
        values.append(new_version)

        now = datetime.now(timezone.utc).isoformat()
        fields.append("updated_at = ?")
        values.append(now)

        values.append(place_id)
        query = f"UPDATE places SET {', '.join(fields)} WHERE id = ?"

        with get_db(self.db_path) as conn:
            conn.execute(query, tuple(values))

        updated = self.get_place(place_id, include_deleted=True)
        if not updated:
            raise RuntimeError(f"Failed to update place {place_id}")

        self._record_history(
            place=updated,
            action="updated",
            changed_by=changed_by,
            summary=change_summary or "Place details updated",
        )

        # Log platform audit record
        try:
            self.audit.log(
                action="UPDATE",
                resource_type="place",
                resource_id=place_id,
                actor_id=changed_by,
                old_state=f"v{current.version}",
                new_state=f"v{updated.version}",
                metadata={"change_summary": change_summary or "Place details updated"},
            )
        except Exception:
            pass

        return updated

    def soft_delete_place(
        self,
        place_id: str,
        changed_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Place:
        """
        Soft delete / archive place. Does NOT remove record physically.
        Maintains history and prevents accidental data loss.
        """
        current = self.get_place(place_id, include_deleted=True)
        if not current:
            raise ValueError(f"Place '{place_id}' not found.")

        now = datetime.now(timezone.utc).isoformat()
        new_version = current.version + 1

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE places
                SET is_deleted = 1, status = 'archived', version = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_version, now, place_id),
            )

        archived = self.get_place(place_id, include_deleted=True)
        if not archived:
            raise RuntimeError(f"Failed to soft delete place {place_id}")

        self._record_history(
            place=archived,
            action="soft_deleted",
            changed_by=changed_by,
            summary=reason or "Place archived / soft deleted",
        )

        # Log platform audit record
        try:
            self.audit.log(
                action="SOFT_DELETE",
                resource_type="place",
                resource_id=place_id,
                actor_id=changed_by,
                old_state="published",
                new_state="archived",
                metadata={"reason": reason or "Place archived / soft deleted"},
            )
        except Exception:
            pass

        return archived

    def archive_place(
        self,
        place_id: str,
        changed_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Place:
        """Archive place record (alias for soft_delete_place)."""
        return self.soft_delete_place(place_id, changed_by=changed_by, reason=reason)

    def restore_place(
        self,
        place_id: str,
        changed_by: Optional[str] = None,
    ) -> Place:
        """
        Restore an archived / soft-deleted place to active published state.
        """
        current = self.get_place(place_id, include_deleted=True)
        if not current:
            raise ValueError(f"Place '{place_id}' not found.")

        now = datetime.now(timezone.utc).isoformat()
        new_version = current.version + 1

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE places
                SET is_deleted = 0, status = 'published', version = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_version, now, place_id),
            )

        restored = self.get_place(place_id, include_deleted=True)
        if not restored:
            raise RuntimeError(f"Failed to restore place {place_id}")

        self._record_history(
            place=restored,
            action="restored",
            changed_by=changed_by,
            summary="Place restored to canonical published status",
        )

        # Log platform audit record
        try:
            self.audit.log(
                action="RESTORE",
                resource_type="place",
                resource_id=place_id,
                actor_id=changed_by,
                old_state="archived",
                new_state="published",
                metadata={"summary": "Place restored to canonical published status"},
            )
        except Exception:
            pass

        return restored

    def list_places(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lon: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Place]:
        """
        List active published places. Excludes soft-deleted / archived records.
        """
        clauses = ["is_deleted = 0", "status = 'published'"]
        params: List[Any] = []

        if category:
            clauses.append("category = ?")
            params.append(category.strip().lower())

        if search:
            clauses.append("(name LIKE ? OR address LIKE ?)")
            term = f"%{search.strip()}%"
            params.extend([term, term])

        if min_lat is not None and max_lat is not None:
            clauses.append("latitude BETWEEN ? AND ?")
            params.extend([min_lat, max_lat])

        if min_lon is not None and max_lon is not None:
            clauses.append("longitude BETWEEN ? AND ?")
            params.extend([min_lon, max_lon])

        where_sql = f"WHERE {' AND '.join(clauses)}"
        query = f"SELECT * FROM places {where_sql} ORDER BY name ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with get_db(self.db_path) as conn:
            cur = conn.execute(query, tuple(params))
            return [Place(**dict(row)) for row in cur.fetchall()]

    def get_place_history(self, place_id: str) -> List[PlaceHistory]:
        """Retrieve full audit history for a place ordered by version descending."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM place_history WHERE place_id = ? ORDER BY version DESC",
                (place_id,),
            )
            return [PlaceHistory(**dict(row)) for row in cur.fetchall()]

    def _record_history(
        self,
        place: Place,
        action: str,
        changed_by: Optional[str],
        summary: Optional[str],
    ) -> None:
        """Internal helper to write an immutable history audit entry and stream to canonical changelog."""
        hid = f"plch_{secrets.token_hex(6)}"
        now = datetime.now(timezone.utc).isoformat()
        snapshot = json.dumps(place.to_dict())

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO place_history (
                    id, place_id, version, action, changed_by,
                    snapshot_json, change_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (hid, place.id, place.version, action, changed_by, snapshot, summary, now),
            )

        # Stream modification into canonical changelog for delta sync
        try:
            from src.db.canonical import CanonicalRepository
            action_map = {
                "created": "create",
                "updated": "update",
                "soft_deleted": "delete",
                "restored": "restore",
            }
            canon_action = action_map.get(action, action)
            canon_repo = CanonicalRepository(self.db_path)
            canon_repo.record_change(
                resource_type="place",
                resource_id=place.id,
                action=canon_action,
                version=place.version,
                data=place.to_dict(),
            )
        except Exception:
            pass

    def publish_from_contribution(self, contribution: Any, publisher: Optional[Any] = None) -> Place:
        """
        Synchronize an approved community contribution into canonical map data.
        If action is 'create', creates a new canonical place.
        If action is 'update', applies proposed edits to the target canonical place.
        If action is 'delete', archives the target canonical place.
        """
        publisher_id = None
        if hasattr(publisher, "user") and publisher.user:
            publisher_id = publisher.user.id
        elif hasattr(publisher, "id"):
            publisher_id = str(publisher.id)
        elif isinstance(publisher, str):
            publisher_id = publisher

        data = contribution.to_dict().get("data", {}) if hasattr(contribution, "to_dict") else getattr(contribution, "data", {})
        action = getattr(contribution, "action", "create")
        target_id = getattr(contribution, "target_resource_id", None)

        if action == "create" or not target_id:
            return self.create_place(
                name=data.get("name") or getattr(contribution, "title", "New Place"),
                category=data.get("category", "landmark"),
                latitude=float(data.get("latitude", 0.0)),
                longitude=float(data.get("longitude", 0.0)),
                address=data.get("address"),
                opening_hours=data.get("opening_hours") or data.get("hours"),
                phone=data.get("phone"),
                website=data.get("website"),
                metadata=data.get("metadata"),
                created_by=getattr(contribution, "owner_id", publisher_id),
            )
        elif action == "update":
            return self.update_place(
                place_id=target_id,
                changed_by=publisher_id,
                name=data.get("name"),
                category=data.get("category"),
                latitude=data.get("latitude"),
                longitude=data.get("longitude"),
                address=data.get("address"),
                opening_hours=data.get("opening_hours") or data.get("hours"),
                phone=data.get("phone"),
                website=data.get("website"),
                metadata=data.get("metadata"),
                change_summary=f"Applied community contribution {getattr(contribution, 'id', '')}",
            )
        elif action == "delete":
            return self.soft_delete_place(
                place_id=target_id,
                changed_by=publisher_id,
                reason=f"Removed per community contribution {getattr(contribution, 'id', '')}",
            )
        else:
            raise ValueError(f"Unknown contribution action '{action}'")
