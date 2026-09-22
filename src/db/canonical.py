"""
Navigators IDR - Canonical Map Repository & Packaging Layer
Manages the canonical map changelog stream and builds standalone offline map packages.
"""

from __future__ import annotations

import json
import sqlite3
import hashlib
import secrets
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from src.db.database import get_db, DEFAULT_DB_PATH


@dataclass
class ChangelogEntry:
    sequence_id: int
    resource_type: str
    resource_id: str
    action: str
    version: int
    data_json: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        try:
            res["data"] = json.loads(self.data_json)
        except (ValueError, TypeError):
            res["data"] = {}
        return res


@dataclass
class OfflinePackage:
    id: str
    package_version: int
    region: str
    format: str
    file_path: str
    checksum_sha256: str
    record_count: int
    size_bytes: int
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CanonicalRepository:
    """Data access layer for canonical map changelog and offline map packaging."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = db_path

    def record_change(
        self,
        resource_type: str,
        resource_id: str,
        action: str,
        version: int,
        data: Dict[str, Any],
    ) -> int:
        """
        Append a modification record to the canonical changelog stream.
        Returns the unique, monotonically increasing sequence_id.
        """
        now = datetime.now(timezone.utc).isoformat()
        data_str = json.dumps(data)

        with get_db(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO canonical_changelog (
                    resource_type, resource_id, action, version, data_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (resource_type.strip().lower(), resource_id.strip(), action.strip().lower(), version, data_str, now),
            )
            sequence_id = cur.lastrowid
            if sequence_id is None:
                row = conn.execute("SELECT MAX(sequence_id) FROM canonical_changelog").fetchone()
                sequence_id = row[0] if row else 1

            return int(sequence_id)

    def get_changes(
        self,
        since_sequence: int = 0,
        limit: int = 100,
    ) -> Tuple[List[ChangelogEntry], int, bool]:
        """
        Query delta changes starting after since_sequence.
        Returns (entries, latest_sequence, has_more).
        """
        limit = max(1, min(limit, 500))
        entries: List[ChangelogEntry] = []

        with get_db(self.db_path) as conn:
            latest_row = conn.execute("SELECT COALESCE(MAX(sequence_id), 0) FROM canonical_changelog").fetchone()
            latest_sequence = int(latest_row[0]) if latest_row else 0


            cur = conn.execute(
                """
                SELECT sequence_id, resource_type, resource_id, action, version, data_json, created_at
                FROM canonical_changelog
                WHERE sequence_id > ?
                ORDER BY sequence_id ASC
                LIMIT ?
                """,
                (int(since_sequence), limit + 1),
            )
            rows = cur.fetchall()

        has_more = len(rows) > limit
        target_rows = rows[:limit]

        for r in target_rows:
            entries.append(ChangelogEntry(
                sequence_id=r["sequence_id"],
                resource_type=r["resource_type"],
                resource_id=r["resource_id"],
                action=r["action"],
                version=r["version"],
                data_json=r["data_json"],
                created_at=r["created_at"],
            ))

        return entries, latest_sequence, has_more

    def get_latest_sequence(self) -> int:
        """Fetch the current highest sequence_id in the changelog."""
        with get_db(self.db_path) as conn:
            row = conn.execute("SELECT COALESCE(MAX(sequence_id), 0) FROM canonical_changelog").fetchone()
            return int(row[0]) if row else 0

    def build_offline_package(
        self,
        region: str = "global",
        format: str = "sqlite",
        export_dir: Optional[str | Path] = None,
    ) -> OfflinePackage:
        """
        Extract active canonical map data into a standalone, compressed offline package.
        Computes SHA-256 checksum and saves package metadata.
        """
        region = region.strip().lower()
        format = format.strip().lower()
        if format not in ("sqlite", "json_bundle"):
            raise ValueError("Offline package format must be 'sqlite' or 'json_bundle'")


        if export_dir:
            pkg_dir = Path(export_dir)
        else:
            base_dir = Path(self.db_path).parent if self.db_path else DEFAULT_DB_PATH.parent
            pkg_dir = base_dir / "packages"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        package_id = f"pkg_{secrets.token_hex(6)}"
        now = datetime.now(timezone.utc).isoformat()

        with get_db(self.db_path) as conn:

            ver_row = conn.execute(
                "SELECT COALESCE(MAX(package_version), 0) FROM offline_map_packages WHERE region = ?",
                (region,),
            ).fetchone()
            next_version = (int(ver_row[0]) if ver_row else 0) + 1


            cur = conn.execute(
                """
                SELECT id, name, category, latitude, longitude, address, opening_hours,
                       phone, website, metadata_json, status, version, created_at, updated_at
                FROM places
                WHERE is_deleted = 0 AND status = 'published'
                ORDER BY id ASC
                """
            )
            places = cur.fetchall()
            latest_seq_row = conn.execute("SELECT COALESCE(MAX(sequence_id), 0) FROM canonical_changelog").fetchone()
            max_seq = int(latest_seq_row[0]) if latest_seq_row else 0

        record_count = len(places)

        if format == "sqlite":
            file_name = f"offline_map_{region}_v{next_version}_{secrets.token_hex(4)}.db"
            file_path = pkg_dir / file_name


            if file_path.exists():
                file_path.unlink()

            pkg_conn = sqlite3.connect(str(file_path))
            try:
                pkg_conn.execute("""
                    CREATE TABLE package_metadata (
                        package_id TEXT PRIMARY KEY,
                        package_version INTEGER NOT NULL,
                        region TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        record_count INTEGER NOT NULL,
                        max_sequence_id INTEGER NOT NULL
                    )
                """)
                pkg_conn.execute("""
                    CREATE TABLE places (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        category TEXT NOT NULL,
                        latitude REAL NOT NULL,
                        longitude REAL NOT NULL,
                        address TEXT,
                        opening_hours TEXT,
                        phone TEXT,
                        website TEXT,
                        metadata_json TEXT,
                        version INTEGER NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                pkg_conn.execute("CREATE INDEX idx_pkg_places_coords ON places(latitude, longitude)")
                pkg_conn.execute("CREATE INDEX idx_pkg_places_cat ON places(category)")

                pkg_conn.execute(
                    """
                    INSERT INTO package_metadata (
                        package_id, package_version, region, created_at, record_count, max_sequence_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (package_id, next_version, region, now, record_count, max_seq),
                )

                for p in places:
                    pkg_conn.execute(
                        """
                        INSERT INTO places (
                            id, name, category, latitude, longitude, address, opening_hours,
                            phone, website, metadata_json, version, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            p["id"], p["name"], p["category"], p["latitude"], p["longitude"],
                            p["address"], p["opening_hours"], p["phone"], p["website"],
                            p["metadata_json"], p["version"], p["updated_at"],
                        ),
                    )
                pkg_conn.commit()
            finally:
                pkg_conn.close()

        else:

            file_name = f"offline_map_{region}_v{next_version}_{secrets.token_hex(4)}.json"
            file_path = pkg_dir / file_name
            bundle_data = {
                "metadata": {
                    "package_id": package_id,
                    "package_version": next_version,
                    "region": region,
                    "created_at": now,
                    "record_count": record_count,
                    "max_sequence_id": max_seq,
                },
                "places": [dict(p) for p in places],
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(bundle_data, f, indent=2)


        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        checksum = hasher.hexdigest()
        size_bytes = file_path.stat().st_size


        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO offline_map_packages (
                    id, package_version, region, format, file_path,
                    checksum_sha256, record_count, size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    package_id, next_version, region, format, str(file_path),
                    checksum, record_count, size_bytes, now
                ),
            )

        return OfflinePackage(
            id=package_id,
            package_version=next_version,
            region=region,
            format=format,
            file_path=str(file_path),
            checksum_sha256=checksum,
            record_count=record_count,
            size_bytes=size_bytes,
            created_at=now,
        )

    def get_latest_package(self, region: str = "global") -> Optional[OfflinePackage]:
        """Fetch latest offline map package for a region."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT id, package_version, region, format, file_path, checksum_sha256,
                       record_count, size_bytes, created_at
                FROM offline_map_packages
                WHERE region = ?
                ORDER BY package_version DESC
                LIMIT 1
                """,
                (region.strip().lower(),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return OfflinePackage(**dict(row))

    def get_package(self, package_id: str) -> Optional[OfflinePackage]:
        """Fetch offline map package by ID."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT id, package_version, region, format, file_path, checksum_sha256,
                       record_count, size_bytes, created_at
                FROM offline_map_packages
                WHERE id = ?
                """,
                (package_id.strip(),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return OfflinePackage(**dict(row))

    def list_packages(self, region: Optional[str] = None, limit: int = 20) -> List[OfflinePackage]:
        """List available offline packages."""
        query = "SELECT * FROM offline_map_packages"
        params: List[Any] = []
        if region:
            query += " WHERE region = ?"
            params.append(region.strip().lower())
        query += " ORDER BY package_version DESC LIMIT ?"
        params.append(limit)

        with get_db(self.db_path) as conn:
            cur = conn.execute(query, tuple(params))
            return [OfflinePackage(**dict(r)) for r in cur.fetchall()]
