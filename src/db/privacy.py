"""
Navigators IDR - Phase 35: Privacy and Data Controls Repository
Manages domain-level privacy settings (stored_locally, synced, sharing_level)
and performs physical data purges / disk file deletions for user accounts.
"""

import os
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.db.database import connect_db, get_db

VALID_DOMAINS = [
    "location_data",
    "sensor_data",
    "navigation_sessions",
    "contribution_data",
    "dataset_contributions",
    "account_data",
]

VALID_SHARING_LEVELS = ["private", "anonymous", "team", "public"]


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")


class PrivacyRepository:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path

    def get_privacy_settings(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves privacy settings for all 6 data domains for a user.
        If a domain has no custom settings row yet, default settings are returned.
        """
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT domain, stored_locally, synced, sharing_level, updated_at
                FROM user_privacy_settings
                WHERE user_id = ?
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
            existing = {row["domain"]: dict(row) for row in rows}

        results = []
        now = _current_timestamp()
        for domain in VALID_DOMAINS:
            if domain in existing:
                row = existing[domain]
                results.append({
                    "domain": domain,
                    "stored_locally": bool(row["stored_locally"]),
                    "synced": bool(row["synced"]),
                    "sharing_level": row["sharing_level"],
                    "updated_at": row["updated_at"],
                })
            else:
                results.append({
                    "domain": domain,
                    "stored_locally": True,
                    "synced": True,
                    "sharing_level": "private",
                    "updated_at": now,
                })

        return results

    def update_domain_setting(
        self,
        user_id: str,
        domain: str,
        stored_locally: Optional[bool] = None,
        synced: Optional[bool] = None,
        sharing_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Updates or inserts domain privacy settings for a user.
        """
        if domain not in VALID_DOMAINS:
            raise ValueError(f"Invalid privacy domain '{domain}'. Must be one of {VALID_DOMAINS}")

        if sharing_level is not None and sharing_level not in VALID_SHARING_LEVELS:
            raise ValueError(f"Invalid sharing level '{sharing_level}'. Must be one of {VALID_SHARING_LEVELS}")

        current_settings = self.get_privacy_settings(user_id)
        current_map = {item["domain"]: item for item in current_settings}
        curr = current_map[domain]

        new_stored_locally = curr["stored_locally"] if stored_locally is None else stored_locally
        new_synced = curr["synced"] if synced is None else synced
        new_sharing_level = curr["sharing_level"] if sharing_level is None else sharing_level
        now = _current_timestamp()

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_privacy_settings (id, user_id, domain, stored_locally, synced, sharing_level, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, domain) DO UPDATE SET
                    stored_locally = excluded.stored_locally,
                    synced = excluded.synced,
                    sharing_level = excluded.sharing_level,
                    updated_at = excluded.updated_at
                """,
                (
                    str(uuid.uuid4()),
                    user_id,
                    domain,
                    1 if new_stored_locally else 0,
                    1 if new_synced else 0,
                    new_sharing_level,
                    now,
                ),
            )

        return {
            "domain": domain,
            "stored_locally": new_stored_locally,
            "synced": new_synced,
            "sharing_level": new_sharing_level,
            "updated_at": now,
        }

    def purge_domain_data(self, user_id: str, domain: str) -> Dict[str, Any]:
        """
        Executes a physical purge for specified domain data.
        Deletes database records and unlinks physical disk files where applicable.
        """
        if domain not in VALID_DOMAINS:
            raise ValueError(f"Invalid privacy domain '{domain}'. Must be one of {VALID_DOMAINS}")

        summary: Dict[str, Any] = {
            "user_id": user_id,
            "domain": domain,
            "purged": True,
            "deleted_records": 0,
            "deleted_files": 0,
            "timestamp": _current_timestamp(),
        }

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            if domain == "location_data":
                cursor.execute("DELETE FROM saved_places WHERE user_id = ?", (user_id,))
                c1 = cursor.rowcount
                cursor.execute("DELETE FROM recent_searches WHERE user_id = ?", (user_id,))
                c2 = cursor.rowcount
                summary["deleted_records"] = c1 + c2

            elif domain == "sensor_data":
                cursor.execute(
                    "SELECT sensor_data_path FROM dataset_sessions WHERE contributor_id = ?",
                    (user_id,),
                )
                paths = cursor.fetchall()
                files_removed = 0
                for row in paths:
                    path = row["sensor_data_path"]
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                            files_removed += 1
                        except OSError:
                            pass

                recordings_dir = os.path.join(os.getcwd(), "data", "recordings")
                if os.path.exists(recordings_dir):
                    for fname in os.listdir(recordings_dir):
                        if user_id in fname:
                            fpath = os.path.join(recordings_dir, fname)
                            try:
                                os.remove(fpath)
                                files_removed += 1
                            except OSError:
                                pass

                summary["deleted_files"] = files_removed

            elif domain == "navigation_sessions":
                cursor.execute("DELETE FROM sos_sessions WHERE user_id = ?", (user_id,))
                summary["deleted_records"] = cursor.rowcount

            elif domain == "contribution_data":
                cursor.execute(
                    "DELETE FROM contributions WHERE owner_id = ? AND status != 'published'",
                    (user_id,),
                )
                c1 = cursor.rowcount
                cursor.execute(
                    "UPDATE contributions SET status = 'withdrawn' WHERE owner_id = ? AND status = 'published'",
                    (user_id,),
                )
                c2 = cursor.rowcount
                summary["deleted_records"] = c1 + c2

            elif domain == "dataset_contributions":
                cursor.execute(
                    "SELECT sensor_data_path FROM dataset_sessions WHERE contributor_id = ?",
                    (user_id,),
                )
                rows = cursor.fetchall()
                files_removed = 0
                for row in rows:
                    path = row["sensor_data_path"]
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                            files_removed += 1
                        except OSError:
                            pass

                cursor.execute("DELETE FROM dataset_sessions WHERE contributor_id = ?", (user_id,))
                summary["deleted_records"] = cursor.rowcount
                summary["deleted_files"] = files_removed

            elif domain == "account_data":
                return self.delete_user_account(user_id)

        return summary

    def delete_user_account(self, user_id: str) -> Dict[str, Any]:
        """
        Executes a complete account wipe for a user.
        Physically deletes dataset files on disk, revokes all sessions,
        and cascades user deletion across all relational tables.
        """
        files_removed = 0
        deleted_records = 0

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()


            cursor.execute(
                "SELECT sensor_data_path FROM dataset_sessions WHERE contributor_id = ?",
                (user_id,),
            )
            rows = cursor.fetchall()
            for row in rows:
                path = row["sensor_data_path"]
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        files_removed += 1
                    except OSError:
                        pass


            now = _current_timestamp()
            cursor.execute(
                "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                (now, user_id),
            )


            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            deleted_records = cursor.rowcount

        return {
            "user_id": user_id,
            "account_deleted": True,
            "deleted_records": deleted_records,
            "deleted_files": files_removed,
            "timestamp": _current_timestamp(),
        }
