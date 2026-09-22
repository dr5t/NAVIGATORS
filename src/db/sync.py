"""
Navigators IDR - Device Offline Sync Repository & Conflict Resolution Layer
Handles offline device queue ingestion, idempotency, version conflict detection,
and synchronization receipts.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from src.db.database import get_db
from src.db.contributions import ContributionRepository
from src.db.places import PlaceRepository
from src.db.reports import ReportRepository
from src.db.state_machine import ContributionState


@dataclass
class DeviceQueueItem:
    id: str
    device_id: str
    client_sequence: int
    operation: str
    payload_json: str
    base_version: Optional[int] = None
    status: str = "pending"
    conflict_reason: Optional[str] = None
    server_resource_id: Optional[str] = None
    created_at: str = ""
    synced_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        try:
            res["payload"] = json.loads(self.payload_json)
        except (ValueError, TypeError):
            res["payload"] = {}
        return res


@dataclass
class SyncReceipt:
    client_id: str
    status: str
    server_resource_id: Optional[str] = None
    conflict_reason: Optional[str] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SyncRepository:
    """Manages offline sync queue processing, conflict resolution, and receipts."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = db_path

    @property
    def db_path(self) -> Optional[str | Path]:
        return self._db_path

    @db_path.setter
    def db_path(self, val: Optional[str | Path]) -> None:
        self._db_path = val
        self.contrib_repo = ContributionRepository(val)
        self.place_repo = PlaceRepository(val)
        self.report_repo = ReportRepository(val)

    def push_device_changes(
        self,
        device_id: str,
        items: List[Dict[str, Any]],
        actor_id: str,
        client_timestamp: Optional[str] = None,
    ) -> List[SyncReceipt]:
        """
        Process a batch of queued offline operations from a client device.
        Performs idempotency checks, conflict detection, and writes to canonical staging.
        """
        receipts: List[SyncReceipt] = []
        now = datetime.now(timezone.utc).isoformat()

        for item in items:
            client_id = str(item.get("client_id") or item.get("id") or f"sync_{secrets.token_hex(6)}")
            client_seq = int(item.get("client_sequence", 1))
            operation = str(item.get("operation", "add_place")).strip().lower()
            payload = item.get("payload", {})
            base_version = item.get("base_version")
            if base_version is not None:
                try:
                    base_version = int(base_version)
                except (ValueError, TypeError):
                    base_version = None


            with get_db(self.db_path) as conn:
                existing = conn.execute(
                    "SELECT status, server_resource_id, conflict_reason FROM device_sync_queue WHERE id = ?",
                    (client_id,),
                ).fetchone()

            if existing:
                receipts.append(SyncReceipt(
                    client_id=client_id,
                    status=existing["status"],
                    server_resource_id=existing["server_resource_id"],
                    conflict_reason=existing["conflict_reason"],
                    message="Idempotent duplicate: operation previously processed",
                ))
                continue


            receipt: SyncReceipt
            if operation == "add_place":
                receipt = self._handle_add_place(
                    client_id=client_id,
                    device_id=device_id,
                    client_seq=client_seq,
                    payload=payload,
                    actor_id=actor_id,
                    now=now,
                )
            elif operation == "suggest_edit":
                receipt = self._handle_suggest_edit(
                    client_id=client_id,
                    device_id=device_id,
                    client_seq=client_seq,
                    payload=payload,
                    base_version=base_version,
                    actor_id=actor_id,
                    now=now,
                )
            elif operation == "report":
                receipt = self._handle_report(
                    client_id=client_id,
                    device_id=device_id,
                    client_seq=client_seq,
                    payload=payload,
                    actor_id=actor_id,
                    now=now,
                )
            else:
                receipt = SyncReceipt(
                    client_id=client_id,
                    status="rejected",
                    conflict_reason=f"Unsupported sync operation '{operation}'",
                    message="Operation rejected",
                )
                self._record_queue_item(
                    item_id=client_id,
                    device_id=device_id,
                    client_seq=client_seq,
                    operation=operation,
                    payload=payload,
                    base_version=base_version,
                    status="rejected",
                    conflict_reason=f"Unsupported sync operation '{operation}'",
                    server_id=None,
                    now=now,
                )

            receipts.append(receipt)

        return receipts

    def _handle_add_place(
        self,
        client_id: str,
        device_id: str,
        client_seq: int,
        payload: Dict[str, Any],
        actor_id: str,
        now: str,
    ) -> SyncReceipt:
        name = payload.get("name", "Offline Place")
        category = payload.get("category", "amenity")
        lat = float(payload.get("latitude", 0.0))
        lon = float(payload.get("longitude", 0.0))

        contrib_data = {
            "name": name,
            "category": category,
            "latitude": lat,
            "longitude": lon,
            "address": payload.get("address"),
            "opening_hours": payload.get("opening_hours"),
            "phone": payload.get("phone"),
            "website": payload.get("website"),
            "metadata": payload.get("metadata", {}),
            "evidence": payload.get("evidence", {}),
        }


        contrib = self.contrib_repo.create_contribution(
            owner_id=actor_id,
            resource_type="place",
            title=f"Add {name}",
            data=contrib_data,
            target_resource_id=None,
            action="create",
        )


        if payload.get("auto_submit", True):
            try:
                self.contrib_repo.transition_state(
                    contribution_id=contrib.id,
                    target_state=ContributionState.PENDING_REVIEW,
                    actor=actor_id,
                )
            except Exception:
                pass

        self._record_queue_item(
            item_id=client_id,
            device_id=device_id,
            client_seq=client_seq,
            operation="add_place",
            payload=payload,
            base_version=None,
            status="synced",
            conflict_reason=None,
            server_id=contrib.id,
            now=now,
        )

        return SyncReceipt(
            client_id=client_id,
            status="synced",
            server_resource_id=contrib.id,
            message="Place contribution successfully created from offline queue",
        )

    def _handle_suggest_edit(
        self,
        client_id: str,
        device_id: str,
        client_seq: int,
        payload: Dict[str, Any],
        base_version: Optional[int],
        actor_id: str,
        now: str,
    ) -> SyncReceipt:
        target_id = payload.get("place_id") or payload.get("target_resource_id")
        if not target_id:
            reason = "Missing target place_id for edit proposal"
            self._record_queue_item(client_id, device_id, client_seq, "suggest_edit", payload, base_version, "rejected", reason, None, now)
            return SyncReceipt(client_id=client_id, status="rejected", conflict_reason=reason, message="Rejected: no place_id")

        place = self.place_repo.get_place(target_id, include_deleted=True)
        if not place:
            reason = f"Target place '{target_id}' does not exist on canonical map"
            self._record_queue_item(client_id, device_id, client_seq, "suggest_edit", payload, base_version, "conflict", reason, None, now)
            return SyncReceipt(client_id=client_id, status="conflict", conflict_reason=reason, message="Conflict: place not found")

        if place.is_deleted or place.status != "published":
            reason = f"Target place '{target_id}' is archived or soft deleted"
            self._record_queue_item(client_id, device_id, client_seq, "suggest_edit", payload, base_version, "conflict", reason, None, now)
            return SyncReceipt(client_id=client_id, status="conflict", conflict_reason=reason, message="Conflict: place archived")


        if base_version is not None and place.version != base_version:
            reason = f"Version conflict: device base version is {base_version}, but canonical version is {place.version}"
            self._record_queue_item(client_id, device_id, client_seq, "suggest_edit", payload, base_version, "conflict", reason, None, now)
            return SyncReceipt(
                client_id=client_id,
                status="conflict",
                server_resource_id=target_id,
                conflict_reason=reason,
                message="Conflict detected: local map data is outdated",
            )


        proposed_data = {
            "name": payload.get("name", place.name),
            "category": payload.get("category", place.category),
            "latitude": float(payload.get("latitude", place.latitude)),
            "longitude": float(payload.get("longitude", place.longitude)),
            "address": payload.get("address", place.address),
            "opening_hours": payload.get("opening_hours", place.opening_hours),
            "phone": payload.get("phone", place.phone),
            "website": payload.get("website", place.website),
            "metadata": payload.get("metadata", {}),
            "evidence": payload.get("evidence", {}),
        }

        contrib = self.contrib_repo.create_contribution(
            owner_id=actor_id,
            resource_type="place",
            title=f"Edit {proposed_data['name']}",
            data=proposed_data,
            target_resource_id=target_id,
            action="update",
        )

        if payload.get("auto_submit", True):
            try:
                self.contrib_repo.transition_state(
                    contribution_id=contrib.id,
                    target_state=ContributionState.PENDING_REVIEW,
                    actor=actor_id,
                )
            except Exception:
                pass

        self._record_queue_item(
            item_id=client_id,
            device_id=device_id,
            client_seq=client_seq,
            operation="suggest_edit",
            payload=payload,
            base_version=base_version,
            status="synced",
            conflict_reason=None,
            server_id=contrib.id,
            now=now,
        )

        return SyncReceipt(
            client_id=client_id,
            status="synced",
            server_resource_id=contrib.id,
            message="Edit proposal created and queued for moderation",
        )

    def _handle_report(
        self,
        client_id: str,
        device_id: str,
        client_seq: int,
        payload: Dict[str, Any],
        actor_id: str,
        now: str,
    ) -> SyncReceipt:
        target_type = payload.get("target_type", "place")
        target_id = payload.get("target_id", "")
        reason = payload.get("reason", "inaccurate_data")
        details = payload.get("details", "")

        rep = self.report_repo.create_report(
            reporter_id=actor_id,
            target_type=target_type,
            target_id=target_id,
            reason=reason,
            details=details,
        )

        self._record_queue_item(
            item_id=client_id,
            device_id=device_id,
            client_seq=client_seq,
            operation="report",
            payload=payload,
            base_version=None,
            status="synced",
            conflict_reason=None,
            server_id=rep.id,
            now=now,
        )

        return SyncReceipt(
            client_id=client_id,
            status="synced",
            server_resource_id=rep.id,
            message="Community report created from offline queue",
        )

    def _record_queue_item(
        self,
        item_id: str,
        device_id: str,
        client_seq: int,
        operation: str,
        payload: Dict[str, Any],
        base_version: Optional[int],
        status: str,
        conflict_reason: Optional[str],
        server_id: Optional[str],
        now: str,
    ) -> None:
        payload_str = json.dumps(payload)
        synced_at = now if status == "synced" else None

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO device_sync_queue (
                    id, device_id, client_sequence, operation, payload_json,
                    base_version, status, conflict_reason, server_resource_id,
                    created_at, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id, device_id, client_seq, operation, payload_str,
                    base_version, status, conflict_reason, server_id,
                    now, synced_at
                ),
            )

    def get_device_queue(self, device_id: str, limit: int = 50) -> List[DeviceQueueItem]:
        """Fetch all synced or pending operations for a given device."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT id, device_id, client_sequence, operation, payload_json,
                       base_version, status, conflict_reason, server_resource_id,
                       created_at, synced_at
                FROM device_sync_queue
                WHERE device_id = ?
                ORDER BY client_sequence ASC
                LIMIT ?
                """,
                (device_id, limit),
            )
            return [DeviceQueueItem(**dict(r)) for r in cur.fetchall()]
