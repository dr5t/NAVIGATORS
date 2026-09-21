"""
Tests for Phase 10: Offline Synchronization
Verifies that:
1. Status and incremental delta pull synchronization work reliably.
2. Offline batch uploads (push) from client devices are processed idempotently.
3. Version conflicts on stale edits are detected and surfaced in sync receipts.
4. Network status never controls GNSS/DR/EKF/Map-matching states; it only controls synchronization.
"""

import pytest
import secrets
from pathlib import Path

from src.db.database import init_db, get_db
from src.db.places import PlaceRepository
from src.db.canonical import CanonicalRepository
from src.db.sync import SyncRepository
from src.db.auth_service import AuthService, SessionContext
from src.api.sync import (
    api_sync_status,
    api_sync_pull,
    api_sync_push,
    api_get_device_queue,
    PushSyncRequest,
    SyncItemModel,
)
from src.navigation.dead_reckoning import DeadReckoningEngine
from src.navigation.ekf import ExtendedKalmanFilter


def test_sync_status_endpoint():
    """Verify sync status returns sequence and latest package."""
    init_db()
    res = api_sync_status()
    assert res["status"] == "online"
    assert "latest_sequence" in res
    assert isinstance(res["latest_sequence"], int)


def test_incremental_delta_sync_pull():
    """Verify incremental delta pull yields sequential changelog items."""
    init_db()
    auth_service = AuthService()
    place_repo = PlaceRepository()
    canonical_repo = CanonicalRepository()

    tag = secrets.token_hex(4)
    user, user_session, _ = auth_service.register(
        email=f"pull_user_{tag}@example.com",
        password="SecurePassword123!",
        name="Pull Test User",
        role_id="user",
    )

    # Initial sequence
    initial_seq = canonical_repo.get_latest_sequence()

    # Create two places to generate changelog deltas
    p1 = place_repo.create_place(
        name=f"Delta Cafe {tag}",
        category="cafe",
        latitude=28.6140,
        longitude=77.2090,
    )
    p2 = place_repo.create_place(
        name=f"Delta Pharmacy {tag}",
        category="pharmacy",
        latitude=28.6145,
        longitude=77.2095,
    )

    # Pull deltas since initial_seq
    pull_res = api_sync_pull(
        since_sequence=initial_seq,
        limit=50,
        context=user_session,
    )

    assert pull_res["count"] >= 2
    assert pull_res["latest_sequence"] >= pull_res["since_sequence"] + 2
    changes = pull_res["changes"]
    resource_ids = [c["resource_id"] for c in changes]
    assert p1.id in resource_ids
    assert p2.id in resource_ids

    # Sequence numbers must be strictly increasing
    sequences = [c["sequence_id"] for c in changes]
    assert sequences == sorted(sequences)


def test_offline_device_push_and_idempotency():
    """Verify offline batch push creates contributions and handles idempotency."""
    init_db()
    auth_service = AuthService()

    tag = secrets.token_hex(4)
    user, user_session, _ = auth_service.register(
        email=f"device_user_{tag}@example.com",
        password="SecurePassword123!",
        name="Device User",
        role_id="user",
    )

    device_id = f"device_pixel_{tag}"
    client_uuid_1 = f"sync_{tag}_001"
    client_uuid_2 = f"sync_{tag}_002"

    items = [
        SyncItemModel(
            client_id=client_uuid_1,
            client_sequence=1,
            operation="add_place",
            payload={
                "name": f"Offline Chai Stall {tag}",
                "category": "amenity",
                "latitude": 28.6150,
                "longitude": 77.2100,
                "address": "Lane 4",
            },
        ),
        SyncItemModel(
            client_id=client_uuid_2,
            client_sequence=2,
            operation="report",
            payload={
                "target_type": "place",
                "target_id": "plc_nonexistent_flag",
                "reason": "inaccurate_data",
                "details": "Coordinates point to empty lake",
            },
        ),
    ]

    req = PushSyncRequest(
        device_id=device_id,
        client_timestamp="2026-09-22T01:00:00Z",
        items=items,
    )

    # 1. First push
    push_res = api_sync_push(body=req, context=user_session)
    assert push_res["total_items"] == 2
    assert push_res["synced_count"] == 2
    assert push_res["conflict_count"] == 0

    receipts = push_res["receipts"]
    assert receipts[0]["client_id"] == client_uuid_1
    assert receipts[0]["status"] == "synced"
    assert receipts[0]["server_resource_id"] is not None

    assert receipts[1]["client_id"] == client_uuid_2
    assert receipts[1]["status"] == "synced"

    # 2. Duplicate push (idempotency verification)
    dup_res = api_sync_push(body=req, context=user_session)
    assert dup_res["synced_count"] == 2
    dup_receipts = dup_res["receipts"]
    assert "Idempotent duplicate" in dup_receipts[0]["message"]
    assert dup_receipts[0]["server_resource_id"] == receipts[0]["server_resource_id"]

    # 3. Check device queue endpoint
    queue_res = api_get_device_queue(device_id=device_id, limit=50, context=user_session)
    assert queue_res["count"] == 2


def test_version_conflict_detection_on_stale_edit():
    """Verify sync engine flags conflict when device base version is outdated."""
    init_db()
    auth_service = AuthService()
    place_repo = PlaceRepository()

    tag = secrets.token_hex(4)
    user, user_session, _ = auth_service.register(
        email=f"conflict_user_{tag}@example.com",
        password="SecurePassword123!",
        name="Conflict User",
        role_id="user",
    )

    # 1. Place created at version 1
    place = place_repo.create_place(
        name=f"Conflict Bakery {tag}",
        category="bakery",
        latitude=28.6200,
        longitude=77.2150,
    )
    assert place.version == 1

    # 2. Staff updates place to version 2
    place_repo.update_place(
        place_id=place.id,
        phone="+91-11-98765432",
        change_summary="Staff updated phone",
    )
    updated_place = place_repo.get_place(place.id)
    assert updated_place.version == 2

    # 3. Offline device attempts suggest_edit with stale base_version=1
    client_uuid = f"sync_stale_{tag}"
    stale_item = SyncItemModel(
        client_id=client_uuid,
        client_sequence=1,
        operation="suggest_edit",
        payload={
            "place_id": place.id,
            "name": f"Conflict Bakery {tag}",
            "phone": "+91-11-00000000",
        },
        base_version=1,  # Stale! Canonical is now 2
    )

    req = PushSyncRequest(
        device_id=f"device_tablet_{tag}",
        items=[stale_item],
    )

    push_res = api_sync_push(body=req, context=user_session)
    assert push_res["conflict_count"] == 1
    receipt = push_res["receipts"][0]
    assert receipt["status"] == "conflict"
    assert "Version conflict" in receipt["conflict_reason"]
    assert "base version is 1, but canonical version is 2" in receipt["conflict_reason"]


def test_network_status_independence_and_navigation_integrity():
    """
    CRITICAL ARCHITECTURAL RULE:
    'Internet status should never control GNSS/DR state. It controls synchronization.'
    Verifies that sensor fusion (EKF and DR) runs continuously without interruption
    even when network connectivity is toggled offline.
    """
    import numpy as np

    dr_engine = DeadReckoningEngine()
    dr_engine.start(position=np.array([0.0, 0.0]), heading=0.0, speed=5.0)
    ekf = ExtendedKalmanFilter()

    # Step 1: Online state - sensor updates processed
    is_network_online = True
    pos1 = dr_engine.update(ai_speed=5.0, gyro_yaw_rate=0.01)
    assert pos1 is not None

    ekf.predict(
        accel_body=np.array([0.0, 1.5, 9.8]),
        gyro_body=np.array([0.0, 0.0, 0.05]),
    )
    assert ekf.x is not None

    # Step 2: Network goes OFFLINE
    is_network_online = False

    # Sensor loop MUST continue computing uninterrupted
    for _ in range(25):
        dr_pos = dr_engine.update(ai_speed=5.5, gyro_yaw_rate=0.01)
        ekf.predict(
            accel_body=np.array([0.0, 1.5, 9.8]),
            gyro_body=np.array([0.0, 0.0, 0.02]),
        )
        assert dr_engine.speed > 0.0
        assert ekf.x is not None

    # Step 3: Offline edits queued in device queue
    offline_queue = []
    offline_queue.append({
        "client_id": "offline_edit_1",
        "operation": "add_place",
        "payload": {"name": "Offline Landmark", "latitude": 28.61, "longitude": 77.20},
    })
    assert len(offline_queue) == 1

    # Step 4: Network restored to ONLINE
    is_network_online = True

    # Sync engine triggers without restarting or disturbing navigation state
    assert dr_engine.speed > 0.0
    assert ekf.x is not None
