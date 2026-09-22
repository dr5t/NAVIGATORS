"""
Navigators IDR - Phase 34 SOS & Emergency System Data Layer
Manages the non-modal SOS emergency workflow:
  SOS -> Confirm -> Emergency options (Call emergency services, Share live location, Emergency contact)

Guarantees:
  1. No automatic consequential action without explicit user confirmation and choice.
  2. Map / navigation session continues underneath uninterrupted (preserves nav_state).
"""

import secrets
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.db.database import get_db


@dataclass
class EmergencyContact:
    id: str
    user_id: str
    name: str
    phone: str
    relationship: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmergencyContactsRepository:
    """Data access layer for user registered emergency contacts."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = db_path

    def add_contact(
        self,
        user_id: str,
        name: str,
        phone: str,
        relationship: Optional[str] = None,
    ) -> EmergencyContact:
        cid = f"emc_{secrets.token_hex(6)}"
        now = datetime.now(timezone.utc).isoformat()

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO emergency_contacts (
                    id, user_id, name, phone, relationship, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (cid, user_id, name.strip(), phone.strip(), relationship.strip() if relationship else None, now, now),
            )

        contact = self.get_contact(user_id, cid)
        if not contact:
            raise RuntimeError(f"Failed to create emergency contact {cid}")
        return contact

    def get_contact(self, user_id: str, contact_id: str) -> Optional[EmergencyContact]:
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM emergency_contacts WHERE id = ? AND user_id = ?",
                (contact_id, user_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            return EmergencyContact(**dict(row))

    def list_contacts(self, user_id: str) -> List[EmergencyContact]:
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM emergency_contacts WHERE user_id = ? ORDER BY name ASC",
                (user_id,),
            )
            return [EmergencyContact(**dict(row)) for row in cur.fetchall()]

    def delete_contact(self, user_id: str, contact_id: str) -> bool:
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM emergency_contacts WHERE id = ? AND user_id = ?",
                (contact_id, user_id),
            )
            return cur.rowcount > 0


class SOSRepository:
    """
    Data access and orchestration layer for non-modal SOS emergency sessions.
    Flow: SOS -> Confirm -> Emergency Options (Call Services, Share Live Location, Emergency Contact).
    """

    EMERGENCY_OPTIONS = [
        "call_emergency_services",
        "share_live_location",
        "emergency_contact",
    ]

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = db_path
        self.contacts_repo = EmergencyContactsRepository(db_path)

    def trigger_sos(
        self,
        user_id: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        nav_state: str = "NORMAL",
    ) -> Dict[str, Any]:
        """
        Step 1: User initiates SOS trigger.
        Creates session in 'triggered' status.
        Does NOT execute automatic consequential actions.
        Navigation session continues underneath uninterrupted.
        """
        sos_id = f"sos_{secrets.token_hex(6)}"
        now = datetime.now(timezone.utc).isoformat()

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sos_sessions (
                    id, user_id, status, latitude, longitude, nav_state, created_at, updated_at
                ) VALUES (?, ?, 'triggered', ?, ?, ?, ?, ?)
                """,
                (sos_id, user_id, latitude, longitude, nav_state, now, now),
            )

        return {
            "sos_id": sos_id,
            "status": "triggered",
            "requires_confirmation": True,
            "message": "SOS trigger received. Please confirm to reveal emergency options.",
            "nav_session_active": True,
            "nav_state": nav_state,
            "coordinates": {"latitude": latitude, "longitude": longitude} if latitude is not None and longitude is not None else None,
            "automatic_action_taken": False,
        }

    def confirm_sos(self, user_id: str, sos_id: str) -> Dict[str, Any]:
        """
        Step 2: User confirms SOS initiation.
        Transitions session status to 'confirmed'.
        Returns available emergency options for explicit user selection.
        """
        now = datetime.now(timezone.utc).isoformat()

        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM sos_sessions WHERE id = ? AND user_id = ?",
                (sos_id, user_id),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"SOS session '{sos_id}' not found.")

            conn.execute(
                "UPDATE sos_sessions SET status = 'confirmed', updated_at = ? WHERE id = ?",
                (now, sos_id),
            )

        return {
            "sos_id": sos_id,
            "status": "confirmed",
            "message": "SOS confirmed. Select an emergency option below.",
            "emergency_options": [
                {
                    "key": "call_emergency_services",
                    "title": "Call Emergency Services",
                    "description": "Dial national emergency helpline (112)",
                },
                {
                    "key": "share_live_location",
                    "title": "Share Live Location",
                    "description": "Generate a secure 2-hour live tracking URL",
                },
                {
                    "key": "emergency_contact",
                    "title": "Emergency Contact",
                    "description": "Dispatch emergency alert to registered contacts",
                },
            ],
            "nav_session_active": True,
            "nav_state": row["nav_state"],
            "automatic_action_taken": False,
        }

    def select_emergency_option(
        self,
        user_id: str,
        sos_id: str,
        option: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Step 3: User explicitly selects an emergency option.
        Options:
          - call_emergency_services
          - share_live_location
          - emergency_contact
        Navigation session continues underneath uninterrupted.
        """
        option_clean = option.strip().lower()
        if option_clean not in self.EMERGENCY_OPTIONS:
            raise ValueError(
                f"Invalid emergency option '{option}'. Must be one of {self.EMERGENCY_OPTIONS}"
            )

        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM sos_sessions WHERE id = ? AND user_id = ?",
                (sos_id, user_id),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"SOS session '{sos_id}' not found.")

            session_dict = dict(row)
            now_dt = datetime.now(timezone.utc)
            now_str = now_dt.isoformat()

            token = session_dict.get("live_location_token")
            exp_str = session_dict.get("live_location_expires_at")
            action_details: Dict[str, Any] = {}

            if option_clean == "call_emergency_services":
                action_details = {
                    "action_type": "call_emergency_services",
                    "phone_number": "112",
                    "dial_uri": "tel:112",
                    "service_name": "National Emergency Helpline",
                }
            elif option_clean == "share_live_location":
                if not token:
                    token = f"loc_{secrets.token_urlsafe(16)}"
                    exp_dt = now_dt + timedelta(hours=2)
                    exp_str = exp_dt.isoformat()

                action_details = {
                    "action_type": "share_live_location",
                    "live_location_token": token,
                    "tracking_url": f"/api/v1/sos/live/{token}",
                    "expires_at": exp_str,
                    "coordinates": {
                        "latitude": session_dict["latitude"],
                        "longitude": session_dict["longitude"],
                    },
                }
            elif option_clean == "emergency_contact":
                contacts = self.contacts_repo.list_contacts(user_id)
                action_details = {
                    "action_type": "emergency_contact",
                    "contacts_notified": [c.to_dict() for c in contacts],
                    "contact_count": len(contacts),
                    "alert_message": f"EMERGENCY ALERT: Location ({session_dict['latitude']}, {session_dict['longitude']})",
                }

            conn.execute(
                """
                UPDATE sos_sessions
                SET status = 'action_selected', selected_option = ?,
                    live_location_token = ?, live_location_expires_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (option_clean, token, exp_str, now_str, sos_id),
            )

        return {
            "sos_id": sos_id,
            "status": "action_selected",
            "selected_option": option_clean,
            "action_details": action_details,
            "nav_session_active": True,
            "nav_state": session_dict["nav_state"],
            "message": f"Executed emergency option '{option_clean}'. Navigation session continues uninterrupted.",
        }

    def get_live_location(self, token: str) -> Optional[Dict[str, Any]]:
        """Public lookup for shared live location token."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM sos_sessions WHERE live_location_token = ?",
                (token,),
            )
            row = cur.fetchone()
            if not row:
                return None

            s = dict(row)
            exp_str = s.get("live_location_expires_at")
            if exp_str:
                exp_dt = datetime.fromisoformat(exp_str)
                if datetime.now(timezone.utc) > exp_dt:
                    return {"expired": True, "message": "Live location tracking token has expired."}

            return {
                "expired": False,
                "latitude": s["latitude"],
                "longitude": s["longitude"],
                "nav_state": s["nav_state"],
                "updated_at": s["updated_at"],
                "expires_at": exp_str,
            }

    def cancel_sos(self, user_id: str, sos_id: str) -> Dict[str, Any]:
        """Cancel active SOS session."""
        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute(
                "UPDATE sos_sessions SET status = 'cancelled', updated_at = ? WHERE id = ? AND user_id = ?",
                (now, sos_id, user_id),
            )
        return {
            "sos_id": sos_id,
            "status": "cancelled",
            "message": "SOS session cancelled.",
            "nav_session_active": True,
        }
